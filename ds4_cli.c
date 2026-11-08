#include "ds4.h"
int ds4_session_llms_rerun_rows(ds4_session *s, const int *tokens, int k, float *row_logits);   /* llms PPL_REPEAT */
void ds4_llms_elide_sweep_set(int layer);   /* llms ELIDE_SWEEP */
int ds4_gpu_llms_alt_off(const char *name);   /* llms PPL_REPEAT */
#include "ds4_distributed.h"
#include "ds4_gpu_args.h"
#include "ds4_tp.h"
#include "ds4_help.h"
#include "ds4_prompt_prefix.h"
#include "linenoise.h"

/* ds4 CLI.
 *
 * One-shot mode builds a single model-family chat prompt and exits.  Interactive
 * mode keeps a rendered token transcript plus one ds4_session, so follow-up
 * turns reuse the live Metal KV checkpoint just like the server does.  The CLI
 * deliberately keeps policy here and leaves graph/cache mechanics inside the
 * engine API. */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

static bool cli_env_flag_enabled(const char *name, bool defval) {
    const char *v = getenv(name);
    if (!v || !v[0]) return defval;
    return strcmp(v, "0") != 0;
}

static bool cli_splitkv_spec_requested(void) {
    if (cli_env_flag_enabled("DS4_CUDA_NO_SPLITKV_SPEC", false)) return false;
    return cli_env_flag_enabled("DS4_CUDA_SPLITKV_SPEC", false);
}

static bool cli_greedy_fast_attention_requested(void) {
    if (!cli_env_flag_enabled("DS4_CUDA_NO_GREEDY_SPLITKV", false) &&
        cli_env_flag_enabled("DS4_CUDA_GREEDY_SPLITKV", false))
    {
        return true;
    }
    if (!cli_env_flag_enabled("DS4_CUDA_NO_GREEDY_VEC4", false) &&
        cli_env_flag_enabled("DS4_CUDA_GREEDY_VEC4", false))
    {
        return true;
    }
    return false;
}

static bool cli_greedy_argmax_requested(bool speculative_requested) {
    if (cli_greedy_fast_attention_requested()) return true;
    if (speculative_requested) return false;
    return cli_env_flag_enabled("DS4_CUDA_GREEDY_TOP1", true);
}

typedef struct {
    const char *prompt;
    const char *system;
    ds4_prompt_prefix prefix;
    bool raw_prompt;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    bool temperature_set;
    bool top_p_set;
    bool min_p_set;
    uint64_t seed;
    bool dump_tokens;
    const char *dump_logits_path;
    const char *dump_logprobs_path;
    int dump_logprobs_top_k;
    int decode_consistency_tokens;
    const char *perplexity_file_path;
    const char *imatrix_dataset_path;
    const char *imatrix_output_path;
    int imatrix_max_prompts;
    int imatrix_max_tokens;
    int imatrix_min_expert_samples;
    ds4_think_mode think_mode;
    bool head_test;
    bool first_token_test;
    bool metal_graph_test;
    bool metal_graph_full_test;
    bool metal_graph_prompt_test;
} cli_generation_options;

typedef struct {
    ds4_engine_options engine;
    ds4_dist_options *dist;
    cli_generation_options gen;
    char *prompt_owned;
    bool inspect;
    /* CLI flag wiring: raw argv values for --gpu-vram and --gpu-devices.
     * Resolved post-parse via parse_gpu_vram_arg(). */
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
} cli_config;

static volatile sig_atomic_t cli_interrupted;
static volatile sig_atomic_t cli_dist_busy;
static volatile sig_atomic_t cli_dist_notice_printed;

static const char cli_dist_drain_msg[] =
    "\nds4: stopping after the distributed cluster finishes the current token/chunk...\n";

static void cli_sigint_handler(int sig) {
    (void)sig;
    cli_interrupted = 1;
    if (cli_dist_busy && !cli_dist_notice_printed) {
        cli_dist_notice_printed = 1;
        ssize_t ignored = write(STDERR_FILENO,
                                cli_dist_drain_msg,
                                sizeof(cli_dist_drain_msg) - 1u);
        (void)ignored;
    }
}

static bool cli_interrupt_requested(void) {
    return cli_interrupted != 0;
}

static void cli_interrupt_clear(void) {
    cli_interrupted = 0;
    cli_dist_notice_printed = 0;
}

static bool cli_distributed_coordinator(const cli_config *cfg) {
    return cfg && cfg->engine.distributed.role == DS4_DISTRIBUTED_COORDINATOR;
}

static void cli_dist_busy_set(const cli_config *cfg, bool busy) {
    if (!cli_distributed_coordinator(cfg)) return;
    cli_dist_busy = busy ? 1 : 0;
    if (!busy) cli_dist_notice_printed = 0;
}

static int cli_wait_distributed_route(const cli_config *cfg, ds4_session *session) {
    if (!cli_distributed_coordinator(cfg)) return 0;

    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }

        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_DS4, topic);
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4: invalid backend: %s\n", s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4: valid backends are: metal, cuda, cpu\n");
#endif
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk,
                               bool        ssd_streaming) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill_mode(backend,
                                                      ctx_size,
                                                      prefill_chunk,
                                                      ssd_streaming);
    const uint64_t kv_bytes = m.raw_bytes + m.compressed_bytes;
    const uint64_t total_bytes = kv_bytes + m.scratch_bytes;
    const bool color = ds4_log_is_tty(stderr);
    const char *green = color ? "\x1b[32m" : "";
    const char *bright_green = color ? "\x1b[1;32m" : "";
    const char *reset = color ? "\x1b[0m" : "";
    fprintf(stderr,
            "%sds4: memory: KV %.2f GiB (raw %.2f + compressed %.2f) "
            "+ buffers %.2f GiB = %s%.2f GiB context%s\n",
            green,
            (double)kv_bytes / 1073741824.0,
            (double)m.raw_bytes / 1073741824.0,
            (double)m.compressed_bytes / 1073741824.0,
            (double)m.scratch_bytes / 1073741824.0,
            bright_green,
            (double)total_bytes / 1073741824.0,
            reset);
    fprintf(stderr,
            "%sds4: memory detail: ctx=%d prefill_cap=%u raw_kv_rows=%u "
            "compressed_kv_rows=%u backend=%s%s\n",
            green,
            ctx_size,
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap,
            ds4_backend_name(backend),
            reset);
}

static ds4_think_mode cli_effective_think_mode(const cli_generation_options *gen) {
    return ds4_think_mode_for_context(gen->think_mode, gen->ctx_size);
}

static bool cli_think_max_downgraded(const cli_generation_options *gen) {
    return gen->think_mode == DS4_THINK_MAX &&
           cli_effective_think_mode(gen) != DS4_THINK_MAX;
}

static void cli_warn_think_max_downgraded(const cli_generation_options *gen, const char *name) {
    if (!cli_think_max_downgraded(gen)) return;
    ds4_log(stderr,
        DS4_LOG_WARNING,
        "ds4: warning: %s needs --ctx >= %u; ctx=%d uses normal thinking instead\n",
        name,
        ds4_think_max_min_context(),
        gen->ctx_size);
}

static double cli_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static char *read_prompt_file(const char *path, bool fatal);

typedef struct {
    int base_tokens;
    int input_tokens;
    bool use_color;
    bool finished;
} cli_prefill_progress;

static void cli_prefill_progress_cb(void *ud, const char *event, int current, int total) {
    (void)total;
    cli_prefill_progress *p = ud;
    if (!p || !event || p->input_tokens <= 0) return;
    const bool is_display =
        strcmp(event, "prefill_display") == 0;
    if (strcmp(event, "prefill_chunk") && !is_display) return;
    if (is_display && !p->use_color) return;

    int processed = current - p->base_tokens;
    if (processed < 0) processed = 0;
    if (processed > p->input_tokens) processed = p->input_tokens;
    double pct = 100.0 * (double)processed / (double)p->input_tokens;
    if (pct > 100.0) pct = 100.0;

    const bool complete = processed >= p->input_tokens;
    if (complete && p->finished) return;

    if (p->use_color) {
        fputc('\r', stderr);
        ds4_log(stderr,
                DS4_LOG_PREFILL,
                "processing %d input tokens: %d/%d (%.1f%%)",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct);
        fputs("\x1b[K", stderr);
        if (complete) fputc('\n', stderr);
    } else {
        fprintf(stderr,
                "processing %d input tokens: %d/%d (%.1f%%)\n",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct);
    }
    if (complete) p->finished = true;
    fflush(stderr);
}

static bool is_rendered_chat_prompt(const char *prompt) {
    static const char *prefixes[] = {
        "<｜begin▁of▁sentence｜>",
        "<｜User｜>",
        "[gMASK]",
        "<sop>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
    };
    if (!prompt) return false;
    for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) {
        size_t n = strlen(prefixes[i]);
        if (strncmp(prompt, prefixes[i], n) == 0) return true;
    }
    return false;
}

typedef struct {
    ds4_engine *engine;
    FILE *fp;
    bool format_thinking;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    char pending[16];
    size_t pending_len;
} token_printer;

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void token_printer_set_grey(token_printer *p) {
    if (p->use_color && !p->color_open) {
        fputs("\x1b[90m", p->fp);
        p->color_open = true;
    }
}

static void token_printer_reset_color(token_printer *p) {
    if (p->use_color && p->color_open) {
        fputs("\x1b[0m", p->fp);
        p->color_open = false;
    }
}

static void token_printer_write_char(token_printer *p, char c) {
    if (p->in_think) token_printer_set_grey(p);
    fputc((unsigned char)c, p->fp);
    p->last_output_newline = c == '\n';
}

static void token_printer_process(token_printer *p, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = p->pending_len + len;
    char *buf = malloc(total ? total : 1);
    if (!buf) return;
    if (p->pending_len) memcpy(buf, p->pending, p->pending_len);
    if (len) memcpy(buf + p->pending_len, text, len);
    p->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        const size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, think_open)) {
            p->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, think_close)) {
            p->in_think = false;
            token_printer_reset_color(p);
            if (!p->last_output_newline) {
                fputc('\n', p->fp);
                p->last_output_newline = true;
            }
            i += strlen(think_close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(p->pending)) {
                memcpy(p->pending, cur, rem);
                p->pending_len = rem;
            }
            break;
        }
        token_printer_write_char(p, cur[0]);
        i++;
    }

    free(buf);
}

static void token_printer_finish(token_printer *p) {
    if (p->format_thinking) {
        token_printer_process(p, NULL, 0, true);
        token_printer_reset_color(p);
    }
    fflush(p->fp);
}

static void generation_done(void *ud) {
    token_printer *p = ud;
    token_printer_finish(p);
    if (!p->last_output_newline) {
        fputc('\n', p->fp);
        p->last_output_newline = true;
    }
    fflush(p->fp);
}

/* LLMS_DS4_CLI_RAW_TRACE: measurement only, off unless a new file path is set. */
static FILE *llms_cli_raw_fp; static uint64_t llms_cli_raw_tokens;
static void llms_cli_raw_close(void) {
    if (llms_cli_raw_fp) { if (fclose(llms_cli_raw_fp)) { perror("llms cli raw close"); _exit(2); }
        fprintf(stderr, "llms cli raw: %llu token-byte records\n", (unsigned long long)llms_cli_raw_tokens); }
}
static void llms_cli_raw_write(const char *text, size_t len) {
    static int initialized;
    if (!initialized) { initialized = 1; const char *path = getenv("LLMS_DS4_CLI_RAW_TRACE");
        if (path && *path) { int fd = open(path, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
            if (fd < 0 || !(llms_cli_raw_fp = fdopen(fd, "wb"))) { perror("llms cli raw open"); exit(2); }
            if (fwrite("D4RAW1\n", 7, 1, llms_cli_raw_fp) != 1) { perror("llms cli raw header"); exit(2); }
            atexit(llms_cli_raw_close); } }
    if (!llms_cli_raw_fp) return;
    if (len > UINT32_MAX) { fprintf(stderr, "llms cli raw: oversized token piece\n"); exit(2); }
    const uint32_t n = (uint32_t)len;
    if (fwrite(&n, sizeof n, 1, llms_cli_raw_fp) != 1 || (len && fwrite(text, len, 1, llms_cli_raw_fp) != 1) || fflush(llms_cli_raw_fp)) {
        perror("llms cli raw record"); exit(2); }
    llms_cli_raw_tokens++;
}
static void token_printer_write_text(token_printer *p, const char *text, size_t len) {
    llms_cli_raw_write(text, len);
    if (p->format_thinking) {
        token_printer_process(p, text, len, false);
    } else if (len) {
        fwrite(text, 1, len, p->fp);
        p->last_output_newline = text[len - 1] == '\n';
    }
}

static void print_generated_token(void *ud, int token) {
    token_printer *p = ud;
    size_t len = 0;
    char *text = ds4_token_text(p->engine, token, &len);
    token_printer_write_text(p, text, len);
    fflush(p->fp);
    free(text);
}

static void build_chat_prompt(ds4_engine *engine,
                              const cli_generation_options *gen,
                              ds4_tokens *out) {
    const ds4_think_mode think_mode = cli_effective_think_mode(gen);
    ds4_chat_begin(engine, out);
    ds4_chat_append_think_prefix(engine, out, think_mode);
    if (gen->system && gen->system[0])
        ds4_chat_append_message(engine, out, "system", gen->system);
    ds4_prompt_prefix_append(engine, out, &gen->prefix);
    ds4_chat_append_message(engine, out, "user", gen->prompt ? gen->prompt : "");
    ds4_chat_append_assistant_prefix(engine, out, think_mode);
}

static void llms_cov_init(const char *prompt);   /* llms THINK_COVER: defined below */
static void build_prompt(ds4_engine *engine, const cli_generation_options *gen, ds4_tokens *out) {
    llms_cov_init(gen->prompt);   /* llms THINK_COVER: parses the finding tags of the prompt once, when the switch is set */
    if (gen->raw_prompt) {
        ds4_tokenize_text(engine, gen->prompt ? gen->prompt : "", out);
    } else if (is_rendered_chat_prompt(gen->prompt)) {
        ds4_tokenize_rendered_chat(engine, gen->prompt, out);
    } else if (gen->prefix.count == 0) {
        ds4_encode_chat_prompt(engine, gen->system, gen->prompt,
                               cli_effective_think_mode(gen), out);
    } else {
        build_chat_prompt(engine, gen, out);
    }
}

static void cli_apply_model_sampling_defaults(
        ds4_engine             *engine,
        cli_generation_options *gen) {
    if (!engine || !gen || !ds4_engine_is_glm_dsa(engine)) return;

    if (!gen->temperature_set) gen->temperature = 1.0f;
    if (!gen->top_p_set) gen->top_p = 0.95f;
    if (!gen->min_p_set) gen->min_p = 0.0f;
}

/* llms THINK_SCRUB (apply_think_scrub.py): the decode's rows are provisional; every W tokens of open thinking the state since the last scrub is rebuilt by the prefill path (restore the snapshot, re-prefill, snapshot again) */
static int llms_scrub_w = -1, llms_scrub_n, llms_scrub_next, llms_scrub_have; static int llms_scrub_native = -1;   /* llms THINK_SCRUB_NATIVE */ static int llms_scrub_close = 0, llms_scrub_last = -1; static long llms_scrub_budget = -1;   /* llms TWOSPEED */ static ds4_session_snapshot llms_scrub_snap; static double llms_scrub_s, llms_scrub_aux_s; static long llms_scrub_rows;
static void llms_scrub_init(void) {
    if (llms_scrub_w >= 0) return;
    const char *e = getenv("LLMS_DS4_THINK_SCRUB"); llms_scrub_w = e && *e ? atoi(e) : 0; if (llms_scrub_w < 0) llms_scrub_w = 0;
    { const char *nv = getenv("LLMS_DS4_THINK_SCRUB_NATIVE"); llms_scrub_native = nv && *nv && *nv != '0'; if (llms_scrub_w && llms_scrub_native) fprintf(stderr, "llms: think scrub native (a graph-position rollback keeps the rows below the mark; no session snapshot) [LLMS_DS4_THINK_SCRUB_NATIVE]\n"); }
    { const char *cv = getenv("LLMS_DS4_THINK_SCRUB_CLOSE"); llms_scrub_close = cv && *cv && *cv != '0' && llms_scrub_w; if (llms_scrub_close) { const char *bv = getenv("LLMS_DS4_THINK_BUDGET"); llms_scrub_budget = bv && *bv ? atol(bv) : -1; fprintf(stderr, "llms: think scrub close (a sampled </think> and the budget's forced close commit first; the close is re-sampled from the committed state) [LLMS_DS4_THINK_SCRUB_CLOSE]\n"); } }
    if (llms_scrub_w) { fprintf(stderr, "llms: think scrub %d (while the thinking is open, every %d generated tokens the state since the last scrub is rebuilt by the prefill path: restore the snapshot, re-prefill, snapshot again; the decode's rows are provisional) -- FIDELITY DIAL [LLMS_DS4_THINK_SCRUB]\n", llms_scrub_w, llms_scrub_w);
        atexit_b(^{ fprintf(stderr, "llms think scrub: %d scrubs, %ld rows re-prefilled in %.1f s, restore + snapshot %.1f s\n", llms_scrub_n, llms_scrub_rows, llms_scrub_s, llms_scrub_aux_s); }); }
}
/* llms IND_PEN (apply_ind_pen.py): a table from the last 4 generated tokens to the token that followed them last time */
#define IND_TAB (1u << 16)
static uint64_t ind_key[IND_TAB]; static int ind_val[IND_TAB]; static int ind_last[4]; static uint32_t ind_n; static float ind_pen = -1.0f; static unsigned long long ind_hits;
static uint64_t ind_gram(void) { uint64_t h = 1469598103934665603ull; for (uint32_t i = 0; i < 4; i++) { h ^= (uint64_t)(uint32_t)ind_last[(ind_n + i) & 3] + 1u; h *= 1099511628211ull; } return h | 1ull; }
static void ind_feed(int tok) { if (ind_n >= 4) { const uint64_t k = ind_gram(); uint32_t s = (uint32_t)(k >> 40) & (IND_TAB - 1); for (int p = 0; p < 16; p++, s = (s + 1) & (IND_TAB - 1)) if (!ind_key[s] || ind_key[s] == k) { ind_key[s] = k; ind_val[s] = tok; break; } } ind_last[ind_n & 3] = tok; ind_n++; }
static int ind_lookup(void) { if (ind_n < 4) return -1; const uint64_t k = ind_gram(); uint32_t s = (uint32_t)(k >> 40) & (IND_TAB - 1); for (int p = 0; p < 16; p++, s = (s + 1) & (IND_TAB - 1)) { if (!ind_key[s]) return -1; if (ind_key[s] == k) return ind_val[s]; } return -1; }
/* llms THINK_LOOP / THINK_REREAD (apply_think_loop_reread.py) */
#define TLB_TAB (1u << 18)
#define TLB_WIN 512
static uint64_t tlb_tab[TLB_TAB]; static uint8_t tlb_ring[TLB_WIN]; static int tlb_hist[8]; static uint32_t tlb_n, tlb_pos, tlb_cnt, tlb_grams; static uint32_t tlb_threshold = (TLB_WIN * 9u) / 10u;
static void tlb_reset(void) { memset(tlb_tab, 0, sizeof tlb_tab); memset(tlb_ring, 0, sizeof tlb_ring); tlb_n = tlb_pos = tlb_cnt = tlb_grams = 0; }
static int tlb_feed(int tok) {
    tlb_hist[tlb_n & 7u] = tok; tlb_n++;
    if (tlb_n < 8u) return 0;
    uint64_t h = 1469598103934665603ull;
    for (uint32_t k = tlb_n - 8u; k < tlb_n; k++) { h ^= (uint64_t)(uint32_t)tlb_hist[k & 7u] + 1ull; h *= 1099511628211ull; }
    h |= 1ull;
    uint32_t i = (uint32_t)(h >> 20) & (TLB_TAB - 1u); int seen = 0;
    for (uint32_t pr = 0; pr < 64u; pr++, i = (i + 1u) & (TLB_TAB - 1u)) { if (tlb_tab[i] == h) { seen = 1; break; } if (tlb_tab[i] == 0) { tlb_tab[i] = h; break; } }
    tlb_cnt -= tlb_ring[tlb_pos]; tlb_ring[tlb_pos] = (uint8_t)seen; tlb_cnt += (uint32_t)seen; tlb_pos = (tlb_pos + 1u) % TLB_WIN; tlb_grams++;
    if (tlb_grams < TLB_WIN) return 0;
    return tlb_cnt >= tlb_threshold;
}
#include <strings.h>
static char llms_stop_prefix[64]; static int llms_stop_on = -1; static char llms_sl[256]; static int llms_sl_len;   /* llms STOP_LINE */
static int llms_stop_feed(const char *p, size_t n) {   /* 1 when a completed line begins with the prefix */
    for (size_t i = 0; i < n; i++) {
        const char c = p[i];
        if (c == '\n') { int k = 0; while (k < llms_sl_len && (llms_sl[k] == ' ' || llms_sl[k] == '*' || llms_sl[k] == '_')) k++;
            const size_t pl = strlen(llms_stop_prefix); const int hit = (size_t)(llms_sl_len - k) >= pl && strncasecmp(llms_sl + k, llms_stop_prefix, pl) == 0; llms_sl_len = 0; if (hit) return 1; }
        else if (llms_sl_len < (int)sizeof llms_sl) llms_sl[llms_sl_len++] = c;
    }
    return 0;
}
static int llms_cli_inside = -1;   /* the thinking span is open (was a block static of the CLI_THINK_BUDGET block) */
static int llms_loop_on = -1, llms_loop_fired, llms_reread_on = -1, llms_think_reread_pending;
static int llms_wrap_tok[256], llms_wrap_n, llms_wrap_cur = -1;
static int llms_wrap_uncap; static unsigned long long llms_wrap_extra, llms_wrap_forced_total;   /* llms THINK_WRAP singleton cap: the control knob, the extra-token counter, the forced-token count */   /* llms THINK_WRAP: the wrap-up text forced inside the thinking before a forced close; cur -1 = not started, -2 = no text */
static int llms_wrap_start(ds4_engine *engine) {   /* llms THINK_WRAP: tokenize the text once; 1 = a wrap starts now */
    const char *fp = getenv("LLMS_DS4_THINK_WRAP"); char buf[1024]; size_t n = 0;
    if (!(fp && *fp)) { llms_wrap_cur = -2; return 0; }
    for (const char *q = fp; *q && n + 1 < sizeof buf; q++) { if (q[0] == '\\' && q[1] == 'n') { buf[n++] = '\n'; q++; } else buf[n++] = *q; }
    buf[n] = 0;
    ds4_tokens tv = {0}; ds4_tokenize_text(engine, buf, &tv);
    llms_wrap_n = 0; for (int k = 0; k < tv.len && k < 256; k++) llms_wrap_tok[llms_wrap_n++] = tv.v[k];
    ds4_tokens_free(&tv);
    if (llms_wrap_n <= 0) { llms_wrap_cur = -2; return 0; }
    llms_wrap_cur = 0; fprintf(stderr, "llms: think wrap (%d tokens forced inside the thinking before the forced close) -- FIDELITY DIAL\n", llms_wrap_n);
    { const char *u = getenv("LLMS_DS4_WRAP_UNCAP"); llms_wrap_uncap = u && *u && *u != '0'; atexit_b(^{ fprintf(stderr, "llms wrap: %llu tokens forced, %llu extra tokens emitted by forced iterations (singleton cap %s)\n", llms_wrap_forced_total, llms_wrap_extra, llms_wrap_uncap ? "OFF" : "on"); }); }   /* llms THINK_WRAP singleton cap */
    return 1;
}
/* llms FINAL_LOOP (apply_final_loop.py): the thinking's LOOPBREAK rule on the FINAL channel; mode 1 = stop the answer, 2 = force '\n\nFLAWS:' */
static int llms_fl_mode, llms_fl_fired, llms_fl_done, llms_fl_started, llms_fl_tok[16], llms_fl_n, llms_fl_cur = -1; static unsigned long long llms_fl_forced;
static int llms_fl_feed(int tok) { if (!llms_fl_started) { tlb_reset(); llms_fl_started = 1; } return tlb_feed(tok); }
/* llms FINAL_PROGRESS (apply_final_progress.py): force the FLAWS: opening when N answer tokens pass without a NEW 'N<digits>' line-start tag; needs FINAL_LOOP */
static long llms_fp_n = -2, llms_fp_idle; static int llms_fp_state, llms_fp_num, llms_fp_fired; static unsigned char llms_fp_seen[1024];
static void llms_fp_init(void) { if (llms_fp_n != -2) return; const char *e = getenv("LLMS_DS4_FINAL_PROGRESS"); llms_fp_n = (e && *e) ? atol(e) : 0; if (llms_fp_n > 0) fprintf(stderr, "llms: final progress %ld (an answer with no new N<digits> line-start tag for %ld tokens forces the FLAWS: opening; needs FINAL_LOOP) -- FIDELITY DIAL\n", llms_fp_n, llms_fp_n); }
static void llms_fp_tag(int num) { if (num >= 0 && num < 1024 && !llms_fp_seen[num]) { llms_fp_seen[num] = 1; llms_fp_idle = 0; } }
static void llms_fp_feed(const char *p, size_t n) {   /* one piece = one token */
    llms_fp_init(); if (llms_fp_n <= 0) return; llms_fp_idle++;
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)p[i]; const int digit = c >= '0' && c <= '9', word = digit || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' || c >= 128;
        if (llms_fp_state == 3) {
            if (digit) { llms_fp_num = llms_fp_num < 100000 ? llms_fp_num * 10 + (c - '0') : llms_fp_num; continue; }
            if (!word) llms_fp_tag(llms_fp_num);
            llms_fp_state = (c == '\n') ? 0 : 1; continue;
        }
        if (c == '\n') { llms_fp_state = 0; continue; }
        if (llms_fp_state == 0) { if (!word) continue; llms_fp_state = (c == 'N') ? 2 : 1; continue; }
        if (llms_fp_state == 2) { if (digit) { llms_fp_state = 3; llms_fp_num = c - '0'; } else llms_fp_state = 1; continue; }
    }
}
static int llms_fp_hit(void) {   /* 1 once, when N answer tokens have passed since the last new tag */
    if (llms_fp_n <= 0 || llms_fp_fired || llms_fp_idle < llms_fp_n) return 0;
    llms_fp_fired = 1; fprintf(stderr, "llms: final progress: %ld answer tokens since the last new tag line, the FLAWS: opening is forced\n", llms_fp_idle); return 1;
}

static int llms_fl_start(ds4_engine *engine) {
    ds4_tokens tv = {0}; ds4_tokenize_text(engine, "\n\nFLAWS:", &tv);
    llms_fl_n = 0; for (int k = 0; k < tv.len && k < 16; k++) llms_fl_tok[llms_fl_n++] = tv.v[k];
    ds4_tokens_free(&tv); llms_fl_cur = llms_fl_n > 0 ? 0 : -2;
    fprintf(stderr, "llms: final loop: %d tokens of the FLAWS: opening forced\n", llms_fl_n);
    return llms_fl_n > 0;
}

/* llms THINK_COVER (apply_think_cover.py): coverage-driven thinking: close g tokens after every finding tag of the prompt opened a line, at the next line end; nudge the loop rule onto the lowest unanalysed finding up to m times; off unless LLMS_DS4_THINK_COVER is set */
static long llms_cov_g = -2, llms_cov_at = -1, llms_cov_min = 0, llms_cov_nl_at_g = -1, llms_cov_nudge_max, llms_cov_nudges, llms_cov_pace, llms_cov_idle; static int llms_cov_n, llms_cov_state, llms_cov_num, llms_cov_seen_n, llms_cov_fired, llms_cov_nudge_tok[48], llms_cov_nudge_n, llms_cov_nudge_cur = -1; static unsigned char llms_cov_seen[1024], llms_cov_inprompt[1024];
static int llms_cov_isw(int c) { return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' || c >= 128; }
static void llms_cov_init(const char *prompt) {
    if (llms_cov_g != -2) return;
    const char *e = getenv("LLMS_DS4_THINK_COVER"), *m = getenv("LLMS_DS4_THINK_COVER_NUDGE"), *mn = getenv("LLMS_DS4_THINK_COVER_MIN");
    llms_cov_g = (e && *e) ? atol(e) : -1; llms_cov_nudge_max = (m && *m) ? atol(m) : 0; llms_cov_min = (mn && *mn) ? atol(mn) : 0; { const char *pc = getenv("LLMS_DS4_THINK_PACE"); llms_cov_pace = (pc && *pc) ? atol(pc) : 0; }
    if (llms_cov_g < 0 || !prompt) { llms_cov_g = -1; return; }
    for (const char *p = prompt; *p; p++) {
        if (*p != 'N' || (p > prompt && llms_cov_isw((unsigned char)p[-1])) || p[1] < '0' || p[1] > '9') continue;
        int v = 0; const char *q = p + 1; while (*q >= '0' && *q <= '9' && v < 100000) { v = v * 10 + (*q - '0'); q++; }
        if (!llms_cov_isw((unsigned char)*q) && v >= 1 && v < 1024) llms_cov_inprompt[v] = 1;
    }
    int n = 0; while (n + 1 < 1024 && llms_cov_inprompt[n + 1]) n++;
    llms_cov_n = n;
    if (n < 3) { fprintf(stderr, "llms: think cover: %d finding tags in the prompt, the rule stays off\n", n); llms_cov_g = -1; return; }
    fprintf(stderr, "llms: think cover (the thinking closes %ld tokens after all %d finding tags of the prompt have opened a line, at the next line end; %ld loop nudges allowed; the loop rule and the budget stay as fallbacks) -- FIDELITY DIAL\n", llms_cov_g, n, llms_cov_nudge_max);
    if (llms_cov_pace > 0) fprintf(stderr, "llms: think pace (%ld thinking tokens without a new finding tag force the next unanalysed finding's tag line, within the nudge budget) -- FIDELITY DIAL\n", llms_cov_pace);
}
static void llms_cov_feed(const char *p, size_t n) {   /* the thinking's pieces, one piece = one token; forced pieces included */
    if (llms_cov_g < 0 || llms_cov_n <= 0) return;
    llms_cov_idle++;   /* THINK_PACE: one piece = one token since the last new finding tag */
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)p[i]; const int digit = c >= '0' && c <= '9', word = llms_cov_isw(c);
        if (llms_cov_state == 3) {
            if (digit) { llms_cov_num = llms_cov_num < 100000 ? llms_cov_num * 10 + (c - '0') : llms_cov_num; continue; }
            if (!word && llms_cov_num >= 1 && llms_cov_num <= llms_cov_n && !llms_cov_seen[llms_cov_num]) { llms_cov_seen[llms_cov_num] = 1; llms_cov_seen_n++; llms_cov_idle = 0; }
            llms_cov_state = (c == '\n') ? 0 : 1; continue;
        }
        if (c == '\n') { llms_cov_state = 0; continue; }
        if (llms_cov_state == 0) { if (!word) continue; llms_cov_state = (c == 'N') ? 2 : 1; continue; }
        if (llms_cov_state == 2) { if (digit) { llms_cov_state = 3; llms_cov_num = c - '0'; } else llms_cov_state = 1; continue; }
    }
}
static int llms_cov_hit(int generated, unsigned long nl_count) {   /* 1 once, when the close is due: coverage complete, g tokens later, at a line end */
    if (llms_cov_g < 0 || llms_cov_fired || llms_cov_n <= 0 || llms_cov_seen_n < llms_cov_n) return 0;
    if (llms_cov_at < 0) llms_cov_at = generated;
    if (generated < llms_cov_min || generated - llms_cov_at < llms_cov_g) return 0;
    if (llms_cov_nl_at_g < 0) llms_cov_nl_at_g = (long)nl_count;
    if ((long)nl_count <= llms_cov_nl_at_g && generated - llms_cov_at < llms_cov_g + 300) return 0;
    llms_cov_fired = 1; fprintf(stderr, "llms: think cover fired at completion %d: all %d finding tags opened by completion %ld, grace %ld\n", generated, llms_cov_n, llms_cov_at, llms_cov_g); return 1;
}
static int llms_cov_nudge_ok(void) { return llms_cov_g >= 0 && llms_cov_n > 0 && llms_cov_nudges < llms_cov_nudge_max && llms_cov_seen_n < llms_cov_n && llms_cov_nudge_cur == -1; }
static int llms_cov_pace_due(void) { return llms_cov_pace > 0 && llms_cov_idle >= llms_cov_pace && llms_cov_seen_n < llms_cov_n; }   /* THINK_PACE */
static int llms_cov_lowest_unseen(void) { for (int k = 1; k <= llms_cov_n; k++) if (!llms_cov_seen[k]) return k; return 0; }

static int llms_cov_nudge_start(ds4_engine *engine) {   /* copies llms_fl_start: the nudge text is tokenized once per nudge */
    const int k = llms_cov_lowest_unseen(); if (k <= 0) return 0;
    char buf[96]; snprintf(buf, sizeof buf, "\n\nStill open; I move on to the next finding.\nN%d:", k);
    ds4_tokens tv = {0}; ds4_tokenize_text(engine, buf, &tv);
    llms_cov_nudge_n = 0; for (int j = 0; j < tv.len && j < 48; j++) llms_cov_nudge_tok[llms_cov_nudge_n++] = tv.v[j];
    ds4_tokens_free(&tv); llms_cov_nudge_cur = llms_cov_nudge_n > 0 ? 0 : -1;
    fprintf(stderr, "llms: think cover nudge %ld of %ld (%s, %ld tokens since the last new finding tag): %d tokens force finding N%d (%d of %d finding tags opened)\n", llms_cov_nudges + 1, llms_cov_nudge_max, llms_loop_fired ? "loop rule" : "pace rule", llms_cov_idle, llms_cov_nudge_n, k, llms_cov_seen_n, llms_cov_n);
    return llms_cov_nudge_n > 0;
}
static int llms_cg_n, llms_cg_left, llms_cx_n, llms_cx_end = -1;
static long llms_eol_n = -2; static unsigned long llms_nl_count, llms_nl_at_budget; static int llms_nl_marked;   /* llms THINK_BUDGET_EOL */
static int llms_budget_hit(long budget, int generated) {   /* llms THINK_BUDGET_EOL: the raw condition (generated >= budget), unless LLMS_DS4_THINK_BUDGET_EOL=N defers the forced close up to N tokens to the end of the current line */
    if (budget <= 0 || generated < budget) return 0;
    if (llms_eol_n == -2) { const char *e = getenv("LLMS_DS4_THINK_BUDGET_EOL"); llms_eol_n = (e && *e) ? atol(e) : 0; if (llms_eol_n > 0) fprintf(stderr, "llms: think budget eol (the budget's forced close waits up to %ld tokens for the end of the line) -- FIDELITY DIAL\n", llms_eol_n); }
    if (llms_eol_n <= 0) return 1;
    if (!llms_nl_marked) { llms_nl_marked = 1; llms_nl_at_budget = llms_nl_count; }
    return llms_nl_count > llms_nl_at_budget || generated >= budget + llms_eol_n;
}
static int llms_ew_from = -1, llms_ew_on, llms_ew_done;   /* llms ESC_WINDOW: exact routing from completion G until the close */   /* llms CLOSE_GREEDY / CLOSE_EXACT */
static int llms_probe_k, llms_isolate_now;   /* llms CLOSE_PROBE; the close token is evaluated alone */
static void llms_probe_print(ds4_engine *engine, ds4_session *session, const char *tag) {
    ds4_token_score sc[32]; const int k = llms_probe_k > 32 ? 32 : llms_probe_k; const int n = ds4_session_top_logprobs(session, sc, k);
    fprintf(stderr, "llms close probe [%s]:", tag);
    for (int i = 0; i < n; i++) { size_t len = 0; char *txt = ds4_token_text(engine, sc[i].id, &len); fprintf(stderr, " %d:%.4f:'", sc[i].id, sc[i].logprob);
        for (size_t q = 0; txt && q < len && q < 14; q++) { const unsigned char c = (unsigned char)txt[q]; fputc(c == '\n' ? '~' : (c == '\'' ? '`' : (c < 32 ? '?' : c)), stderr); } fputc('\'', stderr); free(txt); }
    fputc('\n', stderr);
}
static void llms_snap_dump(ds4_session *session, const char *tag) {   /* llms CLOSE_SNAPSHOT: the session payload (window rows, compressed latents, index keys) to <dir>/<tag>.kv */
    const char *dir = getenv("LLMS_DS4_CLOSE_SNAPSHOT"); if (!dir || !*dir) return;
    ds4_session_snapshot sn = {0}; char serr[160]; serr[0] = 0;
    if (ds4_session_save_snapshot(session, &sn, serr, sizeof serr) != 0) { fprintf(stderr, "llms close snapshot [%s]: failed (%s)\n", tag, serr); return; }
    char path[1024]; snprintf(path, sizeof path, "%s/%s.kv", dir, tag); FILE *sf = fopen(path, "wb");
    if (sf) { fwrite(sn.ptr, 1, (size_t)sn.len, sf); fclose(sf); fprintf(stderr, "llms close snapshot [%s]: %llu bytes -> %s\n", tag, (unsigned long long)sn.len, path); }
    else fprintf(stderr, "llms close snapshot [%s]: cannot open %s\n", tag, path);
    ds4_session_snapshot_free(&sn);
}
static void llms_think_switches_init(void) {
    if (llms_loop_on >= 0) return;
    { const char *e = getenv("LLMS_DS4_THINK_LOOP"); llms_loop_on = e && *e && *e != '0'; if (llms_loop_on) { tlb_reset(); fprintf(stderr, "llms: think loop (LOOPBREAK: the last 512 token 8-grams >= 90 %% repeats close the thinking) -- FIDELITY DIAL\n"); } }
    { const char *e = getenv("LLMS_DS4_STOP_LINE"); llms_stop_on = e && *e && *e != '0'; if (llms_stop_on) { snprintf(llms_stop_prefix, sizeof llms_stop_prefix, "%s", e); fprintf(stderr, "llms: stop line (generation stops at the first completed line that begins with '%s' after the thinking has closed) -- FIDELITY DIAL\n", llms_stop_prefix); } }
    { const char *e = getenv("LLMS_DS4_CLOSE_GREEDY"); llms_cg_n = e && *e ? atoi(e) : 0; if (llms_cg_n < 0) llms_cg_n = 0; if (llms_cg_n) fprintf(stderr, "llms: close greedy %d (after the thinking has closed the first token of each of the next %d decode cycles is the argmax) -- FIDELITY DIAL\n", llms_cg_n, llms_cg_n); }
    { const char *e = getenv("LLMS_DS4_ESC_WINDOW"); if (e && *e) { llms_ew_from = atoi(e); if (llms_ew_from < 0) llms_ew_from = 0; fprintf(stderr, "llms: esc window (exact routing from completion %d until the thinking closes; the shipped dials return at the close) -- FIDELITY DIAL\n", llms_ew_from); } }   /* llms ESC_WINDOW */
        { const char *e = getenv("LLMS_DS4_CLOSE_EXACT"); llms_cx_n = e && *e ? atoi(e) : 0; if (llms_cx_n < 0) llms_cx_n = 0; if (llms_cx_n) fprintf(stderr, "llms: close exact %d (after the thinking has closed the decode runs on exact routing for %d generated tokens) -- FIDELITY DIAL\n", llms_cx_n, llms_cx_n); }
    { const char *e = getenv("LLMS_DS4_CLOSE_PROBE"); llms_probe_k = e && *e ? atoi(e) : 0; if (llms_probe_k < 0) llms_probe_k = 0; if (llms_probe_k) fprintf(stderr, "llms: close probe %d (the top-%d next-token logprobs at the first position after the thinking's close are printed, before and after a rebuild) -- FIDELITY DIAL\n", llms_probe_k, llms_probe_k); }
    { const char *e = getenv("LLMS_DS4_CLOSE_SNAPSHOT"); if (e && *e) fprintf(stderr, "llms: close snapshot (the session payload at the close is written to %s/dec.kv before a rebuild, reb.kv after it, pre.kv for a prompt that ends on the close) -- INSTRUMENT\n", e); }
    { const char *e = getenv("LLMS_DS4_THINK_REREAD"); llms_reread_on = e && *e && *e != '0'; if (llms_reread_on) fprintf(stderr, "llms: think reread (at the thinking's close the session state is rebuilt by a full prefill of its own tokens) -- FIDELITY DIAL\n"); }
    { const char *e = getenv("LLMS_DS4_FINAL_LOOP"); llms_fl_mode = (e && !strcmp(e, "stop")) ? 1 : (e && !strcmp(e, "flaws")) ? 2 : 0; if (llms_fl_mode) { fprintf(stderr, "llms: final loop (%s: the last 512 token 8-grams >= 90 %% repeats after the thinking has closed %s) -- FIDELITY DIAL\n", e, llms_fl_mode == 1 ? "stop the answer" : "force a FLAWS: line"); atexit_b(^{ fprintf(stderr, "llms final loop: fired %d, FLAWS tokens forced %llu\n", llms_fl_fired, llms_fl_forced); }); } }   /* llms FINAL_LOOP */
}
static int run_sampled_generation(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: sampled CLI generation requires a session backend\n");
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        return 1;
    }
    /* Pay the one-time first-submission GPU cost before the prefill timer
     * starts (matches the TP worker's startup warmup). */
    ds4_session_gpu_warmup(session);

    char err[160];
    ds4_think_mode think_mode = cli_effective_think_mode(&cfg->gen);
    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };

    const double t_prefill0 = cli_now_sec();
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    cli_dist_busy_set(cfg, true);
    int sync_rc = ds4_session_sync(session, prompt, err, sizeof(err));
    cli_dist_busy_set(cfg, false);
    if (sync_rc != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();
    llms_scrub_init();   /* llms THINK_SCRUB: the first snapshot is the prompt's own prefill-built state */
    if (llms_scrub_w > 0) { char serr[160]; serr[0] = 0; llms_scrub_have = llms_scrub_native ? ds4_session_llms_scrub_mark(session) == 0 : ds4_session_save_snapshot(session, &llms_scrub_snap, serr, sizeof serr) == 0; llms_scrub_next = llms_scrub_w;
        if (!llms_scrub_have) fprintf(stderr, "llms think scrub: the first snapshot failed (%s); the first scrub will rebuild from the prompt\n", serr); }

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    const bool speculative_argmax = cfg->gen.temperature <= 0.0f &&
        ((ds4_engine_mtp_draft_tokens(engine) > 1 &&
          getenv("DS4_MTP_SPEC_DISABLE") == NULL) ||
         cli_splitkv_spec_requested());
    const bool greedy_argmax = cfg->gen.temperature <= 0.0f &&
        cli_greedy_argmax_requested(speculative_argmax);
    bool have_greedy_next = false;
    int greedy_next = -1;
    const double t_decode0 = cli_now_sec();
    llms_think_switches_init();   /* llms CLOSE_PROBE: a prompt that ends on the close token is already a state at the first-token position */
    if (llms_probe_k > 0 && prompt->len > 0 && prompt->v[prompt->len - 1] == ds4_token_think_end(engine)) llms_probe_print(engine, session, "prefilled");
    if (prompt->len > 0 && prompt->v[prompt->len - 1] == ds4_token_think_end(engine)) llms_snap_dump(session, "pre");
    while (generated < max_tokens && !cli_interrupt_requested()) {
        if (llms_cx_end >= 0 && generated >= llms_cx_end) { ds4_session_escalate(session, 0); llms_cx_end = -1; }   /* llms CLOSE_EXACT: the shipped dials return */
        if (llms_ew_from >= 0 && !llms_ew_on && !llms_ew_done && generated >= llms_ew_from) { ds4_session_escalate(session, 1); llms_ew_on = 1; fprintf(stderr, "llms: esc window opens at completion %d\n", generated); }   /* llms ESC_WINDOW */
        if (llms_think_reread_pending) {   /* llms THINK_REREAD: the thinking has closed; rebuild the state by a full prefill of the session's own tokens */
            llms_think_reread_pending = 0;
            const ds4_tokens *ck = ds4_session_tokens(session);
            if (ck && ck->len > 0) {
                ds4_tokens full = {0}; ds4_tokens_copy(&full, ck);
                const double tr0 = cli_now_sec(); char rerr[160];
                ds4_session_invalidate(session);
                cli_dist_busy_set(cfg, true);
                const int rrc = ds4_session_sync(session, &full, rerr, sizeof(rerr));
                cli_dist_busy_set(cfg, false);
                if (rrc != 0) { fprintf(stderr, "ds4: think reread failed: %s\n", rerr); ds4_tokens_free(&full); ds4_session_free(session); return 1; }
                fprintf(stderr, "llms: think reread: %d tokens re-prefilled after the close in %.1f s\n", full.len, cli_now_sec() - tr0);
                ds4_tokens_free(&full);
                if (llms_probe_k > 0) llms_probe_print(engine, session, "rebuilt");
                llms_snap_dump(session, "reb");
            }
        }
        bool llms_wrap_forced_now = false;   /* llms THINK_WRAP singleton cap: only the forced token may be emitted this iteration */
        int token;
        if (greedy_argmax && have_greedy_next) {
            token = greedy_next;
            have_greedy_next = false;
        } else {
            if (ind_pen < 0.0f) { const char *e = getenv("LLMS_DS4_IND_PEN"); ind_pen = e && *e ? (float)atof(e) : 0.0f; if (ind_pen > 0.0f) fprintf(stderr, "llms: IND_PEN %.3f on the single-token path\n", ind_pen); }
            if (ind_pen > 0.0f) { const int it = ind_lookup(); if (it >= 0) { ds4_session_set_next_bias(session, it, -ind_pen); ind_hits++; } }   /* llms IND_PEN */
            token = ds4_session_sample(session, llms_cg_left > 0 ? 0.0f : cfg->gen.temperature, 0,   /* llms CLOSE_GREEDY */
                                       cfg->gen.top_p, cfg->gen.min_p, &rng);
        }
        if (llms_cg_left > 0) llms_cg_left--;
        if (llms_scrub_w > 0 && (generated >= llms_scrub_next || (llms_scrub_close && generated > llms_scrub_last && (token == ds4_token_think_end(engine) || (llms_scrub_budget > 0 && generated == llms_scrub_budget))))) {   /* llms THINK_SCRUB: placed after the sample; a scrub re-samples from the rebuilt state; TWOSPEED: also at the close */
            llms_scrub_last = generated;
            const int llms_scrub_n_before = llms_scrub_n;
            const ds4_tokens *ck = ds4_session_tokens(session); const int te = ds4_token_think_end(engine); int open = 1;
            for (int q = prompt->len; ck && q < ck->len; q++) if (ck->v[q] == te) { open = 0; break; }
            if (!open) { fprintf(stderr, "llms: think scrub retired at completion %d: the thinking has closed\n", generated); llms_scrub_w = 0; }
            else if (ck && ck->len > prompt->len) {
                ds4_tokens full = {0}; for (int q = 0; q < ck->len; q++) ds4_tokens_push(&full, ck->v[q]);
                char serr[160]; const double s0 = cli_now_sec(); int rows = full.len;
                serr[0] = 0;
                if (llms_scrub_have && (llms_scrub_native ? ds4_session_llms_scrub_restore(session) == 0 : ds4_session_load_snapshot(session, &llms_scrub_snap, serr, sizeof serr) == 0)) rows = full.len - ds4_session_pos(session);
                else { if (llms_scrub_have) fprintf(stderr, "llms think scrub: restore failed (%s): a full rebuild\n", serr); ds4_session_invalidate(session); }
                const double s1 = cli_now_sec();
                cli_dist_busy_set(cfg, true); const int src = ds4_session_sync(session, &full, serr, sizeof serr); cli_dist_busy_set(cfg, false);
                if (src != 0) { fprintf(stderr, "ds4: think scrub failed: %s\n", serr); ds4_tokens_free(&full); ds4_session_free(session); return 1; }
                const double s2 = cli_now_sec();
                llms_scrub_have = llms_scrub_native ? ds4_session_llms_scrub_mark(session) == 0 : ds4_session_save_snapshot(session, &llms_scrub_snap, serr, sizeof serr) == 0;
                if (!llms_scrub_have) fprintf(stderr, "llms think scrub: snapshot failed (%s); the next scrub rebuilds from the prompt\n", serr);
                const double s3 = cli_now_sec(); llms_scrub_n++; llms_scrub_rows += rows; llms_scrub_s += s2 - s1; llms_scrub_aux_s += (s1 - s0) + (s3 - s2);
                fprintf(stderr, "llms: think scrub %d at completion %d: %d rows re-prefilled in %.2f s (restore + snapshot %.2f s); state at %d\n", llms_scrub_n, generated, rows, s2 - s1, (s1 - s0) + (s3 - s2), ds4_session_pos(session));
                ds4_tokens_free(&full);
            }
            if (llms_scrub_w > 0) llms_scrub_next = generated + llms_scrub_w;
            if (llms_scrub_n_before != llms_scrub_n && !(greedy_argmax && have_greedy_next)) token = ds4_session_sample(session, cfg->gen.temperature, 0, cfg->gen.top_p, cfg->gen.min_p, &rng);   /* the token comes from the rebuilt state */
        }
        {   /* llms CLI_THINK_BUDGET (apply_cli_think_budget.py): the server's THINK_BUDGET / THINK_FORCE_PREFIX in the CLI */
            static long think_budget_value = -2; static int forced_opening_tokens[64]; static int forced_opening_count = 0, forced_opening_cursor = -1;
            if (think_budget_value == -2) { const char *e = getenv("LLMS_DS4_THINK_BUDGET"); think_budget_value = (e && *e) ? atol(e) : -1; }
            llms_think_switches_init();
            if (llms_cli_inside < 0) llms_cli_inside = ds4_think_mode_enabled(think_mode) ? 1 : 0;
            const int te = ds4_token_think_end(engine);
            if (forced_opening_cursor >= 0 && forced_opening_cursor < forced_opening_count) token = forced_opening_tokens[forced_opening_cursor++];
            else if (llms_fl_fired && !llms_fl_done && llms_fl_mode == 2 && llms_cli_inside <= 0) { if (llms_fl_cur == -1) llms_fl_start(engine); if (llms_fl_cur >= 0 && llms_fl_cur < llms_fl_n) { token = llms_fl_tok[llms_fl_cur++]; llms_wrap_forced_now = true; llms_fl_forced++; if (llms_fl_cur >= llms_fl_n) llms_fl_done = 1; } else llms_fl_done = 1; }   /* llms FINAL_LOOP: the FLAWS: opening, one forced token an iteration */
            else if (llms_cov_nudge_cur >= 0 && llms_cov_nudge_cur < llms_cov_nudge_n && llms_cli_inside) { token = llms_cov_nudge_tok[llms_cov_nudge_cur++]; llms_wrap_forced_now = true; if (llms_cov_nudge_cur >= llms_cov_nudge_n) { llms_cov_nudge_cur = -1; llms_cov_nudges++; llms_loop_fired = 0; tlb_reset(); } }   /* llms THINK_COVER: the rest of a nudge */
            else if (llms_cov_nudge_ok() && (llms_loop_fired || llms_cov_pace_due()) && !llms_budget_hit(think_budget_value, generated) && llms_cli_inside && te >= 0 && token != te && llms_cov_nudge_start(engine)) { token = llms_cov_nudge_tok[llms_cov_nudge_cur++]; llms_wrap_forced_now = true; if (llms_cov_nudge_cur >= llms_cov_nudge_n) { llms_cov_nudge_cur = -1; llms_cov_nudges++; llms_loop_fired = 0; tlb_reset(); } }   /* llms THINK_COVER: the loop rule fired with findings unanalysed: force the next finding's tag line */
            else if (llms_wrap_cur >= 0 && llms_wrap_cur < llms_wrap_n && llms_cli_inside) { token = llms_wrap_tok[llms_wrap_cur++]; llms_wrap_forced_now = true; llms_wrap_forced_total++; }   /* llms THINK_WRAP: the rest of the wrap text */
            else if (llms_wrap_cur == -1 && (llms_budget_hit(think_budget_value, generated) || llms_loop_fired) && llms_cli_inside && te >= 0 && token != te && llms_wrap_start(engine)) { token = llms_wrap_tok[llms_wrap_cur++]; llms_wrap_forced_now = true; llms_wrap_forced_total++; }   /* llms THINK_WRAP: the first token */
            else if ((llms_budget_hit(think_budget_value, generated) || llms_loop_fired) && llms_cli_inside && te >= 0 && token != te) {
                token = te;
                if (llms_loop_fired && !llms_budget_hit(think_budget_value, generated)) fprintf(stderr, "llms: think loop (LOOPBREAK) fired at completion %d: </think> forced\n", generated);
                else fprintf(stderr, "llms: think budget %ld reached at completion %d: </think> forced\n", think_budget_value, generated);
                const char *fp = getenv("LLMS_DS4_THINK_FORCE_PREFIX");
                if (fp && *fp && forced_opening_cursor < 0) {
                    char buf[256]; size_t n = 0;
                    for (const char *q = fp; *q && n + 1 < sizeof buf; q++) { if (q[0] == '\\' && q[1] == 'n') { buf[n++] = '\n'; q++; } else buf[n++] = *q; }
                    buf[n] = 0;
                    ds4_tokens tv = {0}; ds4_tokenize_text(engine, buf, &tv);
                    forced_opening_count = 0; for (int k = 0; k < tv.len && k < 64; k++) forced_opening_tokens[forced_opening_count++] = tv.v[k];
                    ds4_tokens_free(&tv); forced_opening_cursor = 0;
                    fprintf(stderr, "llms: answer opening forced after the think budget (%d tokens)\n", forced_opening_count);
                }
            }
            if (token == te) { if (llms_ew_on) { ds4_session_escalate(session, 0); llms_ew_on = 0; llms_ew_done = 1; fprintf(stderr, "llms: esc window closes at completion %d (the close row runs on the shipped dials)\n", generated); } if (llms_cli_inside && llms_reread_on) llms_think_reread_pending = 1; if (llms_cli_inside && llms_cg_n > 0) llms_cg_left = llms_cg_n; if (llms_cli_inside && llms_cx_n > 0) { ds4_session_escalate(session, 1); llms_cx_end = generated + llms_cx_n; } if (llms_cli_inside && (llms_reread_on || llms_cg_n > 0 || llms_cx_n > 0 || llms_probe_k > 0)) llms_isolate_now = 1; llms_cli_inside = 0; }
        }
        if (ds4_token_is_stop_for_think_mode(engine, token, think_mode)) break;
        if (llms_isolate_now) {   /* llms CLOSE: the close token is evaluated alone, so the first post-close token is sampled AFTER the rebuild / by the greedy sample, not inside the close token's speculative cycle */
            llms_isolate_now = 0;
            size_t piece_len = 0; char *piece = ds4_token_text(engine, token, &piece_len);
            token_printer_write_text(&printer, piece, piece_len); fflush(stdout); free(piece);
            generated++;
            if (generated >= max_tokens || cli_interrupt_requested()) continue;
            cli_dist_busy_set(cfg, true);
            const int iso_rc = ds4_session_eval(session, token, err, sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (iso_rc != 0) { fprintf(stderr, "ds4: decode failed: %s\n", err); ds4_session_free(session); return 1; }
            if (llms_probe_k > 0) llms_probe_print(engine, session, "decode-built");
            llms_snap_dump(session, "dec");
            continue;
        }
        {   /* llms SPEC_COPY (apply_spec_copy.py): verify [token, copy proposal] as one 2-row step */
            static int spec_copy_state = -1; static int *hist; static int copy_history_count, hcap; static float *rows; static int vocab; static long vsteps, vacc, osteps;
            if (spec_copy_state < 0) { const char *e2 = getenv("LLMS_DS4_SPEC_COPY"); spec_copy_state = e2 && *e2 && *e2 != '0' && cfg->gen.temperature <= 0.0f;
                          if (spec_copy_state) { fprintf(stderr, "llms: spec copy (a copy drafter's proposal verified as a 2-row step)\n");
                                    hcap = prompt->len + max_tokens + 16; hist = malloc((size_t)hcap * sizeof(int)); memcpy(hist, prompt->v, (size_t)prompt->len * sizeof(int)); copy_history_count = prompt->len;
                                    float probe[1]; (void)probe; vocab = 1 << 17; while (vocab <= (1 << 22)) { float *t = malloc((size_t)vocab * sizeof(float)); const int n = ds4_session_copy_logits(session, t, vocab); free(t); if (n > 0) { vocab = n; break; } vocab *= 2; }
                                    rows = malloc((size_t)2 * vocab * sizeof(float));
                                    atexit_b(^{ fprintf(stderr, "llms spec copy: %ld verify steps (%ld accepted), %ld one-token steps\n", vsteps, vacc, osteps); }); } }
            if (spec_copy_state) {
                size_t piece_len = 0; char *piece = ds4_token_text(engine, token, &piece_len);
                token_printer_write_text(&printer, piece, piece_len); fflush(stdout); free(piece);
                generated++; hist[copy_history_count++] = token;
                if (generated >= max_tokens || cli_interrupt_requested()) continue;
                int prop = -1;   /* the longest earlier occurrence of the current suffix (8 / 6 / 4 / 3 tokens), most recent first */
                static const int copy_ngram_sizes[4] = { 8, 6, 4, 3 };
                for (int q = 0; q < 4 && prop < 0; q++) {
                    const int n = copy_ngram_sizes[q]; if (copy_history_count < n + 1) continue;
                    for (int jj = copy_history_count - 1; jj >= n; jj--) {   /* an occurrence ending at jj (exclusive) with a known continuation hist[jj], jj < hn */
                        if (jj == copy_history_count) continue;
                        if (memcmp(&hist[jj - n], &hist[copy_history_count - n], (size_t)n * sizeof(int)) == 0 && jj < copy_history_count) { prop = hist[jj]; break; }
                    }
                }
                cli_dist_busy_set(cfg, true);
                if (prop < 0 || generated + 1 > max_tokens) {
                    const int rc2 = ds4_session_eval(session, token, err, sizeof(err)); osteps++;
                    cli_dist_busy_set(cfg, false);
                    if (rc2 != 0) { fprintf(stderr, "ds4: decode failed: %s\n", err); ds4_session_free(session); return 1; }
                    continue;
                }
                const int vt[2] = { token, prop };
                const int rc3 = ds4_session_eval_rows(session, vt, 2, rows, err, sizeof(err)); vsteps++;
                cli_dist_busy_set(cfg, false);
                if (rc3 != 0) { fprintf(stderr, "ds4: verify failed: %s\n", err); ds4_session_free(session); return 1; }
                int a0 = 0; for (int v = 1; v < vocab; v++) if (rows[v] > rows[a0]) a0 = v;
                if (a0 == prop) {   /* accepted: the proposal is the next token; the second row's logits are in the session already */
                    vacc++;
                    if (ds4_token_is_stop_for_think_mode(engine, prop, think_mode)) break;
                    piece_len = 0; piece = ds4_token_text(engine, prop, &piece_len);
                    token_printer_write_text(&printer, piece, piece_len); fflush(stdout); free(piece);
                    generated++; hist[copy_history_count++] = prop;
                } else {
                    if (ds4_session_rollback_rows(session, 1, 2) != 0) { fprintf(stderr, "ds4: rollback failed\n"); ds4_session_free(session); return 1; }
                    ds4_session_set_logits(session, rows, vocab);
                }
                continue;
            }
        }

        int toks[17];
        int ntok = 0;
        if (ds4_engine_mtp_draft_tokens(engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
            cli_dist_busy_set(cfg, true);
            ntok = ds4_session_eval_speculative(
                session, token, (llms_wrap_forced_now && !llms_wrap_uncap) ? 1 : max_tokens - generated,   /* llms THINK_WRAP singleton cap */
                ds4_token_eos(engine), cfg->gen.temperature, 0,
                cfg->gen.top_p, cfg->gen.min_p, &rng,
                toks, (int)(sizeof(toks) / sizeof(toks[0])),
                err, sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (llms_wrap_forced_now && ntok > 1) llms_wrap_extra += (unsigned long long)(ntok - 1);   /* llms THINK_WRAP singleton cap: the counter */
            if (ntok < 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_free(session);
                return 1;
            }
        } else {
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, token, &piece_len);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            const int llms_hit = llms_stop_on > 0 && llms_cli_inside <= 0 && llms_stop_feed(piece, piece_len);   /* llms STOP_LINE */
            if (piece_len > 0 && memchr(piece, '\n', piece_len)) llms_nl_count++;   /* llms THINK_BUDGET_EOL */
            if (llms_fl_mode > 0 && llms_fp_n != 0 && llms_cli_inside <= 0 && !llms_fl_done && !llms_fl_fired) llms_fp_feed(piece, piece_len);   /* llms FINAL_PROGRESS */
            if (llms_cov_g >= 0 && llms_cli_inside > 0) llms_cov_feed(piece, piece_len);   /* llms THINK_COVER */
            free(piece);
            generated++;
            if (ind_pen > 0.0f) ind_feed(token);   /* llms IND_PEN */
            ds4_session_dry_feed(token, llms_cli_inside > 0);   /* llms DRY */
            if (llms_loop_on > 0 && llms_cli_inside > 0 && token != ds4_token_think_end(engine) && tlb_feed(token)) llms_loop_fired = 1;   /* llms THINK_LOOP */
            if (llms_cov_g >= 0 && llms_cli_inside > 0 && token != ds4_token_think_end(engine) && llms_cov_hit(generated, llms_nl_count)) llms_loop_fired = 1;   /* llms THINK_COVER */
            if (llms_fl_mode > 0 && llms_cli_inside <= 0 && !llms_fl_done && !llms_fl_fired && (llms_fl_feed(token) | llms_fp_hit())) { llms_fl_fired = 1; if (llms_fl_mode == 1) { fprintf(stderr, "llms: final loop (LOOPBREAK) stopped the answer at completion %d\n", generated); break; } fprintf(stderr, "llms: final loop (LOOPBREAK) fired at completion %d\n", generated); }   /* llms FINAL_LOOP */
            if (llms_hit) break;
            if (generated >= max_tokens || cli_interrupt_requested()) {
                continue;
            }

            cli_dist_busy_set(cfg, true);
            int eval_rc = ds4_session_eval(session, token, err, sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (eval_rc != 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_free(session);
                return 1;
            }
            continue;
        }

        bool stop = false;
        for (int j = 0; j < ntok; j++) {
            if (ds4_token_is_stop_for_think_mode(engine, toks[j], think_mode)) {
                stop = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, toks[j], &piece_len);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            const int llms_hit = llms_stop_on > 0 && llms_cli_inside <= 0 && llms_stop_feed(piece, piece_len);   /* llms STOP_LINE */
            if (piece_len > 0 && memchr(piece, '\n', piece_len)) llms_nl_count++;   /* llms THINK_BUDGET_EOL */
            if (llms_fl_mode > 0 && llms_fp_n != 0 && llms_cli_inside <= 0 && !llms_fl_done && !llms_fl_fired) llms_fp_feed(piece, piece_len);   /* llms FINAL_PROGRESS */
            if (llms_cov_g >= 0 && llms_cli_inside > 0) llms_cov_feed(piece, piece_len);   /* llms THINK_COVER */
            free(piece);
            ds4_session_dry_feed(toks[j], llms_cli_inside > 0);   /* llms DRY: a speculative cycle's accepted tokens */
            generated++;
            if (ind_pen > 0.0f) ind_feed(toks[j]);   /* llms IND_PEN: a speculative cycle's accepted tokens */
            if (llms_hit) { stop = true; break; }
            {   /* llms THINK_LOOP / THINK_REREAD: the cycle's tokens while the thinking is open; a close inside the cycle ends it */
                const int te2 = ds4_token_think_end(engine);
                if (toks[j] == te2 && te2 >= 0) { if (llms_ew_on) { ds4_session_escalate(session, 0); llms_ew_on = 0; llms_ew_done = 1; fprintf(stderr, "llms: esc window closes at completion %d (an accepted-draft close)\n", generated); } if (llms_cli_inside > 0 && llms_reread_on) llms_think_reread_pending = 1; if (llms_cli_inside > 0 && llms_cg_n > 0) llms_cg_left = llms_cg_n; if (llms_cli_inside > 0 && llms_cx_n > 0) { ds4_session_escalate(session, 1); llms_cx_end = generated + llms_cx_n; } llms_cli_inside = 0; }
                else if (llms_loop_on > 0 && llms_cli_inside > 0 && tlb_feed(toks[j])) llms_loop_fired = 1;
                if (llms_cov_g >= 0 && llms_cli_inside > 0 && toks[j] != te2 && llms_cov_hit(generated, llms_nl_count)) llms_loop_fired = 1;   /* llms THINK_COVER */
            }
            if (llms_fl_mode > 0 && llms_cli_inside <= 0 && !llms_fl_done && !llms_fl_fired && (llms_fl_feed(toks[j]) | llms_fp_hit())) { llms_fl_fired = 1; if (llms_fl_mode == 1) { fprintf(stderr, "llms: final loop (LOOPBREAK) stopped the answer at completion %d\n", generated); stop = true; break; } fprintf(stderr, "llms: final loop (LOOPBREAK) fired at completion %d\n", generated); }   /* llms FINAL_LOOP */
            if (generated >= max_tokens) break;
        }
        if (stop) break;
    }
    if (ind_pen > 0.0f) fprintf(stderr, "llms ind pen: %.3f, %llu samples biased of %u fed\n", ind_pen, ind_hits, ind_n);   /* llms IND_PEN */
    const double t_decode1 = cli_now_sec();
    generation_done(&printer);
    if (cli_interrupt_requested()) cli_interrupt_clear();

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);

    ds4_session_free(session);
    return 0;
}

static bool json_utf8_valid(const char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i++];
        if (c < 0x80) continue;
        int need = 0;
        if (c >= 0xc2 && c <= 0xdf) need = 1;
        else if (c >= 0xe0 && c <= 0xef) need = 2;
        else if (c >= 0xf0 && c <= 0xf4) need = 3;
        else return false;
        if (i + (size_t)need > n) return false;
        unsigned char c1 = (unsigned char)s[i];
        if (c == 0xe0 && c1 < 0xa0) return false;
        if (c == 0xed && c1 >= 0xa0) return false;
        if (c == 0xf0 && c1 < 0x90) return false;
        if (c == 0xf4 && c1 >= 0x90) return false;
        for (int j = 0; j < need; j++) {
            unsigned char cc = (unsigned char)s[i + (size_t)j];
            if ((cc & 0xc0) != 0x80) return false;
        }
        i += (size_t)need;
    }
    return true;
}

static void json_write_string(FILE *fp, const char *s, size_t n) {
    bool valid_utf8 = json_utf8_valid(s, n);
    fputc('"', fp);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else if (!valid_utf8 && c >= 0x80) {
            /* Tokenizer pieces can be arbitrary byte fragments.  The bytes
             * array is authoritative; this escape keeps the JSON valid. */
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void json_write_token(FILE *fp, ds4_engine *engine, int token) {
    size_t n = 0;
    char *text = ds4_token_text(engine, token, &n);
    fprintf(fp, "{\"id\":%d,\"text\":", token);
    json_write_string(fp, text, n);
    fputs(",\"bytes\":[", fp);
    for (size_t i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "%u", (unsigned)(unsigned char)text[i]);
    }
    fputc(']', fp);
    fputc('}', fp);
    free(text);
}

static int run_logits_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --dump-logits requires a graph session backend\n");
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        return 1;
    }

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        ds4_session_free(session);
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4: failed to copy session logits\n");
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    FILE *fp = fopen(cfg->gen.dump_logits_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open --dump-logits file: %s\n", cfg->gen.dump_logits_path);
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    fprintf(fp, "{\n  \"source\":\"ds4\",\n  \"model\":");
    json_write_string(fp, cfg->engine.model_path, strlen(cfg->engine.model_path));
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quant_bits\":%d,\n"
            "  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"vocab\":%d,\n",
            ds4_backend_name(cfg->engine.backend),
            ds4_engine_routed_quant_bits(engine),
            prompt->len,
            cfg->gen.ctx_size,
            vocab);
    const int argmax = ds4_session_argmax(session);
    fputs("  \"argmax_token\":", fp);
    json_write_token(fp, engine, argmax);
    fprintf(fp, ",\n  \"argmax_logit\":%.9g,\n  \"logits\":[", logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) {
            fprintf(fp, "%.9g", logits[i]);
        } else {
            fputs("null", fp);
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close --dump-logits file: %s\n", cfg->gen.dump_logits_path);
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    free(logits);
    ds4_session_free(session);
    return 0;
}

static int run_logprob_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --dump-logprobs requires a graph session backend\n");
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        return 1;
    }

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);

    FILE *fp = fopen(cfg->gen.dump_logprobs_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        ds4_session_free(session);
        return 1;
    }

    int k = cfg->gen.dump_logprobs_top_k > 0 ? cfg->gen.dump_logprobs_top_k : 20;
    if (k > 128) k = 128;
    ds4_token_score *scores = calloc((size_t)k, sizeof(scores[0]));
    if (!scores) {
        fclose(fp);
        ds4_session_free(session);
        return 1;
    }

    fprintf(fp, "{\n  \"source\":\"ds4\",\n  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"top_k\":%d,\n  \"steps\":[\n",
            prompt->len, cfg->gen.ctx_size, k);
    int generated = 0;
    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    for (; generated < max_tokens; generated++) {
        int n = ds4_session_top_logprobs(session, scores, k);
        int token = ds4_session_argmax(session);
        if (generated) fputs(",\n", fp);
        fprintf(fp, "    {\"step\":%d,\"selected\":", generated);
        json_write_token(fp, engine, token);
        fputs(",\"top_logprobs\":[", fp);
        for (int i = 0; i < n && scores[i].id >= 0; i++) {
            if (i) fputc(',', fp);
            fputs("{\"token\":", fp);
            json_write_token(fp, engine, scores[i].id);
            fprintf(fp, ",\"logit\":%.9g,\"logprob\":%.9g}", scores[i].logit, scores[i].logprob);
        }
        fputs("]}", fp);

        if (ds4_token_is_stop(engine, token)) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: decode failed while dumping logprobs: %s\n", err);
            free(scores);
            fclose(fp);
            ds4_session_free(session);
            return 1;
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        free(scores);
        ds4_session_free(session);
        return 1;
    }
    free(scores);
    ds4_session_free(session);
    return 0;
}

static void print_diag_token(FILE *fp, ds4_engine *engine, int token) {
    fputs("{\"token\":", fp);
    json_write_token(fp, engine, token);
    fputc('}', fp);
}

static void print_diag_top(FILE *fp, ds4_engine *engine, const char *label,
                           const ds4_token_score *scores, int n) {
    fprintf(fp, "%s:", label);
    for (int i = 0; i < n && scores[i].id >= 0; i++) {
        fputc(' ', fp);
        json_write_token(fp, engine, scores[i].id);
        fprintf(fp, "@%.6g", scores[i].logit);
    }
    fputc('\n', fp);
}

static int run_decode_consistency(ds4_engine *engine, const cli_config *cfg,
                                  const ds4_tokens *prompt) {
    if (cfg->dist && cfg->dist->role != DS4_DISTRIBUTED_NONE) {
        fprintf(stderr, "ds4: --decode-consistency is local-session only\n");
        return 1;
    }
    if (cfg->gen.decode_consistency_tokens < 0) {
        fprintf(stderr, "ds4: --decode-consistency requires a non-negative token count\n");
        return 1;
    }

    const int vocab = ds4_engine_vocab_size(engine);
    float *live_logits = malloc((size_t)vocab * sizeof(live_logits[0]));
    float *fresh_logits = malloc((size_t)vocab * sizeof(fresh_logits[0]));
    if (!live_logits || !fresh_logits) {
        free(live_logits);
        free(fresh_logits);
        fprintf(stderr, "ds4: out of memory allocating diagnostic logits\n");
        return 1;
    }

    ds4_tokens prefix = {0};
    ds4_tokens_copy(&prefix, prompt);

    ds4_session *live = NULL;
    char err[160];
    if (ds4_session_create(&live, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --decode-consistency requires a graph session backend\n");
        ds4_tokens_free(&prefix);
        free(live_logits);
        free(fresh_logits);
        return 1;
    }

    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(live, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(live,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(live, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(live, NULL, NULL);
        ds4_session_set_display_progress(live, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(live);
        ds4_tokens_free(&prefix);
        free(live_logits);
        free(fresh_logits);
        return 1;
    }
    ds4_session_set_progress(live, NULL, NULL);
    ds4_session_set_display_progress(live, NULL, NULL);

    fprintf(stderr, "ds4: decode-consistency prompt_tokens=%d check_after=%d\n",
            prompt->len, cfg->gen.decode_consistency_tokens);
    for (int i = 0; i < cfg->gen.decode_consistency_tokens; i++) {
        int token = ds4_session_argmax(live);
        fprintf(stderr, "ds4: decode-consistency selected[%d]=", i);
        print_diag_token(stderr, engine, token);
        fputc('\n', stderr);
        ds4_tokens_push(&prefix, token);
        if (ds4_session_eval(live, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: decode failed during consistency check: %s\n", err);
            ds4_session_free(live);
            ds4_tokens_free(&prefix);
            free(live_logits);
            free(fresh_logits);
            return 1;
        }
        if (ds4_token_is_stop(engine, token)) break;
    }

    ds4_token_score live_top[10];
    int live_top_n = ds4_session_top_logprobs(live, live_top, 10);
    if (ds4_session_copy_logits(live, live_logits, vocab) != vocab) {
        fprintf(stderr, "ds4: failed to copy live logits\n");
        ds4_session_free(live);
        ds4_tokens_free(&prefix);
        free(live_logits);
        free(fresh_logits);
        return 1;
    }
    ds4_session_free(live);
    live = NULL;

    ds4_session *fresh = NULL;
    if (ds4_session_create(&fresh, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: failed to create fresh diagnostic session\n");
        ds4_tokens_free(&prefix);
        free(live_logits);
        free(fresh_logits);
        return 1;
    }
    progress.input_tokens = prefix.len;
    ds4_session_set_progress(fresh, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(fresh,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(fresh, &prefix, err, sizeof(err)) != 0) {
        ds4_session_set_progress(fresh, NULL, NULL);
        ds4_session_set_display_progress(fresh, NULL, NULL);
        fprintf(stderr, "ds4: fresh prompt processing failed: %s\n", err);
        ds4_session_free(fresh);
        ds4_tokens_free(&prefix);
        free(live_logits);
        free(fresh_logits);
        return 1;
    }
    ds4_session_set_progress(fresh, NULL, NULL);
    ds4_session_set_display_progress(fresh, NULL, NULL);

    ds4_token_score fresh_top[10];
    int fresh_top_n = ds4_session_top_logprobs(fresh, fresh_top, 10);
    if (ds4_session_copy_logits(fresh, fresh_logits, vocab) != vocab) {
        fprintf(stderr, "ds4: failed to copy fresh logits\n");
        ds4_session_free(fresh);
        ds4_tokens_free(&prefix);
        free(live_logits);
        free(fresh_logits);
        return 1;
    }
    ds4_session_free(fresh);

    double ss = 0.0;
    float max_abs = 0.0f;
    int max_i = 0;
    for (int i = 0; i < vocab; i++) {
        float d = fabsf(live_logits[i] - fresh_logits[i]);
        if (d > max_abs) {
            max_abs = d;
            max_i = i;
        }
        ss += (double)d * (double)d;
    }

    fprintf(stderr,
            "ds4: decode-consistency compared prefix_tokens=%d vocab=%d "
            "max_abs=%.9g at token=%d live=%.9g fresh=%.9g rms=%.9g\n",
            prefix.len, vocab, max_abs, max_i,
            live_logits[max_i], fresh_logits[max_i],
            sqrt(ss / (double)vocab));
    print_diag_top(stderr, engine, "ds4: live_top", live_top, live_top_n);
    print_diag_top(stderr, engine, "ds4: fresh_top", fresh_top, fresh_top_n);

    ds4_tokens_free(&prefix);
    free(live_logits);
    free(fresh_logits);
    return live_top_n > 0 && fresh_top_n > 0 && live_top[0].id == fresh_top[0].id ? 0 : 1;
}

static int ppl_score_row(ds4_session *session, int token, ds4_token_score *score,
                         FILE *dump, int position, int topk) {
    ds4_token_score top[256];
    int n = 0;
    if (dump && topk) {
        n = ds4_session_top_logprobs_and_token(session, top, topk, token, score);
        if (!n) return 0;
    } else if (!ds4_session_token_logprob(session, token, score)) return 0;
    if (dump) {
        fprintf(dump, "%d %d %.9f\n", position, token, (double)score->logprob);
        if (n) {
            fprintf(dump, "K %d", position);
            for (int q = 0; q < n; q++) fprintf(dump, " %d %.6f", top[q].id, (double)top[q].logprob);
            fputc('\n', dump);
        }
    }
    return 1;
}

static int run_perplexity_file(ds4_engine *engine, const cli_config *cfg) {
    char *text = read_prompt_file(cfg->gen.perplexity_file_path, true);
    ds4_tokens tokens = {0};
    {   /* llms PPL_IDS (drafter fine-tune capture): the file holds whitespace-separated TOKEN IDS, used as is (no * re-tokenization: the agent's special DSML tokens stay single ids); off unless LLMS_DS4_PPL_IDS is set */
        const char *e = getenv("LLMS_DS4_PPL_IDS");
        if (e && *e && *e != '0') {
            char *s = text, *end = NULL;
            for (long v = strtol(s, &end, 10); end != s; s = end, v = strtol(s, &end, 10)) ds4_tokens_push(&tokens, (int)v);
            fprintf(stderr, "llms: perplexity ids (%d token ids read from the file, no tokenization)\n", tokens.len);
        } else ds4_tokenize_text(engine, text, &tokens);
    }
    free(text);

    /* Seed the graph with enough real context to stay on the normal Metal
     * prefill path; scoring starts immediately after this fixed prefix. */
    int prefix_len = 32;
    { const char *e = getenv("LLMS_DS4_PPL_PREFIX"); if (e && *e && atoi(e) >= 32) { prefix_len = atoi(e); fprintf(stderr, "llms: perplexity prefix %d tokens (through the prompt path)\n", prefix_len); } }   /* llms PPL_PREFIX (apply_ppl_prefix.py) */
    {   /* llms PPL_PREFIX_FILE (2026-10-16, ffdp.py): the prefix length = the token count of this file's text (a leading part of the scored
         * text), so only the rest (e.g. a flaw-field verdict) is scored one-token; off unless set */
        const char *pf = getenv("LLMS_DS4_PPL_PREFIX_FILE");
        if (pf && *pf) { char *t2 = read_prompt_file(pf, true); ds4_tokens tp = {0}; ds4_tokenize_text(engine, t2, &tp); free(t2);
                         if (tp.len >= 32) prefix_len = tp.len;
                         fprintf(stderr, "llms: perplexity prefix file %d tokens (of %d)\n", prefix_len, tokens.len); ds4_tokens_free(&tp); }
    }
    if (tokens.len <= prefix_len) {
        fprintf(stderr, "ds4: --perplexity-file needs more than %d tokens\n", prefix_len);
        ds4_tokens_free(&tokens);
        return 1;
    }

    int scored = tokens.len - prefix_len;
    if (cfg->gen.n_predict > 0 && scored > cfg->gen.n_predict) scored = cfg->gen.n_predict;
    if (scored > cfg->gen.ctx_size - prefix_len) scored = cfg->gen.ctx_size - prefix_len;
    if (scored <= 0) {
        fprintf(stderr, "ds4: context too small for perplexity scoring\n");
        ds4_tokens_free(&tokens);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --perplexity-file requires a graph session backend\n");
        ds4_tokens_free(&tokens);
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        ds4_tokens_free(&tokens);
        return 1;
    }

    ds4_tokens prefix = {0};
    for (int i = 0; i < prefix_len; i++) ds4_tokens_push(&prefix, tokens.v[i]);
    char err[160];
    if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: perplexity initial token failed: %s\n", err);
        ds4_tokens_free(&prefix);
        ds4_session_free(session);
        ds4_tokens_free(&tokens);
        return 1;
    }
    ds4_tokens_free(&prefix);

    const char *topk_env = getenv("LLMS_DS4_PPL_TOPK");
    int topk = topk_env ? atoi(topk_env) : 0;
    if (topk < 0) topk = 0;
    if (topk > 256) topk = 256;
    double nll = 0.0;   /* llms VERIFY_ROWS: shared with the one-token loop below */
    {   /* llms PPL_MASKS (apply_dims_zero.py): one pass per lane mask, the prefix re-synced under each */
        const char *mf = getenv("LLMS_DS4_PPL_MASKS"), *pr = getenv("LLMS_DS4_PPL_ROWS"); const int K = pr ? atoi(pr) : 0;
        if (mf && *mf) {
            extern void ds4_llms_dims_zero_set(const uint32_t *lanes, uint32_t n);
            FILE *fm = fopen(mf, "r"); if (!fm || K < 2) { fprintf(stderr, "ds4: PPL_MASKS needs a readable file and LLMS_DS4_PPL_ROWS >= 2\n"); return 1; }
            const int sc_n = (scored / K) * K; if (sc_n < K) { fprintf(stderr, "ds4: PPL_MASKS: fewer than %d scored tokens\n", K); return 1; }
            const char *lpp = getenv("LLMS_DS4_PPL_MASKS_LP"); FILE *flp = lpp && *lpp ? fopen(lpp, "wb") : NULL;
            float *held = NULL; int vocab = 0; while (!ds4_session_copy_logits(session, held, vocab)) { vocab = vocab ? vocab * 2 : 65536; held = realloc(held, (size_t)vocab * sizeof(float)); if (vocab > (1 << 22)) break; }
            vocab = ds4_session_copy_logits(session, held, vocab); free(held);
            float *rows = malloc((size_t)K * (size_t)vocab * sizeof(float)), *lp = malloc((size_t)sc_n * sizeof(float));
            char line[65536]; int mi = 0; static uint32_t lanes[8192];
            fprintf(stderr, "llms: ppl masks (%s; %d scored tokens per mask in %d-row steps)\n", mf, sc_n, K);
            extern int ds4_gpu_llms_route_record(const char *path); extern int ds4_gpu_llms_route_replay(const char *path, int ids_only);   /* llms ROUTE_FREEZE */
            extern void ds4_gpu_llms_route_counters(uint64_t *replayed, uint64_t *mismatched, uint64_t *short_reads, uint64_t *logged);
            const char *fz = getenv("LLMS_DS4_PPL_MASKS_FREEZE"); const char *fzw = getenv("LLMS_DS4_PPL_MASKS_FREEZE_W"); const int fz_ids = !(fzw && !strcmp(fzw, "pin"));
            if (fz && *fz) fprintf(stderr, "llms: ppl masks freeze (mask 0's routing recorded to %s, replayed in every later mask; weights %s)\n", fz, fz_ids ? "recomputed from each pass's scores" : "pinned");
            while (fgets(line, sizeof line, fm)) {
                uint32_t n = 0;
                for (char *p = line; *p && n < 8192u; ) { char *q; const long v = strtol(p, &q, 10); if (q == p) { p++; continue; } if (v >= 0) lanes[n++] = (uint32_t)v; p = q; }
                ds4_llms_dims_zero_set(lanes, n);
                uint64_t c0[4] = {0}; if (fz && *fz) { if (mi == 0) ds4_gpu_llms_route_record(fz); else ds4_gpu_llms_route_replay(fz, fz_ids); ds4_gpu_llms_route_counters(&c0[0], &c0[1], &c0[2], &c0[3]); }   /* llms ROUTE_FREEZE */
                ds4_session_invalidate(session);
                ds4_tokens pfx = {0}; for (int q2 = 0; q2 < prefix_len; q2++) ds4_tokens_push(&pfx, tokens.v[q2]);
                if (ds4_session_sync(session, &pfx, err, sizeof(err)) != 0) { fprintf(stderr, "ds4: PPL_MASKS: prefix sync failed (mask %d): %s\n", mi, err); return 1; }
                ds4_tokens_free(&pfx);
                double mnll = 0.0; ds4_token_score sc;
                for (int j = 0; j < sc_n; j += K) {
                    const int i = prefix_len + j;
                    if (!ppl_score_row(session, tokens.v[i], &sc, NULL, i, 0)) { fprintf(stderr, "ds4: PPL_MASKS: scoring failed at %d\n", i); return 1; }
                    mnll -= (double)sc.logprob; lp[j] = sc.logprob;
                    if (ds4_session_eval_rows(session, &tokens.v[i], K, rows, err, sizeof(err)) != 0) { fprintf(stderr, "ds4: PPL_MASKS: rows failed at %d: %s\n", i, err); return 1; }
                    for (int r = 0; r + 1 < K; r++) {
                        ds4_session_set_logits(session, rows + (size_t)r * vocab, vocab);
                        if (!ppl_score_row(session, tokens.v[i + r + 1], &sc, NULL, i + r + 1, 0)) { fprintf(stderr, "ds4: PPL_MASKS: scoring failed at %d\n", i + r + 1); return 1; }
                        mnll -= (double)sc.logprob; lp[j + r + 1] = sc.logprob;
                    }
                    ds4_session_set_logits(session, rows + (size_t)(K - 1) * vocab, vocab);
                }
                if (fz && *fz) { uint64_t c1[4]; ds4_gpu_llms_route_counters(&c1[0], &c1[1], &c1[2], &c1[3]);   /* llms ROUTE_FREEZE: this pass's counters */
                    fprintf(stderr, "llms ppl masks freeze: mask %d logged %llu, replayed %llu, mismatched %llu, short %llu\n", mi, (unsigned long long)(c1[3] - c0[3]),
                            (unsigned long long)(c1[0] - c0[0]), (unsigned long long)(c1[1] - c0[1]), (unsigned long long)(c1[2] - c0[2]));
                    if (mi == 0) ds4_gpu_llms_route_record(NULL); }
                printf("mask %d lanes %u nll %.9f avg %.9f\n", mi, n, mnll, mnll / (double)sc_n); fflush(stdout);
                if (flp) { fwrite(lp, sizeof(float), (size_t)sc_n, flp); fflush(flp); }
                mi++;
            }
            fclose(fm); if (flp) fclose(flp); free(rows); free(lp);
            ds4_session_free(session); ds4_tokens_free(&tokens);
            return 0;
        }
    }
    {   /* llms VERIFY_ROWS: LLMS_DS4_PPL_ROWS=k scores the text through k-row verify steps (perfect drafts) */
        const char *pr = getenv("LLMS_DS4_PPL_ROWS"); const int K = pr ? atoi(pr) : 0;
        if (K >= 2 || (K == 1 && getenv("LLMS_DS4_VERIFY_ONE"))) {   /* llms VERIFY_ONE: K = 1 through the batched step */
            const char *lp = getenv("LLMS_DS4_PPL_ROWS_LOG"); FILE *lf = lp && *lp ? fopen(lp, "w") : NULL;
            const char *pp = getenv("LLMS_DS4_PPL_LOGPROB_DUMP"); FILE *pd = pp && *pp ? fopen(pp, "w") : NULL;
            const int nv = ds4_session_copy_logits(session, NULL, 0) ;
            (void)nv;
            float *held = malloc((size_t)1 << 20), *rows = NULL; int vocab = 0;
            { float probe[1]; (void)probe; }
            vocab = 0; while (!ds4_session_copy_logits(session, held, vocab)) { vocab = vocab ? vocab * 2 : 65536; held = realloc(held, (size_t)vocab * sizeof(float)); if (vocab > (1 << 22)) break; }
            vocab = ds4_session_copy_logits(session, held, vocab);
            rows = malloc((size_t)K * (size_t)vocab * sizeof(float));
            int j = 0;
            while (j < scored) {
                const int i = prefix_len + j;
                ds4_token_score sc;
                if (!ppl_score_row(session, tokens.v[i], &sc, pd, i, topk)) { fprintf(stderr, "ds4: failed to score token %d\n", i); return 1; }
                nll -= (double)sc.logprob;
                int c = scored - j; if (c > K) c = K;
                if (j + c >= scored) c = scored - j;   /* the last chunk: its final row's logits are never used */
                if (c < (K == 1 ? 1 : 2)) { j += 1; if (j < scored && ds4_session_eval(session, tokens.v[i], err, sizeof(err)) != 0) { fprintf(stderr, "\nds4: eval failed at %d: %s\n", i, err); return 1; } continue; }
                {   /* llms PPL_JUNK (apply_ppl_junk.py): [x_i, junk] rolled back to row 0 */
                    static int ppl_junk_state = -1; if (ppl_junk_state < 0) { ppl_junk_state = getenv("LLMS_DS4_PPL_JUNK") != NULL; if (ppl_junk_state) fprintf(stderr, "llms: ppl junk (2-row steps [x, junk] rolled back to row 0)\n"); }
                    if (ppl_junk_state && K == 2) {
                        const int jt[2] = { tokens.v[i], tokens.v[i] == 13 ? 14 : 13 };
                        if (ds4_session_eval_rows(session, jt, 2, rows, err, sizeof(err)) != 0 || ds4_session_rollback_rows(session, 1, 2) != 0) {
                            fprintf(stderr, "\nds4: junk step failed at %d: %s\n", i, err); return 1; }
                        ds4_session_set_logits(session, rows, vocab);
                        j += 1;
                        continue;
                    }
                }
                struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
                if (ds4_session_eval_rows(session, &tokens.v[i], c, rows, err, sizeof(err)) != 0) { fprintf(stderr, "\nds4: verify rows failed at %d: %s\n", i, err); return 1; }
                clock_gettime(CLOCK_MONOTONIC, &t1);
                if (lf) { fprintf(lf, "%.3f %d %d\n", (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6, c, i); fflush(lf); }
                {   /* llms PPL_REPEAT (apply_rep.py): the same step R - 1 more times, each after a keep-0 rollback */
                    /* llms PPL_REP_NAME may name up to 8 ALTW switches, comma-separated (mgate.py, 2026-10-19 12:5x): the log's last field is the
                       bitmask of their OFF states, bit j = the j-th name -- one name gives 0 / 1 as before */
#define llms_rep_am(nm) ({ int _am = -1; if (nm) { _am = 0; char _nb[256]; snprintf(_nb, sizeof _nb, "%s", nm); int _bi = 0; char *_sv = NULL; \
    for (char *_t = strtok_r(_nb, ",", &_sv); _t && _bi < 8; _t = strtok_r(NULL, ",", &_sv), _bi++) _am |= ds4_gpu_llms_alt_off(_t) << _bi; } _am; })
                    static int ppl_random_state = -1; static FILE *ppl_random_file; static const char *rname; static float *ppl_random_held_logits; static size_t ppl_random_held_floats;
                    extern void ds4_gpu_llms_alt_force(const char *names);
                    static int randrep = -1;
                    if (randrep < 0) { const char *e = getenv("LLMS_DS4_PPL_RANDOM"); randrep = e && *e && *e != '0'; }
                    if (ppl_random_state < 0) { const char *e = getenv("LLMS_DS4_PPL_REPEAT"); ppl_random_state = e && atoi(e) > 1 ? atoi(e) : 0; const char *p = getenv("LLMS_DS4_PPL_REP_LOG");
                                   ppl_random_file = p && *p ? fopen(p, "w") : NULL; rname = getenv("LLMS_DS4_PPL_REP_NAME"); if (rname && !*rname) rname = NULL;
                                   if (ppl_random_state) fprintf(stderr, "llms: ppl repeat %d (each full %d-row step re-run at its position after a keep-0 rollback; LLMS_DS4_DEEPRB=1 needed)\n", ppl_random_state, K); }
                    if (ppl_random_state && c == K) {
                        int phase = 0;
                        if (randrep) {
                            const char *a = getenv("LLMS_DS4_ALTW"), *b = getenv("LLMS_DS4_ALT");
                            if (ppl_random_state != 5 || !ppl_random_file || !rname || strchr(rname, ',') || (a && *a) || (b && *b)) {
                                fprintf(stderr, "random repeat requires R5, log, one ALT name and no periodic ALT schedule\n"); return 1;
                            }
                            phase = arc4random_uniform(2) ? 1 : -1;
                            fprintf(ppl_random_file, "R %d %d\n", i, phase); fflush(ppl_random_file);
                        }
                        const size_t nf = (size_t)c * (size_t)vocab;
                        if (ppl_random_held_floats < nf) { free(ppl_random_held_logits); ppl_random_held_logits = malloc(nf * sizeof(float)); ppl_random_held_floats = ppl_random_held_logits ? nf : 0; }
                        if (ppl_random_held_logits) memcpy(ppl_random_held_logits, rows, nf * sizeof(float));
                        if (ppl_random_file) fprintf(ppl_random_file, "%d 0 %.3f 0 %d\n", i, (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6, llms_rep_am(rname));
                        for (int q = 1; q < ppl_random_state; q++) {
                            if (randrep) {
                                const int off = (q == 1 || q == 4) ? phase == 1 : phase == -1;
                                ds4_gpu_llms_alt_force(off ? rname : "");
                            }
                            struct timespec r0, r1; clock_gettime(CLOCK_MONOTONIC, &r0);
                            const int rr = ds4_session_llms_rerun_rows(session, &tokens.v[i], c, rows);
                            clock_gettime(CLOCK_MONOTONIC, &r1);
                            if (randrep && rr != 0) ds4_gpu_llms_alt_force("");
                            if (rr != 0) { fprintf(stderr, "\nds4: ppl repeat failed at %d (repeat %d, code %d)\n", i, q, rr); return 1; }
                            double mx = 0.0; if (ppl_random_held_logits) for (size_t z = 0; z < nf; z++) { const double dd = fabs((double)rows[z] - (double)ppl_random_held_logits[z]); if (dd > mx) mx = dd; }
                            if (ppl_random_file) fprintf(ppl_random_file, "%d %d %.3f %.3g %d\n", i, q, (r1.tv_sec - r0.tv_sec) * 1e3 + (r1.tv_nsec - r0.tv_nsec) / 1e6, mx, llms_rep_am(rname));
                        }
                        if (randrep) ds4_gpu_llms_alt_force("");
                        if (ppl_random_file) fflush(ppl_random_file);
                    }
                }
                {   /* llms ELIDE_SWEEP (apply_elide_sweep.py): every masked layer dropped alone on a re-run of this step; KL per row; a clean restore */
                    static int sw_state = -1; static unsigned long long sw_mask; static FILE *sw_log; static float *sw_held; static size_t sw_held_n;
                    if (sw_state < 0) { const char *e = getenv("LLMS_DS4_ELIDE_SWEEP"); sw_mask = e && *e ? strtoull(e, NULL, 16) : 0ull; const char *p = getenv("LLMS_DS4_ELIDE_SWEEP_LOG");
                                        sw_log = sw_mask && p && *p ? fopen(p, "w") : NULL; sw_state = sw_mask && sw_log ? 1 : 0;
                                        if (sw_mask) fprintf(stderr, "llms: ELIDE_SWEEP %llx -- MEASUREMENT: each step re-run once per masked layer with its attention output zeroed (LLMS_DS4_DEEPRB=1 needed)%s\n", sw_mask, sw_state ? "" : " -- NO LOG, OFF"); }
                    if (sw_state && c == K) {
                        const size_t nf = (size_t)c * (size_t)vocab;
                        if (sw_held_n < nf) { free(sw_held); sw_held = malloc(nf * sizeof(float)); sw_held_n = sw_held ? nf : 0; }
                        if (!sw_held) { fprintf(stderr, "ds4: sweep allocation failed\n"); return 1; }
                        memcpy(sw_held, rows, nf * sizeof(float));
                        for (int L = 0; L < 64; L++) {
                            if (!((sw_mask >> L) & 1ull)) continue;
                            ds4_llms_elide_sweep_set(L);
                            const int rr = ds4_session_llms_rerun_rows(session, &tokens.v[i], c, rows);
                            ds4_llms_elide_sweep_set(-1);
                            if (rr != 0) { fprintf(stderr, "\nds4: sweep re-run failed at %d layer %d (code %d)\n", i, L, rr); return 1; }
                            for (int r = 0; r < c; r++) {
                                const float *hp = sw_held + (size_t)r * vocab, *qp = rows + (size_t)r * vocab;
                                float hm = -INFINITY, qm = -INFINITY; int ha = 0, qa = 0;
                                for (int v = 0; v < vocab; v++) { if (hp[v] > hm) { hm = hp[v]; ha = v; } if (qp[v] > qm) { qm = qp[v]; qa = v; } }
                                double hz = 0.0, qz = 0.0; for (int v = 0; v < vocab; v++) { hz += exp((double)hp[v] - hm); qz += exp((double)qp[v] - qm); }
                                const double hl = log(hz) + hm, ql = log(qz) + qm; double kl = 0.0;
                                for (int v = 0; v < vocab; v++) { const double lp = (double)hp[v] - hl; const double pr = exp(lp); if (pr > 0.0) kl += pr * (lp - ((double)qp[v] - ql)); }
                                fprintf(sw_log, "%d %d %d %.6g %d\n", i, L, r, kl, ha == qa);
                            }
                        }
                        const int rr = ds4_session_llms_rerun_rows(session, &tokens.v[i], c, rows);   /* the clean restore: the session's KV and the logits as first run */
                        if (rr != 0) { fprintf(stderr, "\nds4: sweep restore failed at %d (code %d)\n", i, rr); return 1; }
                        double mx = 0.0; for (size_t z = 0; z < nf; z++) { const double dd = fabs((double)rows[z] - (double)sw_held[z]); if (dd > mx) mx = dd; }
                        fprintf(sw_log, "%d -1 restore %.3g\n", i, mx); fflush(sw_log);
                        memcpy(rows, sw_held, nf * sizeof(float));
                    }
                }
                for (int r = 0; r + 1 < c; r++) {   /* row r's logits predict token i + r + 1 */
                    ds4_session_set_logits(session, rows + (size_t)r * vocab, vocab);
                    if (!ppl_score_row(session, tokens.v[i + r + 1], &sc, pd, i + r + 1, topk)) { fprintf(stderr, "ds4: failed to score token %d\n", i + r + 1); return 1; }
                    nll -= (double)sc.logprob;
                }
                ds4_session_set_logits(session, rows + (size_t)(c - 1) * vocab, vocab);
                j += c;
                if ((j % 256) < c || j >= scored) { fprintf(stderr, "ds4: perplexity (rows %d) scored %d/%d\r", K, j, scored); fflush(stderr); }
            }
            if (pd) fclose(pd); if (lf) fclose(lf); free(rows); free(held);
            fputc('\n', stderr);
            const double avg_nll = nll / (double)scored;
            printf("tokens=%d scored=%d nll=%.9f avg_nll=%.9f ppl=%.9f (verify rows %d)\n", tokens.len, scored, nll, avg_nll, exp(avg_nll), K);
            ds4_session_free(session); ds4_tokens_free(&tokens);
            return 0;
        }
    }
    int evaluated = 0;
    for (int j = 0; j < scored; j++) {
        const int i = prefix_len + j;
        static FILE *llms_pd = NULL;
        static int llms_pd_init = 0;
        if (!llms_pd_init) {
            const char *pp = getenv("LLMS_DS4_PPL_LOGPROB_DUMP");
            llms_pd = pp && *pp ? fopen(pp, "w") : NULL;
            llms_pd_init = 1;
        }
        ds4_token_score score;
        if (!ppl_score_row(session, tokens.v[i], &score, llms_pd, i, topk)) {
            fprintf(stderr, "ds4: failed to score token %d\n", i);
            ds4_session_free(session);
            ds4_tokens_free(&tokens);
            return 1;
        }
        nll -= (double)score.logprob;
        evaluated++;
        if (llms_pd) fflush(llms_pd);

        if (((j + 1) % 256) == 0 || j + 1 == scored) {
            fprintf(stderr, "ds4: perplexity scored %d/%d\r", j + 1, scored);
            fflush(stderr);
        }

        {   /* llms PPL_APPEND (apply_ppl_append.py): every S scored steps, A tokens through the prompt path */
            static int app_s = -1, app_a = 0; static long app_n = 0;
            if (app_s < 0) { app_s = 0; const char *e = getenv("LLMS_DS4_PPL_APPEND");
                             if (e && *e && strchr(e, ':')) { app_s = atoi(e); app_a = atoi(strchr(e, ':') + 1); if (app_s < 1 || app_a < 2) app_s = 0; }
                             if (app_s) fprintf(stderr, "llms: perplexity append %d:%d (after every %d scored steps, %d tokens through the prompt path, unscored)\n", app_s, app_a, app_s, app_a); }
            if (app_s && j + 1 < scored && ((j + 1) % app_s) == 0 && j + app_a < scored) {
                ds4_tokens pr = {0};
                for (int q = 0; q < i + app_a; q++) ds4_tokens_push(&pr, tokens.v[q]);
                const int rc = ds4_session_sync(session, &pr, err, sizeof(err)); ds4_tokens_free(&pr);
                if (rc != 0) { fprintf(stderr, "\nds4: perplexity append failed at token %d: %s\n", i, err); ds4_session_free(session); ds4_tokens_free(&tokens); return 1; }
                app_n++; if (app_n == 1) atexit_b(^{ fprintf(stderr, "llms ppl append: %ld appends of %d tokens\n", app_n, app_a); });
                j += app_a - 1;
                continue;
            }
        }
        if (j + 1 < scored && ds4_session_eval(session, tokens.v[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "\nds4: perplexity decode failed at token %d: %s\n", i, err);
            ds4_session_free(session);
            ds4_tokens_free(&tokens);
            return 1;
        }
    }
    fputc('\n', stderr);

    const double avg_nll = nll / (double)evaluated;
    printf("tokens=%d scored=%d nll=%.9f avg_nll=%.9f ppl=%.9f\n",
           tokens.len, evaluated, nll, avg_nll, exp(avg_nll));

    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    return 0;
}

static int run_generation(ds4_engine *engine, const cli_config *cfg) {
    ds4_tokens prompt = {0};
    build_prompt(engine, &cfg->gen, &prompt);

    int rc = 0;
    if (cfg->gen.metal_graph_test) {
        rc = ds4_engine_metal_graph_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_full_test) {
        rc = ds4_engine_metal_graph_full_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_prompt_test) {
        rc = ds4_engine_metal_graph_prompt_test(engine, &prompt, cfg->gen.ctx_size);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.dump_logits_path) {
        rc = run_logits_dump(engine, cfg, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.dump_logprobs_path) {
        rc = run_logprob_dump(engine, cfg, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.decode_consistency_tokens > 0) {
        rc = run_decode_consistency(engine, cfg, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }

    const bool diagnostic = cfg->gen.dump_tokens ||
                            cfg->gen.head_test ||
                            cfg->gen.first_token_test;
    if (cfg->gen.head_test) {
        rc = ds4_engine_head_test(engine, &prompt);
    }
    if (rc == 0 && cfg->gen.first_token_test) {
        rc = ds4_engine_first_token_test(engine, &prompt);
    }
    if (cfg->gen.dump_tokens) {
        ds4_engine_dump_tokens(engine, &prompt);
    }

    if (diagnostic) {
        if (rc == 0) {
            fprintf(stderr, "ds4: diagnostic run completed on the native %s path.\n",
                    ds4_backend_name(cfg->engine.backend));
        }
    } else {
        if (cfg->engine.distributed.role == DS4_DISTRIBUTED_COORDINATOR ||
            cfg->engine.tp.role == DS4_TP_LEADER ||
            getenv("DS4_CLI_FORCE_SESSION") != NULL ||
            cfg->gen.temperature > 0.0f ||
            ds4_engine_mtp_draft_tokens(engine) > 1) {
            /* TP leaders always drive the session path: the sync/eval
             * mirroring that keeps the worker in lockstep lives there.
             * The env override exists so TP-vs-single-node validation
             * compares like with like. */
            rc = run_sampled_generation(engine, cfg, &prompt);
        } else {
            token_printer printer = {
                .engine = engine,
                .fp = stdout,
                .format_thinking = ds4_think_mode_enabled(cli_effective_think_mode(&cfg->gen)),
                .in_think = ds4_think_mode_enabled(cli_effective_think_mode(&cfg->gen)),
                .use_color = isatty(fileno(stdout)) != 0,
                .last_output_newline = true,
            };
            cli_prefill_progress progress = {
                .base_tokens = 0,
                .input_tokens = prompt.len,
                .use_color = ds4_log_is_tty(stderr),
            };
            rc = ds4_engine_generate_argmax(engine, &prompt, cfg->gen.n_predict,
                                            cfg->gen.ctx_size,
                                            print_generated_token,
                                            generation_done,
                                            &printer,
                                            cli_prefill_progress_cb,
                                            &progress);
        }
    }

    ds4_tokens_free(&prompt);
    return rc;
}

static char *trim_inplace(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

static void print_repl_help(void) {
    puts("Commands:");
    puts("  /help          Show this help.");
    puts("  /think [N]     Use normal thinking, or V4.1 effort 0..100 (0 disables thinking).");
    puts("  /think-max     Use maximum thinking (V4.1: 100; V4: requires ctx >= 393216).");
    puts("  /nothink       Disable thinking mode.");
    puts("  /ctx N         Set context size for following prompts.");
    puts("  /power N       Set GPU duty cycle percentage, 1..100.");
    puts("  /steer F       Set FFN steering for subsequent tokens; no value shows it.");
    puts("  /read FILE     Submit a text file, PNG, or JPEG.");
    puts("  /quit, /exit   Leave the prompt.");
    puts("  Ctrl+C         Stop generation and return to the prompt.");
}

static bool parse_power_percent(const char *arg, int *out) {
    char *end = NULL;
    long v = strtol(arg, &end, 10);
    if (!arg[0] || *end != '\0' || v < 1 || v > 100) return false;
    *out = (int)v;
    return true;
}

static bool parse_steering_level(const char *arg, float *out) {
    char *end = NULL;
    errno = 0;
    float v = strtof(arg, &end);
    if (!arg[0] || *end != '\0' || errno == ERANGE || !isfinite(v) ||
        v < -100.0f || v > 100.0f) {
        return false;
    }
    *out = v;
    return true;
}

static void history_file_path(char *buf, size_t len) {
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(buf, len, "%s/.ds4_history", home);
}

typedef struct {
    ds4_session *session;
    ds4_tokens transcript;
    ds4_vision_span *images;
    size_t image_count;
    size_t image_cap;
    int ctx_size;
    int think_prefix_pos;
    int think_prefix_tokens;
    bool initial_system_text;
} repl_chat;

static void repl_chat_free(repl_chat *chat);

static void repl_chat_trim_images(repl_chat *chat, size_t count) {
    while (chat->image_count > count) {
        chat->image_count--;
        ds4_vision_embedding_free(&chat->images[chat->image_count].embedding);
    }
}

static ds4_vision_span *repl_chat_add_image(repl_chat *chat) {
    if (chat->image_count == chat->image_cap) {
        size_t cap = chat->image_cap ? chat->image_cap * 2u : 4u;
        if (cap > SIZE_MAX / sizeof(chat->images[0])) return NULL;
        void *next = realloc(chat->images, cap * sizeof(chat->images[0]));
        if (!next) return NULL;
        chat->images = next;
        chat->image_cap = cap;
    }
    ds4_vision_span *span = &chat->images[chat->image_count++];
    memset(span, 0, sizeof(*span));
    return span;
}

static void tokens_insert(ds4_tokens *dst, int pos, const ds4_tokens *src) {
    if (!src || src->len <= 0) return;
    if (pos < 0) pos = 0;
    if (pos > dst->len) pos = dst->len;
    while (dst->len + src->len > dst->cap) {
        dst->cap = dst->cap ? dst->cap * 2 : 64;
        int *next = realloc(dst->v, (size_t)dst->cap * sizeof(dst->v[0]));
        if (!next) {
            perror("ds4: realloc");
            exit(1);
        }
        dst->v = next;
    }
    memmove(dst->v + pos + src->len, dst->v + pos,
            (size_t)(dst->len - pos) * sizeof(dst->v[0]));
    memcpy(dst->v + pos, src->v, (size_t)src->len * sizeof(src->v[0]));
    dst->len += src->len;
}

static void tokens_remove(ds4_tokens *dst, int pos, int n) {
    if (n <= 0 || pos < 0 || pos >= dst->len) return;
    if (pos + n > dst->len) n = dst->len - pos;
    memmove(dst->v + pos, dst->v + pos + n,
            (size_t)(dst->len - pos - n) * sizeof(dst->v[0]));
    dst->len -= n;
}

static void repl_chat_build_think_prefix(ds4_engine *engine,
                                         ds4_think_mode mode,
                                         ds4_tokens *prefix) {
    ds4_chat_append_think_prefix(engine, prefix, mode);
}

/* Insert/replace the model-family thinking prefix inside the existing
 * transcript.  It lives immediately after the BOS sequence and before any
 * user/system text, matching the GGUF chat templates. */
static bool repl_chat_apply_think_prefix(ds4_engine *engine,
                                         repl_chat *chat,
                                         ds4_think_mode mode) {
    ds4_tokens prefix = {0};
    repl_chat_build_think_prefix(engine, mode, &prefix);
    if (ds4_engine_is_deepseek41(engine) && chat->initial_system_text && !prefix.len)
        ds4_chat_append_message(engine, &prefix, "system", "");

    bool same = chat->think_prefix_tokens == prefix.len;
    if (same && prefix.len > 0) {
        same = !memcmp(chat->transcript.v + chat->think_prefix_pos,
                       prefix.v,
                       (size_t)prefix.len * sizeof(prefix.v[0]));
    }
    if (!same) {
        const int old_prefix_tokens = chat->think_prefix_tokens;
        const int shift = prefix.len - old_prefix_tokens;
        if (chat->ctx_size > 0 && (int64_t)chat->transcript.len + shift >= chat->ctx_size) {
            fprintf(stderr, "ds4: no context room to change the thinking prefix\n");
            ds4_tokens_free(&prefix);
            return false;
        }
        tokens_remove(&chat->transcript, chat->think_prefix_pos,
                      chat->think_prefix_tokens);
        tokens_insert(&chat->transcript, chat->think_prefix_pos, &prefix);
        chat->think_prefix_tokens = prefix.len;
        for (size_t i = 0; i < chat->image_count; i++) {
            chat->images[i].token_start =
                (uint32_t)((int64_t)chat->images[i].token_start + shift);
        }
        if (chat->session) ds4_session_invalidate(chat->session);
    }
    ds4_tokens_free(&prefix);
    return true;
}

static int repl_chat_create_session(ds4_engine *engine, repl_chat *chat, int ctx_size) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "ds4: interactive chat KV cache requires a session backend\n");
        return 1;
    }
    if (chat->session) ds4_session_free(chat->session);
    chat->session = session;
    chat->ctx_size = ctx_size;
    return 0;
}

static int repl_chat_init(ds4_engine *engine, repl_chat *chat, const cli_config *cfg) {
    memset(chat, 0, sizeof(*chat));
    chat->initial_system_text = cfg->gen.system && cfg->gen.system[0];
    ds4_chat_begin(engine, &chat->transcript);
    chat->think_prefix_pos = chat->transcript.len;
    repl_chat_apply_think_prefix(engine, chat, cli_effective_think_mode(&cfg->gen));
    if (cfg->gen.system && cfg->gen.system[0]) {
        ds4_chat_append_message(engine, &chat->transcript, "system", cfg->gen.system);
    }
    ds4_prompt_prefix_append(engine, &chat->transcript, &cfg->gen.prefix);
    if (repl_chat_create_session(engine, chat, cfg->gen.ctx_size) != 0) {
        ds4_tokens_free(&chat->transcript);
        return 1;
    }
    if (cfg->gen.prefix.count == 0) return 0;
    if (cli_wait_distributed_route(cfg, chat->session) != 0) {
        repl_chat_free(chat);
        return 1;
    }

    char err[160] = {0};
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = chat->transcript.len,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(chat->session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(chat->session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    cli_dist_busy_set(cfg, true);
    int rc = ds4_session_sync(chat->session, &chat->transcript,
                              err, sizeof(err));
    cli_dist_busy_set(cfg, false);
    ds4_session_set_progress(chat->session, NULL, NULL);
    ds4_session_set_display_progress(chat->session, NULL, NULL);
    if (rc != 0) {
        fprintf(stderr, "ds4: prefix prefill failed: %s\n",
                err[0] ? err : "unknown error");
        repl_chat_free(chat);
        return 1;
    }
    return 0;
}

static void repl_chat_free(repl_chat *chat) {
    if (!chat) return;
    ds4_session_free(chat->session);
    ds4_tokens_free(&chat->transcript);
    repl_chat_trim_images(chat, 0);
    free(chat->images);
    memset(chat, 0, sizeof(*chat));
}

static int repl_chat_set_ctx(ds4_engine *engine, repl_chat *chat, int ctx_size) {
    ds4_session_free(chat->session);
    chat->session = NULL;
    chat->ctx_size = 0;
    return repl_chat_create_session(engine, chat, ctx_size);
}

static bool repl_chat_assistant_turn_uses_eos(ds4_engine *engine) {
    return !ds4_engine_is_glm_dsa(engine);
}

/* Run one interactive turn.  The transcript is tentatively extended with user
 * and assistant markers, then ds4_session_sync() decides whether this is a KV
 * continuation.  If prompt processing fails, the transcript rolls back before
 * returning to the prompt. */
static int run_chat_turn(ds4_engine *engine, cli_config *cfg, repl_chat *chat,
                         const char *user_text,
                         ds4_vision_embedding *image) {
    if (!chat->session) {
        fprintf(stderr, "ds4: no active interactive KV cache\n");
        return 1;
    }

    ds4_think_mode think_mode = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                           chat->ctx_size);
    if (!repl_chat_apply_think_prefix(engine, chat, think_mode)) return 1;
    const int rollback_len = chat->transcript.len;
    const size_t rollback_images = chat->image_count;
    if (image) {
        ds4_vision_span *span = repl_chat_add_image(chat);
        const char *text_parts[] = {"", user_text ? user_text : ""};
        char image_error[160] = {0};
        if (!span || !ds4_chat_append_multimodal_message(
                engine, &chat->transcript, "user", text_parts,
                image, 1, span, image_error, sizeof(image_error))) {
            repl_chat_trim_images(chat, rollback_images);
            chat->transcript.len = rollback_len;
            fprintf(stderr, "ds4: failed to add image: %s\n",
                    image_error[0] ? image_error : "out of memory");
            return 1;
        }
    } else {
        ds4_chat_append_message(engine, &chat->transcript, "user", user_text);
    }
    ds4_chat_append_assistant_prefix(engine, &chat->transcript, think_mode);

    const int old_pos = ds4_session_pos(chat->session);
    const int common = ds4_session_common_prefix(chat->session, &chat->transcript);
    const int cached = common == old_pos && chat->transcript.len >= old_pos ? common : 0;
    const int suffix = chat->transcript.len - cached;

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = cached,
        .input_tokens = suffix,
        .use_color = ds4_log_is_tty(stderr),
    };
    const double t_prefill0 = cli_now_sec();
    ds4_session_set_progress(chat->session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(chat->session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    cli_dist_busy_set(cfg, true);
    int sync_rc;
    {   /* llms PFPAIR (apply_pfpair.py): LLMS_DS4_PFPAIR=name -- the turn prefilled twice on the same tokens, ALT 'name' forced off / on */
        static const char *pf_name = (const char *)1; static int pf_turn;
        if (pf_name == (const char *)1) { pf_name = getenv("LLMS_DS4_PFPAIR"); if (pf_name && !*pf_name) pf_name = NULL;
                                          if (pf_name) fprintf(stderr, "llms: pfpair '%s' (each turn prefilled twice: ALT forced off / on, a snapshot restored between) -- INSTRUMENT\n", pf_name); }
        ds4_session_snapshot pf_snap = {0}; char pf_err[160] = {0};
        if (pf_name && cached > 0 && suffix > 0 && chat->image_count == 0 && ds4_session_save_snapshot(chat->session, &pf_snap, pf_err, sizeof pf_err) == 0) {
            extern void ds4_gpu_llms_alt_force(const char *names);
            const int off_first = (pf_turn & 1) == 0; double ms_off = -1.0, ms_on = -1.0; sync_rc = 0;
            static int pf_warm = -1; if (pf_warm < 0) { const char *e = getenv("LLMS_DS4_PFPAIR_WARM"); pf_warm = e && *e ? atoi(e) : 0; if (pf_warm < 0 || pf_warm > 4) pf_warm = 0;   /* llms PFPAIR2 (apply_pfpair2.py) */
                                                          if (pf_warm) fprintf(stderr, "llms: pfpair warm %d (discarded warm-up prefills of each turn before the measured pair)\n", pf_warm); }
            for (int k = 0; k < pf_warm + 2 && sync_rc == 0; k++) {
                const int m = k - pf_warm;   /* < 0: a warm-up, alternating off / on */
                const int off = m < 0 ? (k & 1) == 0 : (m == 0 ? off_first : !off_first);
                ds4_gpu_llms_alt_force(off ? pf_name : "");
                const double t0 = cli_now_sec();
                sync_rc = ds4_session_sync_multimodal(chat->session, &chat->transcript, chat->images, chat->image_count, err, sizeof(err));
                const double dt = (cli_now_sec() - t0) * 1000.0; if (m >= 0) { if (off) ms_off = dt; else ms_on = dt; }
                if (k < pf_warm + 1 && sync_rc == 0 && ds4_session_load_snapshot(chat->session, &pf_snap, pf_err, sizeof pf_err) != 0) { fprintf(stderr, "llms pfpair: restore failed (%s) -- the turn keeps this run\n", pf_err); break; }
            }
            ds4_gpu_llms_alt_force("");
            if (sync_rc == 0 && ms_off >= 0.0 && ms_on >= 0.0) fprintf(stderr, "llms pfpair: turn %d rows %d first %s ms_off %.3f ms_on %.3f warm %d\n", pf_turn, suffix, off_first ? "off" : "on", ms_off, ms_on, pf_warm);
            ds4_session_snapshot_free(&pf_snap); pf_turn++;
        } else {
            if (pf_name && pf_snap.ptr) ds4_session_snapshot_free(&pf_snap);
            sync_rc = ds4_session_sync_multimodal(chat->session,
                                              &chat->transcript,
                                              chat->images,
                                              chat->image_count,
                                              err,
                                              sizeof(err));
        }
    }
    cli_dist_busy_set(cfg, false);
    if (sync_rc != 0) {
        ds4_session_set_progress(chat->session, NULL, NULL);
        ds4_session_set_display_progress(chat->session, NULL, NULL);
        chat->transcript.len = rollback_len;
        repl_chat_trim_images(chat, rollback_images);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        return 1;
    }
    ds4_session_set_progress(chat->session, NULL, NULL);
    ds4_session_set_display_progress(chat->session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();

    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(chat->session) - ds4_session_pos(chat->session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    const bool speculative_argmax = cfg->gen.temperature <= 0.0f &&
        ((ds4_engine_mtp_draft_tokens(engine) > 1 &&
          getenv("DS4_MTP_SPEC_DISABLE") == NULL) ||
         cli_splitkv_spec_requested());
    const bool greedy_argmax = cfg->gen.temperature <= 0.0f &&
        cli_greedy_argmax_requested(speculative_argmax);
    bool have_greedy_next = false;
    int greedy_next = -1;
    const double t_decode0 = cli_now_sec();
    while (generated < max_tokens && !cli_interrupt_requested()) {
        int token;
        if (greedy_argmax && have_greedy_next) {
            token = greedy_next;
            have_greedy_next = false;
        } else {
            token = ds4_session_sample(chat->session,
                                       cfg->gen.temperature,
                                       0,
                                       cfg->gen.top_p,
                                       cfg->gen.min_p,
                                       &rng);
        }
        if (ds4_token_is_stop_for_think_mode(engine, token, think_mode)) break;

        int toks[17];
        int ntok = 0;
        if (ds4_engine_mtp_draft_tokens(engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
            cli_dist_busy_set(cfg, true);
            ntok = ds4_session_eval_speculative(
                chat->session, token, max_tokens - generated,
                ds4_token_eos(engine), cfg->gen.temperature, 0,
                cfg->gen.top_p, cfg->gen.min_p, &rng,
                toks, (int)(sizeof(toks) / sizeof(toks[0])),
                err, sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (ntok < 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                return 1;
            }
        } else {
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, token, &piece_len);
            ds4_tokens_push(&chat->transcript, token);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            free(piece);
            generated++;

            cli_dist_busy_set(cfg, true);
            int eval_rc = ds4_session_eval(chat->session, token, err, sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (eval_rc != 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_invalidate(chat->session);
                return 1;
            }
            if (generated >= max_tokens) break;
            continue;
        }

        bool stop = false;
        for (int j = 0; j < ntok; j++) {
            if (ds4_token_is_stop_for_think_mode(engine, toks[j], think_mode)) {
                stop = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, toks[j], &piece_len);
            ds4_tokens_push(&chat->transcript, toks[j]);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            free(piece);
            generated++;
            if (generated >= max_tokens) break;
        }
        if (stop) break;
    }
    const double t_decode1 = cli_now_sec();
    generation_done(&printer);

    const bool interrupted = cli_interrupt_requested();
    if (interrupted && generated == 0) {
        chat->transcript.len = rollback_len;
        repl_chat_trim_images(chat, rollback_images);
        ds4_session_invalidate(chat->session);
    } else if (repl_chat_assistant_turn_uses_eos(engine)) {
        ds4_tokens_push(&chat->transcript, ds4_token_eos(engine));
    }

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    if (interrupted) cli_interrupt_clear();
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)suffix / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);
    return 0;
}

static bool cli_file_has_image_magic(const char *path) {
    unsigned char magic[8] = {0};
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    size_t n = fread(magic, 1, sizeof(magic), fp);
    fclose(fp);
    return (n >= 8 && !memcmp(magic, "\x89PNG\r\n\x1a\n", 8)) ||
           (n >= 2 && magic[0] == 0xff && magic[1] == 0xd8);
}

static int run_repl(ds4_engine *engine, cli_config *cfg) {
    repl_chat chat;
    if (repl_chat_init(engine, &chat, cfg) != 0) return 1;

    struct sigaction old_int;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = cli_sigint_handler;
    bool sigint_installed = sigaction(SIGINT, &sa, &old_int) == 0;
    cli_interrupt_clear();

    char hist[PATH_MAX];
    history_file_path(hist, sizeof(hist));
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(512);
    linenoiseHistoryLoad(hist);
    print_repl_help();

    int rc = 0;
    for (;;) {
        errno = 0;
        char *line = linenoise("ds4> ");
        if (!line) {
            if (errno == EAGAIN || cli_interrupt_requested()) {
                cli_interrupt_clear();
                continue;
            }
            break;
        }
        char *cmd = trim_inplace(line);
        if (!cmd[0]) {
            linenoiseFree(line);
            continue;
        }
        linenoiseHistoryAdd(cmd);
        linenoiseHistorySave(hist);

        if (!strcmp(cmd, "/help")) {
            print_repl_help();
        } else if (!strncmp(cmd, "/think", 6) &&
                   (cmd[6] == '\0' || isspace((unsigned char)cmd[6]))) {
            const char *arg = trim_inplace(cmd + 6);
            ds4_think_mode mode = DS4_THINK_HIGH;
            if (arg[0] && (!ds4_engine_is_deepseek41(engine) ||
                           !ds4_think_mode_parse_level(arg, &mode))) {
                fprintf(stderr, "ds4: /think N requires V4.1 and a level from 0 to 100\n");
            } else if (repl_chat_apply_think_prefix(engine, &chat, mode)) {
                cfg->gen.think_mode = mode;
                printf("Thinking mode: %s.\n", ds4_think_mode_name(mode));
            }
        } else if (!strcmp(cmd, "/think-max")) {
            bool active = ds4_think_mode_for_context(DS4_THINK_MAX,
                                                     chat.ctx_size) == DS4_THINK_MAX;
            if (repl_chat_apply_think_prefix(engine, &chat,
                                              active ? DS4_THINK_MAX : DS4_THINK_HIGH)) {
                cfg->gen.think_mode = DS4_THINK_MAX;
                cli_warn_think_max_downgraded(&cfg->gen, "/think-max");
                printf("Thinking mode: %s.\n", active ? "max" : "high (ctx below 393216)");
            }
        } else if (!strcmp(cmd, "/nothink")) {
            if (repl_chat_apply_think_prefix(engine, &chat, DS4_THINK_NONE)) {
                cfg->gen.think_mode = DS4_THINK_NONE;
                puts("Thinking mode: none.");
            }
        } else if (!strncmp(cmd, "/power", 6) && (cmd[6] == '\0' || isspace((unsigned char)cmd[6]))) {
            char *arg = trim_inplace(cmd + 6);
            if (!arg[0]) {
                printf("Power: %d%%.\n", ds4_session_power(chat.session));
            } else {
                int power = 0;
                if (!parse_power_percent(arg, &power)) {
                    fprintf(stderr, "ds4: /power must be between 1 and 100\n");
                } else if (ds4_session_set_power(chat.session, power) != 0) {
                    fprintf(stderr, "ds4: failed to set /power\n");
                } else {
                    cfg->engine.power_percent = power;
                    printf("Power: %d%%.\n", power);
                }
            }
        } else if (!strncmp(cmd, "/steer", 6) &&
                   (cmd[6] == '\0' || isspace((unsigned char)cmd[6]))) {
            char *arg = trim_inplace(cmd + 6);
            if (!arg[0]) {
                printf("Steering FFN: %g.\n",
                       (double)ds4_session_directional_steering_ffn(chat.session));
            } else {
                float scale = 0.0f;
                if (!parse_steering_level(arg, &scale)) {
                    fprintf(stderr, "ds4: /steer must be between -100 and 100\n");
                } else if (ds4_session_set_directional_steering_ffn(
                                   chat.session, scale) == 0) {
                    cfg->engine.directional_steering_ffn = scale;
                    printf("Steering FFN: %g.\n", (double)scale);
                }
            }
        } else if (!strncmp(cmd, "/ctx", 4) && (cmd[4] == '\0' || isspace((unsigned char)cmd[4]))) {
            char *arg = trim_inplace(cmd + 4);
            if (!arg[0]) {
                fprintf(stderr, "ds4: /ctx needs a positive integer\n");
            } else {
                cfg->gen.ctx_size = parse_int(arg, "/ctx");
                log_context_memory(cfg->engine.backend,
                                   cfg->gen.ctx_size,
                                   ds4_engine_prefill_chunk(engine),
                                   cfg->engine.ssd_streaming);
                rc = repl_chat_set_ctx(engine, &chat, cfg->gen.ctx_size);
                if (rc != 0) {
                    linenoiseFree(line);
                    break;
                }
                ds4_think_mode effective = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                                      chat.ctx_size);
                repl_chat_apply_think_prefix(engine, &chat, effective);
                cli_warn_think_max_downgraded(&cfg->gen, "/ctx");
            }
        } else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
            linenoiseFree(line);
            break;
        } else if (!strncmp(cmd, "/read", 5) && (cmd[5] == '\0' || isspace((unsigned char)cmd[5]))) {
            char *path = trim_inplace(cmd + 5);
            if (!path[0]) {
                fprintf(stderr, "ds4: /read needs a file path\n");
            } else if (cli_file_has_image_magic(path)) {
                char image_error[256] = {0};
                ds4_vision_embedding image = {0};
                if (!ds4_engine_vision_encode_file(engine, path, &image,
                                                   image_error,
                                                   sizeof(image_error))) {
                    fprintf(stderr, "ds4: /read image failed: %s\n", image_error);
                } else {
                    fprintf(stderr,
                            "ds4: image %ux%u, %u image tokens\n",
                            image.width, image.height, image.token_count);
                    rc = run_chat_turn(engine, cfg, &chat, "", &image);
                    ds4_vision_embedding_free(&image);
                }
            } else {
                char *prompt = read_prompt_file(path, false);
                if (prompt) {
                    rc = run_chat_turn(engine, cfg, &chat, prompt, NULL);
                    free(prompt);
                }
            }
        } else if (cmd[0] == '/') {
            fprintf(stderr, "ds4: unknown command: %s\n", cmd);
            fprintf(stderr, "ds4: type /help for commands\n");
        } else {
            rc = run_chat_turn(engine, cfg, &chat, cmd, NULL);
        }
        linenoiseFree(line);
    }
    if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
    repl_chat_free(&chat);
    return rc;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static char *read_prompt_file(const char *path, bool fatal) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open prompt file: %s\n", path);
        if (fatal) exit(2);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4: failed to seek prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fprintf(stderr, "ds4: failed to size prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    rewind(fp);

    char *buf = malloc((size_t)len + 1);
    if (!buf) {
        fprintf(stderr, "ds4: out of memory reading prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    size_t nread = fread(buf, 1, (size_t)len, fp);
    if (nread != (size_t)len) {
        fprintf(stderr, "ds4: failed to read prompt file: %s\n", path);
        free(buf);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close prompt file: %s\n", path);
        free(buf);
        if (fatal) exit(2);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

static cli_config parse_options(int argc, char **argv) {
    cli_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
        },
        .gen = {
            .prompt = NULL,
            .system = "You are a helpful assistant",
            .n_predict = 50000,
            .ctx_size = 32768,
            .temperature = DS4_DEFAULT_TEMPERATURE,
            .top_p = DS4_DEFAULT_TOP_P,
            .min_p = DS4_DEFAULT_MIN_P,
            .dump_logprobs_top_k = 20,
            .think_mode = DS4_THINK_HIGH,
        },
    };

    c.dist = ds4_dist_options_create();
    if (!c.dist) {
        fprintf(stderr, "ds4: out of memory creating distributed options\n");
        exit(1);
    }

    bool directional_steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse = ds4_dist_parse_cli_arg(arg,
                                                                      &i,
                                                                      argc,
                                                                      argv,
                                                                      c.dist,
                                                                      dist_parse_err,
                                                                      sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr, "ds4: %s\n", dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        char tp_parse_err[256] = {0};
        ds4_tp_cli_parse_result tp_parse = ds4_tp_parse_cli_arg(arg,
                                                                &i,
                                                                argc,
                                                                argv,
                                                                &c.engine.tp,
                                                                tp_parse_err,
                                                                sizeof(tp_parse_err));
        if (tp_parse == DS4_TP_CLI_ERROR) {
            fprintf(stderr, "ds4: %s\n", tp_parse_err[0] ? tp_parse_err : "invalid tensor-parallel option");
            exit(2);
        }
        if (tp_parse == DS4_TP_CLI_MATCHED) continue;

        if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.gen.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.prompt_owned = read_prompt_file(need_arg(&i, argc, argv, arg), true);
            c.gen.prompt = c.prompt_owned;
        } else if (!strcmp(arg, "--prefix-file")) {
            if (c.gen.prefix.count != 0) {
                fprintf(stderr, "ds4: specify --prefix-file only once\n");
                exit(2);
            }
            const char *path = need_arg(&i, argc, argv, arg);
            char error[256] = {0};
            if (ds4_prompt_prefix_load(&c.gen.prefix, path,
                                       error, sizeof(error)) != 0) {
                fprintf(stderr, "ds4: %s\n",
                        error[0] ? error : "invalid prefix file");
                exit(2);
            }
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--raw") || !strcmp(arg, "--raw-prompt")) {
            c.gen.raw_prompt = true;
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--vision")) {
            c.engine.vision_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.glm_mtp = true;
        } else if (!strcmp(arg, "--mtp-model")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "--mtp-timing")) {
            c.engine.glm_mtp = true;
            c.engine.glm_mtp_timing = true;
        } else if (!strcmp(arg, "--dspark")) {
            c.engine.dspark = true;
        } else if (!strcmp(arg, "--dspark-confidence")) {
            c.engine.dspark = true;
            c.engine.dspark_confidence_threshold =
                parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.engine.dspark_confidence_threshold_set = true;
        } else if (!strcmp(arg, "--dspark-strict")) {
            c.engine.dspark = true;
            c.engine.dspark_strict = true;
        } else if (!strcmp(arg, "--mtp-exact-sampling")) {
            c.engine.dspark_exact_sampling = true;
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
            c.gen.temperature_set = true;
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.gen.top_p_set = true;
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.gen.min_p_set = true;
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.engine.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.engine.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.engine.ssd_streaming_cache_experts = experts;
            c.engine.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.engine.ssd_streaming_full_layers = (uint32_t)v;
            c.engine.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.engine.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.engine.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4: --prefill-chunk must be positive\n");
                exit(2);
            }
            c.engine.prefill_chunk = (uint32_t)v;
        } else if (!strcmp(arg, "--power")) {
            c.engine.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.engine.power_percent < 1 || c.engine.power_percent > 100) {
                fprintf(stderr, "ds4: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-profile")) {
            c.engine.expert_profile_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.engine.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--gpu-vram")) {
            c.gpu_vram_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu-devices")) {
            c.gpu_devices_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--cuda-tensor-parallel")) {
            c.engine.cuda_tensor_parallel = true;
        } else if (!strcmp(arg, "--dump-tokens")) {
            c.gen.dump_tokens = true;
        } else if (!strcmp(arg, "--dump-logits")) {
            c.gen.dump_logits_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-logprobs")) {
            c.gen.dump_logprobs_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--logprobs-top-k")) {
            c.gen.dump_logprobs_top_k = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--decode-consistency")) {
            c.gen.decode_consistency_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--perplexity-file")) {
            c.gen.perplexity_file_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--imatrix-dataset")) {
            c.gen.imatrix_dataset_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--imatrix-out")) {
            c.gen.imatrix_output_path = need_arg(&i, argc, argv, arg);
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--imatrix-max-prompts")) {
            c.gen.imatrix_max_prompts = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--imatrix-max-tokens")) {
            c.gen.imatrix_max_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--imatrix-min-expert-samples")) {
            c.gen.imatrix_min_expert_samples =
                parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think-level")) {
            if (!ds4_think_mode_parse_level(need_arg(&i, argc, argv, arg), &c.gen.think_mode)) {
                fprintf(stderr, "ds4: --think-level requires an integer from 0 to 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--think")) {
            c.gen.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--head-test")) {
            c.gen.head_test = true;
        } else if (!strcmp(arg, "--first-token-test")) {
            c.gen.first_token_test = true;
        } else if (!strcmp(arg, "--metal-graph-test")) {
            c.gen.metal_graph_test = true;
#ifdef DS4_ROCM_BUILD
            c.engine.backend = DS4_BACKEND_CUDA;
#else
            c.engine.backend = DS4_BACKEND_METAL;
#endif
        } else if (!strcmp(arg, "--metal-graph-full-test")) {
            c.gen.metal_graph_full_test = true;
#ifdef DS4_ROCM_BUILD
            c.engine.backend = DS4_BACKEND_CUDA;
#else
            c.engine.backend = DS4_BACKEND_METAL;
#endif
        } else if (!strcmp(arg, "--metal-graph-prompt-test")) {
            c.gen.metal_graph_prompt_test = true;
#ifdef DS4_ROCM_BUILD
            c.engine.backend = DS4_BACKEND_CUDA;
#else
            c.engine.backend = DS4_BACKEND_METAL;
#endif
        } else if (!strcmp(arg, "--metal-graph-generate")) {
            fprintf(stderr, "ds4: --metal-graph-generate was removed; --metal is the graph path\n");
            exit(2);
        } else if (!strcmp(arg, "--inspect")) {
            c.inspect = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--server")) {
            fprintf(stderr, "ds4: use ds4-server for the HTTP server\n");
            exit(2);
        } else {
            fprintf(stderr, "ds4: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    if (c.gen.imatrix_output_path && !c.gen.imatrix_dataset_path) {
        fprintf(stderr, "ds4: --imatrix-out requires --imatrix-dataset\n");
        exit(2);
    }
    if (c.gen.imatrix_dataset_path && !c.gen.imatrix_output_path) {
        fprintf(stderr, "ds4: --imatrix-dataset requires --imatrix-out\n");
        exit(2);
    }
    if (c.gen.imatrix_min_expert_samples < 0) {
        fprintf(stderr, "ds4: --imatrix-min-expert-samples must not be negative\n");
        exit(2);
    }
    if (c.gen.perplexity_file_path && c.gen.prompt) {
        fprintf(stderr, "ds4: --perplexity-file does not use -p/--prompt-file\n");
        exit(2);
    }
    if (c.gen.prefix.count != 0 && c.gen.raw_prompt) {
        fprintf(stderr, "ds4: --prefix-file cannot be combined with --raw-prompt\n");
        exit(2);
    }
    if (c.gen.prefix.count != 0 && is_rendered_chat_prompt(c.gen.prompt)) {
        fprintf(stderr,
                "ds4: --prefix-file cannot be combined with an already-rendered prompt\n");
        exit(2);
    }
    char tp_err[256];
    if (!ds4_tp_adopt_distributed_options(&c.engine.tp, c.dist,
                                          tp_err, sizeof(tp_err))) {
        fprintf(stderr, "ds4: %s\n", tp_err);
        exit(2);
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(c.dist, &c.engine, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4: %s\n", dist_err);
        exit(2);
    }
    if (!ds4_tp_validate_engine_options(&c.engine, tp_err, sizeof(tp_err))) {
        fprintf(stderr, "ds4: %s\n", tp_err);
        exit(2);
    }

    return c;
}

int main(int argc, char **argv) {
    cli_config cfg = parse_options(argc, argv);
    if (cfg.gen.dump_tokens) {
        if (cfg.gen.prefix.count != 0) {
            fprintf(stderr, "ds4: --dump-tokens does not support --prefix-file\n");
            ds4_prompt_prefix_free(&cfg.gen.prefix);
            ds4_dist_options_free(cfg.dist);
            free(cfg.prompt_owned);
            return 2;
        }
        if (cfg.gen.prompt == NULL) {
            fprintf(stderr, "ds4: --dump-tokens requires -p or --prompt-file\n");
            ds4_prompt_prefix_free(&cfg.gen.prefix);
            ds4_dist_options_free(cfg.dist);
            free(cfg.prompt_owned);
            return 2;
        }
        int rc;
        if (cfg.gen.raw_prompt || is_rendered_chat_prompt(cfg.gen.prompt)) {
            rc = ds4_dump_text_tokenization(cfg.engine.model_path,
                                            cfg.gen.prompt,
                                            stdout);
        } else {
            rc = ds4_dump_chat_tokenization(cfg.engine.model_path,
                                            cfg.gen.system,
                                            cfg.gen.prompt,
                                            cfg.gen.think_mode,
                                            cfg.gen.ctx_size,
                                            stdout);
        }
        ds4_dist_options_free(cfg.dist);
        ds4_prompt_prefix_free(&cfg.gen.prefix);
        free(cfg.prompt_owned);
        return rc;
    }
    cfg.engine.inspect_only = cfg.inspect;
    cfg.engine.first_token_test = cfg.gen.first_token_test;
    cfg.engine.metal_graph_test = cfg.gen.metal_graph_test;
    cfg.engine.context_size = cfg.gen.ctx_size;
    cfg.engine.placement_ctx_hint = cfg.gen.ctx_size;
    ds4_engine *engine = NULL;
    if (cfg.gpu_vram_arg || cfg.gpu_devices_arg) {
        ds4_gpu_config gpu_cfg = {0};
        bool skip_cuda = false;
        char errbuf[256];
        if (parse_gpu_vram_arg(cfg.gpu_vram_arg, cfg.gpu_devices_arg,
                               &gpu_cfg, &skip_cuda,
                               errbuf, sizeof(errbuf)) != 0) {
            fprintf(stderr, "ds4: %s\n", errbuf);
            ds4_dist_options_free(cfg.dist);
            free(cfg.prompt_owned);
            return 2;
        }
        cfg.engine.backend = skip_cuda ? DS4_BACKEND_CPU : DS4_BACKEND_CUDA;
        if (skip_cuda) {
            if ((llms_adopted_engine_defaults(&cfg.engine), ds4_engine_open(&engine, &cfg.engine)) != 0) {
                ds4_dist_options_free(cfg.dist);
                free(cfg.prompt_owned);
                return 1;
            }
        } else {
            const bool was_auto =
                (cfg.gpu_vram_arg && !strcmp(cfg.gpu_vram_arg, "auto")) ||
                (!cfg.gpu_vram_arg && cfg.gpu_devices_arg);
            char layout[256];
            if (format_gpu_layout_line(&gpu_cfg, was_auto,
                                       layout, sizeof(layout)) > 0) {
                fprintf(stdout, "%s\n", layout);
                fflush(stdout);
            }
            if (ds4_engine_create_with_gpu_config(&engine, &cfg.engine,
                                                   &gpu_cfg) != 0) {
                ds4_dist_options_free(cfg.dist);
                free(cfg.prompt_owned);
                return 1;
            }
        }
    } else if ((llms_adopted_engine_defaults(&cfg.engine), ds4_engine_open(&engine, &cfg.engine)) != 0) {
        ds4_dist_options_free(cfg.dist);
        free(cfg.prompt_owned);
        return 1;
    }
    if (ds4_think_mode_level(cfg.gen.think_mode) >= 0 && !ds4_engine_is_deepseek41(engine)) {
        fprintf(stderr, "ds4: --think-level requires a DeepSeek V4.1 model\n");
        ds4_engine_close(engine);
        ds4_dist_options_free(cfg.dist);
        ds4_prompt_prefix_free(&cfg.gen.prefix);
        free(cfg.prompt_owned);
        return 2;
    }
    cli_apply_model_sampling_defaults(engine, &cfg.gen);
    if (cfg.engine.tp.role == DS4_TP_WORKER) {
        int rc = ds4_tp_worker_run(engine, &cfg.engine.tp);
        ds4_engine_close(engine);
        ds4_dist_options_free(cfg.dist);
        free(cfg.prompt_owned);
        return rc;
    }
    ds4_tp *tp_leader = NULL;
    if (cfg.engine.tp.role == DS4_TP_LEADER) {
        char tp_err[256] = "";
        ds4_tp_identity tp_id = {
            .gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = (uint32_t)ds4_engine_model_id(engine),
            .n_layer = (uint32_t)ds4_engine_layer_count(engine),
            .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
            .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
            .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
            .ctx_size = (uint32_t)cfg.gen.ctx_size,
        };
        ds4_engine_tp_gate_schedule(engine,
                                    &tp_id.gate_slot_start,
                                    &tp_id.gate_slot_step,
                                    &tp_id.gates_per_token,
                                    tp_id.gate_slot_mask);
        if (!ds4_tp_create(&tp_leader, &cfg.engine.tp, &tp_id, tp_err, sizeof(tp_err)) ||
            !ds4_engine_tp_bind(engine, tp_leader, tp_err, sizeof(tp_err))) {
            fprintf(stderr, "ds4: %s\n", tp_err);
            ds4_tp_free(tp_leader);
            ds4_engine_close(engine);
            ds4_dist_options_free(cfg.dist);
            free(cfg.prompt_owned);
            return 1;
        }
    }
    if (cfg.dist && cfg.dist->role == DS4_DISTRIBUTED_WORKER) {
        ds4_dist_generation_options dist_gen = {
            .prompt = cfg.gen.prompt,
            .system = cfg.gen.system,
            .dump_logits_path = cfg.gen.dump_logits_path,
            .dump_logprobs_path = cfg.gen.dump_logprobs_path,
            .dump_logprobs_top_k = cfg.gen.dump_logprobs_top_k,
            .n_predict = cfg.gen.n_predict,
            .ctx_size = cfg.gen.ctx_size,
            .temperature = cfg.gen.temperature,
            .top_p = cfg.gen.top_p,
            .min_p = cfg.gen.min_p,
            .seed = cfg.gen.seed,
            .think_mode = cfg.gen.think_mode,
        };
        int rc = ds4_dist_run(engine, cfg.dist, &dist_gen);
        ds4_engine_close(engine);
        ds4_dist_options_free(cfg.dist);
        free(cfg.prompt_owned);
        return rc;
    }
    if (!cfg.inspect) {
        log_context_memory(cfg.engine.backend,
                           cfg.gen.ctx_size,
                           ds4_engine_prefill_chunk(engine),
                           cfg.engine.ssd_streaming);
        cli_warn_think_max_downgraded(&cfg.gen, "--think-max");
    }
    int rc = 0;
    if (cfg.inspect) {
        ds4_engine_summary(engine);
    } else if (cfg.gen.imatrix_output_path) {
        rc = ds4_engine_collect_imatrix(engine,
                                        cfg.gen.imatrix_dataset_path,
                                        cfg.gen.imatrix_output_path,
                                        cfg.gen.ctx_size,
                                        cfg.gen.imatrix_max_prompts,
                                        cfg.gen.imatrix_max_tokens,
                                        cfg.gen.imatrix_min_expert_samples);
    } else if (cfg.gen.perplexity_file_path) {
        rc = run_perplexity_file(engine, &cfg);
    } else if (cfg.gen.prompt == NULL) {
        rc = run_repl(engine, &cfg);
    } else {
        rc = run_generation(engine, &cfg);
    }
    if (tp_leader) ds4_tp_send_stop(tp_leader);
    ds4_engine_close(engine);
    ds4_tp_free(tp_leader);
    ds4_dist_options_free(cfg.dist);
    ds4_prompt_prefix_free(&cfg.gen.prefix);
    free(cfg.prompt_owned);
    return rc;
}
