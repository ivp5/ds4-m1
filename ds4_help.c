#include "ds4_help.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    const char *off;
    const char *cyan;
    const char *title;
    const char *yellow;
    const char *grey;
    const char *red;
    const char *white;
    const char *bright;
} help_colors;

static help_colors help_make_colors(FILE *fp) {
    bool color = isatty(fileno(fp));
    help_colors c = {0};
    if (!color) return c;
    c.off = "\x1b[0m";
    c.cyan = "\x1b[38;5;81m";
    c.title = "\x1b[1;38;5;250m";
    c.yellow = "\x1b[38;5;179m";
    c.grey = "\x1b[38;5;240m";
    c.red = "\x1b[38;5;203m";
    c.white = "\x1b[38;5;252m";
    c.bright = "\x1b[1;38;5;231m";
    return c;
}

static void title(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n", c->title ? c->title : "", s, c->off ? c->off : "");
}

static void title_red(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n", c->red ? c->red : "", s, c->off ? c->off : "");
}

static bool option_name_has_switch(const char *name) {
    bool word_start = true;
    while (*name) {
        if (word_start && (*name == '-' || *name == '/')) return true;
        word_start = (*name == ' ');
        name++;
    }
    return false;
}

static void print_colored_option_name(FILE *fp, const help_colors *c, const char *name) {
    bool has_switch = option_name_has_switch(name);
    bool word_start = true;
    while (*name) {
        const char *start = name;
        while (*name && *name != ' ') name++;
        bool is_option = !has_switch || *start == '-' || *start == '/' ||
                         (word_start && has_switch && *start != '[');
        const char *color = is_option ? c->cyan : c->bright;
        if (color) fputs(color, fp);
        fwrite(start, 1, (size_t)(name - start), fp);
        if (color && c->off) fputs(c->off, fp);
        if (*name == ' ') {
            fputc(*name++, fp);
            word_start = false;
        }
    }
}

static void opt(FILE *fp, const help_colors *c, const char *name, const char *desc) {
    if (c->cyan) {
        fputs("  ", fp);
        print_colored_option_name(fp, c, name);
        fprintf(fp, " %s|%s ", c->grey ? c->grey : "", c->grey ? c->off : "");
        fprintf(fp, "%s%s%s\n", c->white ? c->white : "", desc,
                c->white ? c->off : "");
        return;
    }

    const int col = 30;
    int n = (int)strlen(name);
    if (n > col) {
        fprintf(fp, "  %s\n      %s\n", name, desc);
    } else {
        fprintf(fp, "  %-30s %s\n", name, desc);
    }
}

static void para(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n",
            c->yellow ? c->yellow : "", s, c->yellow ? c->off : "");
}

static bool streq(const char *a, const char *b) {
    return a && b && strcmp(a, b) == 0;
}

static bool topic_is(const char *topic, const char *name) {
    return topic && strcmp(topic, name) == 0;
}

static const char *tool_name(ds4_help_tool tool) {
    switch (tool) {
    case DS4_HELP_DS4: return "ds4";
    case DS4_HELP_SERVER: return "ds4-server";
    case DS4_HELP_AGENT: return "ds4-agent";
    case DS4_HELP_BENCH: return "ds4-bench";
    case DS4_HELP_EVAL: return "ds4-eval";
    }
    return "ds4";
}

static const char *tool_usage(ds4_help_tool tool) {
    switch (tool) {
    case DS4_HELP_DS4:
        return "Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]";
    case DS4_HELP_SERVER:
        return "Usage: ds4-server [options]";
    case DS4_HELP_AGENT:
        return "Usage: ds4-agent [options]";
    case DS4_HELP_BENCH:
        return "Usage: ds4-bench (--prompt-file FILE | --chat-prompt-file FILE) [options]";
    case DS4_HELP_EVAL:
        return "Usage: ds4-eval [options]";
    }
    return "Usage: ds4 [options]";
}

static const char *tool_summary(ds4_help_tool tool) {
    switch (tool) {
    case DS4_HELP_DS4:
        return "Chat with a local DwarfStar model, run one-shot prompts, inspect models, or coordinate distributed inference.";
    case DS4_HELP_SERVER:
        return "Serve one loaded DwarfStar model through OpenAI, Responses, Anthropic, and completion-compatible HTTP APIs.";
    case DS4_HELP_AGENT:
        return "Run the native terminal coding agent with live tools, session save/restore, and a responsive prompt while the model works.";
    case DS4_HELP_BENCH:
        return "Measure prefill, decode, context growth, and KV-cache size across repeatable context frontiers.";
    case DS4_HELP_EVAL:
        return "Run the built-in reasoning, math, science, and security evaluation harness with a live terminal UI.";
    }
    return "";
}

static void print_model_runtime(FILE *fp, const help_colors *c,
                                ds4_help_tool tool, bool full) {
    title(fp, c, "Model And Runtime");
    opt(fp, c, "-m, --model FILE", "GGUF model path. Default: ds4flash.gguf");
    if (tool == DS4_HELP_DS4 || tool == DS4_HELP_AGENT || tool == DS4_HELP_SERVER) {
        opt(fp, c, "--vision FILE", "Vision encoder GGUF for the selected model.");
    }
#ifdef DS4_ROCM_BUILD
    opt(fp, c, "--metal | --rocm | --cpu", "Select the backend explicitly.");
    opt(fp, c, "--backend NAME", "Backend name: metal, rocm, or cpu.");
#else
    opt(fp, c, "--metal | --cuda | --cpu", "Select the backend explicitly.");
    opt(fp, c, "--backend NAME", "Backend name: metal, cuda, or cpu.");
    opt(fp, c, "--gpu-vram N[,N,...]|auto", "CUDA VRAM budgets per device, in GiB, or auto-detect free VRAM.");
    opt(fp, c, "--gpu-devices N[,N,...]", "CUDA device indices used by multi-GPU placement.");
    if (tool != DS4_HELP_EVAL) {
        opt(fp, c, "--cuda-tensor-parallel", "Enable the paired DeepSeek tensor/expert path on an even multi-GPU CUDA placement.");
    }
#endif
    if (tool != DS4_HELP_BENCH) {
        opt(fp, c, "-c, --ctx N", "Allocated context tokens.");
    }
    if (tool == DS4_HELP_SERVER) {
        opt(fp, c, "-n, --tokens N", "Default max output tokens when clients omit a limit.");
    }
    opt(fp, c, "-t, --threads N", "CPU helper threads for host-side/reference work.");
    opt(fp, c, "--power N", "GPU duty-cycle target, 1..100. Default: 100");
    opt(fp, c, "--ssd-streaming", "Metal/CUDA/ROCm: opt in to SSD-backed model streaming instead of full residency.");
    opt(fp, c, "--ssd-streaming-cold", "SSD streaming: skip default popularity-based expert-cache preload.");
    opt(fp, c, "--ssd-streaming-cache-experts N|NGB", "SSD streaming cache target. N requests dynamic expert slots; NGB also reserves two full prefill layers. Either may be reduced to fit the model, graph, context, and backend working set.");
    opt(fp, c, "--ssd-streaming-full-layers N", "GLM Metal streaming: keep the first N routed layers fully resident. Default: auto from NGB expert budget; use 0 to disable.");
    opt(fp, c, "--ssd-streaming-preload-experts N", "SSD streaming: upfront popularity preload count. DeepSeek auto-seeds by default; GLM demand-fills unless N is explicit.");
    opt(fp, c, "--simulate-used-memory NGB", "Diagnostic: lock N GiB before model load to simulate a smaller-memory machine.");
    opt(fp, c, "--prefill-chunk N", "Graph prefill chunk size. Default: CUDA TP 2048; PRO long prompts 8192; others 4096.");
    if (full) {
        if (tool == DS4_HELP_EVAL || tool == DS4_HELP_BENCH) {
            opt(fp, c, "--mtp-model FILE", "External MTP or DSpark support GGUF.");
        }
        if (tool == DS4_HELP_DS4 || tool == DS4_HELP_AGENT || tool == DS4_HELP_SERVER) {
            opt(fp, c, "--mtp", "Enable model-embedded MTP speculation.");
            opt(fp, c, "--mtp-model FILE", "External MTP or DSpark support GGUF.");
            opt(fp, c, "--mtp-draft N", "Maximum autoregressive MTP draft tokens. Default: 1");
            opt(fp, c, "--mtp-margin F", "Verifier confidence margin for fast MTP acceptance. Default: 3");
            opt(fp, c, "--mtp-timing", "Enable embedded MTP and print acceptance/timing counters.");
            opt(fp, c, "--dspark", "Enable DSpark using the support GGUF passed with --mtp-model.");
            opt(fp, c, "--dspark-confidence F", "Enable DSpark with confidence pruning threshold 0..1. Greedy/opportunistic default: resident Spark Q2 0.4, two-Spark Flash Q2/MXFP4 0, Metal 0.6, other CUDA/ROCm 0.7; exact sampling: 0.8");
            opt(fp, c, "--mtp-exact-sampling", "Preserve the ordinary temperature distribution instead of accepting target-matching greedy drafts directly.");
            opt(fp, c, "--dspark-strict", "Load DSpark support but keep target-only decode.");
        } else if (tool == DS4_HELP_BENCH) {
            opt(fp, c, "--dspark", "Benchmark greedy DSpark using the support GGUF passed with --mtp-model.");
            opt(fp, c, "--dspark-confidence F", "DSpark confidence pruning threshold 0..1.");
        }
        opt(fp, c, "--quality", "Prefer exact kernels where faster approximate paths exist.");
        opt(fp, c, "--warm-weights", "Touch resident weights at startup to reduce first-use stalls.");
        if (tool == DS4_HELP_DS4 || tool == DS4_HELP_BENCH) {
            opt(fp, c, "--expert-profile FILE", "Metal-only: write routed expert locality/cache simulation JSON.");
        }
    }
    fputc('\n', fp);
}

static void print_sampling(FILE *fp, const help_colors *c, bool full, ds4_help_tool tool) {
    title(fp, c, "Prompt And Sampling");
    opt(fp, c, "-n, --tokens N", "Maximum generated tokens.");
    opt(fp, c, "--temp F", "Sampling temperature. 0 is greedy/deterministic.");
    opt(fp, c, "--top-p F", "Nucleus sampling probability.");
    opt(fp, c, "--min-p F", "Keep tokens scoring at least F times the top token.");
    opt(fp, c, "--seed N", "Sampling seed for reproducible non-greedy runs.");
    para(fp, c, "GLM CLI and agent runs default to temperature 1.0, top-p 0.95, and min-p 0 unless those options are set explicitly.");
    opt(fp, c, "--think", "Use normal thinking mode (V4.1: effort 75).");
    opt(fp, c, "--think-max", "Use maximum thinking (V4.1: 100; V4: requires ctx >= 393216).");
    if (tool == DS4_HELP_DS4 || tool == DS4_HELP_AGENT)
        opt(fp, c, "--think-level N", "V4.1 thinking effort, 1..100; 0 disables thinking.");
    opt(fp, c, "--nothink", "Disable thinking and ask for direct replies.");
    if (full) {
        opt(fp, c, "-sys, --system TEXT", "System prompt. Empty string disables the default where supported.");
        opt(fp, c, "-p, --prompt TEXT", "One-shot prompt text.");
        opt(fp, c, "--prompt-file FILE", "Read one-shot prompt text from FILE.");
        opt(fp, c, "--raw-prompt", "Tokenize the one-shot prompt without chat markers.");
    }
    fputc('\n', fp);
}

static void print_steering(FILE *fp, const help_colors *c) {
    title(fp, c, "Directional Steering");
    opt(fp, c, "--dir-steering-file FILE", "Load one f32 direction vector per layer.");
    opt(fp, c, "--dir-steering-ffn F", "Apply steering after FFN outputs. Default with file: 1");
    opt(fp, c, "--dir-steering-attn F", "Apply steering after attention outputs. Default: 0");
    fputc('\n', fp);
}

static void print_distributed(FILE *fp, const help_colors *c) {
    title(fp, c, "Distributed Inference");
    fputc('\n', fp);
    para(fp, c, "Distributed mode runs one logical session across several machines by assigning contiguous model layer ranges to workers. Workers own their layer slice and KV-cache shard; the coordinator owns the prompt, sampling loop, and client/API flow. Start workers first, then start the coordinator. The coordinator waits for a complete route and streams hidden states through the workers.");
    fputc('\n', fp);
    opt(fp, c, "--role ROLE", "Distributed role: coordinator or worker.");
    opt(fp, c, "--layers A:B", "Inclusive layer slice, e.g. 0:20 or 21:output.");
    opt(fp, c, "--listen HOST PORT", "Coordinator listen address; workers may use it for their data listener.");
    opt(fp, c, "--coordinator HOST PORT", "Coordinator address for --role worker.");
    opt(fp, c, "--dist-prefill-chunk N", "Coordinator prefill pipeline chunk size. Default: session cap.");
    opt(fp, c, "--dist-prefill-window N", "Max prefill chunks in flight. Default: workers+2, capped at 8.");
    opt(fp, c, "--dist-activation-bits N", "Hidden-state transport width: 32, 16, or 8. Default: 32");
    opt(fp, c, "--dist-replay-check", "Diagnostic: reset and replay prompt, then compare logits.");
    opt(fp, c, "--debug", "Print coordinator route/debug logs.");
    fputc('\n', fp);
    title(fp, c, "Tensor Parallelism");
    fputc('\n', fp);
    para(fp, c, "Tensor parallelism uses the same coordinator/worker addresses as distributed mode, but always runs one 50/50 worker. Add --tensor-parallel, omit --layers, start the worker, then start the coordinator.");
    fputc('\n', fp);
    opt(fp, c, "--tensor-parallel", "Switch --role/--listen/--coordinator to two-machine tensor parallelism.");
    opt(fp, c, "--transport auto|rdma|tcp", "Tensor gate transport. Default: auto");
    opt(fp, c, "--rdma-device NAME", "Select a verbs device when auto-detection is ambiguous.");
    opt(fp, c, "--rdma-gid-index N", "Select the local verbs GID index.");
    opt(fp, c, "--tensor-parallel-token-prefill", "GLM diagnostic: prefill one token at a time for exact arithmetic.");
    opt(fp, c, "--debug-hash N", "Cross-check hidden state every N tokens.");
    fputc('\n', fp);
}

static void print_cli_diagnostics(FILE *fp, const help_colors *c);

static void print_cli_specific(FILE *fp, const help_colors *c, bool full) {
    title(fp, c, "CLI Modes");
    opt(fp, c, "ds4", "Start the interactive prompt.");
    opt(fp, c, "ds4 -p TEXT", "Run one prompt and exit.");
    opt(fp, c, "ds4 --prompt-file FILE", "Run a long prompt from a file and exit.");
    opt(fp, c, "--prefix-file FILE", "Preload complete alternating USER:/ASSISTANT: turns before the live conversation.");
    fputc('\n', fp);
    if (full) {
        print_cli_diagnostics(fp, c);
    }
}

static void print_cli_diagnostics(FILE *fp, const help_colors *c) {
    title(fp, c, "Diagnostics And Data Collection");
    opt(fp, c, "--inspect", "Load the model and print a summary only.");
    opt(fp, c, "--dump-tokens", "Print the exact CLI prompt token stream, then exit. Use --raw for literal text.");
    opt(fp, c, "--dump-logits FILE", "Write full next-token logits as JSON.");
    opt(fp, c, "--dump-logprobs FILE", "Write greedy continuation top-logprobs as JSON.");
    opt(fp, c, "--logprobs-top-k N", "Alternatives stored by --dump-logprobs. Default: 20");
    opt(fp, c, "--decode-consistency N", "Compare N-token decode logits with a fresh full prefill.");
    opt(fp, c, "--expert-profile FILE", "Metal-only: write routed expert locality/cache simulation JSON.");
    opt(fp, c, "--perplexity-file FILE", "Score raw text with teacher-forced NLL.");
    opt(fp, c, "--imatrix-dataset FILE", "Rendered prompt dataset for imatrix collection.");
    opt(fp, c, "--imatrix-out FILE", "Write llama-compatible routed-MoE imatrix .dat.");
    opt(fp, c, "--imatrix-max-prompts N", "Stop imatrix collection after N prompts.");
    opt(fp, c, "--imatrix-max-tokens N", "Stop imatrix collection after N prompt tokens.");
    opt(fp, c, "--imatrix-min-expert-samples N", "Continue until every routed expert has N samples.");
    opt(fp, c, "--head-test", "Run the output HC/logits head after the native slice.");
    opt(fp, c, "--first-token-test", "Run exact CPU whole-model pass for the first prompt token.");
    opt(fp, c, "--metal-graph-test", "Compare first GPU-resident graph stages with CPU.");
    opt(fp, c, "--metal-graph-full-test", "Run the GPU-resident self-token graph across all layers.");
    opt(fp, c, "--metal-graph-prompt-test", "Compare CPU and GPU graph logits for the full prompt.");
    fputc('\n', fp);
}

static void print_cli_commands(FILE *fp, const help_colors *c) {
    title_red(fp, c, "Interactive Commands");
    opt(fp, c, "/help", "Show interactive commands.");
    opt(fp, c, "/think [N], /think-max, /nothink", "Switch thinking mode; V4.1 accepts effort 0..100.");
    opt(fp, c, "/ctx N", "Restart the interactive session with a new context size.");
    opt(fp, c, "/power N", "Set GPU duty cycle percentage, 1..100.");
    opt(fp, c, "/read FILE", "Submit a text file, PNG, or JPEG as the next user message.");
    opt(fp, c, "/quit, /exit", "Leave the prompt.");
    opt(fp, c, "Ctrl+C", "Stop current generation and return to ds4>.");
    fputc('\n', fp);
}

static void print_agent_specific(FILE *fp, const help_colors *c) {
    title(fp, c, "Agent Options");
    opt(fp, c, "-p, --prompt TEXT", "Submit an initial prompt after startup.");
    opt(fp, c, "--prompt-file FILE", "Read the initial prompt from FILE.");
    opt(fp, c, "--prefix-file FILE", "Preload complete alternating USER:/ASSISTANT: turns before the live task.");
    opt(fp, c, "--non-interactive", "Run without TUI. With an initial prompt: one turn; otherwise: repeated stdin prompts.");
    opt(fp, c, "--raw-prompt", "Non-interactive initial prompt only: omit agent chat/tool text.");
    opt(fp, c, "--edit-upto", "Enable anchored [upto] edits and automatic marker insertion.");
    opt(fp, c, "-sys, --system TEXT", "Extra system prompt. Empty disables extra text.");
    opt(fp, c, "--trace FILE", "Write prompt, token, and DSML debug trace.");
    opt(fp, c, "--chdir DIR", "Change working directory before loading runtime assets.");
    fputc('\n', fp);
}

static void print_agent_sessions(FILE *fp, const help_colors *c) {
    title(fp, c, "Agent Runtime Commands");
    opt(fp, c, "/save", "Save the current session in ~/.ds4/kvcache.");
    opt(fp, c, "/compact", "Compact the current session context now.");
    opt(fp, c, "/list", "List saved sessions, sorted by recent update time.");
    opt(fp, c, "/switch ID", "Load a saved session and show recent history.");
    opt(fp, c, "/del ID", "Delete a saved session.");
    opt(fp, c, "/strip ID", "Remove KV payload; the text history can be rebuilt later.");
    opt(fp, c, "/history [N]", "Show N recent user turns from the current session.");
    opt(fp, c, "/hints on|off", "Enable or disable brief programming hints. New and resumed sessions start off.");
    opt(fp, c, "/think [N]", "V4.1: set effort 0..100 (default 75) after this turn; rebuilds the prefix.");
    opt(fp, c, "/power N", "Set GPU duty cycle percentage, 1..100.");
    opt(fp, c, "/new", "Start a fresh session from the system prompt.");
    opt(fp, c, "/quit, /exit", "Exit.");
    fputc('\n', fp);
}

static void print_server_api(FILE *fp, const help_colors *c) {
    title(fp, c, "HTTP API");
    opt(fp, c, "--host HOST", "Bind address. Default: 127.0.0.1");
    opt(fp, c, "--port N", "Bind port. Default: 8000");
    opt(fp, c, "--cors", "Add Access-Control-Allow-* headers for browser JS clients.");
    opt(fp, c, "--trace FILE", "Write prompts, cache decisions, output, and tool calls.");
    opt(fp, c, "--batched-session N", "Keep N resident sessions and batch decode-ready requests.");
    opt(fp, c, "--mixed-prefill-quantum N", "Prefill chunk while generations are active. Default: 128; GLM-5.3 minimum: 1024");
    para(fp, c, "Endpoints: /v1/chat/completions, /v1/responses, /v1/completions, and /v1/messages.");
    para(fp, c, "Model endpoint aliases include deepseek-v4-flash and deepseek-v4-pro; both serve the loaded GGUF.");
    fputc('\n', fp);
}

static void print_server_thinking(FILE *fp, const help_colors *c) {
    title(fp, c, "Server Thinking Defaults");
    para(fp, c, "DeepSeek-compatible chat requests default to high-effort thinking.");
    para(fp, c, "reasoning_effort=max or output_config.effort=max requests Think Max.");
    para(fp, c, "Think Max requires --ctx >= 393216; smaller contexts use high.");
    para(fp, c, "thinking={type:disabled}, think=false, or model=deepseek-chat selects non-thinking mode.");
    para(fp, c, "In thinking mode, client sampling knobs are ignored like the official API.");
    fputc('\n', fp);
}

static void print_kv_cache(FILE *fp, const help_colors *c) {
    title(fp, c, "Disk KV Cache");
    opt(fp, c, "--kv-disk-dir DIR", "Enable disk KV checkpoints in DIR.");
    opt(fp, c, "--kv-disk-space-mb N", "Disk budget. Default when enabled: 4096");
    opt(fp, c, "--kv-cache-min-tokens N", "Do not save/load checkpoints shorter than N. Default: 512");
    opt(fp, c, "--kv-cache-cold-max-tokens N", "Save cold first prompts up to N tokens. 0 disables. Default: 30000");
    opt(fp, c, "--kv-cache-continued-interval-tokens N", "Save aligned continued frontiers. 0 disables. Default: 10000");
    opt(fp, c, "--kv-cache-boundary-trim-tokens N", "Trim tail tokens for cold boundary saves. Default: 32");
    opt(fp, c, "--kv-cache-boundary-align-tokens N", "Align cold boundary saves to this multiple. Default: 2048");
    opt(fp, c, "--kv-cache-reject-different-quant", "Reject checkpoints written with different routed-expert quantization.");
    opt(fp, c, "--disable-exact-dsml-tool-replay", "Disable exact sampled DSML tool replay map.");
    opt(fp, c, "--tool-memory-max-ids N", "Exact tool-call IDs kept in RAM. Default: 100000");
    fputc('\n', fp);
}

static void print_bench_specific(FILE *fp, const help_colors *c) {
    title(fp, c, "Benchmark Input");
    opt(fp, c, "--prompt-file FILE", "Raw benchmark text; token sequence is sliced at each frontier.");
    opt(fp, c, "--chat-prompt-file FILE", "Render FILE as one no-thinking chat user message.");
    opt(fp, c, "-sys, --system TEXT", "System prompt used only with --chat-prompt-file.");
    fputc('\n', fp);
    title(fp, c, "Benchmark Sweep");
    opt(fp, c, "--ctx-start N", "First measured frontier. Default: 2048");
    opt(fp, c, "--ctx-max N", "Last measured frontier. Default: 32768");
    opt(fp, c, "--ctx-alloc N", "Allocated context. Default: ctx-max + gen-tokens + 1");
    opt(fp, c, "--step-mul F", "Multiplicative step. Default: 1");
    opt(fp, c, "--step-incr N", "Linear step when --step-mul is 1. Default: 2048");
    opt(fp, c, "--gen-tokens N", "Greedy decode tokens per frontier. 0 for pure prefill. Default: 128");
    opt(fp, c, "--teacher-forced-decode", "Decode the following prompt tokens instead of each predicted argmax.");
    opt(fp, c, "--csv FILE", "Write CSV there instead of stdout.");
    opt(fp, c, "--dump-frontier-logits-dir DIR", "Write one full-logit JSON file per frontier.");
    fputc('\n', fp);
}

static void print_eval_specific(FILE *fp, const help_colors *c) {
    title(fp, c, "Evaluation");
    opt(fp, c, "--suite NAME", "core, hard, all, or hard-smoke. Default: core");
    opt(fp, c, "--source NAME", "Run only cases from this source.");
    opt(fp, c, "--domain NAME", "Run only cases in this domain.");
    opt(fp, c, "--case-id ID", "Run the case with this source ID.");
    opt(fp, c, "--list-cases", "List selected cases without loading a model.");
    opt(fp, c, "--validate-cases", "Validate all embedded cases and exit.");
    opt(fp, c, "-n, --tokens N", "Override the generation budget for every question.");
    opt(fp, c, "--questions N", "Run only the first N selected questions.");
    opt(fp, c, "--case-sequence LIST", "Run 1-based case numbers in this comma-separated order.");
    opt(fp, c, "--retry-incomplete", "Retry a missing final answer once with twice the budget.");
    opt(fp, c, "--trace FILE", "Write questions, outputs, and grading decisions.");
    opt(fp, c, "--regrade-trace FILE", "Regrade a prior trace without loading the model.");
    opt(fp, c, "--soft-limit-reply-budget N", "Soft close thinking near the end of reply budget. Default: 1024");
    opt(fp, c, "--hard-limit-reply-budget N", "Force </think> with N tokens left. Default: 512");
    opt(fp, c, "--soft-limit-think-close-rank N", "Soft-close when </think> is in top N tokens. Default: 3");
    opt(fp, c, "--pause-ms N", "Pause after each result in the TTY UI. Default: 350");
    opt(fp, c, "--plain", "Disable split-screen ANSI UI.");
    opt(fp, c, "--self-test-extractors", "Run answer-extractor self-tests and exit.");
    fputc('\n', fp);
}

static bool tool_has_topic(ds4_help_tool tool, const char *topic) {
    if (!topic) return true;
    if (streq(topic, "all")) return true;
    if (streq(topic, "runtime") || streq(topic, "distributed")) return true;
    if (streq(topic, "sampling"))
        return tool == DS4_HELP_DS4 || tool == DS4_HELP_AGENT || tool == DS4_HELP_EVAL;
    if (streq(topic, "steering"))
        return tool == DS4_HELP_DS4 || tool == DS4_HELP_SERVER || tool == DS4_HELP_AGENT;
    switch (tool) {
    case DS4_HELP_DS4:
        return streq(topic, "diagnostics") || streq(topic, "commands");
    case DS4_HELP_SERVER:
        return streq(topic, "api") || streq(topic, "kv-cache") || streq(topic, "thinking");
    case DS4_HELP_AGENT:
        return streq(topic, "sessions") || streq(topic, "commands") || streq(topic, "tools");
    case DS4_HELP_BENCH:
        return streq(topic, "benchmark");
    case DS4_HELP_EVAL:
        return streq(topic, "evaluation");
    }
    return false;
}

static void more_line(FILE *fp, const help_colors *c, const char *label, const char *topic) {
    static const char *colors[] = {
        "\x1b[38;5;81m", "\x1b[38;5;114m", "\x1b[38;5;179m",
        "\x1b[38;5;141m", "\x1b[38;5;147m"
    };
    static size_t idx;
    const char *on = c->cyan ? colors[idx++ % (sizeof(colors) / sizeof(colors[0]))] : "";
    if (streq(label, "Interactive commands:") && c->red) on = c->red;
    const char *off = c->off ? c->off : "";
    fprintf(fp, "    %s%-26s%s --help %s\n", on, label, off, topic);
}

static void print_more_info(FILE *fp, const help_colors *c, ds4_help_tool tool) {
    title(fp, c, "More Info");
    more_line(fp, c, "Runtime full info:", "runtime");
    if (tool_has_topic(tool, "sampling"))
        more_line(fp, c, "Sampling full info:", "sampling");
    more_line(fp, c, "Distributed inference:", "distributed");
    if (tool_has_topic(tool, "steering"))
        more_line(fp, c, "Steering full info:", "steering");
    if (tool == DS4_HELP_DS4) {
        more_line(fp, c, "Interactive commands:", "commands");
        more_line(fp, c, "Diagnostics:", "diagnostics");
    } else if (tool == DS4_HELP_SERVER) {
        more_line(fp, c, "HTTP API:", "api");
        more_line(fp, c, "Disk KV cache:", "kv-cache");
        more_line(fp, c, "Thinking behavior:", "thinking");
    } else if (tool == DS4_HELP_AGENT) {
        more_line(fp, c, "Agent sessions:", "sessions");
        more_line(fp, c, "Agent commands:", "commands");
        more_line(fp, c, "Agent tool system:", "tools");
    } else if (tool == DS4_HELP_BENCH) {
        more_line(fp, c, "Benchmark sweep:", "benchmark");
    } else if (tool == DS4_HELP_EVAL) {
        more_line(fp, c, "Evaluation options:", "evaluation");
    }
    fputc('\n', fp);
}

static void print_examples(FILE *fp, const help_colors *c, ds4_help_tool tool, const char *topic) {
    title(fp, c, "Examples");
    if (topic_is(topic, "distributed")) {
        opt(fp, c, "worker", "./ds4 --role worker --layers 21:output --coordinator 192.168.0.181 9000 -m ds4flash.gguf");
        opt(fp, c, "coordinator", "./ds4 --role coordinator --layers 0:20 --listen 0.0.0.0 9000 -p \"Hello\" -m ds4flash.gguf");
    } else if (topic_is(topic, "runtime")) {
        if (tool == DS4_HELP_SERVER) {
            opt(fp, c, "Metal API", "./ds4-server -m ds4flash.gguf --metal --ctx 100000");
            opt(fp, c, "quiet API", "./ds4-server --power 60 --host 127.0.0.1 --port 8000");
        } else if (tool == DS4_HELP_AGENT) {
            opt(fp, c, "agent", "./ds4-agent -m ds4flash.gguf --ctx 100000");
            opt(fp, c, "quiet agent", "./ds4-agent --power 50");
        } else if (tool == DS4_HELP_BENCH) {
            opt(fp, c, "bench", "./ds4-bench --prompt-file long.txt --ctx-max 32768");
            opt(fp, c, "quiet bench", "./ds4-bench --prompt-file long.txt --power 70");
        } else if (tool == DS4_HELP_EVAL) {
            opt(fp, c, "eval", "./ds4-eval --questions 10 --ctx 100000");
            opt(fp, c, "CPU debug", "./ds4-eval --cpu --questions 1 --tokens 32");
        } else {
            opt(fp, c, "Metal", "./ds4 -m ds4flash.gguf --metal -c 100000");
            opt(fp, c, "quiet thermals", "./ds4 -p \"Summarize README\" --power 50");
        }
    } else if (topic_is(topic, "steering")) {
        opt(fp, c, "steer FFN", "./ds4 -p \"Write tersely\" --dir-steering-file dir.bin --dir-steering-ffn 0.8");
    } else if (tool == DS4_HELP_SERVER || topic_is(topic, "api") || topic_is(topic, "kv-cache")) {
        opt(fp, c, "local API", "./ds4-server --ctx 100000 --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 8192");
        opt(fp, c, "curl", "curl http://127.0.0.1:8000/v1/models");
    } else if (tool == DS4_HELP_AGENT || topic_is(topic, "sessions") || topic_is(topic, "tools")) {
        opt(fp, c, "interactive", "./ds4-agent");
        opt(fp, c, "one shot", "./ds4-agent --non-interactive -p \"Create /tmp/hello.c\"");
    } else if (tool == DS4_HELP_BENCH || topic_is(topic, "benchmark")) {
        opt(fp, c, "csv", "./ds4-bench --prompt-file long.txt --ctx-max 32768 --csv speed.csv");
        opt(fp, c, "prefill only", "./ds4-bench --prompt-file long.txt --gen-tokens 0");
    } else if (tool == DS4_HELP_EVAL || topic_is(topic, "evaluation")) {
        opt(fp, c, "first 10", "./ds4-eval --questions 10 --trace eval.trace");
        opt(fp, c, "plain", "./ds4-eval --plain --nothink --tokens 512");
    } else {
        opt(fp, c, "chat", "./ds4");
        opt(fp, c, "one shot", "./ds4 -p \"Explain mmap in C\"");
        opt(fp, c, "long prompt", "./ds4 --think-max --prompt-file prompt.txt --ctx 393216");
    }
    fputc('\n', fp);
}

static void print_topic(FILE *fp, const help_colors *c, ds4_help_tool tool, const char *topic) {
    if (streq(topic, "all")) {
        print_model_runtime(fp, c, tool, true);
        if (tool_has_topic(tool, "sampling")) print_sampling(fp, c, true, tool);
        if (tool_has_topic(tool, "steering")) print_steering(fp, c);
        print_distributed(fp, c);
        if (tool == DS4_HELP_DS4) {
            print_cli_specific(fp, c, true);
            print_cli_commands(fp, c);
        } else if (tool == DS4_HELP_SERVER) {
            print_server_api(fp, c);
            print_server_thinking(fp, c);
            print_kv_cache(fp, c);
        } else if (tool == DS4_HELP_AGENT) {
            print_agent_specific(fp, c);
            print_agent_sessions(fp, c);
        } else if (tool == DS4_HELP_BENCH) {
            print_bench_specific(fp, c);
        } else if (tool == DS4_HELP_EVAL) {
            print_eval_specific(fp, c);
        }
        return;
    }

    if (streq(topic, "runtime")) print_model_runtime(fp, c, tool, true);
    else if (streq(topic, "sampling")) print_sampling(fp, c, true, tool);
    else if (streq(topic, "steering")) print_steering(fp, c);
    else if (streq(topic, "distributed")) print_distributed(fp, c);
    else if (tool == DS4_HELP_DS4 && streq(topic, "diagnostics")) print_cli_diagnostics(fp, c);
    else if (tool == DS4_HELP_DS4 && streq(topic, "commands")) print_cli_commands(fp, c);
    else if (tool == DS4_HELP_SERVER && streq(topic, "api")) print_server_api(fp, c);
    else if (tool == DS4_HELP_SERVER && streq(topic, "kv-cache")) print_kv_cache(fp, c);
    else if (tool == DS4_HELP_SERVER && streq(topic, "thinking")) print_server_thinking(fp, c);
    else if (tool == DS4_HELP_AGENT && streq(topic, "sessions")) print_agent_sessions(fp, c);
    else if (tool == DS4_HELP_AGENT && streq(topic, "commands")) print_agent_sessions(fp, c);
    else if (tool == DS4_HELP_AGENT && streq(topic, "tools")) {
        title(fp, c, "Agent Tool System");
        para(fp, c, "The agent can read, search, write, edit, run bash, and browse through Chrome-backed web tools.");
        para(fp, c, "DeepSeek-family models emit DSML tool calls; GLM models use native <tool_call> syntax. Both are rendered live in the terminal.");
        para(fp, c, "Edit uses exact old/new replacement. --edit-upto enables anchored replacements between a unique head and tail.");
        fputc('\n', fp);
    } else if (tool == DS4_HELP_BENCH && streq(topic, "benchmark")) print_bench_specific(fp, c);
    else if (tool == DS4_HELP_EVAL && streq(topic, "evaluation")) print_eval_specific(fp, c);
}

static void print_default(FILE *fp, const help_colors *c, ds4_help_tool tool) {
    print_model_runtime(fp, c, tool, false);

    if (tool == DS4_HELP_DS4) {
        print_cli_specific(fp, c, true);
        print_sampling(fp, c, false, tool);
    } else if (tool == DS4_HELP_SERVER) {
        print_server_api(fp, c);
        print_kv_cache(fp, c);
    } else if (tool == DS4_HELP_AGENT) {
        print_agent_specific(fp, c);
        print_agent_sessions(fp, c);
    } else if (tool == DS4_HELP_BENCH) {
        print_bench_specific(fp, c);
    } else if (tool == DS4_HELP_EVAL) {
        print_eval_specific(fp, c);
    }
}

void ds4_help_print(FILE *fp, ds4_help_tool tool, const char *topic) {
    help_colors c = help_make_colors(fp);
    if (topic && !tool_has_topic(tool, topic)) {
        fprintf(fp, "%s: unknown help topic '%s'\n\n", tool_name(tool), topic);
        topic = NULL;
    }

    fprintf(fp, "%s%s%s\n", c.bright ? c.bright : "", tool_name(tool), c.off ? c.off : "");
    fprintf(fp, "%s\n\n", tool_summary(tool));
    fprintf(fp, "%s\n\n", tool_usage(tool));

    if (topic) print_topic(fp, &c, tool, topic);
    else {
        print_default(fp, &c, tool);
        print_more_info(fp, &c, tool);
    }
    print_examples(fp, &c, tool, topic);
}

#if defined(__APPLE__)   /* llms MEMGUARD (2026-10-20 16:1x): every binary linking this object must link the GUARDED ds4.o -- an engine core
   built from a tree copy older than the memory guard fails to LINK (undefined _llms_mg_holds_engine_lock) instead of running unguarded.
   OUTSIDE the generated 'llms ADOPTED' region on purpose: tools/ds41_adopt_defaults.py rewrites that region and erased the first copy
   of this block (placed inside it at 15:2x; adopt_adapt40 15:59 regenerated the file -- caught by link_canary.py's symbol list). */
extern int holds_engine_lock(void);
__attribute__((used)) int (*const memguard_required)(void) = holds_engine_lock;
#endif


/* llms PROF (apply_prof.py; v3 apply_prof3.py): LLMS_DS4_PROF=<file> -- an in-process statistical profiler of every thread at 1 kHz of CPU time, 16 frames, symbolized at exit; off unless set */
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#include <pthread.h>
#include <mach-o/dyld.h>
#include <stdatomic.h>
#include <dlfcn.h>
#define LLMS_PROF_CAP (1u << 18)
#define LLMS_PROF_DEPTH 16
#define LLMS_PROF_THREADS 64
static uint64_t llms_prof_pc[LLMS_PROF_CAP][LLMS_PROF_DEPTH]; static uint8_t llms_prof_thread[LLMS_PROF_CAP]; static atomic_uint llms_prof_n;
static pthread_t llms_prof_tid[LLMS_PROF_THREADS]; static char llms_prof_tname[LLMS_PROF_THREADS][32]; static atomic_uint llms_prof_nthreads; static const char *llms_prof_path;
static int llms_prof_ok_ptr(uint64_t p) { return p && !(p & 7u) && p > 0x100000000ull && p < 0x0000800000000000ull; }
static void llms_prof_handler(int sig, siginfo_t *si, void *ucv) {
    (void)sig; (void)si; const unsigned k = atomic_fetch_add_explicit(&llms_prof_n, 1u, memory_order_relaxed); if (k >= LLMS_PROF_CAP) return;
    ucontext_t *uc = (ucontext_t *)ucv; uint64_t pc = 0, fp = 0, lr = 0;
#if defined(__arm64__) || defined(__aarch64__)
    if (uc && uc->uc_mcontext) { pc = (uint64_t)__darwin_arm_thread_state64_get_pc(uc->uc_mcontext->__ss); fp = (uint64_t)__darwin_arm_thread_state64_get_fp(uc->uc_mcontext->__ss); lr = (uint64_t)__darwin_arm_thread_state64_get_lr(uc->uc_mcontext->__ss); }
#endif
    const pthread_t me = pthread_self(); const uint64_t hi = (uint64_t)pthread_get_stackaddr_np(me), lo = hi - (uint64_t)pthread_get_stacksize_np(me);
    uint64_t *out = llms_prof_pc[k]; out[0] = pc; out[1] = lr; int d = 2;
    while (d < LLMS_PROF_DEPTH && llms_prof_ok_ptr(fp) && fp >= lo && fp + 16u <= hi) { const uint64_t *f = (const uint64_t *)fp; out[d++] = f[1]; const uint64_t nfp = f[0]; if (nfp <= fp) break; fp = nfp; }
    while (d < LLMS_PROF_DEPTH) out[d++] = 0;
    unsigned n = atomic_load_explicit(&llms_prof_nthreads, memory_order_acquire), t = 0;
    for (; t < n && t < LLMS_PROF_THREADS; t++) if (llms_prof_tid[t] == me) break;
    if (t >= n && n < LLMS_PROF_THREADS) { t = atomic_fetch_add_explicit(&llms_prof_nthreads, 1u, memory_order_acq_rel); if (t < LLMS_PROF_THREADS) { llms_prof_tid[t] = me; pthread_getname_np(me, llms_prof_tname[t], sizeof llms_prof_tname[t]); } else t = LLMS_PROF_THREADS - 1; }
    llms_prof_thread[k] = (uint8_t)(t < LLMS_PROF_THREADS ? t : LLMS_PROF_THREADS - 1);
}
static int llms_prof_cmp(const void *a, const void *b) { const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static void llms_prof_write(void) {
    struct itimerval off = {{0, 0}, {0, 0}}; setitimer(ITIMER_PROF, &off, NULL);
    FILE *f = fopen(llms_prof_path, "wb"); if (!f) return;
    unsigned n = atomic_load(&llms_prof_n); if (n > LLMS_PROF_CAP) n = LLMS_PROF_CAP; unsigned nt = atomic_load(&llms_prof_nthreads); if (nt > LLMS_PROF_THREADS) nt = LLMS_PROF_THREADS;
    const uint64_t slide = (uint64_t)_dyld_get_image_vmaddr_slide(0); const uint32_t hdr[4] = { 0x3352504Cu, n, nt, LLMS_PROF_DEPTH };
    fwrite(hdr, sizeof hdr, 1, f); fwrite(&slide, sizeof slide, 1, f);
    for (unsigned t = 0; t < nt; t++) { char nm[32] = {0}; memcpy(nm, llms_prof_tname[t], sizeof nm); fwrite(nm, 1, sizeof nm, f); }
    fwrite(llms_prof_thread, 1, n, f); fwrite(llms_prof_pc, sizeof llms_prof_pc[0], n, f);
    /* the symbol table: every distinct address (frames >= 1 looked up at addr - 1: the call instruction) */
    uint64_t *u = (uint64_t *)malloc((size_t)n * LLMS_PROF_DEPTH * sizeof(uint64_t)); size_t m = 0;
    if (u) { for (unsigned k = 0; k < n; k++) for (int d = 0; d < LLMS_PROF_DEPTH; d++) if (llms_prof_pc[k][d]) u[m++] = llms_prof_pc[k][d] | (d ? 0ull : 0x8000000000000000ull);
             qsort(u, m, sizeof *u, llms_prof_cmp); size_t w = 0; for (size_t i = 0; i < m; i++) if (!w || u[i] != u[w - 1]) u[w++] = u[i]; m = w; }
    const uint32_t cnt = (uint32_t)m; fwrite(&cnt, 4, 1, f);
    for (size_t i = 0; i < m; i++) {
        const int leaf = (u[i] >> 63) != 0; const uint64_t addr = u[i] & 0x7fffffffffffffffull, q = leaf ? addr : addr - 1u;
        Dl_info di; memset(&di, 0, sizeof di); const char *img = "", *sym = ""; uint32_t off = 0;
        if (dladdr((const void *)q, &di)) { if (di.dli_fname) { img = strrchr(di.dli_fname, '/'); img = img ? img + 1 : di.dli_fname; } if (di.dli_sname) { sym = di.dli_sname; off = (uint32_t)(q - (uint64_t)di.dli_saddr); } }
        const uint16_t li = (uint16_t)strnlen(img, 200), ls = (uint16_t)strnlen(sym, 400);
        fwrite(&u[i], 8, 1, f); fwrite(&li, 2, 1, f); fwrite(img, 1, li, f); fwrite(&ls, 2, 1, f); fwrite(sym, 1, ls, f); fwrite(&off, 4, 1, f);
    }
    free(u); fclose(f);
    fprintf(stderr, "llms prof: %u samples over %u threads, %u symbols -> %s (slide 0x%llx)\n", n, nt, cnt, llms_prof_path, (unsigned long long)slide);
}
__attribute__((constructor)) static void llms_prof_init(void) {
    const char *p = getenv("LLMS_DS4_PROF"); if (!p || !*p) return;
    llms_prof_path = strdup(p);
    { const pthread_t me = pthread_self(); llms_prof_tid[0] = me; snprintf(llms_prof_tname[0], sizeof llms_prof_tname[0], "main"); atomic_store(&llms_prof_nthreads, 1u); }
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = llms_prof_handler; sa.sa_flags = SA_SIGINFO | SA_RESTART; sigemptyset(&sa.sa_mask); sigaction(SIGPROF, &sa, NULL);
    struct itimerval it = {{0, 1000}, {0, 1000}}; setitimer(ITIMER_PROF, &it, NULL);
    atexit(llms_prof_write);
    fprintf(stderr, "llms: prof v3 (SIGPROF every 1 ms of CPU time, every thread, 16 frames) -> %s\n", llms_prof_path);
}
#include <dlfcn.h>
static void llms_prof_wr(const char *b) { size_t n = strlen(b); while (n) { ssize_t w = write(2, b, n); if (w <= 0) break; b += w; n -= (size_t)w; } }
static void llms_prof_crash(int sig, siginfo_t *si, void *ucv) {
    ucontext_t *uc = (ucontext_t *)ucv; uint64_t pc = 0, lr = 0, fp = 0; char b[512], nm[64] = {0};
#if defined(__arm64__) || defined(__aarch64__)
    if (uc && uc->uc_mcontext) { pc = (uint64_t)__darwin_arm_thread_state64_get_pc(uc->uc_mcontext->__ss); lr = (uint64_t)__darwin_arm_thread_state64_get_lr(uc->uc_mcontext->__ss); fp = (uint64_t)__darwin_arm_thread_state64_get_fp(uc->uc_mcontext->__ss); }
#endif
    pthread_getname_np(pthread_self(), nm, sizeof nm);
    snprintf(b, sizeof b, "llms prof crash: signal %d code %d addr %p thread '%s' pc 0x%llx lr 0x%llx fp 0x%llx slide 0x%llx\n", sig, si ? si->si_code : 0, si ? si->si_addr : NULL, nm,
             (unsigned long long)pc, (unsigned long long)lr, (unsigned long long)fp, (unsigned long long)_dyld_get_image_vmaddr_slide(0)); llms_prof_wr(b);
    uint64_t fr[26]; int nf = 0; fr[nf++] = pc; fr[nf++] = lr;
    for (int d = 0; d < 24 && llms_prof_ok_ptr(fp); d++) { const uint64_t *f = (const uint64_t *)fp; fr[nf++] = f[1]; const uint64_t nfp = f[0]; if (nfp <= fp) break; fp = nfp; }
    for (int i = 0; i < nf; i++) {
        Dl_info di; memset(&di, 0, sizeof di); const char *img = "?", *sym = "?"; uint64_t off = 0;
        if (fr[i] && dladdr((const void *)fr[i], &di)) { if (di.dli_fname) { img = strrchr(di.dli_fname, '/'); img = img ? img + 1 : di.dli_fname; } if (di.dli_sname) { sym = di.dli_sname; off = fr[i] - (uint64_t)di.dli_saddr; } }
        snprintf(b, sizeof b, "llms prof crash:   #%02d 0x%llx %s %s+%llu\n", i, (unsigned long long)fr[i], img, sym, (unsigned long long)off); llms_prof_wr(b);
    }
}
__attribute__((constructor)) static void llms_prof_crash_init(void) {
    const char *p = getenv("LLMS_DS4_PROF"); if (!p || !*p) return;
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = llms_prof_crash; sa.sa_flags = SA_SIGINFO | SA_RESETHAND; sigemptyset(&sa.sa_mask);
    const int sigs[5] = { SIGTRAP, SIGSEGV, SIGBUS, SIGILL, SIGABRT }; for (int i = 0; i < 5; i++) sigaction(sigs[i], &sa, NULL);
    char b[160]; snprintf(b, sizeof b, "llms: prof crash reporter on (main image slide 0x%llx)\n", (unsigned long long)_dyld_get_image_vmaddr_slide(0)); llms_prof_wr(b);
}

/* ---- llms ADOPTED (generated by tools/ds41_adopt_defaults.py from ds41_adopted_env.json) ---- */
#include <stdlib.h>
#include <string.h>
#include "ds4.h"
#include <stdio.h>
#include <stdint.h>
/* llms ENV AUDIT + CACHE: getenv, setenv, unsetenv and putenv are interposed for ds4's own objects (these definitions bind every call in ds4's objects; libraries keep libSystem's). (1) AUDIT: the LLMS_* / DS4_* variables the OUTSIDE environment set (a snapshot taken before the defaults below apply) are checked at exit against every name the code looked up -- a set switch whose code never ran never reads its variable (switches read lazily), so 'NEVER READ' names a misspelled switch or one on a path this run never took. (2) CACHE: LLMS_* / DS4_* lookups are served from a table after the first scan of environ (an agent session made 429,094 of them, ~300-660 ns each as a linear scan of ~250 entries, many from per-call reads of the stock DS4_METAL_* knobs); every setenv / unsetenv / putenv bumps a generation that invalidates the whole table, so a cached value is never older than the environment. */
#include <dlfcn.h>
extern char **environ;
static char *llms_ea_out[512]; static int llms_ea_out_n;
#define LLMS_EA_N 4096
static char *llms_ea_name[LLMS_EA_N]; static char *llms_ea_val[LLMS_EA_N]; static uint32_t llms_ea_vgen[LLMS_EA_N]; static uint64_t llms_ea_cnt[LLMS_EA_N];
static uint64_t llms_ea_calls, llms_ea_scans; static uint32_t llms_ea_gen = 1; static volatile int llms_ea_lk;
static uint32_t llms_ea_h(const char *s) { uint32_t h = 2166136261u; while (*s) h = (h ^ (uint8_t)*s++) * 16777619u; return h; }
static char *llms_ea_scan(const char *name) {
    const size_t n = strlen(name);
    if (environ) for (char **e = environ; *e; e++) if (!strncmp(*e, name, n) && (*e)[n] == '=') return *e + n + 1;
    return NULL;
}
static int llms_def_rec; static char *llms_def_names[1024]; static int llms_def_n;   /* llms DEAD DEFAULTS (apply_dead_defaults.py) */
int setenv(const char *name, const char *value, int overwrite) {
    static int (*real)(const char *, const char *, int); if (!real) real = (int (*)(const char *, const char *, int))dlsym(RTLD_NEXT, "setenv");
    if (llms_def_rec && llms_def_n < 1024 && (!strncmp(name, "LLMS_", 5) || !strncmp(name, "DS4_", 4)) && !llms_ea_scan(name)) llms_def_names[llms_def_n++] = strdup(name);
    const int r = real(name, value, overwrite); __atomic_add_fetch(&llms_ea_gen, 1, __ATOMIC_SEQ_CST); return r;
}
int unsetenv(const char *name) {
    static int (*real)(const char *); if (!real) real = (int (*)(const char *))dlsym(RTLD_NEXT, "unsetenv");
    const int r = real(name); __atomic_add_fetch(&llms_ea_gen, 1, __ATOMIC_SEQ_CST); return r;
}
int putenv(char *str) {
    static int (*real)(char *); if (!real) real = (int (*)(char *))dlsym(RTLD_NEXT, "putenv");
    const int r = real(str); __atomic_add_fetch(&llms_ea_gen, 1, __ATOMIC_SEQ_CST); return r;
}
char *getenv(const char *name) {
    if (!name) return NULL;
    if (strncmp(name, "LLMS_", 5) && strncmp(name, "DS4_", 4)) return llms_ea_scan(name);
    __atomic_fetch_add(&llms_ea_calls, 1, __ATOMIC_RELAXED);
    const uint32_t g = __atomic_load_n(&llms_ea_gen, __ATOMIC_ACQUIRE), h0 = llms_ea_h(name) & (LLMS_EA_N - 1);
    for (uint32_t k = 0, h = h0; k < LLMS_EA_N; k++, h = (h + 1) & (LLMS_EA_N - 1)) {   /* the lock-free read path */
        const char *nm = __atomic_load_n(&llms_ea_name[h], __ATOMIC_ACQUIRE);
        if (!nm) break;
        if (!strcmp(nm, name)) {
            __atomic_fetch_add(&llms_ea_cnt[h], 1, __ATOMIC_RELAXED);
            if (__atomic_load_n(&llms_ea_vgen[h], __ATOMIC_ACQUIRE) == g) return __atomic_load_n(&llms_ea_val[h], __ATOMIC_RELAXED);
            break;
        }
    }
    char *v = llms_ea_scan(name); __atomic_fetch_add(&llms_ea_scans, 1, __ATOMIC_RELAXED);
    while (__atomic_exchange_n(&llms_ea_lk, 1, __ATOMIC_ACQUIRE)) { }
    for (uint32_t k = 0, h = h0; k < LLMS_EA_N; k++, h = (h + 1) & (LLMS_EA_N - 1)) {
        if (!llms_ea_name[h]) { llms_ea_val[h] = v; __atomic_store_n(&llms_ea_vgen[h], g, __ATOMIC_RELEASE); __atomic_fetch_add(&llms_ea_cnt[h], 1, __ATOMIC_RELAXED); __atomic_store_n(&llms_ea_name[h], strdup(name), __ATOMIC_RELEASE); break; }
        if (!strcmp(llms_ea_name[h], name)) { __atomic_store_n(&llms_ea_val[h], v, __ATOMIC_RELAXED); __atomic_store_n(&llms_ea_vgen[h], g, __ATOMIC_RELEASE); break; }
    }
    __atomic_store_n(&llms_ea_lk, 0, __ATOMIC_RELEASE);
    return v;
}
static int llms_ea_was_read(const char *name) {
    uint32_t h = llms_ea_h(name) & (LLMS_EA_N - 1);
    for (int k = 0; k < LLMS_EA_N; k++, h = (h + 1) & (LLMS_EA_N - 1)) { if (!llms_ea_name[h]) return 0; if (!strcmp(llms_ea_name[h], name)) return 1; }
    return 0;
}
static void llms_ea_report(void) {
    const char *g = getenv("LLMS_DS4_GUARDS"); if (!(g && *g && *g != '0')) return;
    int unread = 0;
    for (int i = 0; i < llms_ea_out_n; i++) {
        char nm[256]; const char *eq = strchr(llms_ea_out[i], '='); const size_t l = eq ? (size_t)(eq - llms_ea_out[i]) : strlen(llms_ea_out[i]);
        if (l >= sizeof nm) continue;
        memcpy(nm, llms_ea_out[i], l); nm[l] = 0;
        if (!llms_ea_was_read(nm)) { fprintf(stderr, "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!! llms env audit: %s is set in the environment and was NEVER READ by this run -- THIS RUN DID NOT EXERCISE IT\n!!! (a misspelled name, or a switch on a path this run never took)\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", llms_ea_out[i]); unread++; }
    }
    if (unread) fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!! llms env audit: %d SET VARIABLE(S) WERE NEVER READ -- the run tested none of them (see the blocks above)\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", unread);
    fprintf(stderr, "llms env audit: %d LLMS_/DS4_ variables from the environment, %d never read; %llu getenv calls for LLMS_/DS4_ names, %llu environ scans (the rest from the cache), %u environment changes\n",
            llms_ea_out_n, unread, (unsigned long long)llms_ea_calls, (unsigned long long)llms_ea_scans, (unsigned)(llms_ea_gen - 1u));
    {   int top[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
        for (int h = 0; h < LLMS_EA_N; h++) { if (!llms_ea_name[h]) continue; for (int k = 0; k < 8; k++) if (top[k] < 0 || llms_ea_cnt[h] > llms_ea_cnt[top[k]]) { for (int j = 7; j > k; j--) top[j] = top[j - 1]; top[k] = h; break; } }
        fprintf(stderr, "llms env audit: the most-read names:"); for (int k = 0; k < 8 && top[k] >= 0; k++) fprintf(stderr, " %s %llu", llms_ea_name[top[k]], (unsigned long long)llms_ea_cnt[top[k]]); fprintf(stderr, "\n"); }
}
static void llms_def_report(void) {   /* llms DEAD DEFAULTS: the defaults this run's code never read */
    int dead = 0; for (int i = 0; i < llms_def_n; i++) dead += !llms_ea_was_read(llms_def_names[i]);
    fprintf(stderr, "llms defaults audit: %d of %d defaults set by the constructor were NEVER READ:", dead, llms_def_n);
    for (int i = 0; i < llms_def_n; i++) if (!llms_ea_was_read(llms_def_names[i])) fprintf(stderr, " %s", llms_def_names[i]);
    fprintf(stderr, "\n");
}
static const char *const llms_adopted_env_nonserver[][2] = {
    {"DS4_METAL_DISABLE_QUEUE_KEEPALIVE", "1"},
    {"DS4_METAL_DISABLE_V41_BATCH_TOPK", "1"},
    {"DS4_METAL_DISABLE_V41_WIDE_CHUNK", "1"},
    {"DS4_METAL_STREAMING_EXPERT_PREAD_THREADS", "16"},
    {"LLMS_DS4_BATCH_MVEXT", "1"},
    {"LLMS_DS4_BATCH_SPLIT", "listdedup4"},
    {"LLMS_DS4_COMP_ROWS", "1"},
    {"LLMS_DS4_COUNT_OPS", "1"},
    {"LLMS_DS4_DECODE_ROUTE_LATE", "37:1.5"},
    {"LLMS_DS4_ED", "8"},
    {"LLMS_DS4_EMBD_CPU", "1"},
    {"LLMS_DS4_EXPERT_NOCACHE", "1"},
    {"LLMS_DS4_F32_ROWS", "1"},
    {"LLMS_DS4_GATEUP_BS64", "pack"},
    {"LLMS_DS4_KEEPBUSY", "0"},
    {"LLMS_DS4_KV_HALF", "1"},
    {"LLMS_DS4_LOAD_DIRECT", "1"},
    {"LLMS_DS4_PLAIN_MV_LEAN", "1"},
    {"LLMS_DS4_PREAD_ALIGNED", "1"},
    {"LLMS_DS4_PREFILL_ROUTE", "1"},
    {"LLMS_DS4_PREFILL_ROUTE_COH", "1"},
    {"LLMS_DS4_QUEUE_LAYERS", "1"},
    {"LLMS_DS4_ROUTING", "bonus"},
    {"LLMS_DS4_ROWS_T4", "2"},
    {"LLMS_DS4_SHARE_SCRATCH", "1"},
    {"LLMS_DS4_SHORT_PREFILL_MAXPOS", "0"},
    {"LLMS_DS4_V41_ROWS_ATTN", "1"}
};
static const char *const llms_adopted_env_server[][2] = {
    {"DS4_METAL_DISABLE_QUEUE_KEEPALIVE", "1"},
    {"DS4_METAL_DISABLE_QUEUE_RESIDENCY_SET", "1"},
    {"DS4_METAL_DISABLE_V41_BATCH_TOPK", "1"},
    {"DS4_METAL_DISABLE_V41_WIDE_CHUNK", "1"},
    {"DS4_METAL_NORM_SOURCE", "@repo/live/norm_mvb.metal"},
    {"DS4_METAL_STREAMING_EXPERT_PREAD_THREADS", "16"},
    {"LLMS_DS4_ALLHIT_LIST", "1"},
    {"LLMS_DS4_AO_SGMH", "1"},
    {"LLMS_DS4_AO_T32G", "1"},
    {"LLMS_DS4_ATTN_GEMM", "2"},
    {"LLMS_DS4_ATTN_OUT_MT", "8"},
    {"LLMS_DS4_BATCH_ASYNC", "1"},
    {"LLMS_DS4_BATCH_MVEXT", "1"},
    {"LLMS_DS4_BATCH_SPLIT", "listdedup4"},
    {"LLMS_DS4_CF2", "1"},
    {"LLMS_DS4_CF4", "1"},
    {"LLMS_DS4_CF_FORM", "70"},
    {"LLMS_DS4_COMP_ROWS", "1"},
    {"LLMS_DS4_COUNT_OPS", "1"},
    {"LLMS_DS4_DD2C", "1"},
    {"LLMS_DS4_DECODE_ROUTE", "0.20"},
    {"LLMS_DS4_DECODE_ROUTE_LATE", "37:1.5"},
    {"LLMS_DS4_DECODE_ROUTE_TOPK", "5"},
    {"LLMS_DS4_DOWN_T8", "2"},
    {"LLMS_DS4_EMBD_CPU", "1"},
    {"LLMS_DS4_EXPERT_NOCACHE", "1"},
    {"LLMS_DS4_EXT_R8", "1"},
    {"LLMS_DS4_EXT_SGM", "1"},
    {"LLMS_DS4_EXT_SGMH", "1"},
    {"LLMS_DS4_EXT_SGMH16", "1"},
    {"LLMS_DS4_F32_ROWS", "1"},
    {"LLMS_DS4_FAST_SAMPLER", "1"},
    {"LLMS_DS4_GATEUP_BS64", "pack"},
    {"LLMS_DS4_IDX_SCORE_SG", "1"},
    {"LLMS_DS4_KEEPBUSY", "0"},
    {"LLMS_DS4_KV_FP4", "1"},
    {"LLMS_DS4_KV_HALF", "1"},
    {"LLMS_DS4_LOAD_DIRECT", "1"},
    {"LLMS_DS4_LOCK_HEADROOM_MB", "256"},
    {"LLMS_DS4_MLOCK_RELIEF_PCT", "2"},
    {"LLMS_DS4_NO_CARRY", "1"},
    {"LLMS_DS4_NO_PREFILL_RESERVE", "1"},
    {"LLMS_DS4_PIPE_GROUPS", "1"},
    {"LLMS_DS4_PIPE_ODD", "1"},
    {"LLMS_DS4_PLAIN_MV_LEAN", "1"},
    {"LLMS_DS4_PREAD_ALIGNED", "1"},
    {"LLMS_DS4_PREFILL_ROUTE", "1"},
    {"LLMS_DS4_PREFILL_ROUTE_BONUS", "0.2*20,1.2*20"},
    {"LLMS_DS4_PREFILL_ROUTE_COH", "1"},
    {"LLMS_DS4_PREHALF64", "1"},
    {"LLMS_DS4_QUEUE_LAYERS", "1"},
    {"LLMS_DS4_ROUTER_FIRST", "1"},
    {"LLMS_DS4_ROUTING", "bonus"},
    {"LLMS_DS4_ROWS_T4", "2"},
    {"LLMS_DS4_ROWS_WIN", "1"},
    {"LLMS_DS4_SGMH16A", "1"},
    {"LLMS_DS4_SGMH32", "1"},
    {"LLMS_DS4_SGMT32", "2"},
    {"LLMS_DS4_SGMT32X", "3"},
    {"LLMS_DS4_SHARE_SCRATCH", "1"},
    {"LLMS_DS4_SHEXP_PAIR", "1"},
    {"LLMS_DS4_SHORT_PREFILL_MAXPOS", "0"},
    {"LLMS_DS4_STEP_CLOCK", "1"},
    {"LLMS_DS4_T32_BF16", "1"},
    {"LLMS_DS4_TOPK_PAD", "1"},
    {"LLMS_DS4_TOPK_ROWS", "1"},
    {"LLMS_DS4_V41_ROWS_ATTN", "1"}
};
__attribute__((constructor(102))) static void llms_maxdec_profile(void) {   /* 2026-10-21 18:5x : LLMS_DS4_MAXDEC=1 -- the MAX-DECODE profile: the exact decoder pass and exact prefill routing, set BEFORE llms_adopted_defaults (whose setenv never overwrites): decode KL 0.2516 -> 0.0826, generation +2.56 % (s.e. 3.86), prefill -86..-89 %. Its own constructor: the generator parses llms_adopted_defaults' conditions for program classes, and this one depends on the environment, not the program. */
    const char *md = getenv("LLMS_DS4_MAXDEC"); if (!md || !*md || *md == '0') return;
    setenv("LLMS_DS4_CED_SP", "suffix", 0); setenv("LLMS_DS4_APPEND_ROUTE_POLICY", "0", 0);
    fprintf(stderr, "llms: MAXDEC profile (the exact decoder pass + exact prefill routing: decode fidelity first, prefill ~8x slower)\n");
}
__attribute__((constructor(101))) static void llms_def_init(void) {   /* llms DEAD DEFAULTS: armed before the defaults constructor (a separate function: the generator parses that one's conditions) */
    const char *ad = getenv("LLMS_DS4_AUDIT_DEFAULTS"); if (ad && *ad && *ad != '0') { llms_def_rec = 1; atexit(llms_def_report); }
}
__attribute__((constructor)) static void llms_adopted_defaults(void) {
    if (environ) for (char **e = environ; *e && llms_ea_out_n < 512; e++) if (!strncmp(*e, "LLMS_", 5) || !strncmp(*e, "DS4_", 4)) llms_ea_out[llms_ea_out_n++] = strdup(*e);   /* llms ENV AUDIT: before any default */
    atexit(llms_ea_report);
    const char *off = getenv("LLMS_DS4_NO_ADOPTED");
    if (off && *off && *off != '0') return;
    {   /* 2026-10-13 09:0x: ED (the event-driven one-token MoE) is a single-stream program feature: in the server, mixed batched sessions
          can make a listener wait on in-flight cache entries parked behind its own event (x297: GPU timeout). ds4-server: off unless set. */
        const char *pn = getprogname();
        if (pn && !strcmp(pn, "ds4-server")) setenv("LLMS_DS4_ED", "0", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_Q4S8", "1", 0);   /* Q4S8; ADOPTED 2026-10-29: Q4L group scales as one byte; ITS THREE PATHS ARE IN-PLACE VALUES.*/
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_WORKER_RT", "1", 0);   /* WORKER_RT; ADOPTED 2026-10-29: the session thread user-interactive.*/
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_RINGPRE", "1", 0);   /* RINGPRE; ADOPTED 2026-10-29: the drafter ring preload.*/
        /* 2026-10-13 12:2x: the carry (wide prefill: each layer read once; x183 110.6 t/s vs chunked) costs ~0.4 GB per session at ctx
           9,000 -- the table's NO_CARRY=1 is for the multi-session server; one local session keeps the wide prefill (x314) */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_NO_CARRY", "0", 0);
        /* 2026-10-26 M20 (bisect 20:24): PREFILL_CAP 512 -> 256 in the next line, KEPT when M20 was reverted -- wired -0.36 GiB, speed null (half A +0.14 % clock-adjusted) */
        if (pn && !strncmp(pn, "ds4-agent", 9)) setenv("LLMS_DS4_PREFILL_CAP", "256", 0);   /* ADOPTED 2026-10-26: the agent's appends all run 128-row short-prefill calls since SHORT_PREFILL=1000000 (10-25), so its prefill storage needs 512 rows, not 8,192; bit-exact; context buffers 2,676 -> 662 MiB; implies no carry. Restore: remove this line */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_MAINWFE", "1", 0);   /* ADOPTED 2026-10-26: the decode cycle's owned batches end with a one-thread post and the main thread waits on it in WFE instead of waitUntilCompleted (~0.15 ms per wait, 2.4 waits per drafted cycle); exact (EXACT_DET identity); adopt-when-indistinguishable, RETEST.md. Restore: remove this line */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_RAGGED_AO", "1", 0); setenv("LLMS_DS4_RAGGED_Q", "1", 0); }   /* ADOPTED 2026-10-28: the Q4L index literal above points at the ragged index dense_q4l_gptq_gen_alla_eng_qakv_rag_e.idx; member RAGGED RAGGED_E, extended to RAGGED_U 2026-10-28; RAGGED_V 2026-10-28*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_RAGQ_S1_RSG", "1:8", 0); /* RAGQ_S1_RSG (was 4); ADOPTED 2026-10-30 : attn_q_b stage 1 at NR 1: -10.7 % of the stage, a rounding class.*/ }   /* ADOPTED 2026-10-29: RAGGED_Q's unrounded stage 1 (attn_q_b 1280 -> r) on the RSG f body; EXACT (818 / 818, 536 / 536 bit-identical); the stage's kernel -34.6 % a two-row step (0.816 -> 0.534 ms, timeline pair)*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SHM_KEEP_BUILDS", "1", 0); }   /* ADOPTED 2026-10-29: the shm expert cache survives an engine rebuild -- the key's build stamp replaced by a layout tag, every restore first re-reads 16 sampled entries from their source files and compares them byte for byte (any difference: nothing restored); the test: all 4,100 entries equal across a forced rebuild, the canary cold, the switch unset 818 / 818.*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_BASH_POLL", "1", 0); }   /* ADOPTED 2026-10-30: the bash tool's job check and monitor poll at 1 ms instead of 20 / 50 -- bash calls 50.75 / 50.48 -> 38.18 / 33.53 ms, the rest the command itself (python3 test runs); exact (no model output touched).*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_RAGQ_S2_RSG", "8", 0); }   /* RAGQ_S2_RSG; ADOPTED 2026-10-29: attn_q_b stage 2 on the RSG body (NR 4 / S 8); EXACT, null alone.*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_NPL45", "2", 0); }   /* NPL45_DEC; ADOPTED 2026-11-01: the decode gate / up at 4 activation planes (prefill already at 4, NPL_PF); speed -0.114 +- 0.072 ms direct / -0.249 +- 0.121 total a step, decode KL +0.00143 +- 0.00116*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DSP_BS64", "pack", 0); setenv("LLMS_DS4_DSP_G32D", "pack", 0); setenv("LLMS_DS4_DSP_BANK_LEAN", "1", 0); }   /* DSP_LEAN; ADOPTED 2026-11-01: the drafter bank's routed experts on BS64 gate / up + G32 down codes from the FP4 shards, inside the bank segment in place of IQ2 / Q2_K; per drafted cycle -0.180 +- 0.033 and -0.198 +- 0.055 ms, acceptance unchanged*/
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_DSP_HCF", "1", 0);   /* DSP_HCF; ADOPTED 2026-11-01: the drafter's six HC transitions a draft on the fused one-token kernel + the split kernel; drafts 454 / 454 equal under EXACT_DET, per drafted cycle -0.072 +- 0.047 ms*/
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_HX_TAIL", "1", 0);   /* HX_TAIL; ADOPTED 2026-11-01: the heads tail writes the grouped output_a's half copy, its conversion dispatch skipped (needs the _hxt misc source, swapped in place in the non-server DENSE / MISC line; ship value ..._idxr.metal)*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ROWS256", "256", 0); setenv("LLMS_DS4_LIST2048", "1", 0); }   /* ROWS256; ADOPTED 2026-11-01: 256-row prefill calls, the routed list at 2,048 pairs, the policy protected misses fixed (PLAGCAP); KL +0.0015 +- 0.0046 (rows256i), prefill -9.1 % an append (p 0.031); J per row +9.7 % -- the level is the harm gate; SP_ROWS 256 on line 248 */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SP_PUB256", "1", 0); }   /* SP_PUB256; ADOPTED 2026-11-01: a > 128-row call publishes its keys once, the batched indexer in every piece; exact (818 / 818), cycles -4.67 % (s.e. 2.11) -- */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_PLAG_RENORM", "2", 0); }   /* PLAG_RENORM2; ADOPTED 2026-11-01: PLAG rows take PWNORM's norm-matched divisor; KL -0.01149 (s.e. 0.00349) against mode 1, both pairs -- ; replaces M43's mode 1 */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SUB_DROP", "1", 0); }   /* SUB_DROP; ADOPTED 2026-11-01: the decode cut's resident substitutes dropped, a row keeps its heaviest if emptied; KL +0.00435 (s.e. 0.00401), +0.391 ms a two-row step (s.e. 0.141); bisect alone x0.88 of neither */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_GD_DRAIN", "1", 0); }   /* b GD_DRAIN; ADOPTED 2026-11-01: the drafter's pending rows enter its ring before a grammar-depth verify  */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_DENSE", "1", 0); setenv("LLMS_DS4_ANE_DENSE_LAYERS", "20", 0); setenv("LLMS_DS4_ANE_DENSE_WIRED", "1", 0); setenv("LLMS_DS4_ANE_DENSE_WARM", "1", 0); setenv("LLMS_DS4_ANE_DENSE_HANDOFF", "post", 0); setenv("LLMS_DS4_ANE_DENSE_ANY", "32", 0); setenv("LLMS_DS4_ANE_DENSE_DIR", "pack", 0); }   /* T1ANY; ADOPTED 2026-11-01: tier 1 on every 129..256-row call, GPU J per prefill row -5.6 % (s.e. 1.3), KL -0.0045 on dense2s */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_SHEXP_KA", "0", 0); }   /* KAOFF; ADOPTED 2026-11-02: the shared-expert keepalive off -- with tier 1 in every 129..256-row call its fillers delay the real ANE pieces; off, appends -3.6 % cycles at flat J per row */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_DENSE_EARLY", "1", 0); setenv("LLMS_DS4_ANE_DENSE_AOEARLY", "1", 0); }   /* SLACK; ADOPTED 2026-11-02: tier 1's hand-off before piece A's projection + piece A's attention output before the wait for piece B -- the GPU's poll for tier 1 ~1.7 -> ~0.6 ms a piece, cycles -2.45 % (s.e. 0.32), J flat */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_SHEXP_WDPROG", "1", 0); }   /* WDPROG; ADOPTED 2026-11-02: the shared expert's watchdog trips only when B stops advancing for 3 s -- the shipped rule tripped on steady progress with one block outstanding and released B early */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_COLD_REFILL", "1", 0); }   /* COLDREFILL; ADOPTED 2026-11-02: a cold shared expert cache re-reads the last saved expert list from SSD before the first install */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_DENSE_MULTI", "1", 0); }   /* T1MULTI; ADOPTED 2026-11-02: tier 1 from two multi-procedure models + a 2-evaluation keep-warm while it idles */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_DENSE_KEEPWARM_PROCS", "0,3,6,9,12,14,15,18", 0); }   /* KWPROCS; ADOPTED 2026-11-02: M55's keep-warm on one procedure per 120-MiB weight window */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_FIXSET_LEARN", "120", 0); }   /* LEARN; ADOPTED 2026-11-05 under ): FIXSET_LEARN 120 KL -0.0235 (s.e. 0.0047, refreshkl), learnfree r 0.859 unresolved; KVTAIL_LAST exact */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_PAIROAB", "1", 0); setenv("LLMS_DS4_ANE_OAB_LAYERS", "19", 0); setenv("LLMS_DS4_ANE_OAB_WIRED", "0", 0); setenv("LLMS_DS4_ANE_T2", "32", 0); setenv("LLMS_DS4_ANE_OAB_MULTI", "1", 0); setenv("LLMS_DS4_ANE_OAB_MULTI_MODEL", "pack", 0); setenv("LLMS_DS4_ANE_OAB_KEEPWARM_PROCS", "0,2,4,5,7,9,10,13,15,17,18", 0); setenv("LLMS_DS4_ANE_DENSE_QB_MODEL", "pack", 0); setenv("LLMS_DS4_ATT2_H32", "1", 0); setenv("LLMS_DS4_PSO_MAXT", "@repo/live/pso_maxt_h32.txt", 0); }   /* SET61; ADOPTED 2026-11-02: form A: tier 1 q_b qb20h, tier 2 layers 0-14 (full rank 0-9 + ragged 10-14) + H32 (ATT2_H32, its misc source and PSO census) -- cleared its bar against M60 */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_OAB_EAGER", "1", 0); setenv("LLMS_DS4_ANE_T12_BG", "1", 0); /* T12BG */ }   /* T2EAGER; ADOPTED 2026-11-02: tier 2's ANE model loads beside tier 1 at the first prefill, not lazily after round 0 where its load met a full memory (5 of 5 memory faults of M60's / M61's levels fired during it); numerics identical (EXACT_DET 818 / 818) */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ATT2_PV2", "1", 0); }   /* PV2; ADOPTED 2026-11-02: H32 with padded K / V rows and PV register blocking (kernel_llms_att2_h32pv2 in the _h32_pv2 misc source); bit-identical; cycles pv2 - base over 6 rounds: -2.12 % (s.e. 0.26); : roman in-level ship 0 / 8, pv2 0 / 8 failed */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_RESTART_NOTE", "1", 0); setenv("LLMS_DS4_AGENT_RESTART_NOEDIT", "1", 0); }   /* RESTART_NOTE_NOEDIT; ADOPTED 2026-11-03: a restart resubmits its prompt saying an earlier attempt may have changed files, and an attempt that edited a file disarms the restart; : N 16 / 16 53.7 s, E 16 / 16 49.0 s, O 15 / 16 45.1 s per accurate solution with recovery, E vs N order 8 / 8 */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ADD6", "1", 0); setenv("LLMS_DS4_REDU12", "1", 0); }   /* MERGED_DEC2; ADOPTED 2026-11-03: ADD6 v2 + REDU12 -- each exact; NR0 8 left out (closed for harm by nr0c, 16:2x); read in-level against M66 by one random-block factor (ALTW_ALIAS) */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q2S", "pack", 0); setenv("LLMS_DS4_F16K2_NR", "4", 0); }   /* MERGED_DEC3; ADOPTED 2026-11-03: q2s + f16nr -- each exact; Q2S_HEAD KL +0.00003 (q2h), in situ +0.522 +- 0.649 ms (q2fix); restore: remove this line */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_G32D_FOLD", "1", 0); setenv("LLMS_DS4_Q4L_NR0", "8", 0); setenv("LLMS_DS4_G32D_TG8", "1", 0); }   /* MERGED_DEC4; ADOPTED 2026-11-03; the agent line's MoE path and the dense line's path were swapped to the composed sources in the same edit -- restore: remove this line and swap both back */
        if (pn && strcmp(pn, "ds4-server")) { setenv("DS4_METAL_NORM_SOURCE", "@repo/live/norm_mvb_hxn.metal", 0); setenv("LLMS_DS4_HX_NORM", "1", 0); }   /* HX_NORM; ADOPTED 2026-11-01: the attention-side norm writes the half copy attn_q_b reads, its conversion skipped -- EXACT (818 / 818 twice); ALTW +0.017 +- 0.094 / +0.166 +- 0.159 ms*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DSP_PREENC", "1", 0); }   /* ADOPTED 2026-10-27: DSP_PREENC*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_VPIPE", "2", 0); setenv("LLMS_DS4_VPIPE_FIRST", "1", 0); }   /* ADOPTED 2026-10-27: VPIPE_2_FIRST*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_RESCUE_GATE", "0.35:5", 0); }   /* ADOPTED 2026-10-27: RESCUE_GATE*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4L_KGX8", "1", 0); setenv("LLMS_DS4_AO_RSG", "8", 0); }   /* ADOPTED 2026-10-28: members Q4L_KGX8 (+0.302 ms a two-row step, s.e. 0.037) and AO_RSG 8 (+0.096 ms, s.e. 0.037) from ; M24_KGX8_AORSG), or remove this line */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_RB_ASYNC", "skip", 0); }   /* ADOPTED 2026-10-26: the rejected draft's rollback without its dead GPU copies, its residual draw over the support; exact (818 / 818, canary 4 / 818, 90 / 90 draws); ABBA -0.29 ms per rejected cycle; RETEST.md. Restore: remove this line */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_MAINWFE_PF", "1", 0);   /* ADOPTED 2026-10-26: MAINWFE's post and WFE wait outside decode cycles too (the prefill's per-layer waits); exact (EXACT_DET identity); the held-hot CLI prefill A/B not slower; RETEST.md. Restore: remove this line */
        /* 2026-10-13 15:1x: LOCKSTEP_PREFILL (x300 / x322b): prompt suffixes under 256 tokens ride the decode batch one token per step --
           2.35x end to end at N 64 for a rule cost of 1.68 %. Server code only; scoped here so the eval's environment file is unchanged
           (its prompts are >= 256 tokens: lockstep never applies there). */
        if (pn && !strcmp(pn, "ds4-server")) setenv("LLMS_DS4_LOCKSTEP_PREFILL", "256", 0);
        /* 2026-10-13 21:1x: ONE_CLOCK (apply_one_clock.py; found in review): the one-token decode ran with no expert aging (ONE_BS6C returns before the token clock). b1_ab_clock.py, A B B A at batch 1: 93.37 -> 86.05 ms per step (+8.5 %), misses 10.95 -> 6.51 per token; fid_clock.py: paired logprob vs the adopted stack -0.0040 (se 0.0097), rule cost 0.40 %. Scoped to the ED programs where it was measured; ds4-server (ED off, STEP_CLOCK in its batched steps) stays unchanged. */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_DECODE_AGING_CLOCK", "1", 0);
        if (pn) setenv("LLMS_DS4_GUARDS", "1", 0);
        /* 2026-10-19: the.md guards (apply_guards.py, the env audit above), every program */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_PREFILL_ROUTE_BONUS", "1.5*40", 0);   /* OLDP */ setenv("LLMS_DS4_KVTAIL_LAST", "1", 0);   /* KVTAIL_LAST */   /* 2026-10-24 22:4x ADOPTED */   /* 2026-10-24 22:0x ADOPTED: the strong-but-soft resident bonus -- substitution beats LAG deferral on fidelity (pooled KL -0.0176 +- 0.0045) */   /* 2026-10-24 20:4x: the fixed-set adoption REVERTED -- a hard bonus froze a half-empty cache after a rebuild (calc KL 0.349); see */   /* 2026-10-19 03:2x: the prefill bonus 0.4 in the encoder layers (was 0.2; 02:3x-03:2x briefly 2.0). prefill 63.6 -> 72.5 rows/s, generation 25.62 -> 25.18 (2.0: ~80 / ~24.7), warm_kl KL 0.0618 -> 0.0748 (2.0: 0.0982) -- generation is the binding goal, so the weakest bonus that meets 64 with margin */ setenv("LLMS_DS4_MQA7", "1", 0); setenv("LLMS_DS4_ASTRA_INDEX_ALL", "2", 0); setenv("LLMS_DS4_Q4G", "1", 0); /* 2026-10-16 11:3x b2s_ab: SCHED on block 2 +3.6 % (s.e. 2.0 %, n 10); exact */ /* REVERTED 12:1x (agent rewinds into block steps -> full rebuilds; agent 600 s timeout vs 62 s): was 10:4x frontier_speed: L15 (late 1.5) + block 2 +8.6 % (s.e. 1.7 %), KL 0.110 vs 0.059 */ /* 2026-10-16 09:4x q4g_wall: prefill 30.8 -> 28.7 s (-6.7 %, 2 pairs), q4g_price KL 0.017, d logprob -0.003 +- 0.015 */ /* 2026-10-16 09:4x edsas_ab: +7.9 % (s.e. 2.4 %), greedy identical; irdg_gate: both exact (stdout identical), executed*/ /* 2026-10-16 07:0x m7_price: KL 0.0149, d logprob -0.001 +- 0.008 (no measurable loss); bench -11..-20 % per attention */ /* 2026-10-16, executes */ /* 2026-10-16 05:0x pf_price5-7: late prefill bonus 2.0 (64-row short-prefill steps only; SP_ED rows and decode take ONE_RPOL): -7-8 % prefill, KL = shipped on two texts */ setenv("LLMS_DS4_KV_FP4", "0", 0); setenv("LLMS_DS4_ROPE_TAB", "1", 0); setenv("DS4_METAL_DSV41_SOURCE", "@repo/metal/dsv41_ropetab.metal", 0); }   /* 2026-10-16 tlp advice round 7 (2.2 ROPE_TAB, 2.3 KV half): exact long-context levers; identity checked */
        /* 2026-10-13 21:4x: ED_FLAG (apply_ed_flag.py): the per-layer notice by a posted flag + a spinning server instead of encodeSignalEvent + the listener (the event wait stays; the router outputs are checksum-verified). b1_ab_edf.py: text and miss series identical (exact), paired step -3.83 ms (se 0.65) = +4.4 %; one CPU core spins while a token is generated. ED programs only. */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_ED_FLAG", "1", 0); if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_DOORBELL", "1", 0);   /* 2026-10-22 15:0x: the ED path's per-layer event wait as a poll kernel on an MSL 4.1 acquire load -- altw db off +2.449 ms per two-row step (s.e. 0.145); see:3x-15:0x */
        /* 2026-10-13 22:3x: Q8_NSG_K (apply_q8_nsg_k.py): the one-token Q8_0 matvec's simdgroups by row length (K <= 1536 -> 1, <= 3072 -> 2). Per shape (nsg_kernels.py): attn_q_b 232.5 -> 128.2 us, shared down 48.5 -> 41.8; b1_ab_nsgk.py A B B A: zero-miss floor -4.53 / -4.84 ms (+5.7 %). Fidelity on the routing-replay instrument (route_pin.py, bf16_basis.py): +0.0002 +- 0.0039 and -0.0003 +- 0.0043 nats -- inside the null of harmless reorderings; the end-to-end -0.049 reading was trajectory-fork noise. ED programs, as measured. */
        /* 2026-10-13 22:4x: DOWN_COLZ (apply_down_colz.py): the batch-1 routed down projection split over the six expert slots (80 -> 480 threadgroups) into partials + the slot-order sum. b1_ab_replay.py (routing replayed, identical miss series): paired step -2.07 ms (se 0.09) = +2.6 %; fid_colz.py: +0.0018 +- 0.0047 nats (inside the rounding floor). Its MoE source is the adopted one plus the kernel; set here before the table below (whose setenv does not overwrite), for the ED programs only. */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_GTOPK", "0", 0); /* ROUTE_T4TK2 (was 0.95:3 -- it wakes at keep 4) */   /* 2026-10-21 22:5x: +3.14 % generation (s.e. 0.66) on forced trajectories, KL pooled +0.012 +- 0.0056 */ setenv("LLMS_DS4_DOWN_COLZ", "1", 0); setenv("DS4_METAL_MOE_SOURCE", "@repo/live/moe_llms_d8lhd8q4x3wcf_ctx_cffb_cfdec10_bs6c_bs64n_redu_colz_xpar_idx32_g32d2m8xj_xpl5_korder_fold_t8.metal", 0);   /* 2026-10-21 21:4x: korder on the plane builders -- warm -0.646 ms, KL +0.0039 +- 0.0029; was moe_llms_d8lhd8q4x3wcf_ctx_cffb_cfdec10_bs6c_bs64n_redu_colz_xpar_idx32_g32d2m8xj_xpl5.metal */ setenv("LLMS_DS4_DOWN_G32", "pack", 0); /* DOWN_GRID */ setenv("LLMS_DS4_DOWN_G32_NPL", "5", 0); setenv("LLMS_DS4_DOWN_G32_GROUP", "1", 0); }   /* 2026-10-20 (adopt_g32d): B1 -- the source + the G32 side file down, grouped, NPL 5 (KL +0.021; agent score +6.25 % CAP) */
        if (pn && !strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4S8", "1", 0); setenv("LLMS_DS4_RAGGED_AO", "1", 0); setenv("LLMS_DS4_RAGGED_Q", "1", 0); setenv("DS4_METAL_DSV41_SOURCE", "@repo/metal/dsv41_ropetab.metal", 0); setenv("LLMS_DS4_DOWN_COLZ", "1", 0); setenv("DS4_METAL_MOE_SOURCE", "@repo/live/moe_llms_d8lhd8q4x3wcf_ctx_cffb_cfdec10_bs6c_bs64n_redu_colz_xpar_idx32_g32d2m8xj_xpl5_korder.metal", 0); setenv("LLMS_DS4_DOWN_G32", "pack", 0); setenv("LLMS_DS4_DOWN_G32_NPL", "5", 0); setenv("LLMS_DS4_DOWN_G32_GROUP", "1", 0); setenv("DS4_METAL_DENSE_SOURCE", "@repo/live/dense_llms2t_b_over_dense_llms2s_over_dense_llms2r_mvb_q4l_rbf_dual_k_h2_off32k_fold_q4gxt_q4grb_q4grb5_hpf5_aorb5_ksrb5_c32_c16_kspr5_ksp_k2f4_le_rsg_aof_kgh_q4s8e.metal", 0); setenv("DS4_METAL_DSV4_MISC_SOURCE", "@repo/live/dsv4_misc_sg_llms4_llms11_llms13_llms15_llms16_llms17_llms18_tkfast_mqa4_psel_tkdet_idxc_att2_tkcomb_h16_s1_idxr.metal", 0); setenv("LLMS_DS4_ELA_FIT", "pack", 0); setenv("DS4_METAL_DSV4_HC_SOURCE", "@repo/live/dsv4_hc_llms12_llms20_hcf_rows_hx.metal", 0); setenv("LLMS_DS4_Q4L", "pack", 0); }   /* SERVER_DATA */   /* 2026-11-06 cleanup: current grid for all classes; obsolete duplicate retired. */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_BS64_NPL", "5", 0); setenv("LLMS_DS4_BS64_NPL_PF", "4", 0); /* 2026-10-22 02:4x: the linear DSML close scanner -- transcript-identical, parse 51x faster at 16 KiB; see */ setenv("LLMS_DS4_T4PF", "1", 0);   /* 2026-10-22 02:0x: prefill t4 dispatches at NR0 4, idle simdgroups dropped -- bit-identical, prefill +3.54 %; see */   /* 2026-10-22 01:3x: prefill calls at 4 gate / up planes -- agent_kl -0.0061 +- 0.0035; see */ }   /* 2026-10-20 08:5x: the routed gate / up with 5 activation bit-planes (npl_gate ABOVE MARGIN +66 ms per append; NPL=0 restores 6 planes) */
        /* 2026-10-14 00:2x: FA_RED_SG (apply_fa_red_sg.py): the gathered decode attention's split-K reduce on 16 simdgroups per head instead of 32 -- the same 32-lane sums of the same partials, so bit-identical (b1_ab_frs.py: text and per-step misses equal in every arm; -1.1 ms per step against the preceding control, the trailing control drifted, so the size is not pinned). Exact: every program. */
        if (pn && !strcmp(pn, "ds4-server")) setenv("LLMS_DS4_FA_RED_SG", "16", 0);   /* 19:0x: server only -- no non-server mode reads it (dead_defaults_survey2) */
        /* 2026-10-14 00:4x: ED_LA (apply_ed_la.py .. apply_ed_la9.py): the router look-ahead prefetch carried by the ED_FLAG server -- the next layers' routers (l+1, l+2) as a bf16 NEON matvec on layer l's FFN input, scored as ONE_RPOL will score them, and only non-resident experts that survive the policy's top six are read, as utility-tier libdispatch reads into four slots. In-process ALT A/B (ela_alt_slots.py, both phases): -3.81 ms per step (se 1.03), ~5.1 %. Fidelity (fid_ela2.py, 1,945 tokens of the eval's text): B - A -0.0031 nats (se 0.0072), cost 0.3 %. ED programs only (ds4-server runs ED off). */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ED_LA", "1", 0); setenv("LLMS_DS4_ED_LA_AHEAD", "2", 0); }
        /* 2026-10-14 08:2x: EDF_WFE (apply_edf_wfe.py, 2026-10-13: null on an in-process ALT, off): the ED server sleeps in WFE on the
           mailbox line instead of spinning 2.3 M times per notice. This machine runs on a 50 W adapter while the battery charges
           (pmset -g adapter / batt, 08:1x): the power manager holds the GPU below its top clock in 41 % of decode seconds
           (1,126 vs 1,253 MHz), so a spinning core spends the GPU's clock. abab_clk.py wfe, whole-process, 21 pairs over three
           rounds (one through a low-clock regime): -1.59 ms per step (se 0.68, 15 of 21), GPU clock +67 MHz (se 20, t 3.4).
           Exact (only how a CPU thread waits). */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_EDF_WFE", "1", 0);
        /* 2026-10-14 09:1x: ROUTER_BF16 (apply_router_bf16.py): the one-token router reads BF16 copies -- the GGUF's F32 router values are exact BF16 (all 78.6 M have zero low halves), same kernel template / rows / reduction tree, so bit- identical (exact_check.py rbf: text identical, the canary changes it); half of 316 MB per token. abab_clk.py rbf, 14 pairs: -0.92 ms per step (se 0.28, t 3.2, 11 of 14). The json's dense source is the _rbf superset. */
        /* 2026-10-14 10:3x: ROUTER41 (apply_router41.py v2): the one-token V4.1 router (softplus, sqrt, + bias, top six, gather, row sum, clamp, divide, scale: nine dependent one-threadgroup dispatches per layer, because the engine's fused router is written for 256 experts) in ONE dispatch with the same arithmetic (the row sum a 6-lane simd_sum, clamp / divide / scale materialized through threadgroup memory). exact_check.py r41: text identical with the base run twice; the canary (the 6th weight dropped) changes it. alt_long.py r41v2, 64-step blocks: -3.62 ms (se 0.98) and -4.60 ms (se 0.97, phase 1). Lesson: a small dependent kernel costs ~5-10 us in the real stream (its own latency), not the 0.65 us of an empty one. */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_ROUTER41", "1", 0);
        /* 2026-10-14 13:0x: BS64_XPAR (apply_bs64_xpar.py): the bs64 activation planes one thread per value (simd_max, rint per value, simd_sum, simd_ballot per plane) instead of a serial 32-step loop per sub-block -- the tiny prologue before each layer's gate / up cost ~29 us (k1bench.m: 5.85 -> 4.82 ms per 40 layers with the parallel kernel). Bit-exact (dump_cmp.py xparc: 0 differing words; canary moves them). alt_seq.py xparc on the adopted colz source: -0.793 ms (se 0.219). The MOE source above is the _colz_xpar superset. (BS64_XIN -- the serial loop folded into every threadgroup -- lost: +0.76.) */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_BS64_XPAR", "1", 0);
        /* 2026-10-14 11:0x: ED_LA_A3 (the look-ahead prefetch three layers ahead instead of two; apply_ed_la9.py's A3 switch).
           alt_long.py el3, 64-step blocks: -2.41 ms (se 0.96) and -3.33 (se 0.84) raw, -3.2 / -4.2 adjusted for misses (it reads
           ~1.4 more cold experts per step and is faster anyway: the earlier reads are hidden). Not exact (residency feeds the route
           policy): fid_el3.py on the eval's text (715 tokens): B - A +0.0057 nats per token (se 0.0118) -- no likelihood cost even
           at the adverse 95 % edge (-1.8 %) against ~5 % speed. ED programs only (ds4-server runs ED off). */
        /* 2026-10-14 14:0x: MQA_ATTN (apply_mqa_attn.py): the one-token decode attention core as two kernels that read each KV row once for 16 heads (llms_mqa_partial + llms_mqa_reduce) instead of the staging copies, the mask, the pad, kernel_flash_attn_ext_vec (every head re-read every row) and its reduce. Not bit-exact (summation order): verify mode over the eval's text, 4,000 sampled calls, max |d| 2.2e-6 (rel 1.3e-4); fid_mqa.py B - A -0.0006 nats per token (se 0.0051, 1,945 positions). Speed: EDF_TIME's release -> next post 1039.5 -> 1004.9 us per layer (n ~17,500 each); alt_fixed.py -0.47 ms per step (se 0.17, misses-adjusted; raw -0.46, se 0.28; misses +0.02). The misc source is the _tkfast_mqa superset (placed before the generic one: setenv does not overwrite). ED programs only. */
        /* 2026-10-14 15:0x: MQA4 (apply_mqa4.py): the attention reads the converted compressed half rows BY INDEX (no gather pass) with unrolled 16-row loops, plus an unrolled chunk reduce -- the same arithmetic order on the same half values. Bit-exact (dump_cmp.py mqa4: 0 differing words; canary = the last selected row dropped); mqa_bench.m 42.0 vs 51.6 us per layer at 203 rows (81.2 vs 103.8 at 640); EDF_TIME 961.9 -> 942.8 us per layer; alt_fixed -0.310 ms (se 0.167, adjusted; raw -0.08, se 0.25 -- the faster GPU path lengthens the loaded serves: less lead for the look-ahead reads). The misc source becomes the _tkfast_mqa4 superset (it holds MQA_ATTN's kernels too). ED programs only. */
        /* 2026-10-18 08:5x: THE MERGED KERNEL ARM (build_port.py; agent_merged_ab 08:39-08:43, the real agent task, three ABAB pairs on the correct base: generation +8.1 % (paired +1.1 / +2.2 / +2.2 t/s), action +15.4 %, verify -5.0 / -5.4 ms per cycle): HALF2, HX, F16_K2, ATT2 (4 splits), OFF32K, SWIGLU_BF16; the dense source is the _h2_off32k superset, the misc source the _idxc_att2 superset (placed first: setenv does not overwrite). Non-server programs. */
        if (pn && strcmp(pn, "ds4-server")) { setenv("DS4_METAL_DENSE_SOURCE", "@repo/live/dense_llms2t_b_over_dense_llms2s_over_dense_llms2r_mvb_q4l_rbf_dual_k_h2_off32k_fold_q4gxt_q4grb_q4grb5_hpf5_aorb5_ksrb5_c32_c16_kspr5_ksp_k2f4_le_rsg_aof_kgh_q4s8e_nr8_s1nr_aosa_mx8r.metal", 0); setenv("LLMS_DS4_AO_SPLIT_A", "576", 0); setenv("LLMS_DS4_ROUTER_MX8", "1", 0); setenv("LLMS_DS4_ROUTER_MX8K", "1", 0);   /* AOSA_MX8: AO_SPLIT_A 576 + the router on MX8 copies, adopted under ) */   /* 2026-10-26: was the _le source -- + the rsg kernels */   /* 2026-10-25: was dense_llms2t_b_over_dense_llms2s_over_dense_llms2r_mvb_q4l_rbf_dual_k_h2_off32k_fold_q4gxt_q4grb_q4grb5_hpf5_aorb5_ksrb5_c32_c16_kspr5_ksp_k2f4.metal -- the same source + the four lean-epilogue kernels (make_le_source.py) */   /* 2026-10-24 10:3x: was dense_llms2t_b_over_dense_llms2s_over_dense_llms2r_mvb_q4l_rbf_dual_k_h2_off32k_fold_q4gxt_q4grb_ksp_k2f4.metal -- the chain's source on the same kernels + 7 */ setenv("LLMS_DS4_K2F4_AO", "1", 0);   /* 2026-10-23 20:3x: the two-row attention output_b at nr 4, exact, -0.27 ms per step */ setenv("LLMS_DS4_Q4G_KSP", "1", 0);   /* 2026-10-23 16:3x: the xt K-split passes as S partial planes + an ordered reduction dispatch (kernel_llms_q4g64_ksp_nt8_xt + kernel_llms_q4g_ksred) instead of the atomic last-arrival tail -- bit-identical (x302 1,200 / 1,200), +2.87 % per prefill call (s.e. 0.19, n 5) */ setenv("LLMS_DS4_Q4G_RB", "5", 0);   /* 2026-10-24 10:3x: 1 -> 5, the OR-dequant rb5 body (the chain) */   /* 2026-10-22 20:0x: the > 32-token xt passes with out_dim >= 8,192 register-blocked with the activation tile staged (4 simdgroups of 32 x 32, rb3) -- bit-exact (identity +0.0000), +0.62 % prefill (s.e. 0.14, n 3; q_b -20 % in isolation with the staged body) */ setenv("LLMS_DS4_Q4G_XT", "1", 0);   /* 2026-10-22 02:4x: the >32-token hpf / ksahpf nt8 GEMM passes on a K-major half copy -- bit-identical, prefill +1.57 %; see */   /* 2026-10-21 20:5x: the bias fold -- warm -0.451 ms x2, rows_kl -0.0008 +- 0.0028; was dense_llms2t_b_over_dense_llms2s_over_dense_llms2r_mvb_q4l_rbf_dual_k_h2_off32k.metal */ setenv("DS4_METAL_DSV4_MISC_SOURCE", "@repo/live/dsv4_misc_sg_llms4_llms11_llms13_llms15_llms16_llms17_llms18_tkfast_mqa4_psel_tkdet_idxc_att2_tkcomb_h16_s1_idxr_hxt_h32_pv2.metal", 0); setenv("LLMS_DS4_SP_ROWS", "256", 0);   /* M41 ROWS256: was 128 */   /* 2026-10-22: the SHORT-PREFILL steps at most 64 rows (llms_rows_max stays 128 for the row paths) -- prefill +13..+30 % per agent session on top of LIST1024 with generation held; KL +0.0031 +- 0.0062 (the route bonus binds tighter at 64); a fidelity dial */   /* 2026-10-22 03:3x: 128-row short-prefill calls -- agent KL -0.0289 +- 0.0105, score pairs in */ /* 2026-10-22 03:2x: the short prefill indexer batched -- bit-identical, +0.57 / +0.29 % (each FUTILE); RETEST */ setenv("LLMS_DS4_ATT2_H16", "2", 0); /* 2026-10-22 00:5x: the prefill attention on one split, finalized in-kernel -- see:5x; was dsv4_misc_sg_llms4_llms11_llms13_llms15_llms16_llms17_llms18_tkfast_mqa4_psel_tkdet_idxc_att2_tkcomb_h16.metal */   /* 2026-10-21 22:5x: the split attention with 16 heads per threadgroup -- bit-identical, prefill +2.04 % (s.e. 0.37); was dsv4_misc_sg_llms4_llms11_llms13_llms15_llms16_llms17_llms18_tkfast_mqa4_psel_tkdet_idxc_att2_tkcomb.metal */ setenv("LLMS_DS4_ATT2", "4", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ATT2_BYROWS", "16:8:5:4", 0); }   /* c ATT2_BYR3 (bisect: back to "16:8:4"); ADOPTED 2026-11-01: three-row calls at 5 splits, -0.939 ms a 3-row step (s.e. 0.168); before: ADOPTED 2026-10-26: the decode attention's splits by rows, 16 / 8 / 4; ALTW -0.93 ms per two-row step held on; KL pooled -0.0051 +- 0.0028 against the warm ship. Restore: remove this line */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DSP_TOPK", "2", 0); }   /* ADOPTED 2026-10-26: the drafter's stages on their top 2 experts; ALTW dr_gpu -0.525 ms per cycle, acceptance -0.0008 +- 0.028; KL pooled -0.0025 +- 0.0028 against the warm ship. Restore: remove this line */
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SERVE_FAST", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_HEAD_RS", "256", 0); setenv("LLMS_DS4_ENGRAM_MM", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DSP_SS", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_CB_LAZYABORT", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_AO_RB5", "1", 0); setenv("LLMS_DS4_Q4G_KSRB5", "1", 0); setenv("LLMS_DS4_Q4G_C32", "1", 0); setenv("LLMS_DS4_Q4G_KSPLIT_BY_SHAPE", "5120:1280=8;5120:512=16;5120:32=32;2304:5120=4;8192:5120=4", 0); setenv("LLMS_DS4_AO_OB_KS", "4", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DSP41_SHORT", "pack", 0); }   /* 2026-10-19 05:0x: the drafter's head over 31,457 corpus ids + the context ( drafter -0.86 ms per cycle, exact sampling) */
        if (pn && strcmp(pn, "ds4-server")) { }   /* 2026-10-19 05:5x: no mlock on a fresh expert slot ( -946 ms of mlock per session, round 0 -8 %, wired unchanged) */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4L_K2DUAL", "1", 0); }   /* 2026-10-19 11:4x: q_a + kv and the shared gate + up as one two-row dispatch ( identity exact, -0.39 ms per two-row step) */
        if (pn && strcmp(pn, "ds4-server")) { }   /* 2026-10-19 14:0x: A1X v3, the two-row HC transitions fused with the mixers carried by the duals ( exact, -0.63 ms per two-row step) */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ENG_PAR", "1", 0); }   /* 2026-10-19 15:0x: the Engram gated add on 1024 threads per row and stream, bit-identical ( -0.33 ms per two-row step) */
        if (pn && strcmp(pn, "ds4-server")) { }   /* 2026-10-19 17:1x: E1, the rows attention reads its window from the ring and the new rows from batch.kv (no snapshot / append copy), bit-identical */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_TK_IDXORD", "1", 0); }   /* 2026-10-19 18:0x: the indexer top-k set in index order, no sort; rounding class, no acceptance tax measured */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_PREFILL_ROUTE_TOPK", "3", 0); /* APPEND_TOPK3 */ setenv("LLMS_DS4_SP_TAIL", "8:r", 0); setenv("LLMS_DS4_PWNORM", "1", 0); /* 2026-10-27 ADOPTED: the weight rule of a prefill row that lost picks (the shipped renormalization boosts a two-pick row by ~2.08; this rule: KL -0.01100); restore: remove this setenv */ /* 2026-10-27 ADOPTED: prefill top 3 -> 2 paid for by the resident-only tail (an append's last 8 rows on their resident router picks); restore: LLMS_DS4_PREFILL_ROUTE_TOPK=3 LLMS_DS4_SP_TAIL=0 */ setenv("LLMS_DS4_PLAG_DROP", "2", 0); }   /* 2026-10-21 07:5x: its fidelity-free half -- prefill top 4 -> 3 (KL -0.0060), PLAG_DROP 2 (KL -0.0018), SEED_EARLY (exact) and the K <= 2 grouped down on the XJ source (g32d4m8xj; exact) -- adopted; the K <= 2 down REVERTED 08:1x (adopt_m8free's clock-controlled verify +3.43 +- 1.72 %, its 10-20 production A/B -2.10 +- 1.46 %; pair_steps drifted, inconclusive) -- the source line above is g32d2m8xj again; M8 whole: FUTILE +3.23 +- 1.01 vs its 2.62 % price, prefill-only +1.69 +- 0.63, cycle-level verify -1.23 +- 0.84 ms. Its priced half (LAG 2:37, EVICT_LAYER_W 13-19:2 + HOTNESS_DECAY 8) is not adopted. Reverts: the source back to g32d2m8xj, PREFILL_ROUTE_TOPK 4, PLAG_DROP 0, SEED_EARLY 0. */   /* 2026-10-20 07:1x (adopt_mpf, merged with CED_SP br:32; was 5 at 05:4x): the prefill's > 4-row steps keep each row's top 4 experts ( merged_kl rpol5 gate; -4.7 % prefill wall 10-15) */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DECODE_ROUTE_MISS_SPLIT", "2", 0); }   /* 2026-10-20 07:1x (adopt_split): a non-resident decode pick outside the row's two heaviest is dropped */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_SHM_CACHE", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_FLIPKEEP", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_MIN_P", "0.2", 0); }   /* ADOPTED 2026-10-30: the agent samples with min_p 0.2 unless --min-p is given; bisect: remove this line */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_EDIT_PYCHECK", "1", 0); setenv("LLMS_DS4_AGENT_EDIT_PYCHECK_DETAIL", "2", 0); }   /* ADOPTED 2026-10-31 */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_RESTART_S", "60", 0); }   /* AGENT_RESTART_S; ADOPTED 2026-10-31: a prompt still running at 60 s is interrupted, reset, resubmitted once salted.*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SP_BATON", "0", 0); }   /* 2026-10-31 REVERTED; was ADOPTED 2026-10-31 (M31_PYFIX_BATON) */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_SYSPROMPT_EXACT", "1", 0); }   /* ADOPTED 2026-10-31: the system-prompt checkpoint built with EXACT_DET's prefill dials, one per rendered prompt; : a warm ship-built prefix +5.36 alien per 1k / +0.0217 KL on calc, the agent's exact build within the A/A of the pin; ~50 s once per prompt change. Bisect: LLMS_DS4_AGENT_SYSPROMPT_EXACT=0 */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_LATCRIT", "2", 0); }   /* 2026-10-20 (adopt_latcrit): latency-critical activity + the time-constraint policy on the encode thread (CAP +2.92 % of score, exact) */   /* 2026-10-20 (adopt_flipkeep): keep the accepted draft across a sampling-mode flip when the new mode draws it (exact: shadow 30 / 30) */   /* 2026-10-20 05:4x: the expert cache outlives the agent process ( restore bit-identical, canary fires; round 0 -20 %); agent only -- the CLI instruments stay cold */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_DSML_HINT", "1", 0); }   /* 2026-10-20 (adopt_dsmlhint): the DSML41 grammar's fixed next token as the block-1 draft on greedy markup steps (exact) */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_DSP_BANK_SHM", "1", 0); }   /* 2026-10-20 (adopt_bankshm): the drafter bank in shared memory across agent sessions (exact; -173 ms of round 0) */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_R0PF", "1", 0); }   /* 2026-10-20 (adopt_r0pf): round 0's experts loaded at the previous agent's exit (exact; round 0 ~-270 ms) */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_R0PF_DEC", "16", 0); }   /* 2026-10-20 (adopt_r0pfdec): the R0PF recording through the first 16 decode cycles of round 0 (exact) */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_GRAMMAR_DEPTH", "1", 0); setenv("LLMS_DS4_K3X", "1", 0); setenv("LLMS_DS4_K3DUAL", "1", 0); setenv("LLMS_DS4_K3HC", "1", 0); setenv("LLMS_DS4_K4X", "1", 0); setenv("LLMS_DS4_G32D_XJ4", "1", 0); setenv("LLMS_DS4_DSPB_FULLMK", "2", 0); setenv("LLMS_DS4_DSPB_BATCH", "1", 0); }   /* 2026-10-20 (adopt_gd): the DSML grammar's fixed tokens as one k-row verify + the third row on the two-row fast paths (exact; gdepth_ab +3.99 % generation) */ /* 2026-10-21 15:0x: K3DUAL + K3HC -- the two-row duals and HC fusions at three rows, EXACT (three-row dumps identical, canaries differ); repeat instrument -0.605 / -1.764 ms per three-row step. Revert: LLMS_DS4_K3DUAL=0, LLMS_DS4_K3HC=0. */ /* 2026-10-21 15:5x: K4X -- four-row parity (HX / HXO / KGX / F16_K2 / dual / HC at four rows); repeat instrument -15.64 +- 0.54 ms per four-row step, four-row KL -0.0039 +- 0.0066; G32D_XJ4 exact, -1.590 +- 1.092 ms. Revert: LLMS_DS4_K4X=0 / LLMS_DS4_G32D_XJ4=0. */ /* 2026-10-21 17:1x: DSPB_FULLMK 2 + DSPB_BATCH 1 -- block-mode only (inert at DSP41_BLOCK 1): the later drafts over the full vocabulary with the Markov correction, the batched block drafter. Revert: LLMS_DS4_DSPB_FULLMK=0 / LLMS_DS4_DSPB_BATCH=0. */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_GRAMMAR_CHAIN", "1", 0); /* 2026-10-31 BACK ON; was: GRAMMAR_CHAIN OFF ADOPTED 2026-10-30 (courageous, member 2): -6.05 % speed (lagc_speed) for predicted tokens x0.830.*/ }   /* ADOPTED 2026-10-24 19:5x: the DSML41 fixed text chained across tags, 3 drafts (4-row verifies); +3.26 points (s.e. 1.81, n 4), generation +2.95 %; exact (verified drafts); LLMS_DS4_GRAMMAR_CHAIN=0 restores */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_GFORCE", "1", 0); setenv("LLMS_DS4_GFORCE_OPEN", "1", 0); setenv("LLMS_DS4_AGENT_SYNTAX_RETRY", "2", 0); setenv("LLMS_DS4_GFORCE_NAME", "1", 0); }   /* GFORCE_NAME ADOPTED 2026-10-30 13:1x. Restore: LLMS_DS4_GFORCE_NAME=0 */   /* SYNTAX_RETRY 2 ADOPTED 2026-10-27: the detector also catches the ASCII DSML marker and tags named after the agent tools; LLMS_DS4_AGENT_SYNTAX_RETRY=1 restores mode 1 */   /* SYNTAX_RETRY ADOPTED 2026-10-27: a round with tool markup but no DSML call is regenerated through the loopbreak retry; LLMS_DS4_AGENT_SYNTAX_RETRY=0 restores */   /* GFORCE_OPEN ADOPTED 2026-10-27: after the full DSML marker a token fitting neither start form is replaced by the form's next text; LLMS_DS4_GFORCE_OPEN=0 restores */   /* ADOPTED 2026-10-25: a token off the DSML41 grammar's fixed text is replaced by the fixed text (STRUCTURAL parser state only); LLMS_DS4_GFORCE=0 restores */
        if (pn && strcmp(pn, "ds4-server")) { }   /* ADOPTED 2026-10-25: the prefill GEMMs on their two-half-epilogue kernels at 9,216 bytes of threadgroup memory, over TG16K; exact; restore: the old dense source, and LEANEPI set to 0 */
        if (pn && strcmp(pn, "ds4-server")) { }   /* ADOPTED 2026-10-25: the owner layers' per-row publication batched; exact; restore: set LLMS_DS4_SP_PUBLISH to 0 */
        if (pn && strcmp(pn, "ds4-server")) { }   /* ADOPTED 2026-10-25: the ratio-0 layers' short-prefill attention batched on att2; a rounding class (pooled KL -0.0004 +- 0.0006); restore: set LLMS_DS4_R0ATT2 to 0 */
        if (pn && strcmp(pn, "ds4-server")) { }   /* ADOPTED 2026-10-25 */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_COROUTE", "0.2", 0); }   /* ADOPTED 2026-10-25 */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_SHEXP", "1", 0); setenv("LLMS_DS4_ANE_SHEXP_DIR", "pack", 0); setenv("LLMS_DS4_ANE_SHEXP_HANDOFF", "cpu", 0); setenv("LLMS_DS4_ANE_SHEXP_LAYERS", "20", 0); setenv("LLMS_DS4_ANE_PRELOAD", "1", 0); setenv("LLMS_DS4_ANE_PRELOAD_NOWAIT", "1", 0); setenv("LLMS_DS4_ANE_SHEXP_UNWIRED", "1", 0); }   /* ADOPTED 2026-10-26: the shared expert of layers 0..19 on the ANE (i8t); held hot -7.04 % prefill */
        if (pn && strcmp(pn, "ds4-server")) { }   /* ADOPTED 2026-10-26: the drafter's 4-bit table mlock'd */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_SHEXP_ROUND", "128", 0); }   /* ADOPTED 2026-10-26: appends under 128 rows keep the shared expert on the GPU */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4L_RSG", "4", 0); /* 2026-10-27 ADOPTED: the Engram projection on the half-x rsg kernel, the bf16 rounding folded -- KL WITHIN (-0.0024 +- 0.0024), ALTW per-step price positive; restore: LLMS_DS4_PROJ_RSG=0 */ }   /* ADOPTED 2026-10-26: the K-split dense matvecs at 2-4 rows and the one-row matvec on row-per-simdgroup kernels */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_LOOPBREAK", "retry", 0); }   /* ADOPTED 2026-10-25: a round whose last 512 8-grams are >= 90 % repeats of earlier ones is stopped, its message dropped and regenerated (at most twice); LLMS_DS4_AGENT_LOOPBREAK=0 restores */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_FIXSET_FIT", "0.3", 0); }   /* ADOPTED 2026-10-25: a HARD guard whose restored cache serves round 0 badly (installs / touches > 0.3) turns the decode bonus and the prefill schedule SOFT -- a full cache of the wrong experts (after an exact-routing session: 26 of 28 such sessions in (0.3, 0.95], 0 of 203 after ship-like sessions); restore: =0 */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_LB_SNAP", "1", 0); }   /* ADOPTED 2026-10-25: a loopbreak retry restores the round-start snapshot instead of invalidating the session (whose next sync re-prefilled the whole transcript: 5.8 % of the ship sessions' wall); restore: =0 */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_GREEDY_LT", "1", 0); }   /* ADOPTED 2026-10-25: the token after a line-start '<' is the argmax -- the deviant tool-result opener ('<' + 72461): loopfix stress sessions with any deviant start, L 0 against S 3 of the first 7 (the session is the unit); restore: =0 */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_LOOPBREAK_SHARE", "0.6", 0); }   /* ADOPTED 2026-10-25: a round stops at 307 of 512 repeated 8-grams (healthy max 0.402); restore: =0 (the old 0.9) */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_THINK_BUDGET", "3000", 0); }   /* ADOPTED 2026-10-25: </think> forced after 3,000 thinking tokens in a round (1 healthy round in 785); restore: =0 */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_PAD_TIGHT", "1", 0); }   /* ADOPTED 2026-10-24 19:5x: +2.63 points (s.e. 3.33, n 4), pooled KL -0.00401 (bound +0.00007); the members: LAG 2:36:3:0 on the generic LAG line, DECODE_ROUTE_LATE 20:0.6 -> 37:0.6 in ds41_adopted_env.json (both binaries), PAD_TIGHT 1 here; restore: LLMS_DS4_LAG=2:37:2:1 DECODE_ROUTE_LATE=20:0.6 PAD_TIGHT=0 */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_LAG", "0", 0); /* LAG OFF ADOPTED 2026-10-30 : predicted tokens x0.790, no speed cost.*/ }   /* 2026-10-24 19:5x: d 2 -> 3, band 2..36, mode 1 -> 0 (was 2:37:2:1); 2026-10-21 10:0x: M9 v2 widened the band 20:34 -> 2:37; before that, 2026-10-21 05:4x: LAG v2 mode 1 -- a non-resident decode pick in layers 20-34 leaves its call, its output added 2 layers later; per-step -1.087 ms (s.e. 0.296), KL ~0; LLMS_DS4_LAG=0 restores */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_PLAG", "0:17:2", 0); setenv("LLMS_DS4_PFU_PRECISE", "1", 0); }   /* 2026-10-21 07:1x: PLAG -- a non-resident prefill pick in layers 0-17 leaves the batched call, its expert read in the background (PFU; LLMS_DS4_PLAG_CAP 64 loads per ticket), its output added 2 layers later; PFU_PRECISE: the PFU guard waits only for the prepared layer's ticket and a landing entry is never evicted; PLAG_NW: the lag call keeps the open batch. A/B n 4: prefill +20.5 / +22.1 / +15.7 / +10.1 %, generation -8.5 / -0.2 / +5.2 / -1.5 % (FUTILE +2.15 +- 2.72 vs margin 1.0); KL x302 all-600 +0.0053 +- 0.0036. RETEST row: generation. LLMS_DS4_PLAG=0 reverts. */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DECODE_ROUTE_TOPK_LATE", "30:4", 0); setenv("LLMS_DS4_DECODE_ROUTE_BAND", "12-19:1.5", 0);   /* 2026-10-24 22:4x ADOPTED */ }   /* 2026-10-21 07:3x: M7's decode dials -- top-4 of the re-picked six in layers >= 30, the resident bonus 0.5 in layers 12-19 (fewer experts and installs per token: fewer SSD bytes, the larger thermal owner). m7d FUTILE +3.49 +- 2.29 % vs its 2.65 % price (point net +0.84); m7b (with prefill top-3) +2.46 +- 1.73 vs 2.94, prefill-only +2.28 +- 0.78 (the dials cut the prefill tails' decoder-layer picks). KL +0.012-0.017. Reverts: LLMS_DS4_DECODE_ROUTE_TOPK_LATE=0, LLMS_DS4_DECODE_ROUTE_BAND=0. */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DECODE_ROUTE_CUM", "0.90", 0); setenv("LLMS_DS4_DECODE_ROUTE", "1.5", 0);   /* 2026-10-24 22:4x ADOPTED: the graded bonus */ setenv("LLMS_DS4_FIXSET_GUARD", "0.95", 0); setenv("LLMS_DS4_FIXSET_AGE", "0", 0);   /* 2026-10-24 22:3x ADOPTED: the hard bonus only after a >= 95 % restore; the old soft values refill a partial cache (the b5 level restored 2,312 of 4,100) */   /* 2026-10-24 22:0x ADOPTED: decode rows with a miss 4,549 -> 160 a calc session */   /* 2026-10-24 20:4x: fixed-set adoption REVERTED (see the PREFILL_ROUTE_BONUS line) */ setenv("LLMS_DS4_DECODE_ROUTE_TOPK", "4", 0); setenv("LLMS_DS4_THINK_K", "2", 0); /* ROUTE_T4TK2 */ /* 2026-10-23 19:5x: 4 -> 3 under W0 + the cap */ }   /* 2026-10-21 13:2x: THE MERGED ARM M10 on the courageous-adoption rule -- KL +0.0329, session +5.09 % (FUTILE vs 5.59), cycle level +8.33 %; SP_VROWS 3 -> 4 below is its fifth member. Revert: LLMS_DS4_CED_SFX=0, LLMS_DS4_DECODE_ROUTE_CUM=0, LLMS_DS4_DECODE_ROUTE=0.20, LLMS_DS4_DECODE_ROUTE_TOPK=5, LLMS_DS4_SP_VROWS=3. */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DECODE_COROUTE", "0.1", 0); }   /* 2026-10-21 08:4x: a verify row's score bonus 0.1 on its earlier rows' final picks -- distinct experts per two-row call ~8.5 -> ~7.9 (the top sink is bytes); two-row verify KL +0.0021 +- 0.0031. LLMS_DS4_DECODE_COROUTE=0 reverts. */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DECODE_ROUTE_UNION", "0", 0); /* DECODE_ROUTE_UNION (was 4); ADOPTED 2026-10-30 : the union cap OFF: offline -21 % divergence from exact, -36 % sampled exact-improbable tokens (alien_mass.py, tv_survival.py 10-30).*/ setenv("LLMS_DS4_UCAP_GATE", "0.35:3", 0); setenv("LLMS_DS4_ROUTE_FILL", "1", 0); setenv("LLMS_DS4_HX_PRODUCER", "1", 0); /* 2026-10-27 ADOPTED: the producers write the half copy of their output (swiglu, hcq_apply, the kg twin) and the consumers skip their f32 -> f16 conversion -- bit-identical; LLMS_DS4_HX_PRODUCER=0 restores the conversions */ /* 2026-10-27 ADOPTED: each row of a two-row verify takes every expert of the kept set that is among its own six, weights norm-matched -- KL -0.0112 +- 0.0043, -0.18 ms per step; restore: LLMS_DS4_ROUTE_FILL=0 */ /* 2026-10-27 ADOPTED: cap 3 on a verify whose sampled draft has q entropy <= 0.35 nats -- KL +0.0030 +- 0.0015, -0.31 ms per step; restore: LLMS_DS4_UCAP_GATE=0 */ /* 2026-10-25 03:2x ADOPTED: the cap back at 4 -- the fixed-token tail */ /* 2026-10-23 20:1x: 4 -> 3 */   /* 2026-10-23 18:5x: 5 -> 4, ucap3 run C + */ setenv("LLMS_DS4_UCAP_RES", "0.5", 0); setenv("LLMS_DS4_UCAP_SQ", "2", 0); /* 2026-10-23 19:5x: */ setenv("LLMS_DS4_ROUTE_WNORM", "1", 0); /* 2026-10-27 00:0x ADOPTED: the kept experts x sqrt(sum six w^2 / sum kept w^2) -- pooled KL -0.0089 (s.e. 0.0034), the fixed-token tail better; restore: LLMS_DS4_ROUTE_WNORM=0 */ }   /* 2026-10-23: the two-row verify step's distinct routed experts capped at 5 on top of the per-row cuts, ranked by summed weight + 0.5 for a resident expert -- per step in production (altw, ucap2 run B) -2.56 ms per two-row step (s.e. 0.31) with -2.3 installs (cap 5, bonus 0.5; cap 4 read -4.26 but carried a DSML error); with it LLMS_DS4_ROUTE_W0 1 -- KL on the exact arm: the ship's cuts 0.0283 -> 0.0217 under W0, the cap 5 + bonus with W0 0.0394 (+0.0111 over rp, paired s.e. 0.0027). Revert: LLMS_DS4_DECODE_ROUTE_UNION=0, LLMS_DS4_ROUTE_W0=0. */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_EDF_NAP", "0.5", 0); setenv("LLMS_DS4_EDF_NAP_ZONE", "0.5", 0); setenv("LLMS_DS4_EVICT_LAYER_W", "13-19:2", 0); setenv("LLMS_DS4_HOTNESS_DECAY", "8", 0); }   /* 2026-10-21 10:0x: THE MERGED ARM M9 v2 -- members retired for insufficient gain, merged: ABOVE MARGIN +6.07 % (s.e. 1.68, n 8), agent-regime KL +0.0014 +- 0.0032; with LAG 2:37 above. Revert: LLMS_DS4_EDF_NAP=0, LLMS_DS4_EVICT_LAYER_W=0, LLMS_DS4_HOTNESS_DECAY=16, LLMS_DS4_LAG=20:34:2:1. */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_G32D_XJ", "1", 0); }   /* 2026-10-21 03:3x (adopt_xj.py): the down's job builder folded into its planes dispatch -- per-step ALT +0.904 ms (s.e. 0.189) saved, dumps identical */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_PREAD_CHUNK", "1048576", 0); setenv("LLMS_DS4_ELA_PSTAR", "0.6", 0); }   /* 2026-10-21 03:3x (adopt_halfy.py): the merged arm half Y, CAP +1.26 % (s.e. 1.28, n 6); with the tkcomb misc source */
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_ED_LA_CPU_SPLIT", "1", 0); }   /* 2026-10-20 (adopt_split1): the look-ahead prediction on one worker, not four (exact; -0.19 busy P-cores, clock +1.7 %, generation +0.77 % CAP) */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_ATT2_ROWSCALE", "1", 0);   /* 2026-10-20 (adopt_att2rs): ATT2's online-softmax rescale as an elementwise row scale, not a diagonal MMA (exact: att2rs_ppl byte-identical, canary differs) */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_ELA_FIT", "pack", 0);   /* 2026-10-20 (adopt_elafit; 10-21 adopt_elafit5: the 26-session refit): the fitted look-ahead (A5; ela_agent generation +6.81 % s.e. 2.08 n 8, wasted prefetches -60 %) */
        if (pn && strcmp(pn, "ds4-server")) { }   /* 2026-10-19 14:2x: the exact lean two-simdgroup epilogue for the half-x kernels ( -0.40 ms per two-row step) */
        if (pn && strcmp(pn, "ds4-server")) { }   /* 2026-10-19 14:2x: HX + the one-simdgroup direct epilogue ( q_b 89.7 -> 73.7 us, exact, -0.65 ms per two-row step) */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4L_KGX", "1", 0); }   /* 2026-10-19 13:3x: the grouped output_a on half heads ( -0.60 ms per two-row step; rounding-class, KL equal to the adopted colz2) */
        if (pn && strcmp(pn, "ds4-server")) { }   /* 2026-10-19 12:2x: the head and the attention output_b on half x (hxo2_gate.py: identical, -0.26 ms per two-row step) */
        if (pn && strcmp(pn, "ds4-server")) { }   /* 2026-10-19 11:4x: the two-row HC split carried by the dual (split2_gate.py) */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DOWN_COLZ2", "1", 0); }   /* 2026-10-19 11:4x: the two-row routed down on the slot-parallel kernels (colz2_gate.py: -0.22 ms per two-row step, rounding-class) */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ED_LA_I8", "1", 0); }   /* 2026-10-19 04:0x: the look-ahead's CPU router matvec on int8 copies (vdotq_s32): i8_gate -1.03 ms per step (2 s.e. 1.01; drift-corrected ~-0.9), m2b merged -0.66; warm_kl2 (two rows, where it acts) KL 0.0778 vs ship 0.0779 */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4G_PF3", "2", 0); setenv("LLMS_DS4_DOWN_T8", "3", 0); }   /* 2026-10-18 23:4x: the merged arm of three sub-resolution prefill levers */
        if (pn && strcmp(pn, "ds4-server")) { }   /* 2026-10-18 22:3x: the prefetch GEMMs on half, padded tiles; rounding class */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4G_PF", "1", 0); setenv("LLMS_DS4_Q4G_PF2", "1", 0); }   /* 2026-10-18 22:1x: the dense GEMMs' register prefetch, bit-identical */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DOWN_T8W", "1", 0); }   /* 2026-10-18 20:3x: the down tiles in K blocks of 64, padded A tile */
        if (pn && strcmp(pn, "ds4-server")) { }   /* 2026-10-18 19:1x: all-hit layers take the split path -- the 8-pair down tiles */
        if (pn && strcmp(pn, "ds4-server")) { }   /* 2026-10-18 17:4x: -- the call ends at the bounded replay start*/
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4G_KSPLIT_DEFAULT", "4", 0);   /* 2026-10-24 10:3x: auto -> 4 (the chain) */ setenv("LLMS_DS4_Q4G_KSA", "1", 0); setenv("LLMS_DS4_AOG_KSA", "1", 0); }   /* 2026-10-18 16:2x: the atomic K-split + the bounded attention output */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_LIST512", "1", 0); setenv("LLMS_DS4_LIST1024", "1", 0);   /* 2026-10-22 20:4x: the routed pair lists up to 1,024 pairs -- 128-row calls keep the list paths; a rounding class, -0.0011 +- 0.0046 in production's regime; +5.97 % prefill (s.e. 0.51, n 3) */ setenv("LLMS_DS4_MQA_ATTN", "1", 0); setenv("LLMS_DS4_CED_SP", "br:1", 0);   /* 2026-10-24 22:0x ADOPTED: KL -0.0017 +- 0.0032, prefill +4.66 % */   /* 2026-10-21 18:5x: was br:4 -- decode tokens paid ~0.11 nats each in the cold window; score +9.03 % on */   /* 2026-10-20 (adopt_br; was br:32 at 07:14): the agent's replay call is its per-round fixed prefill cost (round_fit 582 ms per round); screen_br all-600 / cold over br:32 for br:1; br_agent ABOVE MARGIN mean +5.24 % vs margin 1.51 */ setenv("LLMS_DS4_Q4L_DSP", "pack", 0);   /* 2026-10-15 adopt_q4ld.py: the drafter on 4-bit copies (+3.5 %, no Q8_0 left in decode) */   /* 2026-10-15 q4c2_test: wide verify steps never read Q8_0 (6-row FORBID run clean), speed neutral */   /* 2026-10-15 adopt_dsp41.py: DSP41 block 1 (+13-15 % on agent turns) */ setenv("LLMS_DS4_ELA_PROTECT", "1", 0); setenv("LLMS_DS4_ROUTER41_ROWS", "1", 0); setenv("DS4_METAL_DSV4_HC_SOURCE", "@repo/live/dsv4_hc_llms12_llms20_hcf_rows_hx_hcq1p.metal", 0); setenv("LLMS_DS4_HCQ_ONEPASS", "1", 0); /* members M80 MERGED: HCQ_ONEPASS; SHEXP_NOWAIT REMOVED 11-04 by (identity vacuous under EXACT_DET, speed nil) */ setenv("LLMS_DS4_DSP41", "pack", 0)   /* ADOPTED 2026-10-30 12:1x (courageous, controlled folly): ft4 = the OVERLAP-trained drafter, offline held-out +0.0380 vs ft3 +0.0317; native ft4 - ft3 +0.0019 +- 0.0052 overlap., Q4L_DSP -> engine/live/dsp_ft3_q4s8e.idx. PREVIOUSLY ADOPTED 2026-10-30 10:1x: the FINE-TUNED drafter -- QAT 4-bit codes on the shipped scales + trained floats, 99 generic prompts; offline held-out +0.0317 overlap, native +0.0096 +- 0.0135 conditional (underpowered); exact (drafts only).*/; setenv("LLMS_DS4_DSP41_BANK", "pack", 0); /* 21:2x tk_ho: strong 0.8/1.5 reverted (KL 0.209, -0.155 nats/token) */ /* 21:4x b2_weak: block 2 REVERTED under the weak bonus (read 21.99 vs B1 24.18, write 18.56 vs 20.74) */ setenv("LLMS_DS4_BATCH_PREFETCH", "0", 0); /* NOPF */   /* 2026-10-15 tk_test.py: verify per-row rope / quantize / indexer batched; greedy text identical, canary differs; +1.5 / +2.3 % */   /* 2026-10-15 b2_strong.py: block 2 under the strong bonus, read 27.05 vs 25.35, write 24.89 vs 24.13 t/s */   /* beta_dsp.py: strong */ setenv("LLMS_DS4_DSP41_ADAPT", "0.40", 0);   /* 19:4x: DeepSeek bounded replay (R 128): sessions 2.5x for -0.0225 nats/token */   /* 2026-10-15 adopt_ced.py: exact CED (900 / 900 bit-identical; canary 141 / 900) */   /* 2026-10-15 adopt_c2h.py: exact (0 of 3.62e9 words differ; canary differs) */ setenv("LLMS_DS4_SP_VROWS", "4", 0);   /* 2026-10-21 13:4x: SP_VROWS 2 REVERTED: the record A/B FUTILE -4.11 %, acceptance -0.0535 (4 / 4 pairs) for prefill +6.45 % -- M10's level 4 */ setenv("LLMS_DS4_SP_COLD", "2,20", 0); setenv("LLMS_DS4_DEC_COLD", "3,1", 0); setenv("LLMS_DS4_ASTRA_ELA_SHIFT", "2", 0); setenv("LLMS_DS4_BS_NOPREHALF", "1", 0);   /* 2026-10-17 adopt_rtpersist.py: the rope cos / sin table kept across command buffers (identity xc_rtpersist; RETEST.md) */   /* 2026-10-17 adopt_noprehalf.py:; RETEST.md) */   /* 2026-10-17 adopt_elashift.py: the decode look-ahead forecast two layers back instead of one  */   /* 2026-10-17 adopt_deccold.py: the decode's resident bonus x3 while the miss EMA per policy call is >= 1 (cold decode); fresh prompt +9-12 % generation, agent +4 % */   /* 2026-10-17 adopt_spcold.py: the short prefill's resident bonus x2 on a layer once the step's misses so far reach 20 per layer done (cold steps); cold12 -8.7 % SP wall; x302 KL +0.020 / perplexity -6.8 % */ /* 2026-10-17 adopt_sppipe.py: the short-prefill step under BATCH_PIPE semantics (no per-layer batch end before the MoE, exit keeps the batch open) -- exact: identical text / installs on the harness, identical x302 dumps; -6 % per full-decoder step */ /* 2026-10-17 readopt_spnpw.py: the short-prefill step keeps the batch open after each layer's MoE (slots guarded in flight): -22..-41 ms per full-decoder 64-row step; policy-off identity 2/2; the install-order variation is the stack's own */ setenv("LLMS_DS4_BATCH_PREFETCH_MAX", "384", 0); setenv("LLMS_DS4_PFU_LOADS", "8", 0); /* 2026-10-16 adopt_spattnv.py: the batched short-prefill attention also for the verify's rows; spattnv_gate.py bit-identical, canary fails */ setenv("LLMS_DS4_DSP41_RING", "64", 0);   /* 2026-10-16 adopt_ring64.py: the drafter's post-append ring rebuild in 64-row chunks (ring64_agent.py) */   /* 2026-10-16 adopt_pfu.py: the prefill look-ahead unions each row's predicted top-6 (was top-1, 8 ids), ordered by rows voting, 8 loads per ticket; bp_ab.py -3.1 %, x302 vs exact unchanged */   /* 2026-10-16 adopt_spvrows3.py: the short prefill's q / kv rope + kv FP8 batched over its rows and its indexer / compressor projections as 4-row groups (VROWS_PRE's parts, the verify's exact k-row path); bit-identical, ROWS_ROPE canary fails; level 2 (64-row F16 kernel) parked: +0.014 KL vs exact */   /* 2026-10-16 adopt_spattn.py: the short prefill's indexed attention batched <= 16 rows per split call ; bit-identical, GPU-ordered canary fails; agent-like appends -1.9 %; agent task PASS */ }
        /* 2026-10-14 14:4x: Q4L_DUAL (apply_q4l_dual.py): q_a + kv and the shared expert's gate + up as single dispatches (kernel_llms_q4l_mv_b2: each threadgroup runs the single q4l kernel's code on its matrix), each carrying its sublayer's HC split (deferred from after hc_pre_fused) as one more threadgroup -- the split's ~13 us serial loop and the kv / up dispatches leave the chain, and the FFN split leaves the path to the router's post. Bit-exact (dump_cmp.py dual: 0 differing words in every field over 240 records; the canary -- halved Sinkhorn iterations in the deferred split -- moves them all). EDF_TIME release -> next post 994.4 -> 974.0 us per layer; alt_fixed.py -0.834 ms per step (se 0.170, misses-adjusted; raw -0.743, se 0.323; misses +0.16). The dense source is the _dual superset. ED programs only. */
        /* 2026-10-14 19:1x: SLAB_ONLY (apply_slab_only.py) -- a CORRECTNESS fix. With ED_LA on, the look-ahead's in-flight prefetches hold slab slots without being cache entries, so the slabs ran dry below the budget and prepare_load_buffers made STANDALONE expert buffers; the ED path makes only the slabs resident and the bs64 kernel reads weights by raw GPU address, so an expert placed there read as ZEROS on the GPU (PEEK: GPU all-zero, CPU data) -- 5-27 dropped expert computations per 40-token run (dump_zero*.py), the decode non-repeatable with every route policy off (vdiv1.py). The fix evicts instead of allocating outside the slabs: 0 zeroed, 0 standalone buffers, two runs identical with ppl equal to the ED_LA-off value to the last digit (vdiv6.py: 27.390786695). Speed: slab_ab.py whole-process ABBA, 4 quads: +0.17 ms per step (se 0.25). ED programs. */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_SLAB_ONLY", "1", 0);
        /* 2026-10-14 01:1x: TOPK_FAST (make_topk_fast.py): kernel_llms_topk_radix with a parallel digit scan (one thread walked 2,048 bins serially, three times) and a k >= n shortcut; census 276.1 -> 17.0 us per call (1.44 -> 0.09 ms per token at decode). Exact: text identical at 96 tokens and after a 3,200-token prompt (19,669 calls with k < n, 7,598 with k >= n); the radix canary changes the text. A misc source beside the adopted one; every program (exact). */
        /* SERVER_DATA: the server's old misc pin removed */
        /* 2026-10-14 02:1x: the token boundary (apply_eng_split.py + apply_boundary.py): ED_PHASE commits after layers 0, 8, ... (the adopted ED=8 committed after 7, 15, ...: the GPU waited for 8 layers' encode), ENG_ASYNC reads the Engram rows on a background block behind per-layer event gates (the host read them before encoding, 1.2-1.5 ms), HEAD_MERGE puts the head in the last chunk's buffer (no drain, one buffer fewer). Exact (text identical). ledger_split.py, in-process ALT: starved -1.95 ms per step, GPU busy -0.99, wall paired -3.10 ms (se 0.53). ED programs only (ENG_ASYNC needs ENG2, adopted). */
        if (pn && strcmp(pn, "ds4-server")) { }
        /* 2026-10-14 02:2x: ARGSORT_SHUFFLE (apply_argsort_shuffle.py): the non-causal argsort (the router's top-k, 46 calls per token) on the stock kernel's own shuffle path; census 25.58 -> 19.92 us per call; exact (exact_check.py: text identical, the ascending canary changes it). Every program. */
        if (pn) { setenv("DS4_METAL_ARGSORT_SOURCE", "@repo/live/argsort_shuffle.metal", 0); }
        /* 2026-10-14 02:2x: KEEPBUSY off for the ED programs (eal_seq.py kb: in-process ALT, time + GPU energy): with the filler, steps
           +4.36 ms (se 1.10; phases +0.9 / +7.8, larger when hotter) and +81 mJ GPU energy per step (se 21.6) at unchanged power -- on this
           power-capped 14-inch M1 Max the filler's ~1.1 W is taken from real work. Adopted on 2026-10-08 when the GPU idled at low clocks;
           the busy, capped decode now loses by it. ED programs only (the server's 64-row regime is not measured here). */
        /* 2026-10-14 05:4x the reversal REVERTED: whole-process pairs show the filler is insurance against a GPU regime the machine enters as its cooling changes: abab_gen.py kb runs 5-7: no filler 73-74 ms median vs 58.5 with it (the GPU OFF in 32 % of samples without it, never with it); abab_regime.py, 9 pairs in the normal regime: +0.37 ms (se 0.24). Break-even at that regime 2.5 % of the time; tonight ~12 %. The in-process ALT +4.36 ms was one regime. */
        /* if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_KEEPBUSY", "0", 0); */
        /* 2026-10-14 03:0x: Q4L (apply_q4l.py, make_dense_q4l.py): the one-token matvecs of attn_q_b, attn_output_b, the shared expert and the head read 4-bit copies (stage A's Q4_0 values, models/DeepSeek-V4.1-Flash-1017-ds4/dense_q4l.bin) through the q4l kernel (throughput/ds41_kbench_q4d.py: 0.53-0.64x the Q8_0 kernel's time). eal_seq.py q4l (in-process A/B, 18 pairs): -8.14 ms per step (se 0.88) of 70.0, GPU energy -178 mJ per step (se 31) of 1,245. fid_q4l.py: cost 5.71 % +- 5.52 (3 se, 350 tokens; the rule stopped it below the 11.6 % break-even). Prefill keeps the Q8_0 weights. The json's dense source is the _q4l superset. ED programs. */
        /* 2026-10-14 04:3x: the msei values: through the real kernel KL 0.0314 vs the reference rounding's 0.0383, -5.3 se paired (fid_q4lk.py, kl_pair.py). */
        /* 2026-10-14 07:5x: + attn_output_a at 4 bits (dense_q4l_msei_alla.idx: the 40 grouped attn_output_a tensors, msei values, the
           grouped q4la kernel; was parked at 03:28 on an in-process ALT A/B of +8.89 ms). abab_clk.py q4la, whole-process, a
           warm-up run discarded, 15 pairs: -2.26 ms per step (se 0.40, t 5.6, 14 of 15 pairs) -- the GPU ran 49 MHz faster with it
           (se 16; the clock-scaled base moved +0.06, se 0.37): on this power-capped machine the 4-bit read saves energy and the
           cap turns it into clock. fid_kl_inc.py outa: KL +0.0194 (se 0.0018) -- 1.9 % against 4.0 % faster. */
        /* 2026-10-14 09:4x: + the Engram projection at 4 bits (make_engram_q4l.py: engram_kv F16 6,144 -> 25,600 of layers 1 and 14,
           least-squares 32-blocks, relative RMS 0.086; apply_q4l_f16.py hooks the one-token F16 matvec; the side file is
           dense_q4l_msei_eng.bin = dense_q4l_msei.bin + the two tensors). 629 -> 177 MB per token; census 880.7 -> 261.1 us per call.
           fid_kl_inc.py eng4: dKL -0.0047 (se 0.0057). alt_long.py (in-process, 64-step blocks, 17 pairs per run, two runs in
           opposite phases, one at full power and one power-zone limited): -1.56 (se 0.50) and -1.78 (se 1.28) ms per step. */
        /* 2026-10-14 12:0x: q_a / kv at 4 bits too (dense_q4l_msei_alla_eng_qakv.idx: the adopted 243 entries + the 80 q_a / kv tensors,
           msei values already in the side file, tagged ALT q4lq). alt_seq.py q4lq (raw: it changes the numbers): -0.861 ms per step
           (se 0.239, t -3.6, decided after 8 blocks / 53 s) = 1.70 % of a 50.6 ms step -> break-even 0.017 nats; fid_kl_inc.py qakv
           KL +0.0080 (se 0.0015): +3 se = 0.0125 < 0.017. The Q8_0 originals become decode-dead (Q4L_UNLOCK releases them). */

        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_Q4L", "pack", 0);
        /* 2026-10-14 07:0x: Q4L_UNLOCK (apply_q4l_unlock.py): the 201 Q8_0 originals of the Q4L tensors (5.38 GiB, read only by prefill) leave the static-weight mlock and get views of their own that decode never binds and the warmup skips. mem_unlock.py: decode wired 52.31 -> 46.96 GiB (the global user wire limit is 52.48 GiB and counts the GPU's wired buffers: the adopted run sat 0.17 GiB under it, and one run in three capped the cache at 3,588 slots). Bit-exact (exact_check.py q4u: text identical with the base run twice). Speed: abab_long.py unl +2.10 ms (se 3.59) for the old state -- no change resolved. The room is NOT spent on cache slots: cache 4,000 (abab_long.py unl4k) cut misses 8.02 -> 6.75 per step but the step by +0.22 ms (se 0.67) only, and every 4,000-slot run made the system compress (0.9-11 GB; 0 in every 3,663 run). ED programs (the server runs without Q4L). */
        /* 2026-10-14 05:01: HC_FUSE1 (apply_hc_fuse1.py): the one-token HC pre-sublayer's flat norm, mix, weighted sum and block norm in one dispatch (25 threadgroups; V4.1 carries pre one sublayer ahead, so the block input does not wait for the mix), then the original split kernel: bit-exact (hcf_diff.py: residual, mix, split, pre, x, norm identical; exact_check.py: text identical, the canary changes it). eal_seq2.py (stopping rule): -0.97 ms (se 1.41) and -3.31 ms (se 1.29), inverse-variance -2.2 ms (se 0.95). The json's HC source is the _hcf superset. ED programs. */
        /* 2026-10-14 05:3x REVERTED: whole-process A B A B in generation mode (abab_gen.py, the use case, tightly alternated against the
           user's cooling changes): HC_FUSE1 off is FASTER by 4.65 ms per step (se 0.60; pairs -5.39 -5.87 -4.08 -3.26); the perplexity
           replay A/B (rab.py) agrees (+12-17 ms with it on). The in-process ALT A/B that adopted it measured the wrong thing. */
        /* if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_HC_FUSE1", "1", 0); */
        /* 2026-10-14 06:2x: HC_FUSE1=3 -- the same fused HC pre-sublayer at 256 threads per threadgroup: the 1,024-thread threadgroups (21 KB
           of threadgroup memory each) were the whole-process slowdown (they need an empty core). Bit-exact: each of the 256 threads
           computes four virtual threads' partials of the 1,024-thread norm kernels, so every simd_sum and the reduction tree match
           (hcf_diff.py: 80 records identical at positions 60-86; exact_check.py with the base twice: text identical, canary differs).
           abab_gen.py hcf3, whole-process pairs: +1.82 +1.52 +1.11 +0.98 ms for the unfused chain (mean 1.36, se 0.19). */
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_HC_FUSE1", "3", 0);
        /* 2026-10-13 17:xx: SHORT_PREFILL for one-session programs (x323-x333, x332): an append under 4,096 tokens ending at position
           <= 40,000 (19:3x; was 20,000) runs as the session's causal rows of 64-row batched steps with the resident bonus (SP_RPOL) and the scoped split
           (SP_SPLIT): five ~520-token appends in one session 88 s instead of 420 s (the layer-major sweep slows 2.3-2.6x after the first
           prefill in a process); continuation logprob unchanged within +-1 s.e. at 512 / 2,048-token prefixes and at position 16,640. */
        /* 17:3x: ds4-server too (x334: 16 x ~2K-token prompts under load, 982 s; the adopted server took them one token at a time,
           ~3.5 min each -- ~55-60 min projected; answers coherent). The eval runs its own server build (preflight-gated on rebuild). */
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SHORT_PREFILL", "1000000", 0); }   /* ADOPTED 2026-10-25: every append on the short path -- an 8,677-row append 262 rows/s against the wide pass's 123 and its GGUF-codec fallback; ds4-server keeps 4096 (the line below); restore: set LLMS_DS4_SHORT_PREFILL to 4096 */
        if (pn && !strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SHORT_PREFILL", "4096", 0); }   /* ds4-server keeps the 10-13 threshold */
        if (pn) {
            setenv("LLMS_DS4_APPEND_ROUTE_POLICY", "1", 0);   /* 2026-10-15 19:4x: .02-0.04) */   /* 2026-10-15 12:2x: exact routing is the default (held-out KL lower bounds 0.017-0.044 for any bonus; fid_ho / kl_bound_ho) */
            /* 2026-10-17 12:xx: SHORT_PREFILL_MAXPOS=0 (no cap) comes from the JSON: past 40,000 the tail went layer-major, the slab stayed empty and the first decode died ; the gate maxpos_gate priced it */
            /* 20:0x: the layer-major prefill's cache seed off (x339 / x341 / x342: it preads ~37.6 GB from the bs64 side file per prefill;
               off, consecutive 2K prefills run 34 s each instead of 47 then ~105 s) */
            setenv("DS4_METAL_DISABLE_STREAMING_PREFILL_CACHE_SEED", "1", 0);
        }
    }
    {   /* the JSON table, per program class: a name that a line above already sets for that class is left out (tools/default_of.py) */
        const char *pn4 = getprogname();
        if (pn4 && strcmp(pn4, "ds4-server")) for (size_t i = 0; i < sizeof llms_adopted_env_nonserver / sizeof llms_adopted_env_nonserver[0]; i++) setenv(llms_adopted_env_nonserver[i][0], llms_adopted_env_nonserver[i][1], 0);
        if (pn4 && !strcmp(pn4, "ds4-server")) for (size_t i = 0; i < sizeof llms_adopted_env_server / sizeof llms_adopted_env_server[0]; i++) setenv(llms_adopted_env_server[i][0], llms_adopted_env_server[i][1], 0);
    }
    /* THE EXPERT CACHE IS A BYTE BUDGET -- the only cache-size knob is LLMS_DS4_EXPERT_CACHE_GIB; the slot count is DERIVED (llms_adopted_engine_defaults: floor(GiB x 2^30 / LLMS_DS4_EXPERT_CACHE_SLOT_BYTES)); the old slot-count name is refused. 38.225 GiB (41.04 GB) for every program but the server: the footprint production has run at since 10-18 (3,950 slots of 9.91 MiB, then 4,100 of 9.55 MiB after B1 shrank the slot). LARGER LOST ON THE AGENT, TWICE: 10-18 ( 4,300 slots faster on matched text, slower on the agent -- its tool runs evict the unlocked dense pages) and 10-21 (slots_block: 4,350 slots -6.5 % generation in production while pair_steps read -10 ms per step). The admission guard refuses any other budget unless LLMS_DS4_EXPERT_CACHE_EXPERIMENT=1. ds4-server: 35.44 GiB = 3,663 slots of 9.91 MiB (BS64 + the q2_K down; its dense Q8_0 is wired; 4,200 did not fit there). */
    { const char *pn3 = getprogname(); if (pn3 && strcmp(pn3, "ds4-server")) setenv("LLMS_DS4_EXPERT_CACHE_GIB", "38.225", 0); if (pn3 && strcmp(pn3, "ds4-server")) setenv("LLMS_DS4_EXPERT_CACHE_SLOT_BYTES", "10010624", 0); if (pn3 && !strcmp(pn3, "ds4-server")) setenv("LLMS_DS4_EXPERT_CACHE_GIB", "35.44", 0); if (pn3 && !strcmp(pn3, "ds4-server")) setenv("LLMS_DS4_EXPERT_CACHE_SLOT_BYTES", "10387456", 0); }
    llms_def_rec = 0;   /* llms DEAD DEFAULTS: the constructor's own setenvs end here */
}
const char LLMS_ADOPTED_CACHE_GIB_NONSERVER[] = "38.225", LLMS_ADOPTED_CACHE_GIB_SERVER[] = "35.44";   /* the budgets, for the admission guard (ds4.c) */
void llms_adopted_engine_defaults(ds4_engine_options *e) {
    const char *off = getenv("LLMS_DS4_NO_ADOPTED"), *old = getenv("LLMS_DS4_ADOPTED_SSD_CACHE"), *gib = getenv("LLMS_DS4_EXPERT_CACHE_GIB"), *sb = getenv("LLMS_DS4_EXPERT_CACHE_SLOT_BYTES");
    if (old && *old) { fprintf(stderr, "llms: REFUSING TO RUN -- LLMS_DS4_ADOPTED_SSD_CACHE (a slot COUNT) is retired: the expert cache is a BYTE budget, LLMS_DS4_EXPERT_CACHE_GIB "
                                   "(the slot count is derived from it). (exit 78)\n"); _exit(78); }
    if ((off && *off && *off != '0') || !e || !gib || !*gib || !sb || !*sb) return;
    e->ssd_streaming = true;   /* 2026-10-13 09:0x: the adopted cache size also when --ssd-streaming is given without a size (x297 sized 4,275 by itself) */
    const double g = atof(gib); const unsigned long long b = strtoull(sb, NULL, 10);
    if (!e->ssd_streaming_cache_experts && !e->ssd_streaming_cache_bytes && g > 0.0 && b > 0ull) e->ssd_streaming_cache_experts = (uint32_t)((g * 1073741824.0) / (double)b);
    fprintf(stderr, "llms adopted: SSD streaming on by default, expert cache %.3f GiB = %u slots of %llu B (flags or LLMS_DS4_NO_ADOPTED=1 override)\n", g, e->ssd_streaming_cache_experts, b);
}
/* ---- end llms ADOPTED ---- */

/* ds4m1: an engine asset named '@repo/<path>' in the defaults above is resolved against the executable's directory, or DS4_ASSET_ROOT when set */
#include <limits.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
__attribute__((constructor)) static void ds4m1_resolve_repo_paths(void) {
    char root[PATH_MAX] = {0}; const char *ar = getenv("DS4_ASSET_ROOT");
    if (ar && *ar) { if (!realpath(ar, root)) return; }
    else {
#if defined(__APPLE__)
        char exe[PATH_MAX]; uint32_t n = sizeof exe;
        if (_NSGetExecutablePath(exe, &n) || !realpath(exe, root)) return;
#else
        return;
#endif
        char *sl = strrchr(root, '/'); if (!sl) return; *sl = 0;
    }
    for (int changed = 1; changed; ) {   /* setenv may move environ: rescan until no '@repo/' value is left */
        changed = 0;
        for (char **e = environ; e && *e; e++) {
            const char *eq = strchr(*e, '='); if (!eq || strncmp(eq + 1, "@repo/", 6)) continue;
            char name[256], val[PATH_MAX]; const size_t nl = (size_t)(eq - *e); if (nl >= sizeof name) continue;
            memcpy(name, *e, nl); name[nl] = 0; snprintf(val, sizeof val, "%s/%s", root, eq + 7);
            setenv(name, val, 1); changed = 1; break;
        }
    }
}

