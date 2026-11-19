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
}
__attribute__((constructor)) static void llms_prof_init(void) {
    const char *p = getenv("LLMS_DS4_PROF"); if (!p || !*p) return;
    llms_prof_path = strdup(p);
    { const pthread_t me = pthread_self(); llms_prof_tid[0] = me; snprintf(llms_prof_tname[0], sizeof llms_prof_tname[0], "main"); atomic_store(&llms_prof_nthreads, 1u); }
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = llms_prof_handler; sa.sa_flags = SA_SIGINFO | SA_RESTART; sigemptyset(&sa.sa_mask); sigaction(SIGPROF, &sa, NULL);
    struct itimerval it = {{0, 1000}, {0, 1000}}; setitimer(ITIMER_PROF, &it, NULL);
    atexit(llms_prof_write);
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

#include <stdlib.h>
#include <string.h>
#include "ds4.h"
#include <stdio.h>
#include <stdint.h>
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
static int llms_def_rec; static char *llms_def_names[1024]; static int llms_def_n;
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
    for (uint32_t k = 0, h = h0; k < LLMS_EA_N; k++, h = (h + 1) & (LLMS_EA_N - 1)) {
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
        if (!llms_ea_was_read(nm)) { ((void)0); unread++; }
    }
    if (unread) ((void)0);
    {   int top[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
        for (int h = 0; h < LLMS_EA_N; h++) { if (!llms_ea_name[h]) continue; for (int k = 0; k < 8; k++) if (top[k] < 0 || llms_ea_cnt[h] > llms_ea_cnt[top[k]]) { for (int j = 7; j > k; j--) top[j] = top[j - 1]; top[k] = h; break; } }
        ((void)0); for (int k = 0; k < 8 && top[k] >= 0; k++) ((void)0); ((void)0); }
}
static void llms_def_report(void) {
    int dead = 0; for (int i = 0; i < llms_def_n; i++) dead += !llms_ea_was_read(llms_def_names[i]);
    for (int i = 0; i < llms_def_n; i++) if (!llms_ea_was_read(llms_def_names[i])) ((void)0);
    ((void)0);
}
static const char *const llms_adopted_env_nonserver[][2] = {
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
__attribute__((constructor(102))) static void llms_maxdec_profile(void) {
    const char *md = getenv("LLMS_DS4_MAXDEC"); if (!md || !*md || *md == '0') return;
    setenv("LLMS_DS4_CED_SP", "suffix", 0); setenv("LLMS_DS4_APPEND_ROUTE_POLICY", "0", 0);
}
__attribute__((constructor(101))) static void llms_def_init(void) {
    const char *ad = getenv("LLMS_DS4_AUDIT_DEFAULTS"); if (ad && *ad && *ad != '0') { llms_def_rec = 1; atexit(llms_def_report); }
}
__attribute__((constructor)) static void llms_adopted_defaults(void) {
    if (environ) for (char **e = environ; *e && llms_ea_out_n < 512; e++) if (!strncmp(*e, "LLMS_", 5) || !strncmp(*e, "DS4_", 4)) llms_ea_out[llms_ea_out_n++] = strdup(*e);
    atexit(llms_ea_report);
    const char *off = getenv("LLMS_DS4_NO_ADOPTED");
    if (off && *off && *off != '0') return;
    {
        const char *pn = getprogname();
        if (pn && !strcmp(pn, "ds4-server")) setenv("LLMS_DS4_ED", "0", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_Q4S8", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_WORKER_RT", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_RINGPRE", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_NO_CARRY", "0", 0);
        if (pn && !strncmp(pn, "ds4-agent", 9)) setenv("LLMS_DS4_PREFILL_CAP", "256", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_MAINWFE", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_RAGGED_AO", "1", 0); setenv("LLMS_DS4_RAGGED_Q", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_RAGQ_S1_RSG", "1:8", 0);  }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SHM_KEEP_BUILDS", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_BASH_POLL", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_RAGQ_S2_RSG", "8", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_NPL45", "2", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DSP_BS64", "pack", 0); setenv("LLMS_DS4_DSP_G32D", "pack", 0); setenv("LLMS_DS4_DSP_BANK_LEAN", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_DSP_HCF", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_HX_TAIL", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ROWS256", "256", 0); setenv("LLMS_DS4_LIST2048", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SP_PUB256", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_A2R_BASE0", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_PARAM_UNIQ", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_PLAG_RENORM", "2", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SUB_DROP", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_GD_DRAIN", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_DENSE", "1", 0); setenv("LLMS_DS4_ANE_DENSE_LAYERS", "20", 0); setenv("LLMS_DS4_ANE_DENSE_WIRED", "1", 0); setenv("LLMS_DS4_ANE_DENSE_WARM", "1", 0); setenv("LLMS_DS4_ANE_DENSE_HANDOFF", "post", 0); setenv("LLMS_DS4_ANE_DENSE_ANY", "32", 0); setenv("LLMS_DS4_ANE_DENSE_DIR", "pack", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_SHEXP_KA", "0", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_DENSE_EARLY", "1", 0); setenv("LLMS_DS4_ANE_DENSE_AOEARLY", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_SHEXP_WDPROG", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_COLD_REFILL", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_DENSE_MULTI", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_DENSE_KEEPWARM_PROCS", "0,3,6,9,12,14,15,18", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_FIXSET_LEARN", "120", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_PAIROAB", "1", 0); setenv("LLMS_DS4_ANE_OAB_LAYERS", "19", 0); setenv("LLMS_DS4_ANE_OAB_WIRED", "0", 0); setenv("LLMS_DS4_ANE_T2", "32", 0); setenv("LLMS_DS4_ANE_OAB_MULTI", "1", 0); setenv("LLMS_DS4_ANE_OAB_MULTI_MODEL", "pack", 0); setenv("LLMS_DS4_ANE_OAB_KEEPWARM_PROCS", "0,2,4,5,7,9,10,13,15,17,18", 0); setenv("LLMS_DS4_ANE_DENSE_QB_MODEL", "pack", 0); setenv("LLMS_DS4_ATT2_H32", "1", 0); setenv("LLMS_DS4_PSO_MAXT", "@repo/live/pso_maxt_h32.txt", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_OAB_EAGER", "1", 0); setenv("LLMS_DS4_ANE_T12_BG", "1", 0); setenv("LLMS_DS4_ANE_NOCOMPILE", "1", 0);   }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ATT2_PV2", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_RESTART_NOTE", "1", 0); setenv("LLMS_DS4_AGENT_RESTART_NOEDIT", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ADD6", "1", 0); setenv("LLMS_DS4_REDU12", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q2S", "pack", 0); setenv("LLMS_DS4_F16K2_NR", "4", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_G32D_FOLD", "1", 0); setenv("LLMS_DS4_Q4L_NR0", "8", 0); setenv("LLMS_DS4_G32D_TG8", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("DS4_METAL_NORM_SOURCE", "@repo/live/norm_mvb_hxn.metal", 0); setenv("LLMS_DS4_HX_NORM", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DSP_PREENC", "0", 0);    setenv("LLMS_DS4_ROUTE_GAIN", "1.11", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_VPIPE", "2", 0); setenv("LLMS_DS4_VPIPE_FIRST", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_RESCUE_GATE", "0.35:5", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4L_KGX8", "1", 0); setenv("LLMS_DS4_AO_RSG", "8", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_RB_ASYNC", "skip", 0); }
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_MAINWFE_PF", "1", 0);
        if (pn && !strcmp(pn, "ds4-server")) setenv("LLMS_DS4_LOCKSTEP_PREFILL", "256", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_DECODE_AGING_CLOCK", "1", 0);
        if (pn) setenv("LLMS_DS4_GUARDS", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_PREFILL_ROUTE_BONUS", "1.5*40", 0);    setenv("LLMS_DS4_KVTAIL_LAST", "1", 0);                setenv("LLMS_DS4_MQA7", "1", 0); setenv("LLMS_DS4_INDEX_ALL", "2", 0); setenv("LLMS_DS4_Q4G", "1", 0);        setenv("LLMS_DS4_KV_FP4", "0", 0); setenv("LLMS_DS4_ROPE_TAB", "1", 0); setenv("DS4_METAL_DSV41_SOURCE", "@repo/metal/dsv41_ropetab.metal", 0); }
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_ED_FLAG", "1", 0); if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_DOORBELL", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_GTOPK", "0", 0);     setenv("LLMS_DS4_DOWN_COLZ", "1", 0); setenv("DS4_METAL_MOE_SOURCE", "@repo/live/moe_llms_d8lhd8q4x3wcf_ctx_cffb_cfdec10_bs6c_bs64n_redu_colz_xpar_idx32_g32d2m8xj_xpl5_korder_fold_t8_rb_ne_gdd2.metal", 0); setenv("LLMS_DS4_BS64_GDD", "3", 0);     setenv("LLMS_DS4_BS64_RB", "1", 0);     setenv("LLMS_DS4_DOWN_G32", "pack/g32_down_fw20.bin", 0); setenv("LLMS_DS4_GATEUP_BS64", "pack/bs64_gateup_fw20.bin", 0); setenv("LLMS_DS4_ROUTE_GAIN_LAYERS", "4-19:1.0,32-35:1.0", 0); if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_SYSPROMPT_PAYLOAD", "1", 0); }      setenv("LLMS_DS4_DOWN_G32_NPL", "5", 0); setenv("LLMS_DS4_DOWN_G32_GROUP", "1", 0); }
        if (pn && !strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4S8", "1", 0); setenv("LLMS_DS4_RAGGED_AO", "1", 0); setenv("LLMS_DS4_RAGGED_Q", "1", 0); setenv("DS4_METAL_DSV41_SOURCE", "@repo/metal/dsv41_ropetab.metal", 0); setenv("LLMS_DS4_DOWN_COLZ", "1", 0); setenv("DS4_METAL_MOE_SOURCE", "@repo/live/moe_llms_d8lhd8q4x3wcf_ctx_cffb_cfdec10_bs6c_bs64n_redu_colz_xpar_idx32_g32d2m8xj_xpl5_korder.metal", 0); setenv("LLMS_DS4_DOWN_G32", "pack/g32_down_grid.bin", 0); setenv("LLMS_DS4_DOWN_G32_NPL", "5", 0); setenv("LLMS_DS4_DOWN_G32_GROUP", "1", 0); setenv("DS4_METAL_DENSE_SOURCE", "@repo/live/dense_llms2t_b_over_dense_llms2s_over_dense_llms2r_mvb_q4l_rbf_dual_k_h2_off32k_fold_q4gxt_q4grb_q4grb5_hpf5_aorb5_ksrb5_c32_c16_kspr5_ksp_k2f4_le_rsg_aof_kgh_q4s8e.metal", 0); setenv("DS4_METAL_DSV4_MISC_SOURCE", "@repo/live/dsv4_misc_sg_llms4_llms11_llms13_llms15_llms16_llms17_llms18_tkfast_mqa4_psel_tkdet_idxc_att2_tkcomb_h16_s1_idxr.metal", 0); setenv("LLMS_DS4_ELA_FIT", "pack", 0); setenv("DS4_METAL_DSV4_HC_SOURCE", "@repo/live/dsv4_hc_llms12_llms20_hcf_rows_hx.metal", 0); setenv("LLMS_DS4_Q4L", "pack/dense_q4l_rag_v_q4s8e.idx", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_BS64_NPL", "5", 0); setenv("LLMS_DS4_BS64_NPL_PF", "4", 0);  setenv("LLMS_DS4_T4PF", "1", 0);       }
        if (pn && !strcmp(pn, "ds4-server")) setenv("LLMS_DS4_FA_RED_SG", "16", 0);
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ED_LA", "1", 0); setenv("LLMS_DS4_ED_LA_AHEAD", "2", 0); }
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_EDF_WFE", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_ROUTER41", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_BS64_XPAR", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) { setenv("DS4_METAL_DENSE_SOURCE", "@repo/live/dense_llms2t_b_over_dense_llms2s_over_dense_llms2r_mvb_q4l_rbf_dual_k_h2_off32k_fold_q4gxt_q4grb_q4grb5_hpf5_aorb5_ksrb5_c32_c16_kspr5_ksp_k2f4_le_rsg_aof_kgh_q4s8e_nr8_s1nr_aosa_mx8r.metal", 0); setenv("LLMS_DS4_AO_SPLIT_A", "576", 0); setenv("LLMS_DS4_ROUTER_MX8", "1", 0); setenv("LLMS_DS4_ROUTER_MX8K", "1", 0);             setenv("LLMS_DS4_K2F4_AO", "1", 0);    setenv("LLMS_DS4_Q4G_KSP", "1", 0);    setenv("LLMS_DS4_Q4G_RB", "5", 0);       setenv("LLMS_DS4_Q4G_XT", "1", 0);       setenv("DS4_METAL_DSV4_MISC_SOURCE", "@repo/live/dsv4_misc_sg_llms4_llms11_llms13_llms15_llms16_llms17_llms18_tkfast_mqa4_psel_tkdet_idxc_att2_tkcomb_h16_s1_idxr_hxt_h32_pv2.metal", 0); setenv("LLMS_DS4_SP_ROWS", "256", 0);           setenv("LLMS_DS4_ATT2_H16", "2", 0);     setenv("LLMS_DS4_ATT2", "4", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ATT2_BYROWS", "16:8:5:4", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DSP_TOPK", "2", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SERVE_FAST", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_HEAD_RS", "256", 0); setenv("LLMS_DS4_ENGRAM_MM", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DSP_SS", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_CB_LAZYABORT", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_AO_RB5", "1", 0); setenv("LLMS_DS4_Q4G_KSRB5", "1", 0); setenv("LLMS_DS4_Q4G_C32", "1", 0); setenv("LLMS_DS4_Q4G_KSPLIT_BY_SHAPE", "5120:1280=8;5120:512=16;5120:32=32;2304:5120=4;8192:5120=4", 0); setenv("LLMS_DS4_AO_OB_KS", "4", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DSP41_SHORT", "pack", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4L_K2DUAL", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ENG_PAR", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_TK_IDXORD", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_PREFILL_ROUTE_TOPK", "3", 0);       setenv("LLMS_DS4_SP_TAIL", "8:r", 0); setenv("LLMS_DS4_PWNORM", "1", 0);   setenv("LLMS_DS4_PLAG_DROP", "2", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DECODE_ROUTE_MISS_SPLIT", "2", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_SHM_CACHE", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_DECODE_ROUTE_HT", "0", 0);    }
        if (pn && strcmp(pn, "ds4-server")) { setenv("DS4_METAL_DISABLE_QUEUE_KEEPALIVE", "0", 0); setenv("LLMS_DS4_QKA_IDLE", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_DSP_SHORT_Q2S", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_ANE_POST_WAIT_MS", "60000", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_EROWD_GD", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_INDEX_TOPK", "512", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_FLIPKEEP", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_COMPACT_LB", "128,0.75", 0); setenv("LLMS_DS4_AGENT_COMPACT_RETRY", "1", 0); setenv("LLMS_DS4_AGENT_EDIT_HINT", "1", 0);  }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_MIN_P", "0.2", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_EDIT_PYCHECK", "1", 0); setenv("LLMS_DS4_AGENT_EDIT_PYCHECK_DETAIL", "2", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_RESTART_S", "90", 0); setenv("LLMS_DS4_AGENT_LB_RESTART", "3", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SP_BATON", "0", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_SYSPROMPT_EXACT", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_LATCRIT", "2", 0); }      
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_DSML_HINT", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_DSP_BANK_SHM", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_R0PF", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_R0PF_DEC", "16", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_GRAMMAR_DEPTH", "1", 0); setenv("LLMS_DS4_K3X", "1", 0); setenv("LLMS_DS4_K3DUAL", "1", 0); setenv("LLMS_DS4_K3HC", "1", 0); setenv("LLMS_DS4_K4X", "1", 0); setenv("LLMS_DS4_G32D_XJ4", "1", 0); setenv("LLMS_DS4_DSPB_FULLMK", "2", 0); setenv("LLMS_DS4_DSPB_BATCH", "1", 0); }     
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_GRAMMAR_CHAIN", "1", 0);  }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_GFORCE", "1", 0); setenv("LLMS_DS4_GFORCE_OPEN", "1", 0); setenv("LLMS_DS4_AGENT_SYNTAX_RETRY", "2", 0); setenv("LLMS_DS4_GFORCE_NAME", "1", 0); setenv("LLMS_DS4_GFORCE_LT", "1", 0); setenv("LLMS_DS4_DECODE_ROUTE_TOPK", "0", 0); setenv("LLMS_DS4_DECODE_ROUTE_TOPK_LATE", "0:0", 0); setenv("LLMS_DS4_DECODE_ROUTE_CUM", "0", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_COROUTE", "0.2lpm1.5", 0); setenv("LLMS_DS4_COROUTE_CAP", "32", 0); setenv("LLMS_DS4_COROUTE_CAP_MASS", "1", 0); setenv("LLMS_DS4_COROUTE_CAP_QUAD", "2", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_SHEXP", "1", 0); setenv("LLMS_DS4_ANE_SHEXP_DIR", "pack", 0); setenv("LLMS_DS4_ANE_SHEXP_HANDOFF", "cpu", 0); setenv("LLMS_DS4_ANE_SHEXP_LAYERS", "20", 0); setenv("LLMS_DS4_ANE_PRELOAD", "1", 0); setenv("LLMS_DS4_ANE_PRELOAD_NOWAIT", "1", 0); setenv("LLMS_DS4_ANE_SHEXP_UNWIRED", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ANE_SHEXP_ROUND", "128lpm24", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4L_RSG", "4", 0);  }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_LOOPBREAK", "retry", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_FIXSET_FIT", "0.3", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_LB_SNAP", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_GREEDY_LT", "1", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_LOOPBREAK_SHARE", "0.6", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_AGENT_THINK_BUDGET", "3000", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_PAD_TIGHT", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_LAG", "0", 0);  }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_PLAG", "0:17:2", 0); setenv("LLMS_DS4_PFU_PRECISE", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DECODE_ROUTE_BAND", "12-19:1.5", 0);    }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DECODE_ROUTE", "1.5", 0);    setenv("LLMS_DS4_FIXSET_GUARD", "0.95", 0); setenv("LLMS_DS4_FIXSET_AGE", "0", 0);                  }
        if (pn && strcmp(pn, "ds4-server") && strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_DECODE_ROUTE_TOPK", "4", 0); setenv("LLMS_DS4_DECODE_ROUTE_TOPK_LATE", "30:4", 0); setenv("LLMS_DS4_DECODE_ROUTE_CUM", "0.90", 0); }
        if (pn && strcmp(pn, "ds4-server") && strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_THINK_K", "2", 0); setenv("LLMS_DS4_THINK_K_UNTIL", "300", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DECODE_COROUTE", "0", 0);  }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DECODE_ROUTE_UNION", "0", 0);  setenv("LLMS_DS4_ROUTE_FILL", "1", 0); setenv("LLMS_DS4_HX_PRODUCER", "1", 0);          setenv("LLMS_DS4_ROUTE_WNORM", "1", 0);  }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_EDF_NAP", "0.5", 0); setenv("LLMS_DS4_EDF_NAP_ZONE", "0.5", 0); setenv("LLMS_DS4_EDF_NAP_READER", "1", 0);  setenv("LLMS_DS4_EVICT_LAYER_W", "13-19:2", 0); setenv("LLMS_DS4_HOTNESS_DECAY", "8", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_G32D_XJ", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_PREAD_CHUNK", "1048576", 0); setenv("LLMS_DS4_ELA_PSTAR", "0.6", 0); }
        if (pn && !strncmp(pn, "ds4-agent", 9)) { setenv("LLMS_DS4_ED_LA_CPU_SPLIT", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_ATT2_ROWSCALE", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_ELA_FIT", "pack", 0);
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4L_KGX", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DOWN_COLZ2", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_ED_LA_I8", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4G_PF3", "2", 0); setenv("LLMS_DS4_DOWN_T8", "3", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4G_PF", "1", 0); setenv("LLMS_DS4_Q4G_PF2", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_DOWN_T8W", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_Q4G_KSPLIT_DEFAULT", "4", 0);    setenv("LLMS_DS4_Q4G_KSA", "1", 0); setenv("LLMS_DS4_AOG_KSA", "1", 0); }
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_LIST512", "1", 0); setenv("LLMS_DS4_LIST1024", "1", 0);    setenv("LLMS_DS4_MQA_ATTN", "1", 0); setenv("LLMS_DS4_CED_SP", "br:1", 0);             setenv("LLMS_DS4_Q4L_DSP", "pack", 0);          setenv("LLMS_DS4_ELA_PROTECT", "1", 0); setenv("LLMS_DS4_ROUTER41_ROWS", "1", 0); setenv("DS4_METAL_DSV4_HC_SOURCE", "@repo/live/dsv4_hc_llms12_llms20_hcf_rows_hx_hcq1p.metal", 0); setenv("LLMS_DS4_HCQ_ONEPASS", "1", 0);  setenv("LLMS_DS4_DSP41", "pack", 0)   ; setenv("LLMS_DS4_DSP41_BANK", "pack", 0);   setenv("LLMS_DS4_BATCH_PREFETCH", "0", 0);           setenv("LLMS_DS4_DSP41_ADAPT", "0.40", 0);          setenv("LLMS_DS4_SP_VROWS", "4", 0);    setenv("LLMS_DS4_SP_COLD", "2,20", 0); setenv("LLMS_DS4_DEC_COLD", "3,1", 0); setenv("LLMS_DS4_ELA_SHIFT", "2", 0); setenv("LLMS_DS4_BS_NOPREHALF", "1", 0);                  setenv("LLMS_DS4_BATCH_PREFETCH_MAX", "384", 0); setenv("LLMS_DS4_PFU_LOADS", "8", 0);  setenv("LLMS_DS4_DSP41_RING", "64", 0);             }
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_SLAB_ONLY", "1", 0);
        if (pn && strcmp(pn, "ds4-server")) { }
        if (pn) { setenv("DS4_METAL_ARGSORT_SOURCE", "@repo/live/argsort_shuffle.metal", 0); }

        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_Q4L", "pack/dense_q4l_fq4both_q4s8e.idx", 0);
        if (pn && strcmp(pn, "ds4-server")) setenv("LLMS_DS4_HC_FUSE1", "3", 0);
        if (pn && strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SHORT_PREFILL", "1000000", 0); }
        if (pn && !strcmp(pn, "ds4-server")) { setenv("LLMS_DS4_SHORT_PREFILL", "4096", 0); }
        if (pn) {
            setenv("LLMS_DS4_APPEND_ROUTE_POLICY", "1", 0);   
            setenv("DS4_METAL_DISABLE_STREAMING_PREFILL_CACHE_SEED", "1", 0);
        }
    }
    {
        const char *pn4 = getprogname();
        if (pn4 && strcmp(pn4, "ds4-server")) for (size_t i = 0; i < sizeof llms_adopted_env_nonserver / sizeof llms_adopted_env_nonserver[0]; i++) setenv(llms_adopted_env_nonserver[i][0], llms_adopted_env_nonserver[i][1], 0);
        if (pn4 && !strcmp(pn4, "ds4-server")) for (size_t i = 0; i < sizeof llms_adopted_env_server / sizeof llms_adopted_env_server[0]; i++) setenv(llms_adopted_env_server[i][0], llms_adopted_env_server[i][1], 0);
    }
    { const char *pn3 = getprogname(); if (pn3 && strcmp(pn3, "ds4-server")) setenv("LLMS_DS4_EXPERT_CACHE_GIB", "38.225", 0); if (pn3 && strcmp(pn3, "ds4-server")) setenv("LLMS_DS4_EXPERT_CACHE_SLOT_BYTES", "10010624", 0); if (pn3 && !strcmp(pn3, "ds4-server")) setenv("LLMS_DS4_EXPERT_CACHE_GIB", "35.44", 0); if (pn3 && !strcmp(pn3, "ds4-server")) setenv("LLMS_DS4_EXPERT_CACHE_SLOT_BYTES", "10387456", 0); }
    llms_def_rec = 0;
}
const char LLMS_ADOPTED_CACHE_GIB_NONSERVER[] = "38.225", LLMS_ADOPTED_CACHE_GIB_SERVER[] = "35.44";
void llms_adopted_engine_defaults(ds4_engine_options *e) {
    const char *off = getenv("LLMS_DS4_NO_ADOPTED"), *old = getenv("LLMS_DS4_ADOPTED_SSD_CACHE"), *gib = getenv("LLMS_DS4_EXPERT_CACHE_GIB"), *sb = getenv("LLMS_DS4_EXPERT_CACHE_SLOT_BYTES");
    if (old && *old) { fprintf(stderr, "llms: REFUSING TO RUN -- LLMS_DS4_ADOPTED_SSD_CACHE (a slot COUNT) is retired: the expert cache is a BYTE budget, LLMS_DS4_EXPERT_CACHE_GIB "
                                   "(the slot count is derived from it). (exit 78)\n"); _exit(78); }
    if ((off && *off && *off != '0') || !e || !gib || !*gib || !sb || !*sb) return;
    e->ssd_streaming = true;
    const double g = atof(gib); const unsigned long long b = strtoull(sb, NULL, 10);
    if (!e->ssd_streaming_cache_experts && !e->ssd_streaming_cache_bytes && g > 0.0 && b > 0ull) e->ssd_streaming_cache_experts = (uint32_t)((g * 1073741824.0) / (double)b);
}

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
    for (int changed = 1; changed; ) {
        changed = 0;
        for (char **e = environ; e && *e; e++) {
            const char *eq = strchr(*e, '='); if (!eq || strncmp(eq + 1, "@repo/", 6)) continue;
            char name[256], val[PATH_MAX]; const size_t nl = (size_t)(eq - *e); if (nl >= sizeof name) continue;
            memcpy(name, *e, nl); name[nl] = 0; snprintf(val, sizeof val, "%s/%s", root, eq + 7);
            setenv(name, val, 1); changed = 1; break;
        }
    }
}

