// llama-eval: simple-mode port of ds4-eval to llama.cpp.
//
// Runs a set of embedded benchmark questions (92 cases: GPQA Diamond,
// SuperGPQA, AIME2025, COMPSEC) through the model with greedy decoding
// and reports pass/fail per case + summary.  No TUI, no thinking-mode
// soft-close, no JSON trace -- just the minimum needed to exercise the
// model and check answers.
//
// The eval_cases[] data and answer-extraction helpers are ported verbatim
// from ds4_eval.c (Andrew Yourtchenko, deepseek4 project).
//
// Honors LLAMA_REP_GUARD=1 transparently via common_sampler.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "chat.h"
#include "llama.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// embedded case data (ported verbatim from ds4_eval.c)
// ---------------------------------------------------------------------------

#define EVAL_MAX_CHOICES 10

struct eval_case {
    const char * source;
    const char * id;
    const char * domain;
    const char * title;
    const char * question;
    const char * choice[EVAL_MAX_CHOICES];
    const char * answer;
};

#include "eval_cases_data.inc"

static int n_eval_cases() {
    return (int)(sizeof(eval_cases) / sizeof(eval_cases[0]));
}

static int eval_case_nchoices(const eval_case * tc) {
    int n = 0;
    while (n < EVAL_MAX_CHOICES && tc->choice[n]) n++;
    return n;
}

static bool eval_case_is_multiple_choice(const eval_case * tc) {
    return eval_case_nchoices(tc) > 0;
}

static bool eval_case_is_compsec(const eval_case * tc) {
    return tc->source && !strcmp(tc->source, "COMPSEC");
}

// ---------------------------------------------------------------------------
// prompt construction (ported from ds4_eval.c::build_question_prompt)
// ---------------------------------------------------------------------------

static const char * eval_system_prompt() {
    return "You are solving a hard benchmark question. Reason carefully. "
           "The final answer must follow the requested format exactly.";
}

static std::string build_question_prompt(const eval_case * tc) {
    std::string s = tc->question;
    s += "\n";
    const int nchoices = eval_case_nchoices(tc);
    if (nchoices > 0) {
        s += "\nChoices:\n";
        for (int i = 0; i < nchoices; i++) {
            s += (char)('A' + i);
            s += ". ";
            s += tc->choice[i];
            s += "\n";
        }
        s += "\nSolve the question. At the end, write exactly one final line in this "
             "format and do not write anything after it:\n"
             "Answer: <letter>";
    } else if (eval_case_is_compsec(tc)) {
        s += "\nAt the end, write exactly one final line in this format and do not "
             "write anything after it:\n"
             "Answer: <line number or comma-separated line numbers>";
    } else {
        s += "\nSolve the problem. At the end, write exactly one final line in this "
             "format and do not write anything after it:\n"
             "Answer: <integer>";
    }
    return s;
}

// ---------------------------------------------------------------------------
// answer extraction & matching (ported from ds4_eval.c)
// ---------------------------------------------------------------------------

static const char * strcasestr_local(const char * hay, const char * needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0) return hay;
    for (; *hay; hay++) {
        if (tolower((unsigned char)*hay) == tolower((unsigned char)needle[0]) &&
            strncasecmp(hay, needle, nlen) == 0) {
            return hay;
        }
    }
    return nullptr;
}

static bool is_letter_boundary(char before, char after) {
    return !isalpha((unsigned char)before) && !isalpha((unsigned char)after);
}

static char find_answer_letter(const char * generated, int nchoices) {
    if (nchoices <= 0) return '?';
    const char * visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;
    char max_answer = (char)('A' + nchoices - 1);

    const char * answer = strcasestr_local(visible, "answer");
    if (answer) {
        const char * end = answer + strlen(answer);
        if (strlen(answer) > 96) end = answer + 96;
        for (const char * p = answer; p < end && *p; p++) {
            char c = (char)toupper((unsigned char)*p);
            if (c >= 'A' && c <= max_answer) {
                char before = (p == visible) ? ' ' : p[-1];
                char after  = p[1];
                if (is_letter_boundary(before, after)) return c;
            }
        }
    }

    for (const char * p = visible + strlen(visible); p > visible; p--) {
        char c = (char)toupper((unsigned char)p[-1]);
        if (c >= 'A' && c <= max_answer) {
            char before = (p - 1 == visible) ? ' ' : p[-2];
            char after  = p[0];
            if (is_letter_boundary(before, after)) return c;
        }
    }
    return '?';
}

static void normalize_integer_answer(const char * p, size_t len, char * dst, size_t dstlen) {
    while (len > 1 && *p == '0') { p++; len--; }
    if (dstlen == 0) return;
    size_t n = len < dstlen - 1 ? len : dstlen - 1;
    memcpy(dst, p, n);
    dst[n] = '\0';
}

static bool scan_first_integer(const char * start, const char * end, char * dst, size_t dstlen) {
    const char * p = start;
    while (p < end && *p) {
        if (isdigit((unsigned char)*p)) {
            const char * q = p + 1;
            while (q < end && isdigit((unsigned char)*q)) q++;
            normalize_integer_answer(p, (size_t)(q - p), dst, dstlen);
            return true;
        }
        p++;
    }
    return false;
}

static void find_integer_answer(const char * generated, char * dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char * visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;

    const char * answer = strcasestr_local(visible, "answer");
    if (answer) {
        const char * end = answer + strlen(answer);
        if (strlen(answer) > 160) end = answer + 160;
        if (scan_first_integer(answer, end, dst, dstlen)) return;
    }

    const char * last_start = nullptr;
    const char * last_end   = nullptr;
    for (const char * p = visible; *p; p++) {
        if (isdigit((unsigned char)*p)) {
            const char * q = p + 1;
            while (isdigit((unsigned char)*q)) q++;
            last_start = p;
            last_end   = q;
            p = q - 1;
        }
    }
    if (last_start && last_end) {
        normalize_integer_answer(last_start, (size_t)(last_end - last_start), dst, dstlen);
    }
}

static void normalize_compsec_line_spec(const char * p, const char * end, char * dst, size_t dstlen) {
    if (dstlen == 0) return;
    size_t n = 0;
    for (; p < end && *p; p++) {
        if (!isdigit((unsigned char)*p)) continue;
        if (n > 0 && n + 1 < dstlen) dst[n++] = ',';
        while (p < end && isdigit((unsigned char)*p)) {
            if (n + 1 < dstlen) dst[n++] = *p;
            p++;
        }
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p < end && *p == '-') {
            if (n + 1 < dstlen) dst[n++] = '-';
            p++;
            while (p < end && isspace((unsigned char)*p)) p++;
            while (p < end && isdigit((unsigned char)*p)) {
                if (n + 1 < dstlen) dst[n++] = *p;
                p++;
            }
        }
        if (p >= end || !*p) break;
    }
    while (n > 0 && (dst[n - 1] == ',' || dst[n - 1] == '-')) n--;
    dst[n] = '\0';
    if (n == 0) snprintf(dst, dstlen, "?");
}

static void find_compsec_answer(const char * generated, char * dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char * visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;

    const char * answer = strcasestr_local(visible, "answer");
    if (answer) {
        const char * end = answer + strlen(answer);
        if (strlen(answer) > 160) end = answer + 160;
        const char * newline = (const char *)memchr(answer, '\n', (size_t)(end - answer));
        if (newline) end = newline;
        normalize_compsec_line_spec(answer, end, dst, dstlen);
        if (strcmp(dst, "?") != 0) return;
    }
    find_integer_answer(generated, dst, dstlen);
}

static bool parse_line_spec(const char * spec, bool * set, size_t setlen) {
    bool any = false;
    const char * p = spec;
    while (p && *p) {
        while (*p && !isdigit((unsigned char)*p)) p++;
        if (!*p) break;
        char * end = nullptr;
        long a = strtol(p, &end, 10);
        long b = a;
        p = end;
        if (*p == '-') {
            p++;
            b = strtol(p, &end, 10);
            p = end;
        }
        if (a > b) { long tmp = a; a = b; b = tmp; }
        if (a < 0) a = 0;
        if (b >= (long)setlen) b = (long)setlen - 1;
        for (long i = a; i <= b; i++) { set[i] = true; any = true; }
    }
    return any;
}

static bool compsec_answer_matches(const char * expected_spec, const char * got_spec) {
    bool expected[256] = {0};
    bool got[256] = {0};
    if (!parse_line_spec(expected_spec, expected, sizeof(expected))) return false;
    if (!parse_line_spec(got_spec, got, sizeof(got))) return false;
    bool hit = false;
    for (size_t i = 0; i < sizeof(got); i++) {
        if (!got[i]) continue;
        if (!expected[i]) return false;
        hit = true;
    }
    return hit;
}

static void find_case_answer(const eval_case * tc, const char * generated, char * dst, size_t dstlen) {
    if (dstlen == 0) return;
    // If the model is in thinking mode and never closed </think>, we should
    // NOT scan the reasoning text for an answer -- the model is liable to use
    // "answer" / digits / letters in its reasoning, producing bogus matches.
    // Report '?' so the case shows up as ERROR rather than a misleading FAIL.
    // (Assumes thinking-mode model.  For non-thinking models, run with a tool
    // that doesn't apply enable_thinking=true in the chat template.)
    if (strstr(generated, "</think>") == nullptr) {
        snprintf(dst, dstlen, "?");
        return;
    }
    if (eval_case_is_multiple_choice(tc)) {
        dst[0] = find_answer_letter(generated, eval_case_nchoices(tc));
        if (dstlen > 1) dst[1] = '\0';
    } else if (eval_case_is_compsec(tc)) {
        find_compsec_answer(generated, dst, dstlen);
    } else {
        find_integer_answer(generated, dst, dstlen);
    }
}

static bool answer_matches(const eval_case * tc, const char * got) {
    if (eval_case_is_multiple_choice(tc)) {
        return got && got[0] && tc->answer && got[0] == tc->answer[0];
    }
    if (eval_case_is_compsec(tc)) {
        return got && tc->answer && compsec_answer_matches(tc->answer, got);
    }
    char expected[64];
    normalize_integer_answer(tc->answer, strlen(tc->answer), expected, sizeof(expected));
    return got && strcmp(got, expected) == 0;
}

// ---------------------------------------------------------------------------
// custom args
// ---------------------------------------------------------------------------

struct eval_extra_args {
    int         start       = 0;
    int         n_questions = -1;   // -1 = all from start
    int         max_tokens  = 8192; // generation cap per question
    std::string dump_dir;           // if set, dump each case's raw text to <dir>/<i>_<id>.txt
};

static void strip_eval_args(int & argc, char ** argv, eval_extra_args & ex) {
    int w = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--start") && i + 1 < argc) {
            ex.start = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--n-questions") && i + 1 < argc) {
            ex.n_questions = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) {
            ex.max_tokens = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--dump-raw") && i + 1 < argc) {
            ex.dump_dir = argv[++i];
        } else {
            argv[w++] = argv[i];
        }
    }
    argc = w;
}

// ---------------------------------------------------------------------------
// per-case run: format chat, tokenize, generate, capture
// ---------------------------------------------------------------------------

struct case_result {
    bool        passed;
    std::string got;          // extracted answer (e.g. "B", "468", "10")
    std::string raw;          // raw generated text (for dumping)
    int         n_generated;
    double      seconds;
};

static case_result run_one(
        llama_context           * ctx,
        const llama_model       * model,
        const llama_vocab       * vocab,
        common_sampler          * smpl,
        common_chat_templates   * tmpls,
        const eval_case         * tc,
        int                       max_tokens,
        bool                      verbose,
        FILE *                    dump_fp) {

    case_result r{};
    r.passed = false;
    r.got    = "?";

    // 1) build the formatted prompt (system + user, with assistant turn opened)
    common_chat_templates_inputs inputs;
    inputs.use_jinja             = true;
    inputs.add_generation_prompt = true;
    inputs.enable_thinking       = true;
    inputs.force_pure_content    = true; // skip the auto-parser path that chokes on this template

    common_chat_msg sys_msg;
    sys_msg.role    = "system";
    sys_msg.content = eval_system_prompt();
    inputs.messages.push_back(sys_msg);

    common_chat_msg user_msg;
    user_msg.role    = "user";
    user_msg.content = build_question_prompt(tc);
    inputs.messages.push_back(user_msg);

    common_chat_params chat = common_chat_templates_apply(tmpls, inputs);
    const std::string & prompt = chat.prompt;

    // 2) tokenize
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, /*add_special=*/true, /*parse_special=*/true);
    if (tokens.empty()) {
        fprintf(stderr, "  ERROR: empty tokenization for case %s\n", tc->id ? tc->id : "?");
        return r;
    }

    // 3) reset KV state for a fresh per-question session
    llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
    common_sampler_reset(smpl);

    auto t0 = std::chrono::steady_clock::now();

    // 4) decode the prompt in batches
    const int n_batch = (int)llama_n_batch(ctx);
    int n_past = 0;
    for (size_t i = 0; i < tokens.size(); i += (size_t)n_batch) {
        const size_t chunk = std::min((size_t)n_batch, tokens.size() - i);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, (int)chunk);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "  ERROR: llama_decode failed on prompt\n");
            return r;
        }
        n_past += (int)chunk;
    }

    // 5) sample-decode loop
    std::string raw;
    raw.reserve(8192);
    llama_token id;
    for (int step = 0; step < max_tokens; step++) {
        id = common_sampler_sample(smpl, ctx, /*idx=*/-1);
        common_sampler_accept(smpl, id, /*accept_grammar=*/true);

        if (llama_vocab_is_eog(vocab, id)) break;

        // append token text to raw
        // special=true so we preserve </think> in the raw buffer -- find_case_answer()
        // splits raw on </think> to skip reasoning text when scanning for the answer.
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), /*lstrip=*/0, /*special=*/true);
        if (n < 0) n = 0;
        raw.append(buf, n);

        if (verbose) {
            fwrite(buf, 1, n, stdout);
            fflush(stdout);
        }
        if (dump_fp && n > 0) {
            fwrite(buf, 1, n, dump_fp);
            fflush(dump_fp);
        }

        // feed it back
        llama_batch nb = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, nb) != 0) {
            fprintf(stderr, "  ERROR: llama_decode failed on generation step %d\n", step);
            return r;
        }
        r.n_generated++;
        n_past++;
    }
    auto t1 = std::chrono::steady_clock::now();
    r.seconds = std::chrono::duration<double>(t1 - t0).count();

    char got[64] = {0};
    find_case_answer(tc, raw.c_str(), got, sizeof(got));
    r.got    = got;
    r.raw    = raw;
    r.passed = answer_matches(tc, got);
    return r;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void print_usage_extras(const char * argv0) {
    fprintf(stderr,
        "\nllama-eval extra options:\n"
        "  --start N           index of first question to run (default: 0)\n"
        "  --n-questions N     number of questions (default: all from --start)\n"
        "  --max-tokens N      per-question generation cap (default: 8192)\n"
        "\nUse standard llama-completion flags for model, context, sampling, etc.\n"
        "Example:\n"
        "  %s -m model.gguf -ngl 999 -c 16384 --start 0 --n-questions 3\n",
        argv0);
}

int main(int argc, char ** argv) {
    eval_extra_args extra;
    strip_eval_args(argc, argv, extra);

    common_params params;
    common_init();

    // Parse standard llama-completion args.  We use the completion example flag
    // set because it includes chat-template / sampling / generation flags.
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION)) {
        print_usage_extras(argv[0]);
        return 1;
    }

    // jinja is required for the chat-template path we use.
    params.use_jinja = true;
    // NOTE: this tool defaults to whatever common_params default sampling is
    // (temp=0.80 by default).  For reproducible / greedy runs, pass --temp 0.

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    if (!llama_init || !llama_init->context() || !llama_init->model()) {
        fprintf(stderr, "FATAL: failed to load model\n");
        return 1;
    }
    llama_model       * model = llama_init->model();
    llama_context     * ctx   = llama_init->context();
    const llama_vocab * vocab = llama_model_get_vocab(model);

    common_chat_templates_ptr tmpls = common_chat_templates_init(model, params.chat_template);

    // Populate reasoning-budget thinking-tag tokens BEFORE sampler creation, so
    // that the rbudget sampler actually gets initialized in common_sampler_init.
    // (llama-cli does this per-task; we only need to do it once.)  Without this
    // step, --reasoning-budget is silently ignored.
    //
    // We probe the chat template *without* force_pure_content (so the
    // auto-parser path runs and populates thinking_start_tag /
    // thinking_end_tag).  If the auto-parser throws (some templates do on
    // empty/probe messages), fall back to hardcoded "<think>" / "</think>" --
    // which is the format Qwen3.5 / 3.6 and other modern reasoning models use.
    std::string think_start_str;
    std::string think_end_str;
    std::string gen_prompt_str;
    try {
        common_chat_templates_inputs probe;
        probe.use_jinja             = true;
        probe.add_generation_prompt = true;
        probe.enable_thinking       = true;
        probe.force_pure_content    = false;  // need auto-parser path to get tags
        common_chat_msg sys, user;
        sys.role = "system"; sys.content = eval_system_prompt();
        user.role = "user";  user.content = "probe";
        probe.messages = {sys, user};
        common_chat_params cp = common_chat_templates_apply(tmpls.get(), probe);
        think_start_str = cp.thinking_start_tag;
        think_end_str   = cp.thinking_end_tag;
        gen_prompt_str  = cp.generation_prompt;
    } catch (const std::exception & e) {
        fprintf(stderr, "rbudget: chat-template probe failed (%s); falling back to <think>/</think>\n", e.what());
    }
    if (think_end_str.empty()) {
        think_start_str = "<think>";
        think_end_str   = "</think>";
    }
    // generation_prompt is what the rbudget sampler sees as "prefill" tokens --
    // it MUST include the thinking_start tag so rbudget transitions from IDLE
    // to COUNTING.  Fall back to just the think_start tag if the probe didn't
    // give us a full generation prompt.
    if (gen_prompt_str.empty()) {
        gen_prompt_str = think_start_str;
    }
    params.sampling.generation_prompt = gen_prompt_str;
    params.sampling.reasoning_budget_start =
        common_tokenize(vocab, think_start_str, false, true);
    params.sampling.reasoning_budget_end =
        common_tokenize(vocab, think_end_str, false, true);
    params.sampling.reasoning_budget_forced =
        common_tokenize(vocab, params.sampling.reasoning_budget_message + think_end_str, false, true);
    fprintf(stderr, "rbudget: think_start=%s (%zu tok) think_end=%s (%zu tok) forced_wrap=%zu tok budget=%d gen_prompt='%s' (%zu chars)\n",
            think_start_str.c_str(), params.sampling.reasoning_budget_start.size(),
            think_end_str.c_str(),   params.sampling.reasoning_budget_end.size(),
            params.sampling.reasoning_budget_forced.size(),
            params.sampling.reasoning_budget_tokens,
            gen_prompt_str.c_str(),  gen_prompt_str.size());

    // Build our own sampler with rbudget tags now populated.  (The one inside
    // llama_init was created before tag population; we ignore it.)
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        fprintf(stderr, "FATAL: failed to init sampler\n");
        return 1;
    }

    // Determine question range.
    const int total = n_eval_cases();
    int start = extra.start;
    if (start < 0) start = 0;
    if (start >= total) {
        fprintf(stderr, "--start %d out of range (have %d cases)\n", start, total);
        return 1;
    }
    int n = extra.n_questions;
    if (n < 0 || start + n > total) n = total - start;

    fprintf(stderr, "\nllama-eval: running %d/%d cases starting at index %d\n",
            n, total, start);
    fprintf(stderr, "LLAMA_REP_GUARD=%s\n", getenv("LLAMA_REP_GUARD") ? getenv("LLAMA_REP_GUARD") : "(unset)");

    int passed = 0;
    int errors = 0;
    auto t_start = std::chrono::steady_clock::now();

    for (int i = 0; i < n; i++) {
        const eval_case * tc = &eval_cases[start + i];
        fprintf(stderr, "\n[%d/%d] %s / %s (%s)\n",
                i + 1, n,
                tc->source ? tc->source : "?",
                tc->id     ? tc->id     : "?",
                tc->domain ? tc->domain : "");
        fprintf(stderr, "  title: %s\n", tc->title ? tc->title : "");

        FILE * dump_fp = nullptr;
        if (!extra.dump_dir.empty()) {
            char path[512];
            snprintf(path, sizeof(path), "%s/%03d_%s.txt",
                     extra.dump_dir.c_str(), start + i,
                     tc->id ? tc->id : "noid");
            dump_fp = fopen(path, "w");
            if (!dump_fp) {
                fprintf(stderr, "  WARN: could not open dump file %s\n", path);
            }
        }

        case_result r = run_one(ctx, model, vocab, smpl, tmpls.get(), tc, extra.max_tokens, /*verbose=*/false, dump_fp);

        if (dump_fp) fclose(dump_fp);

        const char * verdict;
        if (r.got == "?")          { verdict = "ERROR"; errors++; }
        else if (r.passed)         { verdict = "PASS";  passed++; }
        else                       { verdict = "FAIL"; }

        fprintf(stderr, "  -> %s   got=%s expected=%s   (%d tokens, %.1fs)\n",
                verdict, r.got.c_str(), tc->answer ? tc->answer : "?",
                r.n_generated, r.seconds);
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    fprintf(stderr, "\n=== summary ===\n");
    fprintf(stderr, "pass:   %d / %d (%.1f%%)\n", passed, n, n > 0 ? 100.0 * passed / n : 0.0);
    fprintf(stderr, "errors: %d\n", errors);
    fprintf(stderr, "wall:   %.1fs (%.1fs avg)\n", total_sec, n > 0 ? total_sec / n : 0.0);
    fprintf(stderr, "LLAMA_REP_GUARD swaps: see sampler stats (not exposed yet)\n");

    common_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
