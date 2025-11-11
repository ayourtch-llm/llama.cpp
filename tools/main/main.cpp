#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting  = false;
static bool need_insert_eot = false;

// State save/load flags for interactive commands
static bool g_save_state_next = false;
static std::string g_state_save_path = "";

// Tool execution tracking to prevent duplicate executions
static std::string g_last_executed_tool_signature = "";

// Token storage for context compression - stores all tokens in order
static std::vector<llama_token> g_all_tokens;

// Idle timeout tracking
static time_t g_last_activity_time = 0;

// Check if idle timeout has elapsed and we should auto-submit empty input
static bool should_auto_submit_on_idle(int idle_interval_minutes) {
    if (idle_interval_minutes <= 0) {
        return false; // Feature disabled
    }

    time_t current_time = time(nullptr);
    if (g_last_activity_time == 0) {
        g_last_activity_time = current_time;
        return false;
    }

    int elapsed_seconds = (int)(current_time - g_last_activity_time);
    int idle_threshold_seconds = idle_interval_minutes * 60;

    return elapsed_seconds >= idle_threshold_seconds;
}

// Update activity timestamp
static void update_activity_time() {
    g_last_activity_time = time(nullptr);
}

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -sys \"You are a helpful assistant\"\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

// Tool calling support functions
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
static bool is_executable(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return false;
    }
    return (st.st_mode & S_IXUSR) != 0;
}

static std::string execute_command(const std::string & command) {
    std::string result;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return "Error: Failed to execute command\n";
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }

    int status = pclose(pipe);
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        result += "\n[Tool exited with code " + std::to_string(WEXITSTATUS(status)) + "]\n";
    }

    return result;
}

static std::vector<std::string> get_tool_executables(const std::string & tools_dir) {
    std::vector<std::string> executables;

    DIR* dir = opendir(tools_dir.c_str());
    if (!dir) {
        return executables;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') {
            continue; // Skip hidden files and . / ..
        }

        std::string full_path = tools_dir + "/" + entry->d_name;
        if (is_executable(full_path)) {
            executables.push_back(entry->d_name);
        }
    }

    closedir(dir);

    // Sort alphabetically
    std::sort(executables.begin(), executables.end());

    return executables;
}

static std::string collect_tools_help(const std::string & tools_dir) {
    std::vector<std::string> executables = get_tool_executables(tools_dir);

    if (executables.empty()) {
        return "No executable tools found in the 'tools' directory.\n";
    }

    std::ostringstream help_text;
    help_text << "Available tools:\n\n";

    for (const auto & tool_name : executables) {
        help_text << "=== " << tool_name << " ===\n";
        std::string command = tools_dir + "/" + tool_name + " help";
        std::string output = execute_command(command);
        help_text << output;
        if (!output.empty() && output.back() != '\n') {
            help_text << "\n";
        }
        help_text << "\nTo use this tool: <tool-launch>" << tool_name << " [arguments]</tool-launch>\n\n";
    }

    return help_text.str();
}

static std::string execute_tool(const std::string & tools_dir, const std::string & tool_name, const std::string & args) {
    std::string full_path = tools_dir + "/" + tool_name;

    if (!is_executable(full_path)) {
        return "Error: Tool '" + tool_name + "' not found or not executable\n";
    }

    std::string command = full_path;
    if (!args.empty()) {
        // Simple shell escaping - wrap in quotes if contains spaces
        command += " " + args;
    }

    LOG("\n[Executing tool: %s]\n", command.c_str());
    std::string output = execute_command(command);
    LOG("[Tool output follows]\n");

    return output;
}
#elif defined (_WIN32)
// Windows implementations (simplified - no tool support on Windows for now)
static bool is_executable(const std::string & path) {
    return false;
}

static std::string execute_command(const std::string & command) {
    return "Error: Tool execution not supported on Windows\n";
}

static std::vector<std::string> get_tool_executables(const std::string & tools_dir) {
    return std::vector<std::string>();
}

static std::string collect_tools_help(const std::string & tools_dir) {
    return "Tool execution is not supported on Windows.\n";
}

static std::string execute_tool(const std::string & tools_dir, const std::string & tool_name, const std::string & args) {
    return "Error: Tool execution not supported on Windows\n";
}
#endif

// Check if a position in text is inside <think>...</think> tags
static bool is_inside_think_tags(const std::string & text, size_t pos) {
    // Find the most recent <think> before pos
    size_t think_start = text.rfind("<think>", pos);
    if (think_start == std::string::npos) {
        return false; // No <think> tag before this position
    }

    // Check if there's a </think> between think_start and pos
    size_t think_end = text.find("</think>", think_start);
    if (think_end == std::string::npos || think_end > pos) {
        return true; // We're inside an unclosed or currently open think block
    }

    return false; // The think block was closed before pos
}

// Check if the recent output contains <tools-help/> (outside of think tags)
static bool check_for_tools_help(const std::string & text) {
    size_t pos = text.find("<tools-help/>");
    if (pos == std::string::npos) {
        return false;
    }

    // Make sure it's not inside think tags
    return !is_inside_think_tags(text, pos);
}

// Check if the recent output contains <tool-launch>...</tool-launch> and extract tool name and args
// Returns false if inside think tags or if already processed
static bool check_for_tool_launch(const std::string & text, std::string & tool_name, std::string & args, size_t search_from = 0) {
    size_t start = text.find("<tool-launch>", search_from);
    if (start == std::string::npos) {
        return false;
    }

    // Check if this tag is inside think tags
    if (is_inside_think_tags(text, start)) {
        // Try to find the next one after this
        return check_for_tool_launch(text, tool_name, args, start + 1);
    }

    size_t end = text.find("</tool-launch>", start);
    if (end == std::string::npos) {
        return false;
    }

    // Extract the content between tags
    start += 13; // length of "<tool-launch>"
    std::string content = text.substr(start, end - start);

    // Trim whitespace
    content.erase(0, content.find_first_not_of(" \t\n\r"));
    content.erase(content.find_last_not_of(" \t\n\r") + 1);

    // Split into tool name and args
    size_t space_pos = content.find(' ');
    if (space_pos == std::string::npos) {
        tool_name = content;
        args = "";
    } else {
        tool_name = content.substr(0, space_pos);
        args = content.substr(space_pos + 1);
        // Trim args
        args.erase(0, args.find_first_not_of(" \t\n\r"));
        args.erase(args.find_last_not_of(" \t\n\r") + 1);
    }

    return !tool_name.empty();
}

// Save complete LLM state (KV cache + RNG + logits + embeddings) to GGUF file
static bool save_llm_state_to_gguf(llama_context * ctx, const std::string & filename) {
    LOG("\nSaving LLM state to %s...\n", filename.c_str());

    // Get the size of the state
    const size_t state_size = llama_state_get_size(ctx);
    LOG("State size: %zu bytes (%.2f MB)\n", state_size, state_size / (1024.0 * 1024.0));

    // Allocate buffer and get state data
    std::vector<uint8_t> state_data(state_size);
    const size_t written = llama_state_get_data(ctx, state_data.data(), state_size);

    if (written != state_size) {
        LOG_ERR("Failed to get state data: got %zu bytes, expected %zu\n", written, state_size);
        return false;
    }

    // Create GGUF context
    struct gguf_context * gguf_ctx = gguf_init_empty();

    // Add metadata
    gguf_set_val_u32(gguf_ctx, "llm_state.version", 2);  // Bump version for token ID support
    gguf_set_val_u64(gguf_ctx, "llm_state.size", state_size);
    gguf_set_val_str(gguf_ctx, "llm_state.type", "kv_cache_rng_logits_embeddings");

    // Save token IDs count
    gguf_set_val_u64(gguf_ctx, "llm_state.n_tokens", g_all_tokens.size());

    // Save token IDs as array (if any exist)
    if (!g_all_tokens.empty()) {
        LOG("Saving %zu token IDs...\n", g_all_tokens.size());
        // Convert to int32_t for GGUF compatibility
        std::vector<int32_t> tokens_i32(g_all_tokens.begin(), g_all_tokens.end());
        gguf_set_arr_data(gguf_ctx, "llm_state.tokens", GGUF_TYPE_INT32, tokens_i32.data(), tokens_i32.size());
    }

    // Create a ggml context for the tensor
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_size + 1024*1024,  // Extra space for tensor metadata
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,  // We already have the data
    };

    struct ggml_context * ggml_ctx = ggml_init(params);

    // Create a 1D tensor to hold the state data
    int64_t ne[4] = {(int64_t)state_size, 1, 1, 1};
    struct ggml_tensor * state_tensor = ggml_new_tensor(ggml_ctx, GGML_TYPE_I8, 1, ne);
    ggml_set_name(state_tensor, "llm_state_data");
    state_tensor->data = state_data.data();

    // Add tensor to GGUF
    gguf_add_tensor(gguf_ctx, state_tensor);

    // Write to file
    gguf_write_to_file(gguf_ctx, filename.c_str(), false);

    LOG("Successfully saved LLM state (%zu bytes", written);
    if (!g_all_tokens.empty()) {
        LOG(" + %zu tokens", g_all_tokens.size());
    }
    LOG(")\n");

    // Cleanup
    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);

    return true;
}

// Load complete LLM state from GGUF file
static bool load_llm_state_from_gguf(llama_context * ctx, const std::string & filename) {
    LOG("\nLoading LLM state from %s...\n", filename.c_str());

    struct ggml_context * ggml_ctx = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ggml_ctx,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(filename.c_str(), params);

    if (!gguf_ctx) {
        LOG_ERR("Failed to load state file: %s\n", filename.c_str());
        return false;
    }

    // Read metadata
    const int n_kv = gguf_get_n_kv(gguf_ctx);
    uint32_t version = 0;
    uint64_t state_size = 0;
    uint64_t n_tokens = 0;
    int tokens_arr_idx = -1;

    for (int i = 0; i < n_kv; i++) {
        const char * key = gguf_get_key(gguf_ctx, i);
        const enum gguf_type type = gguf_get_kv_type(gguf_ctx, i);

        if (strcmp(key, "llm_state.version") == 0 && type == GGUF_TYPE_UINT32) {
            version = gguf_get_val_u32(gguf_ctx, i);
        } else if (strcmp(key, "llm_state.size") == 0 && type == GGUF_TYPE_UINT64) {
            state_size = gguf_get_val_u64(gguf_ctx, i);
        } else if (strcmp(key, "llm_state.n_tokens") == 0 && type == GGUF_TYPE_UINT64) {
            n_tokens = gguf_get_val_u64(gguf_ctx, i);
        } else if (strcmp(key, "llm_state.tokens") == 0 && type == GGUF_TYPE_ARRAY) {
            tokens_arr_idx = i;
        }
    }

    LOG("State version: %u, size: %lu bytes (%.2f MB)\n", version, state_size, state_size / (1024.0 * 1024.0));

    // Restore token IDs if present (version 2+)
    if (version >= 2 && n_tokens > 0 && tokens_arr_idx >= 0) {
        LOG("Loading %lu token IDs...\n", n_tokens);

        const enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx, tokens_arr_idx);
        const uint64_t arr_n = gguf_get_arr_n(gguf_ctx, tokens_arr_idx);

        if (arr_type == GGUF_TYPE_INT32 && arr_n == n_tokens) {
            const int32_t * tokens_data = (const int32_t *)gguf_get_arr_data(gguf_ctx, tokens_arr_idx);
            g_all_tokens.clear();
            g_all_tokens.reserve(arr_n);
            for (uint64_t i = 0; i < arr_n; i++) {
                g_all_tokens.push_back((llama_token)tokens_data[i]);
            }
            LOG("Successfully restored %zu token IDs\n", g_all_tokens.size());
        } else {
            LOG_WRN("Token array format mismatch, skipping token restore\n");
        }
    } else if (version >= 2) {
        LOG("No token IDs in saved state (empty session)\n");
        g_all_tokens.clear();
    } else {
        LOG_WRN("Old state version %u, no token IDs available (compression will not work)\n", version);
        g_all_tokens.clear();
    }

    // Get the state tensor
    struct ggml_tensor * state_tensor = ggml_get_tensor(ggml_ctx, "llm_state_data");
    if (!state_tensor) {
        LOG_ERR("State tensor not found in file\n");
        gguf_free(gguf_ctx);
        return false;
    }

    // Set the state
    const size_t loaded = llama_state_set_data(ctx, (const uint8_t*)state_tensor->data, ggml_nbytes(state_tensor));

    if (loaded == 0) {
        LOG_ERR("Failed to set state data\n");
        gguf_free(gguf_ctx);
        return false;
    }

    LOG("Successfully loaded LLM state (%zu bytes)\n", loaded);
    LOG("LLM has been restored to the exact state when the save was made\n");

    gguf_free(gguf_ctx);

    return true;
}

// Context compression: Use LLM to summarize tokens that would be discarded
// Returns the compressed tokens, or empty vector on failure
static std::vector<llama_token> compress_context(
        llama_context * ctx,
        const llama_vocab * vocab,
        common_sampler * smpl,
        llama_pos start_pos,
        llama_pos end_pos,
        int target_length) {

    // Show user-facing progress indicator
    LOG("\n");
    LOG("🔄 Context Compression Started\n");
    LOG("   Compressing %d tokens → target %d tokens (%.0f%% ratio)\n",
        (int)(end_pos - start_pos), target_length,
        (float)target_length / (end_pos - start_pos) * 100.0f);
    LOG("   This may take 10-30 seconds depending on model size...\n");

    LOG_INF("=== Context Compression Starting ===\n");
    LOG_INF("Compressing positions %d to %d (%d tokens) down to ~%d tokens\n",
            (int)start_pos, (int)end_pos, (int)(end_pos - start_pos), target_length);

    // Step 1: Extract tokens from the stored token history
    if (start_pos < 0 || end_pos > (llama_pos)g_all_tokens.size()) {
        LOG_ERR("Invalid range for compression: start=%d, end=%d, available=%zu\n",
                (int)start_pos, (int)end_pos, g_all_tokens.size());
        return {};
    }

    std::vector<llama_token> tokens_to_compress(
        g_all_tokens.begin() + start_pos,
        g_all_tokens.begin() + end_pos
    );

    LOG_INF("Extracted %zu tokens from storage\n", tokens_to_compress.size());

    // Step 2: Convert tokens to text
    std::string text_to_compress;
    for (auto token : tokens_to_compress) {
        text_to_compress += common_token_to_piece(ctx, token);
    }

    LOG_INF("Converted to text: %zu characters\n", text_to_compress.size());
    LOG_DBG("Text to compress:\n%s\n", text_to_compress.c_str());

    // Step 3: Create summarization prompt
    std::string summary_prompt =
        "\n\n<|COMPRESSION_TASK|>\n"
        "You are compressing conversation history. Create a dense summary of the following text.\n"
        "Target length: approximately " + std::to_string(target_length) + " tokens.\n"
        "Focus on: key facts, decisions, important context, actionable information.\n"
        "Omit: pleasantries, redundant information, unnecessary details.\n"
        "Output ONLY the compressed summary, nothing else.\n\n"
        "TEXT TO COMPRESS:\n" + text_to_compress + "\n\n"
        "COMPRESSED SUMMARY:\n";

    LOG_INF("Created summarization prompt: %zu characters\n", summary_prompt.size());

    // Step 4: Tokenize the prompt
    auto prompt_tokens = common_tokenize(ctx, summary_prompt, false, true);
    LOG_INF("Prompt tokenized: %zu tokens\n", prompt_tokens.size());

    // Step 5: Decode the summarization prompt (no need to save sampler state -
    // we'll just reset it after by clearing the sampling context)
    LOG("   [1/3] Decoding prompt: %zu tokens...\n", prompt_tokens.size());
    LOG_INF("Decoding summarization prompt...\n");

    for (size_t i = 0; i < prompt_tokens.size(); i += g_params->n_batch) {
        int n_eval = std::min((int)(prompt_tokens.size() - i), g_params->n_batch);

        if (llama_decode(ctx, llama_batch_get_one(&prompt_tokens[i], n_eval))) {
            LOG_ERR("Failed to decode summarization prompt\n");
            LOG("   ❌ Compression failed during prompt decode\n");
            return {};
        }

        // Show progress for long prompts
        if (prompt_tokens.size() > 100 && (i % 100 == 0 || i + n_eval >= prompt_tokens.size())) {
            LOG("      Progress: %zu/%zu tokens\r", i + n_eval, prompt_tokens.size());
            fflush(stdout);
        }
    }
    if (prompt_tokens.size() > 100) {
        LOG("\n");
    }

    // Step 7: Generate the summary
    LOG("   [2/3] Generating summary: target %d tokens...\n", target_length);
    LOG_INF("Generating summary (target: %d tokens)...\n", target_length);
    std::vector<llama_token> summary_tokens;
    std::string summary_text;

    // Generate tokens up to target length (with some margin)
    const int max_summary_tokens = target_length * 2; // Allow up to 2x target
    bool summary_complete = false;

    for (int i = 0; i < max_summary_tokens && !summary_complete; i++) {
        // Show progress every 10 tokens
        if (i > 0 && i % 10 == 0) {
            LOG("      Generated: %d/%d tokens (%.0f%%)\r",
                i, target_length, (float)i / target_length * 100.0f);
            fflush(stdout);
        }

        // Sample next token
        const llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);

        // Check for end of generation
        if (llama_vocab_is_eog(vocab, id)) {
            summary_complete = true;
            break;
        }

        summary_tokens.push_back(id);
        summary_text += common_token_to_piece(ctx, id);

        // Stop if we've reached a reasonable summary length
        if ((int)summary_tokens.size() >= target_length) {
            // Look for a good stopping point (end of sentence)
            std::string token_str = common_token_to_piece(ctx, id);
            if (token_str.find('.') != std::string::npos ||
                token_str.find('!') != std::string::npos ||
                token_str.find('?') != std::string::npos ||
                token_str.find('\n') != std::string::npos) {
                summary_complete = true;
                break;
            }
        }

        // Decode this token for next iteration
        llama_token id_copy = id;  // Need non-const for batch API
        if (llama_decode(ctx, llama_batch_get_one(&id_copy, 1))) {
            LOG_ERR("Failed to decode summary token\n");
            break;
        }
    }

    LOG("\n   [3/3] Finalizing: %zu tokens generated\n", summary_tokens.size());
    LOG_INF("Generated summary: %zu tokens, %zu characters\n",
            summary_tokens.size(), summary_text.size());
    LOG_DBG("Summary text:\n%s\n", summary_text.c_str());

    // Step 8: Reset sampler by clearing its history (simple approach)
    common_sampler_reset(smpl);
    LOG_DBG("Reset sampler\n");

    // Step 9: Add a marker to make the compression visible
    std::string final_summary =
        "\n[COMPRESSED: " + std::to_string((int)(end_pos - start_pos)) + " → " +
        std::to_string((int)summary_tokens.size()) + " tokens]: " + summary_text + "\n";

    auto final_tokens = common_tokenize(ctx, final_summary, false, true);

    LOG_INF("=== Compression Complete: %zu input tokens → %zu output tokens (%.1f%% of original) ===\n",
            tokens_to_compress.size(), final_tokens.size(),
            (float)final_tokens.size() / tokens_to_compress.size() * 100.0f);

    LOG("✅ Compression Complete: %zu → %zu tokens (saved %zu tokens)\n\n",
        tokens_to_compress.size(), final_tokens.size(),
        tokens_to_compress.size() - final_tokens.size());

    return final_tokens;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting  = true;
            need_insert_eot = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*g_ctx, *g_smpl);

            // make sure all logs are flushed
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    common_params params;
    g_params = &params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sampling;

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_INF("%s: llama backend init\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;

    g_model = &model;
    g_ctx = &ctx;
    g_smpl = &smpl;

    std::vector<common_chat_msg> chat_msgs;

    // load the model and apply lora adapter, if any
    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result llama_init = common_init_from_params(params);

    model = llama_init.model.get();
    ctx = llama_init.context.get();

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    // Handle state loading
    if (!params.path_load_activations.empty()) {
        if (!load_llm_state_from_gguf(ctx, params.path_load_activations)) {
            LOG_ERR("%s: failed to load LLM state\n", __func__);
            return 1;
        }
    }

    auto * mem = llama_get_memory(ctx);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto chat_templates = common_chat_templates_init(model, params.chat_template);

    LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, (int) params.cpuparams.n_threads);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        LOG_ERR("%s: no CPU backend found\n", __func__);
        return 1;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    set_process_priority(params.cpuparams.priority);

    struct ggml_threadpool * threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            return 1;
        }

        // start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return 1;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    // auto enable conversation mode if chat template is available
    const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        if (has_chat_template) {
            LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
            params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        } else {
            params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        }
    }

    // in case user force-activate conversation mode (via -cnv) without proper chat template, we show a warning
    if (params.conversation_mode && !has_chat_template) {
        LOG_WRN("%s: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n", __func__);
    }

    // print chat template example in conversation mode
    if (params.conversation_mode) {
        if (params.enable_chat_template) {
            if (!params.prompt.empty() && params.system_prompt.empty()) {
                LOG_WRN("*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?\n");
            }

            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(chat_templates.get(), params.use_jinja, params.default_template_kwargs).c_str());
        } else {
            LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG_INF("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            LOG_INF("%s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(path_session)) {
            LOG_INF("%s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_ERR("%s: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            LOG_INF("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

    LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

    std::vector<llama_token> embd_inp;

    bool waiting_for_first_input = false;
    auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
        common_chat_msg new_msg;
        new_msg.role = role;
        new_msg.content = content;
        auto formatted = common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
        chat_msgs.push_back(new_msg);
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
    };

    std::string prompt;
    {
        if (params.conversation_mode && params.enable_chat_template) {
            if (!params.system_prompt.empty()) {
                // format the system prompt (will use template default if empty)
                chat_add_and_format("system", params.system_prompt);
            }

            if (!params.prompt.empty()) {
                // format and append the user prompt
                chat_add_and_format("user", params.prompt);
            } else {
                waiting_for_first_input = true;
            }

            if (!params.system_prompt.empty() || !params.prompt.empty()) {
                common_chat_templates_inputs inputs;
                inputs.use_jinja = g_params->use_jinja;
                inputs.messages = chat_msgs;
                inputs.add_generation_prompt = !params.prompt.empty();

                prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            }
        } else {
            // otherwise use the prompt as is
            prompt = params.prompt;
        }

        if (params.interactive_first || !prompt.empty() || session_tokens.empty()) {
            LOG_DBG("tokenize the prompt\n");
            embd_inp = common_tokenize(ctx, prompt, true, true);
        } else {
            LOG_DBG("use session tokens\n");
            embd_inp = session_tokens;
        }

        LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
        LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());
    }

    // Should not run without any tokens
    if (!waiting_for_first_input && embd_inp.empty()) {
        if (add_bos) {
            embd_inp.push_back(llama_vocab_bos(vocab));
            LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
        } else {
            LOG_ERR("input is empty\n");
            return -1;
        }
    }

    // Tokenize negative prompt
    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_INF("%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_INF("%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_WRN("%s: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG_INF("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        llama_memory_seq_rm(mem, -1, n_matching_session_tokens, -1);
    }

    LOG_DBG("recalculate the cached logits (check): embd_inp.size() %zu, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu\n",
         embd_inp.size(), n_matching_session_tokens, embd_inp.size(), session_tokens.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOG_DBG("recalculate the cached logits (do): session_tokens.resize( %zu )\n", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    if (params.conversation_mode) {
        if (params.single_turn && !params.prompt.empty()) {
            params.interactive = false;
            params.interactive_first = false;
        } else {
            params.interactive_first = true;
        }
    }

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_INF("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > add_bos) {
            LOG_INF("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_INF("\n");
    }

    // ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (params.interactive) {
        LOG_INF("%s: interactive mode on.\n", __func__);
        LOG_INF("Special commands:\n");
        LOG_INF("  /\\/save <filename> - Save complete LLM state (KV cache, etc.) to GGUF file\n");
        LOG_INF("  /\\/load <filename> - Load LLM state from GGUF file to restore exact conversation state\n");
        LOG_INF("  /\\/temp            - Show current temperature setting\n");
        LOG_INF("  /\\/temp <value>    - Set temperature to a new value (e.g., /\\/temp 0.7)\n");
        LOG_INF("  /\\/timeout         - Show or disable idle timeout (0 = disabled)\n");
        LOG_INF("  /\\/timeout <mins>  - Set idle timeout to N minutes (e.g., /\\/timeout 5)\n");
        LOG_INF("\n");
        LOG_INF("Tool calling (when 'tools' directory exists):\n");
        LOG_INF("  Model can output <tools-help/> to get list of available tools\n");
        LOG_INF("  Model can output <tool-launch>tool-name args</tool-launch> to execute a tool\n");

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_INF("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    auto tmp = common_tokenize(ctx, antiprompt, false, true);
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            LOG_INF("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_INF("Input prefix: '%s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_prefix, true, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            LOG_INF("Input suffix: '%s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_suffix, false, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }
    }

    smpl = common_sampler_init(model, sparams);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        return 1;
    }

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl));
    LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl).c_str());

    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
      //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
      //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_INF("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_INF("\n");

    if (params.interactive) {
        const char * control_message;
        if (params.multiline_input) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to the AI.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_INF("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_INF(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_INF(       "%s", control_message);
        if (params.conversation_mode && params.enable_chat_template && params.system_prompt.empty()) {
            LOG_INF(   " - Not using system message. To change it, set a different value via -sys PROMPT\n");
        }
        LOG_INF("\n");

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool display              = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;
    std::ostringstream assistant_ss; // for storing current assistant message, used in conversation mode

    // the first thing we will do is to output the prompt, so set color accordingly
    console::set_display(console::prompt);
    display = params.display_prompt;

    std::vector<llama_token> embd;

    // single-token antiprompts
    std::vector<llama_token> antiprompt_token;

    for (const std::string & antiprompt : params.antiprompt) {
        auto ids = ::common_tokenize(ctx, antiprompt, false, true);
        if (ids.size() == 1) {
            antiprompt_token.push_back(ids[0]);
        }
    }

    if (llama_model_has_encoder(model)) {
        int enc_input_size = embd_inp.size();
        llama_token * enc_input_buf = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                console::set_display(console::error);
                LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(console::reset);
            }

            if (ga_n == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

                // Calculate effective context limit based on reserve parameter
                // If compression is enabled, ensure we have enough reserve for the compression process
                float effective_reserve = params.ctx_reserve;
                if (params.ctx_compress) {
                    // Compression needs space for:
                    // 1. The compression operation itself (~1000 tokens for prompt + generation)
                    // 2. The compressed result
                    // We compress from 90% down to 75% utilization = discard ~15% of context
                    // Compressed to compress_ratio (default 0.25) = ~3.75% of context
                    const int operation_overhead = 1000;
                    const int max_discard = (int)(n_ctx * 0.15f); // 15% = (90% - 75%)
                    const int max_result_size = (int)(max_discard * params.ctx_compress_ratio);
                    const int total_space_needed = operation_overhead + max_result_size;
                    const float compression_reserve = (float)total_space_needed / n_ctx;

                    // Use larger of user's reserve or compression minimum
                    if (effective_reserve < compression_reserve) {
                        LOG_INF("Increasing context reserve from %.0f%% to %.0f%% for compression (needs %d tokens: %d operation + %d result)\n",
                                params.ctx_reserve * 100, compression_reserve * 100,
                                total_space_needed, operation_overhead, max_result_size);
                        effective_reserve = compression_reserve;
                    }
                }
                const int n_ctx_effective = (int)(n_ctx * (1.0f - effective_reserve));

                // Use actual KV cache position to handle loaded sessions correctly
                const llama_pos kv_pos_max = llama_memory_seq_pos_max(mem, 0);
                const int n_ctx_current = kv_pos_max >= 0 ? (int)(kv_pos_max + 1) : n_past;

                if (n_ctx_current + (int) embd.size() >= n_ctx_effective) {
                    if (!params.ctx_shift){
                        LOG_WRN("\n\n%s: context limit reached (used: %d, limit: %d/%d with %.0f%% reserve) and context shift is disabled => stopping\n",
                                __func__, n_ctx_current + (int) embd.size(), n_ctx_effective, n_ctx, params.ctx_reserve * 100);
                        break;
                    }

                    if (params.n_predict == -2) {
                        LOG_WRN("\n\n%s: context limit reached (used: %d, limit: %d/%d with %.0f%% reserve) and n_predict == %d => stopping\n",
                                __func__, n_ctx_current + (int) embd.size(), n_ctx_effective, n_ctx, params.ctx_reserve * 100, params.n_predict);
                        break;
                    }

                    // Instead of discarding half, only discard enough to get back to 75% utilization
                    // This is less aggressive and leaves more headroom
                    const int target_tokens = (int)(n_ctx_effective * 0.75f);
                    const int tokens_to_free = n_ctx_current - target_tokens;
                    const int n_discard = tokens_to_free > 0 ? tokens_to_free : (n_ctx_current - params.n_keep) / 4; // fallback to 25% if calc fails

                    if (params.ctx_compress) {
                        // Compress discarded context instead of just removing it
                        // Thanks to auto-increased reserve, we have enough space for compression
                        const int compress_start = params.n_keep;
                        const int compress_end = params.n_keep + n_discard;
                        const int compress_target = (int)(n_discard * params.ctx_compress_ratio);

                        const int space_available = n_ctx - n_ctx_current;
                        LOG_INF("Context window compressing: used %d tokens, limit %d/%d (%.0f%% reserve = %d tokens), compressing %d tokens to ~%d\n",
                                n_ctx_current + (int) embd.size(), n_ctx_effective, n_ctx,
                                effective_reserve * 100, space_available, n_discard, compress_target);

                        // Compress using the available reserve space (no need to clear first)
                        auto compressed = compress_context(ctx, vocab, smpl, compress_start, compress_end, compress_target);

                        if (!compressed.empty() && (int)compressed.size() < n_discard) {
                            LOG_INF("Compression successful: %d → %zu tokens\n", n_discard, compressed.size());

                            // Remove the old range from KV cache
                            LOG("   Replacing old context with compressed version in KV cache...\n");
                            llama_memory_seq_rm(mem, 0, params.n_keep, params.n_keep + n_discard);

                            // Re-decode compressed tokens at the keep position
                            LOG_INF("Re-decoding %zu compressed tokens at position %d\n",
                                    compressed.size(), params.n_keep);

                            for (size_t i = 0; i < compressed.size(); i += params.n_batch) {
                                int n_eval = std::min((int)(compressed.size() - i), params.n_batch);

                                // Create batch with correct positions
                                llama_batch batch = llama_batch_get_one(&compressed[i], n_eval);
                                for (int j = 0; j < n_eval; j++) {
                                    batch.pos[j] = params.n_keep + i + j;
                                }

                                if (llama_decode(ctx, batch)) {
                                    LOG_ERR("Failed to re-decode compressed tokens, falling back to simple discard\n");
                                    goto compression_fallback;
                                }
                            }

                            // Shift remaining context
                            const int shift_amount = n_discard - (int)compressed.size();
                            llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_ctx_current, -shift_amount);

                            // Update token storage
                            g_all_tokens.erase(
                                g_all_tokens.begin() + compress_start,
                                g_all_tokens.begin() + compress_end
                            );
                            g_all_tokens.insert(
                                g_all_tokens.begin() + params.n_keep,
                                compressed.begin(),
                                compressed.end()
                            );

                            n_past -= shift_amount;

                            LOG("✅ Context shift with compression complete!\n");
                            LOG("   Saved %d tokens compared to simple discard\n\n", shift_amount);
                            LOG_INF("=== Compression saved %d tokens of context space ===\n", shift_amount);
                        } else {
                            compression_fallback:
                            LOG_WRN("Context compression didn't help, falling back to simple discard\n");

                            // Fall back to simple discard
                            llama_memory_seq_rm(mem, 0, params.n_keep, params.n_keep + n_discard);
                            llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_ctx_current, -n_discard);

                            g_all_tokens.erase(
                                g_all_tokens.begin() + compress_start,
                                g_all_tokens.begin() + compress_end
                            );

                            n_past -= n_discard;
                        }
                    } else {
                        // Simple discard (original behavior)
                        LOG_INF("Context window shifting: used %d tokens, limit %d/%d (%.0f%% reserve), discarding %d tokens\n",
                                n_ctx_current + (int) embd.size(), n_ctx_effective, n_ctx, params.ctx_reserve * 100, n_discard);

                        LOG_DBG("context full, swapping: n_ctx_current = %d, n_past = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                                n_ctx_current, n_past, n_ctx, params.n_keep, n_discard);

                        llama_memory_seq_rm (mem, 0, params.n_keep            , params.n_keep + n_discard);
                        llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_ctx_current, -n_discard);

                        // Update token storage
                        const int compress_start = params.n_keep;
                        const int compress_end = params.n_keep + n_discard;
                        g_all_tokens.erase(
                            g_all_tokens.begin() + compress_start,
                            g_all_tokens.begin() + compress_end
                        );

                        n_past -= n_discard;
                    }

                    LOG_DBG("after swap: n_past = %d\n", n_past);

                    LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());

                    LOG_DBG("clear session path\n");
                    path_session.clear();
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    LOG_DBG("\n");
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    LOG_DBG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_memory_seq_add(mem, 0, ga_i,                n_past,              ib*bd);
                    llama_memory_seq_div(mem, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_memory_seq_add(mem, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    LOG_DBG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                    LOG_ERR("%s : failed to eval\n", __func__);
                    return 1;
                }

                // Track tokens for compression - add to global token storage
                for (int j = 0; j < n_eval; j++) {
                    g_all_tokens.push_back(embd[i + j]);
                }

                n_past += n_eval;

                LOG_DBG("n_past = %d\n", n_past);
                // Display total tokens alongside total time
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG_DBG("saved session to %s\n", path_session.c_str());
            }

            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, /* accept_grammar= */ true);

            // LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

            embd.push_back(id);

            if (params.conversation_mode && !waiting_for_first_input && !llama_vocab_is_eog(vocab, id)) {
                assistant_ss << common_token_to_piece(ctx, id, false);
            }

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG_DBG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                // Console/Stream Output
                LOG("%s", token_str.c_str());

                // Record Displayed Tokens To Log
                // Note: Generated tokens are created one by one hence this check
                if (embd.size() > 1) {
                    // Incoming Requested Tokens
                    input_tokens.push_back(id);
                } else {
                    // Outgoing Generated Tokens
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }

        // reset color to default if there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(console::reset);
            display = true;
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // Check for tool requests in recent output
            const int n_prev = 128; // Look back further to catch full tags
            const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

            // Check for <tools-help/> request
            // Note: Only one tool action per iteration to prevent help examples from being executed
            if (check_for_tools_help(last_output)) {
                LOG_DBG("Detected <tools-help/> request\n");

                // Check if tools directory exists
                if (file_exists("tools")) {
                    std::string help_text = collect_tools_help("tools");

                    LOG("\n[Tools Help Requested]\n");
                    LOG("%s", help_text.c_str());
                    LOG("[End of Tools Help]\n\n");

                    // Inject the help text back into the conversation with structured wrapper
                    std::string wrapped_help = "\n<tool-response type=\"help\">\n" + help_text + "</tool-response>\n";
                    auto help_tokens = common_tokenize(ctx, wrapped_help, false, true);

                    // Check if injecting help would exceed context window
                    int tokens_remaining = n_ctx - n_past - ((int)embd_inp.size() - n_consumed) - 4;
                    if ((int)help_tokens.size() > tokens_remaining) {
                        LOG_WRN("Tool help too large (%zu tokens, only %d available). Truncating...\n",
                                help_tokens.size(), tokens_remaining);
                        if (tokens_remaining > 0) {
                            help_tokens.resize(tokens_remaining);
                        } else {
                            LOG_ERR("No context space available for tool help.\n");
                            help_tokens.clear();
                        }
                    }

                    if (!help_tokens.empty()) {
                        embd_inp.insert(embd_inp.end(), help_tokens.begin(), help_tokens.end());
                    }

                    // Continue generation after injecting help
                    is_interacting = false;
                } else {
                    LOG("\n[Tools Help Requested but 'tools' directory not found]\n\n");
                    std::string error_msg = "\n<tool-response type=\"error\">\nNo 'tools' directory found.\n</tool-response>\n";
                    auto msg_tokens = common_tokenize(ctx, error_msg, false, true);
                    embd_inp.insert(embd_inp.end(), msg_tokens.begin(), msg_tokens.end());
                }
            } else {
                // Check for <tool-launch>...</tool-launch> request only if we didn't handle tools-help
                std::string tool_name, tool_args;
                if (check_for_tool_launch(last_output, tool_name, tool_args)) {
                    // Create signature to check for duplicate execution
                    std::string tool_signature = tool_name + "|" + tool_args;

                    // Only execute if this is a new tool call (not the same as last execution)
                    if (tool_signature != g_last_executed_tool_signature) {
                        LOG_DBG("Detected <tool-launch> request: tool=%s, args=%s\n", tool_name.c_str(), tool_args.c_str());

                        // Execute the tool
                        std::string tool_output = execute_tool("tools", tool_name, tool_args);

                        LOG("%s", tool_output.c_str());
                        LOG("[End of Tool Output]\n\n");

                        // Inject the tool output back into the conversation with structured wrapper
                        // This helps the LLM clearly identify tool output vs its own generated text
                        std::string wrapped_output = "\n<tool-response tool=\"" + tool_name + "\">\n" + tool_output;
                        // Ensure output ends with newline before closing tag
                        if (!tool_output.empty() && tool_output.back() != '\n') {
                            wrapped_output += "\n";
                        }
                        wrapped_output += "</tool-response>\n";

                        auto output_tokens = common_tokenize(ctx, wrapped_output, false, true);
                        LOG_DBG("Tool output: %zu tokens (including wrapper)\n", output_tokens.size());

                        // Check if injecting tool output would exceed context window
                        // Account for: current position (n_past) + unconsumed input (embd_inp - n_consumed) + new tokens
                        int tokens_remaining = n_ctx - n_past - ((int)embd_inp.size() - n_consumed) - 4;

                        if ((int)output_tokens.size() > tokens_remaining) {
                            LOG_WRN("Tool output too large (%zu tokens, only %d available). Truncating...\n",
                                    output_tokens.size(), tokens_remaining);
                            if (tokens_remaining > 0) {
                                output_tokens.resize(tokens_remaining);
                                // Add truncation marker
                                std::string truncated_msg = "\n[Output truncated - exceeded context limit]\n</tool-response>\n";
                                auto truncated_tokens = common_tokenize(ctx, truncated_msg, false, true);
                                output_tokens.insert(output_tokens.end(), truncated_tokens.begin(), truncated_tokens.end());
                            } else {
                                LOG_ERR("No context space available for tool output. Skipping injection.\n");
                                output_tokens.clear();
                            }
                        }

                        if (!output_tokens.empty()) {
                            embd_inp.insert(embd_inp.end(), output_tokens.begin(), output_tokens.end());
                        }

                        // Remember this execution to prevent duplicates
                        g_last_executed_tool_signature = tool_signature;

                        // Continue generation after injecting output
                        is_interacting = false;
                    } else {
                        LOG_DBG("Skipping duplicate tool execution: tool=%s, args=%s\n", tool_name.c_str(), tool_args.c_str());
                    }
                }
            }

            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const std::string last_output_for_antiprompt = common_sampler_prev_str(smpl, ctx, 32);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output_for_antiprompt.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output_for_antiprompt.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output_for_antiprompt.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                // avoid calling common_sampler_last() if last_output_for_antiprompt is empty
                if (!last_output_for_antiprompt.empty()) {
                    llama_token last_token = common_sampler_last(smpl);
                    for (auto token : antiprompt_token) {
                        if (token == last_token) {
                            if (params.interactive) {
                                is_interacting = true;
                            }
                            is_antiprompt = true;
                            break;
                        }
                    }
                }

                if (is_antiprompt) {
                    LOG_DBG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of generation tokens in interactive mode
            if (!waiting_for_first_input && llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
                LOG_DBG("found an EOG token\n");

                if (params.interactive) {
                    // Save LLM state if requested
                    if (g_save_state_next && !g_state_save_path.empty()) {
                        if (!save_llm_state_to_gguf(ctx, g_state_save_path)) {
                            LOG_ERR("Failed to save LLM state to %s\n", g_state_save_path.c_str());
                        }
                        g_save_state_next = false;
                        g_state_save_path = "";
                    }

                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    if (params.enable_chat_template) {
                        chat_add_and_format("assistant", assistant_ss.str());
                    }
                    is_interacting = true;
                    LOG("\n");
                }
            }

            if (params.conversation_mode && !waiting_for_first_input) {
                if (!prompt.empty()) {
                    prompt.clear();
                    is_interacting = false;
                }
            }

            if ((n_past > 0 || waiting_for_first_input) && is_interacting) {
                LOG_DBG("waiting for user input\n");

                // Reset idle timer when we start waiting for user input
                // This ensures we only count time spent waiting, not time spent generating
                update_activity_time();

                // Display context usage status line
                // Use actual KV cache position to account for loaded sessions
                const int n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1; // +1 because positions are 0-indexed
                const int n_ctx_effective = (int)(n_ctx * (1.0f - params.ctx_reserve));
                const int usage_percent = n_ctx_used > 0 ? (n_ctx_used * 100) / n_ctx_effective : 0;
                LOG("\n\033[2m[Context: %d/%d tokens used (%d%%), %d total, %.0f%% reserve]\033[0m\n",
                    n_ctx_used, n_ctx_effective, usage_percent, n_ctx, params.ctx_reserve * 100);

                if (params.conversation_mode) {
                    LOG("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG_DBG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_vocab_bos(vocab));
                }

                std::string buffer;
                if (!params.input_prefix.empty() && !params.conversation_mode) {
                    LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    LOG("%s", params.input_prefix.c_str());
                }

                // color user input only
                console::set_display(console::user_input);
                display = params.display_prompt;

                // Calculate remaining timeout for readline
                int timeout_seconds = 0;
                if (params.idle_action_interval > 0) {
                    time_t current_time = time(nullptr);
                    int elapsed_seconds = (int)(current_time - g_last_activity_time);
                    int idle_threshold_seconds = params.idle_action_interval * 60;
                    int remaining_seconds = idle_threshold_seconds - elapsed_seconds;

                    if (remaining_seconds > 0) {
                        timeout_seconds = remaining_seconds;
                    } else {
                        timeout_seconds = 1; // Will timeout immediately
                    }
                }

                // Read input with timeout support
                std::string line;
                bool another_line = true;
                bool timed_out = false;

                do {
                    another_line = console::readline_with_timeout(line, params.multiline_input, timeout_seconds, timed_out);
                    buffer += line;

                    if (timed_out) {
                        // Idle timeout occurred
                        LOG_DBG("Idle timeout triggered during input wait\n");
                        LOG("\n[Idle timeout - continuing thought process]\n");
                        update_activity_time(); // Reset timer for next iteration

                        // Reset tool execution tracking to allow tools during idle thinking
                        g_last_executed_tool_signature = "";

                        // Insert continuation marker so model knows to keep thinking
                        buffer = "<continue>";

                        another_line = false; // Stop reading more lines
                        break;
                    }

                    // User provided input, update activity time and disable timeout for continuation lines
                    update_activity_time();
                    timeout_seconds = 0; // No timeout for continuation lines
                } while (another_line);

                // done taking input, reset color
                console::set_display(console::reset);
                display = true;

                // Debug: log what readline returned
                LOG_DBG("readline completed - buffer: '%s' (length: %zu), timed_out: %d\n",
                        buffer.c_str(), buffer.length(), timed_out);

                if (buffer.empty() && !timed_out) { // Ctrl+D on empty line exits (but not timeout)
                    LOG("EOF by user\n");
                    break;
                }

                // Process newline handling only if buffer is not empty
                if (!buffer.empty() && buffer.back() == '\n') {
                    // Implement #587:
                    // If the user wants the text to end in a newline,
                    // this should be accomplished by explicitly adding a newline by using \ followed by return,
                    // then returning control by pressing return again.
                    buffer.pop_back();
                }

                // Handle special state save/load commands
                if (buffer.rfind("/\\/save ", 0) == 0) {
                    // Extract filename
                    std::string filename = buffer.substr(8); // Skip "/\/save "
                    // Trim whitespace
                    filename.erase(0, filename.find_first_not_of(" \t\n\r\f\v"));
                    filename.erase(filename.find_last_not_of(" \t\n\r\f\v") + 1);

                    if (!filename.empty()) {
                        LOG("\n");
                        LOG("LLM state will be saved to: %s\n", filename.c_str());
                        LOG("State will be saved after your next prompt and response.\n");

                        g_state_save_path = filename;
                        g_save_state_next = true;
                    } else {
                        LOG_ERR("Error: No filename specified for /\\/save command\n");
                    }
                    // Keep is_interacting true and continue to wait for next input
                    is_interacting = true;
                    continue;
                } else if (buffer.rfind("/\\/load ", 0) == 0) {
                    // Extract filename
                    std::string filename = buffer.substr(8); // Skip "/\/load "
                    // Trim whitespace
                    filename.erase(0, filename.find_first_not_of(" \t\n\r\f\v"));
                    filename.erase(filename.find_last_not_of(" \t\n\r\f\v") + 1);

                    if (!filename.empty()) {
                        LOG("\n");
                        if (!load_llm_state_from_gguf(ctx, filename)) {
                            LOG_ERR("Failed to load LLM state from: %s\n", filename.c_str());
                        }
                    } else {
                        LOG_ERR("Error: No filename specified for /\\/load command\n");
                    }
                    // Keep is_interacting true and continue to wait for next input
                    is_interacting = true;
                    continue;
                } else if (buffer.rfind("/\\/temp", 0) == 0) {
                    // Handle temperature get/set command
                    std::string temp_arg = buffer.substr(7); // Skip "/\/temp"
                    // Trim whitespace
                    temp_arg.erase(0, temp_arg.find_first_not_of(" \t\n\r\f\v"));
                    temp_arg.erase(temp_arg.find_last_not_of(" \t\n\r\f\v") + 1);

                    if (temp_arg.empty()) {
                        // Show current temperature
                        LOG("\n");
                        LOG("Current temperature: %.2f\n", common_sampler_get_temp(smpl));
                    } else {
                        // Set new temperature
                        try {
                            float new_temp = std::stof(temp_arg);
                            if (new_temp < 0.0f) {
                                LOG_ERR("Error: Temperature must be >= 0.0\n");
                            } else {
                                LOG("\n");
                                float old_temp = common_sampler_get_temp(smpl);
                                LOG("Changing temperature from %.2f to %.2f\n", old_temp, new_temp);
                                if (common_sampler_set_temp(smpl, new_temp)) {
                                    LOG("Temperature successfully updated to %.2f\n", new_temp);
                                } else {
                                    LOG_ERR("Failed to update temperature\n");
                                }
                            }
                        } catch (const std::exception & e) {
                            LOG_ERR("Error: Invalid temperature value '%s'\n", temp_arg.c_str());
                        }
                    }
                    // Keep is_interacting true and continue to wait for next input
                    is_interacting = true;
                    continue;
                } else if (buffer.rfind("/\\/timeout", 0) == 0) {
                    // Handle idle timeout get/set command
                    std::string timeout_arg = buffer.substr(10); // Skip "/\/timeout"
                    // Trim whitespace
                    timeout_arg.erase(0, timeout_arg.find_first_not_of(" \t\n\r\f\v"));
                    timeout_arg.erase(timeout_arg.find_last_not_of(" \t\n\r\f\v") + 1);

                    if (timeout_arg.empty()) {
                        // Show current timeout or disable it
                        LOG("\n");
                        if (params.idle_action_interval > 0) {
                            LOG("Current idle timeout: %d minutes\n", params.idle_action_interval);
                            LOG("Disabling idle timeout\n");
                            params.idle_action_interval = 0;
                        } else {
                            LOG("Idle timeout is currently disabled (0 minutes)\n");
                        }
                    } else {
                        // Set new timeout
                        try {
                            int new_timeout = std::stoi(timeout_arg);
                            if (new_timeout < 0) {
                                LOG_ERR("Error: Timeout must be >= 0\n");
                            } else {
                                LOG("\n");
                                int old_timeout = params.idle_action_interval;
                                LOG("Changing idle timeout from %d to %d minutes\n", old_timeout, new_timeout);
                                params.idle_action_interval = new_timeout;
                                if (new_timeout == 0) {
                                    LOG("Idle timeout disabled\n");
                                } else {
                                    LOG("Idle timeout set to %d minutes\n", new_timeout);
                                    // Reset timer to start counting from now
                                    update_activity_time();
                                }
                            }
                        } catch (const std::exception & e) {
                            LOG_ERR("Error: Invalid timeout value '%s'\n", timeout_arg.c_str());
                        }
                    }
                    // Keep is_interacting true and continue to wait for next input
                    is_interacting = true;
                    continue;
                } else if (buffer.rfind("/\\/compress", 0) == 0 && buffer.rfind("/\\/compress-kv", 0) != 0) {
                    // Handle manual compression command (but not /\/compress-kv)
                    if (!params.ctx_compress) {
                        LOG_ERR("\nError: Compression is not enabled. Use --ctx-compress flag.\n");
                        is_interacting = true;
                        continue;
                    }

                    std::string compress_arg = buffer.substr(11); // Skip "/\/compress"
                    // Trim whitespace
                    compress_arg.erase(0, compress_arg.find_first_not_of(" \t\n\r\f\v"));
                    compress_arg.erase(compress_arg.find_last_not_of(" \t\n\r\f\v") + 1);

                    // Get current context usage
                    const llama_pos kv_pos_max = llama_memory_seq_pos_max(mem, 0);
                    const int n_ctx_current = kv_pos_max >= 0 ? (int)(kv_pos_max + 1) : n_past;

                    // Determine how many tokens to compress
                    int tokens_to_compress;
                    if (compress_arg.empty()) {
                        // Default: compress oldest 25% of context (excluding n_keep)
                        tokens_to_compress = (n_ctx_current - params.n_keep) / 4;
                    } else {
                        try {
                            tokens_to_compress = std::stoi(compress_arg);
                            if (tokens_to_compress <= 0) {
                                LOG_ERR("Error: Token count must be > 0\n");
                                is_interacting = true;
                                continue;
                            }
                        } catch (const std::exception & e) {
                            LOG_ERR("Error: Invalid token count '%s'\n", compress_arg.c_str());
                            is_interacting = true;
                            continue;
                        }
                    }

                    // Validate there are enough tokens to compress
                    const int available_tokens = n_ctx_current - params.n_keep;
                    if (tokens_to_compress > available_tokens) {
                        LOG_ERR("Error: Cannot compress %d tokens, only %d available (keeping first %d)\n",
                                tokens_to_compress, available_tokens, params.n_keep);
                        is_interacting = true;
                        continue;
                    }

                    LOG("\n");
                    LOG("Manual compression triggered: compressing %d tokens from position %d-%d\n",
                        tokens_to_compress, params.n_keep, params.n_keep + tokens_to_compress);

                    // Perform compression
                    const int compress_start = params.n_keep;
                    const int compress_end = params.n_keep + tokens_to_compress;
                    const int compress_target = (int)(tokens_to_compress * params.ctx_compress_ratio);

                    auto compressed = compress_context(ctx, vocab, smpl, compress_start, compress_end, compress_target);

                    if (!compressed.empty() && (int)compressed.size() < tokens_to_compress) {
                        LOG("Compression successful: %d → %zu tokens\n", tokens_to_compress, compressed.size());

                        // Remove the old range from KV cache
                        llama_memory_seq_rm(mem, 0, params.n_keep, params.n_keep + tokens_to_compress);

                        // Re-decode compressed tokens at correct positions
                        for (size_t i = 0; i < compressed.size(); i += params.n_batch) {
                            int n_eval = std::min((int)(compressed.size() - i), params.n_batch);
                            llama_batch batch = llama_batch_get_one(&compressed[i], n_eval);
                            for (int j = 0; j < n_eval; j++) {
                                batch.pos[j] = params.n_keep + i + j;
                            }
                            if (llama_decode(ctx, batch)) {
                                LOG_ERR("Failed to decode compressed tokens\n");
                                break;
                            }
                        }

                        // Shift remaining context
                        const int shift_amount = tokens_to_compress - (int)compressed.size();
                        llama_memory_seq_add(mem, 0, params.n_keep + tokens_to_compress, n_ctx_current, -shift_amount);
                        n_past -= shift_amount;

                        // Update token storage
                        g_all_tokens.erase(g_all_tokens.begin() + compress_start, g_all_tokens.begin() + compress_end);
                        g_all_tokens.insert(g_all_tokens.begin() + params.n_keep, compressed.begin(), compressed.end());

                        LOG("Context compressed successfully. Saved %d tokens.\n", shift_amount);
                    } else {
                        LOG_ERR("Compression failed or didn't save space\n");
                    }

                    // Keep is_interacting true and continue to wait for next input
                    is_interacting = true;
                    continue;
                } else if (buffer.rfind("/\\/compress-kv", 0) == 0) {
                    // Handle KV-only compression (works without token IDs)
                    std::string compress_arg = buffer.substr(14); // Skip "/\/compress-kv"
                    // Trim whitespace
                    compress_arg.erase(0, compress_arg.find_first_not_of(" \t\n\r\f\v"));
                    compress_arg.erase(compress_arg.find_last_not_of(" \t\n\r\f\v") + 1);

                    // Get current context usage
                    const llama_pos kv_pos_max = llama_memory_seq_pos_max(mem, 0);
                    const int n_ctx_current = kv_pos_max >= 0 ? (int)(kv_pos_max + 1) : n_past;

                    // Determine compression parameters
                    int positions_to_remove;
                    if (compress_arg.empty()) {
                        // Default: remove oldest 50% of positions (excluding n_keep)
                        // This gives 2x compression by keeping every other position
                        positions_to_remove = (n_ctx_current - params.n_keep) / 2;
                    } else {
                        try {
                            positions_to_remove = std::stoi(compress_arg);
                            if (positions_to_remove <= 0) {
                                LOG_ERR("Error: Position count must be > 0\n");
                                is_interacting = true;
                                continue;
                            }
                        } catch (const std::exception & e) {
                            LOG_ERR("Error: Invalid position count '%s'\n", compress_arg.c_str());
                            is_interacting = true;
                            continue;
                        }
                    }

                    // Validate there are enough positions
                    const int available_positions = n_ctx_current - params.n_keep;
                    if (positions_to_remove > available_positions) {
                        LOG_ERR("Error: Cannot remove %d positions, only %d available (keeping first %d)\n",
                                positions_to_remove, available_positions, params.n_keep);
                        is_interacting = true;
                        continue;
                    }

                    LOG("\n");
                    LOG("KV-only compression: removing %d positions from %d-%d\n",
                        positions_to_remove, params.n_keep, params.n_keep + positions_to_remove);
                    LOG("Strategy: Remove oldest contiguous block\n");

                    // Remove the oldest N positions (simple contiguous removal)
                    llama_memory_seq_rm(mem, 0, params.n_keep, params.n_keep + positions_to_remove);

                    // Shift remaining positions down to fill the gap
                    llama_memory_seq_add(mem, 0, params.n_keep + positions_to_remove, n_ctx_current, -positions_to_remove);

                    n_past -= positions_to_remove;
                    const int removed = positions_to_remove;

                    // Update token storage if it exists
                    if (!g_all_tokens.empty() && (int)g_all_tokens.size() >= params.n_keep + removed) {
                        g_all_tokens.erase(
                            g_all_tokens.begin() + params.n_keep,
                            g_all_tokens.begin() + params.n_keep + removed
                        );
                    }

                    LOG("KV compression complete: removed %d positions, saved %d tokens\n", removed, removed);
                    LOG("New context size: %d positions\n", n_ctx_current - removed);

                    // Keep is_interacting true and continue to wait for next input
                    is_interacting = true;
                    continue;
                }

                if (buffer.empty()) { // Enter key on empty line lets the user pass control back
                    LOG_DBG("empty line, passing control back\n");
                } else { // Add tokens to embd only if the input buffer is non-empty
                    // append input suffix if any
                    if (!params.input_suffix.empty() && !params.conversation_mode) {
                        LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        LOG("%s", params.input_suffix.c_str());
                    }

                    LOG_DBG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    bool format_chat = params.conversation_mode && params.enable_chat_template;
                    std::string user_inp = format_chat
                        ? chat_add_and_format("user", std::move(buffer))
                        : std::move(buffer);
                    // TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                    LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                    // if user stop generation mid-way, we must add EOT to finish model's last response
                    if (need_insert_eot && format_chat) {
                        llama_token eot = llama_vocab_eot(vocab);
                        embd_inp.push_back(eot == LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    if (params.verbose_prompt) {
                        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size() - original_size);
                    }

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        const std::string token_str = common_token_to_piece(ctx, token);
                        output_tokens.push_back(token);
                        output_ss << token_str;

                        if (params.verbose_prompt) {
                            LOG_INF("%6d -> '%s'\n", token, token_str.c_str());
                        }
                    }

                    // reset assistant message
                    assistant_ss.str("");

                    // Reset tool execution tracking on new user input
                    g_last_executed_tool_signature = "";

                    n_remain -= line_inp.size();
                    LOG_DBG("n_remain: %d\n", n_remain);
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0 || waiting_for_first_input) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;

                if (waiting_for_first_input && params.single_turn) {
                    params.interactive = false;
                    params.interactive_first = false;
                }
                waiting_for_first_input = false;
            }
        }

        // end of generation
        if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !(params.interactive)) {
            LOG(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    LOG("\n\n");

    // Save LLM state if dumping was enabled via CLI flag
    if (!params.path_dump_activations.empty()) {
        if (!save_llm_state_to_gguf(ctx, params.path_dump_activations)) {
            LOG_ERR("%s: failed to save LLM state\n", __func__);
        }
    }

    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);

    llama_backend_free();

    ggml_threadpool_free_fn(threadpool);
    ggml_threadpool_free_fn(threadpool_batch);

    return 0;
}
