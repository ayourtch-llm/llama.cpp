#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64)
#include <nmmintrin.h> // SSE4.2 hardware CRC32C
#endif

using json = nlohmann::ordered_json;

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//

// result_timings
//

json result_timings::to_json() const {
    json base = {
        {"cache_n",                cache_n},

        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    return base;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // nlohmann::json converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             timings.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (timings.prompt_n >= 0) {
        deltas.back().push_back({"timings", timings.to_json()});
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (timings.prompt_n > 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            last_json.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            last_json.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (n_decoded == 1) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }
    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_metrics::to_json() {
    return json {
        { "idle",                            n_idle_slots },
        { "processing",                      n_processing_slots },
        { "deferred",                        n_tasks_deferred },
        { "t_start",                         t_start },

        { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
        { "t_tokens_generation_total",       t_tokens_generation_total },
        { "n_tokens_predicted_total",        n_tokens_predicted_total },
        { "t_prompt_processing_total",       t_prompt_processing_total },

        { "n_tokens_max",                    n_tokens_max },

        { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
        { "t_prompt_processing",             t_prompt_processing },
        { "n_tokens_predicted",              n_tokens_predicted },
        { "t_tokens_generation",             t_tokens_generation },

        { "n_decode_total",                  n_decode_total },
        { "n_busy_slots_total",              n_busy_slots_total },

        { "slots",                           slots_data },
    };
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.n_tokens();
    }

    return res;
}

server_prompt * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // next, remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.tokens      =*/ prompt.tokens.clone(),
        /*.data        =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
        /*.checkpoints =*/ prompt.checkpoints,
    });

    return &states.back();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float sim_best    = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->tokens.size();
        const float sim_cur    = float(lcp_cur) / tokens_new.size();

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && sim_best < sim_cur) {
            f_keep_best = f_keep_cur;
            sim_best    = sim_cur;

            it_best = it;
        }
    }

    // consult the level-2 (disk) tier: it wins only if it strictly beats the best RAM/base candidate
    // found above (same f_keep >= 0.25 threshold as the RAM tier).
    if (disk) {
        int   lcp_disk    = 0;
        float f_keep_disk = 0.0f;
        float sim_disk    = 0.0f;

        const std::string path_disk = disk->find_best(tokens_new, 0.25f, lcp_disk, f_keep_disk, sim_disk);

        if (!path_disk.empty() && f_keep_best < f_keep_disk && sim_best < sim_disk) {
            SRV_INF(" - disk tier: found better prompt (lcp = %d, f_keep = %.3f, sim = %.3f), restoring from disk\n",
                    lcp_disk, f_keep_disk, sim_disk);

            const int64_t t_start = ggml_time_us();

            server_prompt dp;
            // load_blob validates + always consumes (removes index entry and unlinks) the disk copy
            bool ok = disk->load_blob(path_disk, dp);

            if (ok) {
                const size_t size = dp.data.main.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_tgt, dp.data.main.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_ERR("failed to restore disk state with size %zu (geometry mismatch) - discarded\n", size);
                    ok = false;
                }
            }

            if (ok && !dp.data.drft.empty()) {
                GGML_ASSERT(ctx_dft);
                const size_t size = dp.data.drft.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, dp.data.drft.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore disk draft state with size %zu - discarded\n", size);
                    ok = false;
                }
            }

            if (ok) {
                dp.data.main.clear(); dp.data.main.shrink_to_fit();
                dp.data.drft.clear(); dp.data.drft.shrink_to_fit();

                prompt = std::move(dp);

                SRV_INF(" - disk tier: restore of %d tokens took %.2f ms\n",
                        prompt.n_tokens(), (ggml_time_us() - t_start) / 1000.0);

                return true;
            }

            SRV_WRN("%s", " - disk tier: restore failed, falling back to RAM tier\n");
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(*it_best);

        states.erase(it_best);
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        // always keep at least one state, regardless of the limits
        while (states.size() > 1 && size() > limit_size) {
            if (states.empty()) {
                break;
            }

            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            if (disk) {
                disk->offload(std::move(states.front()));
            }
            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (states.size() > 1 && n_tokens() > limit_tokens_cur) {
            if (states.empty()) {
                break;
            }

            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            if (disk) {
                disk->offload(std::move(states.front()));
            }
            states.pop_front();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.n_tokens(), state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

//
// server_prompt_disk_cache (level-2 / disk tier)
//

namespace {

constexpr uint32_t KVBLOB_MAGIC   = 0x564C4B42u; // "BKLV"
constexpr uint32_t KVBLOB_VERSION = 2u;           // v2: CRC32C (was bit-serial CRC32); v1 blobs are ignored
constexpr uint32_t KVMETA_MAGIC   = 0x4D4C4B42u; // "BKLM"
constexpr uint32_t KVMETA_VERSION = 2u;

const char * const KVBLOB_EXT = ".kvblob";
const char * const KVMETA_EXT = ".kvmeta";

// CRC32C (Castagnoli, reflected poly 0x82F63B78). These blobs are ephemeral local
// cache, so the polynomial need not interop with anything external - only that write
// and read use the same algorithm. Runs at ~memory bandwidth: hardware SSE4.2 when
// available, slice-by-8 software table otherwise.
struct crc32c_table {
    uint32_t t[8][256];
    crc32c_table() {
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = n;
            for (int k = 0; k < 8; ++k) {
                c = (c >> 1) ^ (0x82F63B78u & (~(c & 1) + 1));
            }
            t[0][n] = c;
        }
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = t[0][n];
            for (int s = 1; s < 8; ++s) {
                c = t[0][c & 0xFF] ^ (c >> 8);
                t[s][n] = c;
            }
        }
    }
};

uint32_t crc32c_sw(uint32_t crc, const uint8_t * p, size_t n) {
    static const crc32c_table tab;
    while (n >= 8) {
        uint32_t lo, hi;
        memcpy(&lo, p,     4);
        memcpy(&hi, p + 4, 4);
        lo ^= crc;
        crc = tab.t[7][lo & 0xFF] ^ tab.t[6][(lo >> 8) & 0xFF] ^ tab.t[5][(lo >> 16) & 0xFF] ^ tab.t[4][lo >> 24]
            ^ tab.t[3][hi & 0xFF] ^ tab.t[2][(hi >> 8) & 0xFF] ^ tab.t[1][(hi >> 16) & 0xFF] ^ tab.t[0][hi >> 24];
        p += 8; n -= 8;
    }
    while (n--) {
        crc = tab.t[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);
    }
    return crc;
}

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("sse4.2")))
#endif
uint32_t crc32c_hw(uint32_t crc, const uint8_t * p, size_t n) {
    uint64_t c = crc;
    while (n >= 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        c = _mm_crc32_u64(c, v);
        p += 8; n -= 8;
    }
    uint32_t c32 = (uint32_t) c;
    while (n--) {
        c32 = _mm_crc32_u8(c32, *p++);
    }
    return c32;
}
#endif

// incremental CRC32C: invert on entry/exit so chained calls match one continuous pass
uint32_t crc32_update(uint32_t crc, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    crc = ~crc;
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("sse4.2")) {
        return ~crc32c_hw(crc, p, n);
    }
#endif
#endif
    return ~crc32c_sw(crc, p, n);
}

// FNV-1a 64-bit hash of the token vector -> stable content-addressed file name
std::string tokens_hash_hex(const llama_tokens & toks) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * p = (const uint8_t *) toks.data();
    const size_t    n = toks.size() * sizeof(llama_token);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    // also fold in the length so different-length prompts that alias never collide
    h ^= (uint64_t) toks.size();
    h *= 1099511628211ull;

    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return std::string(buf);
}

// buffered writer that tracks a running CRC32 over everything written
struct crc_ofstream {
    std::ofstream os;
    uint32_t      crc = 0;
    bool          ok  = true;

    explicit crc_ofstream(const std::string & path) : os(path, std::ios::binary | std::ios::trunc) {
        ok = (bool) os;
    }

    void write(const void * data, size_t n) {
        if (!ok) return;
        os.write((const char *) data, n);
        if (!os) { ok = false; return; }
        crc = crc32_update(crc, data, n);
    }

    template <typename T> void put(const T & v) { write(&v, sizeof(T)); }

    void put_vec(const std::vector<uint8_t> & v) {
        uint64_t n = v.size();
        put(n);
        if (n) write(v.data(), n);
    }
};

// buffered reader that tracks a running CRC32 over everything read
struct crc_ifstream {
    std::ifstream is;
    uint32_t      crc = 0;
    bool          ok  = true;

    explicit crc_ifstream(const std::string & path) : is(path, std::ios::binary) {
        ok = (bool) is;
    }

    void read(void * data, size_t n) {
        if (!ok) return;
        is.read((char *) data, n);
        if ((size_t) is.gcount() != n) { ok = false; return; }
        crc = crc32_update(crc, data, n);
    }

    template <typename T> T get() { T v{}; read(&v, sizeof(T)); return v; }

    bool get_vec(std::vector<uint8_t> & v, uint64_t max_bytes) {
        uint64_t n = get<uint64_t>();
        if (!ok || n > max_bytes) { ok = false; return false; }
        try { v.resize(n); } catch (...) { ok = false; return false; }
        if (n) read(v.data(), n);
        return ok;
    }
};

int64_t ftime_to_i64(std::filesystem::file_time_type t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

// last path component, for compact logging
std::string dir_basename(const std::string & path) {
    const auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

} // namespace

server_prompt_disk_cache::server_prompt_disk_cache(const std::string & dir, int32_t limit_size_mib) {
    this->dir        = dir;
    this->limit_size = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
    this->stop       = false;

    writer = std::thread([this]() { writer_loop(); });
}

server_prompt_disk_cache::~server_prompt_disk_cache() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        stop = true;
    }
    cv.notify_all();
    if (writer.joinable()) {
        writer.join();
    }
}

size_t server_prompt_disk_cache::index_size() const {
    std::lock_guard<std::mutex> lk(mtx);
    return index.size();
}

size_t server_prompt_disk_cache::index_bytes() const {
    std::lock_guard<std::mutex> lk(mtx);
    size_t res = 0;
    for (const auto & e : index) res += e.size;
    return res;
}

void server_prompt_disk_cache::init() {
    namespace fs = std::filesystem;

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        SRV_ERR("disk tier: failed to create/open cache dir '%s': %s\n", dir.c_str(), ec.message().c_str());
        return;
    }

    std::lock_guard<std::mutex> lk(mtx);
    index.clear();

    size_t n_orphan = 0;
    for (const auto & de : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        const auto p = de.path();
        if (p.extension() != KVMETA_EXT) {
            continue;
        }

        const std::string stem      = (dir + "/" + p.stem().string());
        const std::string blob_path = stem + KVBLOB_EXT;

        // read sidecar meta (small, fast) to recover the token vector + expected blob size
        crc_ifstream in(p.string());
        const uint32_t magic = in.get<uint32_t>();
        const uint32_t ver   = in.get<uint32_t>();
        if (!in.ok || magic != KVMETA_MAGIC || ver != KVMETA_VERSION) {
            continue;
        }
        const uint64_t blob_size = in.get<uint64_t>();
        const uint64_t n_tokens  = in.get<uint64_t>();
        if (!in.ok || n_tokens > (1ull << 32)) {
            continue;
        }
        entry e;
        e.path = blob_path;
        try { e.tokens.resize(n_tokens); } catch (...) { continue; }
        if (n_tokens) {
            in.read(e.tokens.data(), n_tokens * sizeof(llama_token));
        }
        if (!in.ok) {
            continue;
        }

        // validate the blob exists and matches the recorded size (cheap corruption/partial-write check)
        std::error_code ec2;
        const auto st = fs::status(blob_path, ec2);
        if (ec2 || !fs::exists(st)) {
            // orphaned meta with no blob -> clean up
            fs::remove(p, ec2);
            n_orphan++;
            continue;
        }
        const auto fsz = fs::file_size(blob_path, ec2);
        if (ec2 || fsz != blob_size) {
            fs::remove(p, ec2);
            fs::remove(blob_path, ec2);
            n_orphan++;
            continue;
        }

        e.size      = blob_size;
        auto wt     = fs::last_write_time(blob_path, ec2);
        e.last_used = ec2 ? 0 : ftime_to_i64(wt);

        index.push_back(std::move(e));
    }

    size_t total = 0;
    for (const auto & e : index) total += e.size;

    char limbuf[32];
    if (limit_size == 0) {
        snprintf(limbuf, sizeof(limbuf), "unlimited");
    } else {
        snprintf(limbuf, sizeof(limbuf), "%.3f MiB", limit_size / (1024.0 * 1024.0));
    }
    SRV_INF("disk tier: initialized at '%s' - %zu entries, %.3f MiB, limit %s%s%s\n",
            dir.c_str(), index.size(), total / (1024.0 * 1024.0), limbuf,
            n_orphan ? ", cleaned orphans: " : "", n_orphan ? std::to_string(n_orphan).c_str() : "");

    evict_to_limit_locked();
}

void server_prompt_disk_cache::offload(server_prompt && p) {
    if (p.tokens.empty() || p.tokens.has_mtmd) {
        // media prompts cannot be safely serialized (image chunk data is not part of the KV blob)
        return;
    }

    {
        std::lock_guard<std::mutex> lk(mtx);
        if (stop) return;
        wq.push_back(std::move(p));
    }
    cv.notify_one();
}

// single writer thread: serializes offloads. Under heavy churn many multi-second
// writes can back up here; with the fast CRC32C above each write is ~10x shorter so
// the backlog is far smaller, but a pool could be added if it ever becomes a bottleneck.
void server_prompt_disk_cache::writer_loop() {
    for (;;) {
        server_prompt job;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [this]() { return stop || !wq.empty(); });
            if (stop && wq.empty()) {
                return;
            }
            job = std::move(wq.front());
            wq.pop_front();
        }
        write_one(job);
    }
}

void server_prompt_disk_cache::write_one(const server_prompt & p) {
    namespace fs = std::filesystem;

    const llama_tokens toks = p.tokens.get_tokens();
    if (toks.empty()) {
        return;
    }

    const std::string hex       = tokens_hash_hex(toks);
    const std::string stem      = dir + "/" + hex;
    const std::string blob_path = stem + KVBLOB_EXT;
    const std::string meta_path = stem + KVMETA_EXT;

    {
        std::lock_guard<std::mutex> lk(mtx);
        for (const auto & e : index) {
            if (e.path == blob_path) {
                // identical content already on disk -> nothing to do
                return;
            }
        }
    }

    const int64_t t_start = ggml_time_us();

    const std::string blob_tmp = blob_path + ".tmp";
    const std::string meta_tmp = meta_path + ".tmp";

    // --- write the blob (streamed, so we never hold a second copy of the multi-GB state) ---
    {
        crc_ofstream out(blob_tmp);
        out.put(KVBLOB_MAGIC);
        out.put(KVBLOB_VERSION);

        const uint64_t n_tokens = toks.size();
        out.put(n_tokens);
        out.write(toks.data(), n_tokens * sizeof(llama_token));

        out.put_vec(p.data.main);
        out.put_vec(p.data.drft);

        const uint64_t n_ckpt = p.checkpoints.size();
        out.put(n_ckpt);
        for (const auto & c : p.checkpoints) {
            const int64_t   ntok = c.n_tokens;
            const llama_pos pmin = c.pos_min;
            const llama_pos pmax = c.pos_max;
            out.put(ntok);
            out.put(pmin);
            out.put(pmax);
            out.put_vec(c.data_tgt);
            out.put_vec(c.data_dft);
            out.put_vec(c.data_spec);
        }

        const uint32_t crc = out.crc;
        out.put(crc);
        out.os.flush();

        if (!out.ok || !out.os) {
            SRV_ERR("disk tier: failed to write blob '%s' - discarding\n", blob_tmp.c_str());
            std::error_code ec; fs::remove(blob_tmp, ec);
            return;
        }
    }

    std::error_code ec;
    const auto blob_size = fs::file_size(blob_tmp, ec);
    if (ec) {
        fs::remove(blob_tmp, ec);
        return;
    }

    // --- write the sidecar meta (small: magic + blob size + tokens) ---
    {
        crc_ofstream out(meta_tmp);
        out.put(KVMETA_MAGIC);
        out.put(KVMETA_VERSION);
        const uint64_t bs = blob_size;
        const uint64_t nt = toks.size();
        out.put(bs);
        out.put(nt);
        out.write(toks.data(), nt * sizeof(llama_token));
        out.os.flush();
        if (!out.ok || !out.os) {
            SRV_ERR("disk tier: failed to write meta '%s' - discarding\n", meta_tmp.c_str());
            fs::remove(meta_tmp, ec);
            fs::remove(blob_tmp, ec);
            return;
        }
    }

    // atomically publish (blob first, then meta - init() treats meta-without-blob as an orphan)
    fs::rename(blob_tmp, blob_path, ec);
    if (ec) { fs::remove(blob_tmp, ec); fs::remove(meta_tmp, ec); return; }
    fs::rename(meta_tmp, meta_path, ec);
    if (ec) { fs::remove(blob_path, ec); fs::remove(meta_tmp, ec); return; }

    {
        std::lock_guard<std::mutex> lk(mtx);
        entry e;
        e.path      = blob_path;
        e.tokens    = toks;
        e.size      = blob_size;
        e.last_used = ftime_to_i64(std::filesystem::file_time_type::clock::now());
        index.push_back(std::move(e));

        size_t total = 0;
        for (const auto & x : index) total += x.size;

        SRV_INF("disk tier: wrote %llu tokens -> %s (%.3f MiB) in %.2f ms [total %zu entries, %.3f MiB]\n",
                (unsigned long long) toks.size(), (dir_basename(blob_path)).c_str(),
                blob_size / (1024.0 * 1024.0), (ggml_time_us() - t_start) / 1000.0,
                index.size(), total / (1024.0 * 1024.0));

        evict_to_limit_locked();
    }
}

std::string server_prompt_disk_cache::find_best(const server_tokens & tokens_new, float f_keep_min,
        int & lcp_out, float & f_keep_out, float & sim_out) const {
    std::lock_guard<std::mutex> lk(mtx);

    const llama_tokens & tn = tokens_new.get_tokens();
    if (tn.empty()) {
        return "";
    }

    std::string best_path;
    float f_keep_best = -1.0f;
    float sim_best    = -1.0f;
    int   lcp_best    = 0;

    for (const auto & e : index) {
        if (e.tokens.empty()) continue;

        size_t lcp = 0;
        const size_t n = std::min(tn.size(), e.tokens.size());
        while (lcp < n && tn[lcp] == e.tokens[lcp]) ++lcp;

        const float f_keep_cur = float(lcp) / e.tokens.size();
        const float sim_cur    = float(lcp) / tn.size();

        if (f_keep_cur < f_keep_min) {
            continue;
        }

        // same "must improve both" rule the RAM tier uses
        if (f_keep_best < f_keep_cur && sim_best < sim_cur) {
            f_keep_best = f_keep_cur;
            sim_best    = sim_cur;
            lcp_best    = (int) lcp;
            best_path   = e.path;
        }
    }

    lcp_out    = lcp_best;
    f_keep_out = f_keep_best;
    sim_out    = sim_best;

    return best_path;
}

bool server_prompt_disk_cache::load_blob(const std::string & path, server_prompt & p) {
    namespace fs = std::filesystem;

    bool ok = true;
    llama_tokens toks;

    {
        crc_ifstream in(path);
        if (!in.ok) { ok = false; }

        const uint32_t magic = ok ? in.get<uint32_t>() : 0;
        const uint32_t ver   = ok ? in.get<uint32_t>() : 0;
        if (!in.ok || magic != KVBLOB_MAGIC || ver != KVBLOB_VERSION) {
            ok = false;
        }

        if (ok) {
            const uint64_t n_tokens = in.get<uint64_t>();
            if (in.ok && n_tokens <= (1ull << 32)) {
                try { toks.resize(n_tokens); } catch (...) { ok = false; }
                if (ok && n_tokens) in.read(toks.data(), n_tokens * sizeof(llama_token));
            } else {
                ok = false;
            }
        }

        // cap individual buffers at 64 GiB to guard against corrupt length fields
        const uint64_t max_buf = 64ull*1024*1024*1024;

        if (ok) ok = in.get_vec(p.data.main, max_buf);
        if (ok) ok = in.get_vec(p.data.drft, max_buf);

        if (ok) {
            const uint64_t n_ckpt = in.get<uint64_t>();
            if (!in.ok || n_ckpt > (1ull << 20)) {
                ok = false;
            } else {
                for (uint64_t i = 0; ok && i < n_ckpt; ++i) {
                    common_prompt_checkpoint c;
                    c.n_tokens = in.get<int64_t>();
                    c.pos_min  = in.get<llama_pos>();
                    c.pos_max  = in.get<llama_pos>();
                    if (!in.ok) { ok = false; break; }
                    ok = in.get_vec(c.data_tgt, max_buf)
                      && in.get_vec(c.data_dft, max_buf)
                      && in.get_vec(c.data_spec, max_buf);
                    if (ok) p.checkpoints.push_back(std::move(c));
                }
            }
        }

        if (ok) {
            const uint32_t crc_calc   = in.crc; // running crc over all payload read so far
            const uint32_t crc_stored = in.get<uint32_t>();
            if (!in.ok || crc_calc != crc_stored) {
                SRV_ERR("disk tier: CRC mismatch on '%s' - corrupt, discarding\n", path.c_str());
                ok = false;
            }
        }
    }

    // a disk hit always consumes the entry: it becomes resident on success, and a bad file must be purged
    {
        std::lock_guard<std::mutex> lk(mtx);
        remove_entry_locked(path);
    }

    if (!ok) {
        p.data.main.clear();
        p.data.drft.clear();
        p.checkpoints.clear();
        return false;
    }

    p.tokens = server_tokens(toks, false);
    return true;
}

void server_prompt_disk_cache::remove_entry_locked(const std::string & path) {
    namespace fs = std::filesystem;
    std::error_code ec;

    // copy first: `path` may alias the string stored inside the entry we are about to erase
    const std::string blob = path;

    for (auto it = index.begin(); it != index.end(); ++it) {
        if (it->path == blob) {
            index.erase(it);
            break;
        }
    }

    // path is "<stem>.kvblob"; derive the sidecar
    std::string meta = blob;
    const std::string ext = KVBLOB_EXT;
    if (meta.size() >= ext.size() && meta.compare(meta.size() - ext.size(), ext.size(), ext) == 0) {
        meta = meta.substr(0, meta.size() - ext.size()) + KVMETA_EXT;
    }
    fs::remove(blob, ec);
    fs::remove(meta, ec);
}

void server_prompt_disk_cache::evict_to_limit_locked() {
    if (limit_size == 0) {
        return;
    }

    size_t total = 0;
    for (const auto & e : index) total += e.size;

    while (total > limit_size && index.size() > 0) {
        // find least-recently-used entry
        auto lru = index.begin();
        for (auto it = index.begin(); it != index.end(); ++it) {
            if (it->last_used < lru->last_used) {
                lru = it;
            }
        }

        SRV_INF("disk tier: size limit reached (%.3f/%.3f MiB), evicting LRU entry %s (%.3f MiB)\n",
                total / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0),
                dir_basename(lru->path).c_str(), lru->size / (1024.0 * 1024.0));

        total -= lru->size;
        remove_entry_locked(lru->path);
    }
}
