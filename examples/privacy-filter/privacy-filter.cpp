#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// openai/privacy-filter emits BIOES tags over 8 PII types. The tag sequence is decoded
// with a Viterbi pass that only allows transitions producing well-formed spans.

enum tag_kind { TAG_O, TAG_B, TAG_I, TAG_E, TAG_S };

struct label_info {
    tag_kind    kind = TAG_O;
    int         type = -1; // index into type_names, -1 for O
    std::string name;
};

struct span {
    int   tok_beg;
    int   tok_end; // inclusive
    int   type;
    float score;
};

static bool can_follow(const label_info & prev, const label_info & cur) {
    const bool prev_closed = prev.kind == TAG_O || prev.kind == TAG_E || prev.kind == TAG_S;

    if (prev_closed) {
        return cur.kind == TAG_O || cur.kind == TAG_B || cur.kind == TAG_S;
    }

    // inside a span: only continue or close the same type
    return (cur.kind == TAG_I || cur.kind == TAG_E) && cur.type == prev.type;
}

static void log_softmax_inplace(float * v, int n) {
    float max = v[0];
    for (int i = 1; i < n; ++i) {
        max = std::max(max, v[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += expf(v[i] - max);
    }

    const float log_z = max + logf(sum);
    for (int i = 0; i < n; ++i) {
        v[i] -= log_z;
    }
}

static std::vector<int> viterbi(const std::vector<float> & logp, int n_tok, const std::vector<label_info> & labels) {
    const int n_lab = (int) labels.size();
    const float neg_inf = -1e30f;

    std::vector<float> dp(n_tok*n_lab, neg_inf);
    std::vector<int>   bp(n_tok*n_lab, 0);

    for (int l = 0; l < n_lab; ++l) {
        if (labels[l].kind == TAG_O || labels[l].kind == TAG_B || labels[l].kind == TAG_S) {
            dp[l] = logp[l];
        }
    }

    for (int t = 1; t < n_tok; ++t) {
        for (int cur = 0; cur < n_lab; ++cur) {
            float best = neg_inf;
            int   arg  = 0;

            for (int prev = 0; prev < n_lab; ++prev) {
                if (dp[(t-1)*n_lab + prev] <= neg_inf || !can_follow(labels[prev], labels[cur])) {
                    continue;
                }

                const float s = dp[(t-1)*n_lab + prev];
                if (s > best) {
                    best = s;
                    arg  = prev;
                }
            }

            if (best > neg_inf) {
                dp[t*n_lab + cur] = best + logp[t*n_lab + cur];
                bp[t*n_lab + cur] = arg;
            }
        }
    }

    // a span cannot be left open at the end
    float best = neg_inf;
    int   last = 0;
    for (int l = 0; l < n_lab; ++l) {
        if (labels[l].kind == TAG_B || labels[l].kind == TAG_I) {
            continue;
        }
        if (dp[(n_tok-1)*n_lab + l] > best) {
            best = dp[(n_tok-1)*n_lab + l];
            last = l;
        }
    }

    std::vector<int> path(n_tok);
    path[n_tok-1] = last;
    for (int t = n_tok-1; t > 0; --t) {
        path[t-1] = bp[t*n_lab + path[t]];
    }

    return path;
}

int main(int argc, char ** argv) {
    common_params params;

    params.embedding    = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.n_ubatch     = params.n_batch;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    const int n_lab = (int) llama_model_n_cls_out(model);
    if (n_lab <= 1) {
        LOG_ERR("%s: model has no classifier labels, is this a privacy-filter model?\n", __func__);
        return 1;
    }

    std::vector<label_info>  labels(n_lab);
    std::vector<std::string> type_names;

    for (int i = 0; i < n_lab; ++i) {
        const char * raw = llama_model_cls_label(model, i);
        label_info & li = labels[i];
        li.name = raw ? raw : "";

        if (li.name.size() < 2 || li.name[1] != '-') {
            li.kind = TAG_O;
            continue;
        }

        switch (li.name[0]) {
            case 'B': li.kind = TAG_B; break;
            case 'I': li.kind = TAG_I; break;
            case 'E': li.kind = TAG_E; break;
            case 'S': li.kind = TAG_S; break;
            default:  li.kind = TAG_O; break;
        }

        const std::string type = li.name.substr(2);
        int idx = -1;
        for (int t = 0; t < (int) type_names.size(); ++t) {
            if (type_names[t] == type) {
                idx = t;
                break;
            }
        }
        if (idx < 0) {
            idx = (int) type_names.size();
            type_names.push_back(type);
        }
        li.type = idx;
    }

    // the HF tokenizer adds no special tokens for this model
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, false, false);
    const int n_tok = (int) tokens.size();

    if (n_tok == 0) {
        LOG_ERR("%s: empty prompt\n", __func__);
        return 1;
    }

    if (n_tok > (int) params.n_ubatch) {
        LOG_ERR("%s: prompt has %d tokens, exceeds n_ubatch %d (non-causal models need the whole "
                "sequence in one ubatch, pass -ub %d)\n", __func__, n_tok, params.n_ubatch, n_tok);
        return 1;
    }

    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        common_batch_add(batch, tokens[i], i, { 0 }, true);
    }

    if (llama_decode(ctx, batch) < 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    std::vector<float> logp(n_tok*n_lab);
    for (int i = 0; i < n_tok; ++i) {
        const float * src = llama_get_embeddings_ith(ctx, i);
        if (src == nullptr) {
            LOG_ERR("%s: no embeddings for token %d\n", __func__, i);
            return 1;
        }
        std::copy(src, src + n_lab, logp.begin() + i*n_lab);
        log_softmax_inplace(logp.data() + i*n_lab, n_lab);
    }

    const std::vector<int> path = viterbi(logp, n_tok, labels);

    // byte-level BPE: concatenating the pieces reproduces the input exactly
    std::vector<std::string> pieces(n_tok);
    std::vector<int>         offsets(n_tok + 1, 0);
    for (int i = 0; i < n_tok; ++i) {
        pieces[i] = common_token_to_piece(ctx, tokens[i]);
        offsets[i+1] = offsets[i] + (int) pieces[i].size();
    }

    std::string text;
    for (const auto & p : pieces) {
        text += p;
    }

    // a token carries its leading space, which is not part of the entity
    const auto span_beg = [&](const span & sp) {
        int beg = offsets[sp.tok_beg];
        while (beg < offsets[sp.tok_end + 1] && text[beg] == ' ') {
            beg++;
        }
        return beg;
    };

    std::vector<span> spans;
    for (int t = 0; t < n_tok; ++t) {
        const label_info & li = labels[path[t]];
        if (li.kind == TAG_B || li.kind == TAG_S) {
            span sp = { t, t, li.type, 0.0f };
            while (labels[path[sp.tok_end]].kind != TAG_E && labels[path[sp.tok_end]].kind != TAG_S) {
                sp.tok_end++;
            }
            float sum = 0.0f;
            for (int i = sp.tok_beg; i <= sp.tok_end; ++i) {
                sum += logp[i*n_lab + path[i]];
            }
            sp.score = expf(sum / (sp.tok_end - sp.tok_beg + 1));
            spans.push_back(sp);
            t = sp.tok_end;
        }
    }

    printf("%s\n", text.c_str());

    if (spans.empty()) {
        printf("  (no PII detected)\n");
    }

    for (const auto & sp : spans) {
        const int beg = span_beg(sp);
        const int end = offsets[sp.tok_end + 1];
        printf("  %-16s %.4f  %s\n", type_names[sp.type].c_str(), sp.score,
            text.substr(beg, end - beg).c_str());
    }

    std::string redacted;
    int pos = 0;
    for (const auto & sp : spans) {
        const int beg = span_beg(sp);
        const int end = offsets[sp.tok_end + 1];
        redacted += text.substr(pos, beg - pos);
        redacted += "[" + type_names[sp.type] + "]";
        pos = end;
    }
    redacted += text.substr(pos);

    printf("\n  -> %s\n", redacted.c_str());

    llama_batch_free(batch);
    llama_backend_free();

    return 0;
}
