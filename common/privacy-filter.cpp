#include "privacy-filter.h"

#include <algorithm>
#include <cmath>

bool pf_tagset::init(const llama_model * model) {
    const int n_lab = (int) llama_model_n_cls_out(model);
    if (n_lab <= 1) {
        return false;
    }

    labels.assign(n_lab, {});

    for (int i = 0; i < n_lab; ++i) {
        const char * raw = llama_model_cls_label(model, i);
        const std::string name = raw ? raw : "";

        pf_label & li = labels[i];

        if (name.size() < 2 || name[1] != '-') {
            li.kind = PF_TAG_O;
            continue;
        }

        switch (name[0]) {
            case 'B': li.kind = PF_TAG_B; break;
            case 'I': li.kind = PF_TAG_I; break;
            case 'E': li.kind = PF_TAG_E; break;
            case 'S': li.kind = PF_TAG_S; break;
            default:  li.kind = PF_TAG_O; continue;
        }

        const std::string type = name.substr(2);

        auto it = std::find(type_names.begin(), type_names.end(), type);
        if (it == type_names.end()) {
            li.type = (int) type_names.size();
            type_names.push_back(type);
        } else {
            li.type = (int) std::distance(type_names.begin(), it);
        }
    }

    return true;
}

pf_chunking pf_plan(const llama_model * model, int n_tok, int budget) {
    if (n_tok <= budget) {
        return { 0, n_tok };
    }

    const int halo = llama_model_n_layer(model)*(llama_model_n_swa(model)/2);

    return { halo, budget - 2*halo };
}

void pf_log_softmax(float * logits, int n_lab) {
    float max = logits[0];
    for (int i = 1; i < n_lab; ++i) {
        max = std::max(max, logits[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < n_lab; ++i) {
        sum += expf(logits[i] - max);
    }

    const float log_z = max + logf(sum);
    for (int i = 0; i < n_lab; ++i) {
        logits[i] -= log_z;
    }
}

static bool pf_can_follow(const pf_label & prev, const pf_label & cur) {
    const bool prev_closed = prev.kind == PF_TAG_O || prev.kind == PF_TAG_E || prev.kind == PF_TAG_S;

    if (prev_closed) {
        return cur.kind == PF_TAG_O || cur.kind == PF_TAG_B || cur.kind == PF_TAG_S;
    }

    // inside a span: only continue or close the same type
    return (cur.kind == PF_TAG_I || cur.kind == PF_TAG_E) && cur.type == prev.type;
}

std::vector<pf_span> pf_decode(const std::vector<float> & logp, int n_tok, const pf_tagset & ts) {
    const int   n_lab   = (int) ts.labels.size();
    const float neg_inf = -1e30f;

    const auto & labels = ts.labels;

    std::vector<float> dp(n_tok*n_lab, neg_inf);
    std::vector<int>   bp(n_tok*n_lab, 0);

    for (int l = 0; l < n_lab; ++l) {
        if (labels[l].kind != PF_TAG_I && labels[l].kind != PF_TAG_E) {
            dp[l] = logp[l];
        }
    }

    for (int t = 1; t < n_tok; ++t) {
        for (int cur = 0; cur < n_lab; ++cur) {
            float best = neg_inf;
            int   arg  = 0;

            for (int prev = 0; prev < n_lab; ++prev) {
                if (dp[(t-1)*n_lab + prev] <= neg_inf || !pf_can_follow(labels[prev], labels[cur])) {
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
        if (labels[l].kind == PF_TAG_B || labels[l].kind == PF_TAG_I) {
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

    std::vector<pf_span> spans;
    for (int t = 0; t < n_tok; ++t) {
        const pf_label & li = labels[path[t]];
        if (li.kind != PF_TAG_B && li.kind != PF_TAG_S) {
            continue;
        }

        pf_span sp = { t, t, li.type, 0.0f };
        while (labels[path[sp.tok_end]].kind != PF_TAG_E && labels[path[sp.tok_end]].kind != PF_TAG_S) {
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

    return spans;
}

int pf_span_begin(const pf_span & sp, const std::vector<int> & offsets, const std::string & text) {
    int beg = offsets[sp.tok_beg];
    while (beg < offsets[sp.tok_end + 1] && text[beg] == ' ') {
        beg++;
    }
    return beg;
}

std::string pf_redact(const std::vector<pf_span> & spans, const std::vector<int> & offsets,
                      const std::string & text, const pf_tagset & ts) {
    std::string out;
    int pos = 0;

    for (const auto & sp : spans) {
        const int beg = pf_span_begin(sp, offsets, text);
        const int end = offsets[sp.tok_end + 1];

        out += text.substr(pos, beg - pos);
        out += "[" + ts.type_names[sp.type] + "]";
        pos = end;
    }

    out += text.substr(pos);

    return out;
}
