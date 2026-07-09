#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "privacy-filter.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;

    params.embedding    = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;

    // room for a core plus a halo on each side; attention is dense, so this bounds kq at
    // n_head * n_ubatch^2 floats
    params.n_ctx    = 3072;
    params.n_batch  = 3072;
    params.n_ubatch = 3072;

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

    pf_tagset ts;
    if (!ts.init(model)) {
        LOG_ERR("%s: model has no classifier labels, is this a privacy-filter model?\n", __func__);
        return 1;
    }

    const int n_lab = (int) ts.n_labels();

    // the HF tokenizer adds no special tokens for this model
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, false, false);
    const int n_tok = (int) tokens.size();

    if (n_tok == 0) {
        LOG_ERR("%s: empty prompt\n", __func__);
        return 1;
    }

    const int n_ubatch = (int) params.n_ubatch;

    const pf_chunking plan = pf_plan(model, n_tok, n_ubatch);
    if (plan.core <= 0) {
        LOG_ERR("%s: n_ubatch %d is too small, needs to exceed 2*%d for a document of %d tokens "
                "(pass -ub %d)\n", __func__, n_ubatch, plan.halo, n_tok, 2*plan.halo + 512);
        return 1;
    }

    if (plan.halo > 0) {
        LOG_INF("%s: %d tokens, decoding in cores of %d with a %d token halo\n",
                __func__, n_tok, plan.core, plan.halo);
    }

    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);

    std::vector<float> logp(n_tok*n_lab);

    for (int beg = 0; beg < n_tok; beg += plan.core) {
        const int end = std::min(beg + plan.core, n_tok);
        const int lo  = std::max(0,     beg - plan.halo);
        const int hi  = std::min(n_tok, end + plan.halo);

        common_batch_clear(batch);
        for (int i = lo; i < hi; ++i) {
            // rope is relative, so the chunk restarts its positions at 0
            // only the core needs an output row, the halo is context
            common_batch_add(batch, tokens[i], i - lo, { 0 }, i >= beg && i < end);
        }

        if (llama_decode(ctx, batch) < 0) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            llama_batch_free(batch);
            return 1;
        }

        for (int i = beg; i < end; ++i) {
            const float * src = llama_get_embeddings_ith(ctx, i - lo);
            if (src == nullptr) {
                LOG_ERR("%s: no embeddings for token %d\n", __func__, i);
                llama_batch_free(batch);
                return 1;
            }
            std::copy(src, src + n_lab, logp.begin() + i*n_lab);
            pf_log_softmax(logp.data() + i*n_lab, n_lab);
        }
    }

    llama_batch_free(batch);

    const std::vector<pf_span> spans = pf_decode(logp, n_tok, ts);

    // byte-level BPE: concatenating the pieces reproduces the input exactly
    std::vector<int> offsets(n_tok + 1, 0);
    std::string text;
    for (int i = 0; i < n_tok; ++i) {
        text += common_token_to_piece(ctx, tokens[i]);
        offsets[i+1] = (int) text.size();
    }

    printf("%s\n", text.c_str());

    if (spans.empty()) {
        printf("  (no PII detected)\n");
    }

    for (const auto & sp : spans) {
        const int beg = pf_span_begin(sp, offsets, text);
        const int end = offsets[sp.tok_end + 1];
        printf("  %-16s %.4f  %s\n", ts.type_names[sp.type].c_str(), sp.score,
            text.substr(beg, end - beg).c_str());
    }

    printf("\n  -> %s\n", pf_redact(spans, offsets, text, ts).c_str());

    llama_backend_free();

    return 0;
}
