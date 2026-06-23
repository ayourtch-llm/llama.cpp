#include "models.h"

#include <cstdlib>

#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"

void llama_model_deepseek32::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,     hparams.n_ff_exp);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
    hparams.f_norm_eps = 1e-6;  // eps for layer norm
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    // MoE parameters
    ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

    // deepseek MLA parameters
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);

    // DSA parameters
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    // Expert gating function
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, 0.0f)) {
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
        // cancel the factor from the convert script
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    // NextN/MTP parameters
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer");

    switch (hparams.n_layer()) {
        case 62: type = LLM_TYPE_685B_A37B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deepseek32::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const bool is_mla = hparams.is_mla();
    if (!is_mla) {
        throw std::runtime_error("DEEPSEEK32 architecture requires MLA");
    }

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // try to load output.weight, if not found, use token_embd (tied embeddings)
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer_all; ++i) {
        int flags = 0;
        if (i >= n_layer) {
            // skip all tensors in the NextN layers
            // TODO @ngxson : TENSOR_NOT_REQUIRED was a hack, need to remove it later
            flags |= TENSOR_SKIP | TENSOR_NOT_REQUIRED;
        }

        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, flags);

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);

        // note: only old legacy GGUF files will have the unsplit wkv_b tensor in
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, flags);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, flags);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        // DSA indexer
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {hparams.indexer_head_size}, flags);
        layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {hparams.indexer_head_size}, flags);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, hparams.indexer_head_size}, flags);
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size}, flags);
        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        // NextN/MTP tensors (preserved but unused) - conditionally load for last nextn_predict_layers
        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);

            // Optional tensors
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek32::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_deepseek32::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const bool is_mla = hparams.is_mla();
    GGML_ASSERT(is_mla);

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();
    GGML_UNUSED(n_embd_head_v);

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const uint32_t n_indexer_top_k = hparams.indexer_top_k;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    // And also: https://github.com/ggml-org/llama.cpp/pull/17945 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]

    // first cancel the adjustment from llama_hparams::yarn_attn_factor_adjust to get the original attn_factor
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    // use the original attn_factor to pre-scale the kq_scale
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_dsa * inp_attn_dsa = build_attn_inp_k_dsa();

    // Length-gated hybrid attention (TASK 05). DSA sparse attention has a high flat cost
    // (indexer scan + sparse_mla_attn) that loses to dense MLA below a measured crossover
    // (~110k tokens on this HW) and wins above it. So: run dense MLA when the current KV
    // length is at/below a threshold and the DSA sparse path above it, chosen per forward.
    //
    // n_kv here is the padded KV length the graph is built around (== MLA attention mask
    // ne[0]); the graph is rebuilt whenever it changes, so the gate is always consistent
    // with the live graph (it can never go stale on a reused graph). The indexer-K
    // projection is always cached (see the full-layer block below) so the path can flip
    // mid-sequence without invalidating prior indexer scores.
    //
    // Override the threshold with GLM_DSA_DENSE_BELOW (token count). 0 => always sparse
    // (== previous behavior); a very large value => always dense.
    const uint32_t n_kv_dsa = (uint32_t) inp_attn_dsa->get_kq_mask_mla()->ne[0];
    const char * env_dense_below = std::getenv("GLM_DSA_DENSE_BELOW");
    const uint32_t dense_below_dsa = env_dense_below && env_dense_below[0]
        ? (uint32_t) std::atol(env_dense_below) : 110000;
    const bool use_dense_dsa = n_kv_dsa <= dense_below_dsa;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // GLM-5.2 cross-layer indexer top-k sharing (HF "MAIN DIFF with DSV3.2"): only the "full"
    // indexer layers compute a top-k selection; the "shared" layers in between reuse the most
    // recent full layer's top-k. DeepSeek-V3.2 runs the indexer on every layer (no sharing).
    ggml_tensor * last_top_k = nullptr;
    // DEBUG: DSA_IDX_NOSHARE=1 forces every layer "full" (per-layer recompute, no sharing).
    const bool idx_noshare = std::getenv("DSA_IDX_NOSHARE");

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * qr = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
            cb(qr, "qr", il);

            qr = build_norm(qr, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(qr, "qr", il);

            ggml_tensor * top_k = nullptr;

            // "full" indexer layers compute a fresh top-k; "shared" layers reuse the previous
            // full layer's selection. Pattern is data-driven from hparams (GLM-5.2: freq=4,
            // offset=3 => layers 0,1,2 then every 4th; DeepSeek-V3.2: all full).
            const bool idx_is_full = idx_noshare || hparams.indexer_layer_is_full(il);

            // lightning indexer
            if (idx_is_full) {
                // Indexer RoPE type is arch-dependent: GLM-5.2 uses interleaved rope
                // (config.json indexer_rope_interleave=true) = ggml NORM; DeepSeek-V3.2 uses NEOX.
                // DSA_IDX_ROPE_NORM=1 forces NORM (override, e.g. for A/B testing on DeepSeek).
                const int idx_rope_type = (model.arch == LLM_ARCH_GLM_DSA || std::getenv("DSA_IDX_ROPE_NORM"))
                                              ? LLAMA_ROPE_TYPE_NORM
                                              : LLAMA_ROPE_TYPE_NEOX;
                // DEBUG: DSA_IDX_NOYARN=1 disables YaRN extension on the indexer rope
                // (plain rope: freq_scale=1, ext_factor=0, attn_factor=1).
                const bool  idx_noyarn   = std::getenv("DSA_IDX_NOYARN");
                const float idx_freq_scale  = idx_noyarn ? 1.0f : freq_scale;
                const float idx_ext_factor  = idx_noyarn ? 0.0f : ext_factor;
                const float idx_attn_factor = idx_noyarn ? 1.0f : attn_factor;

                // Indexer-K projection + cache: ALWAYS computed, on both the dense and sparse
                // branches. The indexer-K is a cheap per-token projection, but it must exist for
                // ALL prior positions the first time a generation crosses the dense/sparse
                // threshold, otherwise the indexer scores (and thus the top-k selection) would
                // be computed against a partial history. Only the downstream scoring + top_k
                // are gated to the sparse branch below.
                const auto * mctx_lid   = inp_attn_dsa->mctx->get_lid();
                const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();

                ggml_tensor * indexer_k = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_k, cur);
                cb(indexer_k, "indexer_k", il);

                indexer_k = build_norm(indexer_k, model.layers[il].indexer_k_norm, model.layers[il].indexer_k_norm_b, LLM_NORM, il);
                cb(indexer_k, "indexer_k", il);

                // split into {n_embd_indexer_head_rope, 1, n_tokens}
                ggml_tensor * indexer_k_pe =
                    ggml_view_3d(ctx0, indexer_k, n_embd_indexer_head_rope, 1, n_tokens,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head) * 1, 0);
                cb(indexer_k_pe, "indexer_k_pe", il);

                // and {n_embd_indexer_head_nope, 1, n_tokens}
                ggml_tensor * indexer_k_nope =
                    ggml_view_3d(ctx0, indexer_k, n_embd_indexer_head_nope, 1, n_tokens,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head) * 1,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head_nope));
                cb(indexer_k_nope, "indexer_k_nope", il);

                indexer_k_pe = ggml_rope_ext(ctx0, indexer_k_pe, inp_pos, nullptr, n_rot,
                                     idx_rope_type, n_ctx_orig, freq_base, idx_freq_scale,
                                     idx_ext_factor, idx_attn_factor, beta_fast, beta_slow);
                cb(indexer_k_pe, "indexer_k_pe", il);

                // {n_embd_indexer_head_rope + n_embd_indexer_head_nope, 1, n_tokens}
                indexer_k = ggml_concat(ctx0, indexer_k_pe, indexer_k_nope, 0);
                cb(indexer_k, "indexer_k", il);

                // perform Hadamard transform on indexer k
                indexer_k = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_k);
                cb(indexer_k, "indexer_k", il);

                // store indexer keys to KV cache
                ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, indexer_k, k_idxs_lid, il));

                if (!use_dense_dsa) {
                    ggml_tensor * indexer_q = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, qr);
                    cb(indexer_q, "indexer_q", il);

                    // split into {n_embd_indexer_head_rope, n_indexer_head, n_tokens}
                    ggml_tensor * indexer_q_pe =
                        ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, n_tokens,
                                     ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                     ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head, 0);
                    cb(indexer_q_pe, "indexer_q_pe", il);

                    // and {n_embd_indexer_head_nope, n_indexer_head, n_tokens}
                    ggml_tensor * indexer_q_nope =
                        ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, n_tokens,
                                     ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                     ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head,
                                     ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));
                    cb(indexer_q_nope, "indexer_q_nope", il);

                    indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, n_rot,
                                         idx_rope_type, n_ctx_orig, freq_base, idx_freq_scale,
                                         idx_ext_factor, idx_attn_factor, beta_fast, beta_slow);
                    cb(indexer_q_pe, "indexer_q_pe", il);

                    // {n_embd_indexer_head_rope + n_embd_indexer_head_nope, n_head, n_tokens}
                    indexer_q = ggml_concat(ctx0, indexer_q_pe, indexer_q_nope, 0);
                    cb(indexer_q, "indexer_q", il);

                    // perform Hadamard transform on indexer q
                    indexer_q = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_q);
                    cb(indexer_q, "indexer_q", il);

                    // prepare indexer weights
                    ggml_tensor * indexer_weights = ggml_mul_mat(ctx0, model.layers[il].indexer_proj, cur);
                    cb(indexer_weights, "indexer_weights", il);

                    // get cached indexer keys
                    indexer_k = mctx_lid->get_k(ctx0, il);

                    // split the batch into streams if needed
                    const auto n_stream = indexer_k->ne[3];
                    indexer_q = ggml_view_4d(ctx0, indexer_q, indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream, indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
                    indexer_weights = ggml_view_4d(ctx0, indexer_weights, indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream, indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

                    // fused indexer score: sum_h relu(q.k) * w, reduced over heads without
                    // materializing the [n_kv, n_tokens, n_head] product (see ggml_indexer_score)
                    indexer_q = ggml_cont(ctx0, ggml_permute(ctx0, indexer_q, 0, 2, 1, 3));
                    cb(indexer_q, "indexer_q", il);
                    // q8_0 indexer KV cache is read directly (dequant-on-read in ggml_indexer_score);
                    // the per-token ggml_cast -> f32 + ggml_cont was the decode cpy_q_f32 bottleneck.
                    // Only the free permute (rewrites strides) remains - no copy node.
                    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
                    cb(indexer_k, "indexer_k", il);

                    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f / sqrtf(float(n_embd_indexer_head * n_indexer_head)));
                    indexer_weights = ggml_cont(ctx0, indexer_weights);
                    cb(indexer_weights, "indexer_weights", il);

                    ggml_tensor * indexer_score = ggml_indexer_score(ctx0, indexer_k, indexer_q, indexer_weights);
                    cb(indexer_score, "indexer_score", il);

                    // mask indexer scores
                    ggml_tensor * indexer_kq_mask = inp_attn_dsa->get_kq_mask_lid();
                    indexer_score = ggml_add(ctx0, indexer_score, indexer_kq_mask);
                    cb(indexer_score, "indexer_score", il);

                    // get indices of top k indexer scores
                    uint32_t n_top_k = indexer_score->ne[0] < n_indexer_top_k ? indexer_score->ne[0] : n_indexer_top_k;
                    top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
                    cb(top_k, "top_k", il);
                    last_top_k = top_k;
                }
            } else {
                // shared layer: reuse the previous full layer's top-k selection (sparse branch only)
                if (!use_dense_dsa) {
                    top_k = last_top_k;
                }
            }

            ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_b, qr);
            cb(q, "q", il);

            // split into {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            // and {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // and {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            // MLA attention
            {
                // {n_embd_head_qk_nope, n_tokens, n_head}
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                // {kv_lora_rank, n_head, n_tokens}
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
                // note: rope must go first for in-place context shifting in build_rope_shift()
                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                cb(Kcur, "Kcur", il);

                // {kv_lora_rank, 1, n_tokens}
                ggml_tensor * Vcur = kv_cmpr;
                cb(Vcur, "Vcur", il);

                // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
                cur = build_attn(inp_attn_dsa,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, top_k, kq_scale, il);
            }
        }
        // When MTP/NextN is active (embeddings_nextn), keep the FULL per-token hidden
        // state at the last layer so res->t_h_nextn below covers every token; the MTP
        // draft head reads it per-position. The inp_out_ids reduction is deferred until
        // after t_h_nextn is exported. Mirrors deepseek2.cpp.
        if (il == n_layer - 1 && inp_out_ids && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked)) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                model.layers[il].ffn_gate_up_exps,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                        model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                        model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    // Export the final hidden state for the MTP/NextN draft head (deepseek2::graph_mtp
    // consumes res->t_h_nextn). Without this the MTP draft on the DSA build runs from an
    // unset hidden state -> drafts are near-random and accept length collapses. This is
    // the FULL per-token hidden state when MTP is active (see the deferred reduction at
    // the last layer above). Mirrors deepseek2.cpp main graph; harmless when MTP is off.
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    // Deferred inp_out_ids reduction (skipped at the last layer when MTP is active).
    if (cparams.embeddings_nextn && !cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
