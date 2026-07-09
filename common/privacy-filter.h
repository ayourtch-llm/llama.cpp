#pragma once

#include "llama.h"

#include <string>
#include <vector>

// shared decoding for openai/privacy-filter, used by the CLI example and the server.
// the model emits BIOES tags over PII types; the tag sequence is decoded with a Viterbi
// pass that only allows transitions producing well-formed spans.

enum pf_tag { PF_TAG_O, PF_TAG_B, PF_TAG_I, PF_TAG_E, PF_TAG_S };

struct pf_label {
    pf_tag kind = PF_TAG_O;
    int    type = -1; // index into pf_tagset::type_names, -1 for O
};

struct pf_span {
    int   tok_beg;
    int   tok_end; // inclusive
    int   type;
    float score;
};

struct pf_tagset {
    std::vector<pf_label>    labels;
    std::vector<std::string> type_names;

    // reads the classifier labels stored in the GGUF, returns false if the model has none
    bool init(const llama_model * model);

    size_t n_labels() const { return labels.size(); }
};

// the model is non-causal, so a pass must fit one ubatch. a token's receptive field is
// n_layer stacked windows, so a chunk carrying that much context on each side produces the
// same logits for its core as a full-document pass would.
struct pf_chunking {
    int halo;
    int core; // <= 0 means the budget is too small
};

pf_chunking pf_plan(const llama_model * model, int n_tok, int budget);

// turns one row of raw classifier logits into log-probabilities, in place
void pf_log_softmax(float * logits, int n_lab);

// one global pass over the whole document, so a span may cross a chunk boundary
std::vector<pf_span> pf_decode(const std::vector<float> & logp, int n_tok, const pf_tagset & ts);

// a token carries its leading space, which is not part of the entity
int pf_span_begin(const pf_span & sp, const std::vector<int> & offsets, const std::string & text);

std::string pf_redact(const std::vector<pf_span> & spans, const std::vector<int> & offsets,
                      const std::string & text, const pf_tagset & ts);
