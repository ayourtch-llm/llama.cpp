#!/usr/bin/env bash
# One-shot rep-guard smoke test: feeds ds4-eval Q1 (the LMC astronaut question)
# to Qwen3.5-27B at temp=0 and runs once without LLAMA_REP_GUARD and once with.
# Greedy decoding maximizes the chance of an exact-repeat loop materializing.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HERE/build/bin/llama-completion"
MODEL="${MODEL:-$HERE/../models/Qwen3.5-27B-UD-IQ2_XXS.gguf}"
N_PREDICT="${N_PREDICT:-4096}"
N_CTX="${N_CTX:-8192}"
SEED="${SEED:-1}"
OUT_DIR="${OUT_DIR:-$HERE/rep-guard-test}"

mkdir -p "$OUT_DIR"

SYS='You are solving a hard benchmark question. Reason carefully. The final answer must follow the requested format exactly.'

read -r -d '' USER <<'EOF' || true
An intelligent civilization in the Large Magellanic Cloud has engineered an extraordinary spacecraft capable of traveling at a substantial fraction of the speed of light. The average lifetime of these aliens is roughly 150 solar years. Now, having Earth as their destination in mind, they are determined to travel with this spacecraft at a constant speed of 0.99999987*c, where c is the speed of light. Approximately, how long will it take for their 22 years old astronaut (from the point of view of the astronaut) to reach the Earth using this incredibly fast spacecraft?

Choices:
A. 72 years
B. 81 years
C. The astronaut will die before reaching to the Earth.
D. 77 years

Solve the question. At the end, write exactly one final line in this format and do not write anything after it:
Answer: <letter>
EOF

run_one() {
    local tag="$1"; shift
    local env_pair="$1"; shift
    local logf="$OUT_DIR/q1.${tag}.log"
    echo "=== run: $tag (env: ${env_pair:-<none>}) ==="
    echo "    log: $logf"
    env $env_pair "$BIN" \
        -m "$MODEL" \
        -ngl 999 \
        -c "$N_CTX" \
        -n "$N_PREDICT" \
        --temp 0 \
        --seed "$SEED" \
        --no-display-prompt \
        --jinja \
        -no-cnv \
        -sys "$SYS" \
        -p "$USER" \
        > "$logf" 2>&1
    local rc=$?
    echo "    exit: $rc, bytes: $(wc -c < "$logf"), lines: $(wc -l < "$logf")"
}

run_one no-guard ""
run_one with-guard "LLAMA_REP_GUARD=1"

echo
echo "=== look for loops (lines repeated >= 5x) ==="
for f in "$OUT_DIR"/q1.*.log; do
    echo "--- $f"
    sort "$f" | uniq -c | sort -rn | awk '$1 >= 5 {print}' | head -10
done
