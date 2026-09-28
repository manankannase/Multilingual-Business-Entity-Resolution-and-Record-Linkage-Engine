#!/bin/bash
# Body of every SLURM job:  run_stage.sh <prep|cand|dense|feat|xenc|llm|stage2|all>
# Each stage is resumable (cached outputs are skipped) and leaves WORK_DIR/.done_<stage> when it succeeds.
set -Eeo pipefail
source "${BER_SLURM_DIR:?submit through submit_all.sh or export BER_SLURM_DIR}/env.sh"
cd "$SRC"
STAGE="$1"
trap 'echo "=== stage $STAGE FAILED (exit $?) at line $LINENO: $BASH_COMMAND" | tee /dev/stderr' ERR
W="$BER_WORK_DIR"

fresh() { [ -f "$1" ] && [ "$1" -nt "$2" ]; }     # output exists and is newer than the model that made it

need_gpu() { [ "$NGPU" -gt 0 ] || { echo "stage $STAGE needs a GPU but none is visible"; exit 1; }; }

features() {                     # exit code 3 = more chunks to do (process recycled to bound memory)
    while true; do
        rc=0; python -u pipeline.py features "$1" || rc=$?
        [ $rc -eq 0 ] && return 0
        [ $rc -eq 3 ] || return $rc
    done
}

run() {
    case "$1" in
    prep)
        python -u translit.py
        python -u prep.py train test ;;
    cand)
        python -u pipeline.py candidates train
        python -u pipeline.py candidates test ;;
    dense)
        need_gpu
        [ -f "$W/models/dense/config.json" ] || $TORCHRUN gpu_dense.py train
        for f in "$W"/dense/*.parquet; do        # search results of an older dense model are stale
            [ -e "$f" ] && ! fresh "$f" "$W/models/dense/config.json" && rm -f "$f" "$W/.merged_dense"
        done
        $TORCHRUN gpu_dense.py search train
        $TORCHRUN gpu_dense.py search test ;;
    feat)
        if ls "$W"/dense/train_*.parquet >/dev/null 2>&1 && [ ! -f "$W/.merged_dense" ]; then
            python -u pipeline.py merge_dense train
            python -u pipeline.py merge_dense test
            touch "$W/.merged_dense"
        fi
        features train
        features test
        python -u pipeline.py fit
        python -u pipeline.py predict
        cp "$BER_OUT_DIR/matching_results.tsv" "$W/matching_results_stage1.tsv"
        python -u error_analysis.py stage1 || true
        python -u stage2.py oof ;;
    xenc)
        need_gpu
        for k in 0 1; do
            [ -f "$W/models/xenc_fold$k/config.json" ] || $TORCHRUN gpu_xenc.py train $k
        done
        for s in train val test; do
            fresh "$W/xenc/xenc_$s.parquet" "$W/models/xenc_fold1/config.json" || $TORCHRUN gpu_xenc.py score $s
        done ;;
    llm)                         # decoder-LLM judge on the uncertain band (same script, different tag/model)
        need_gpu
        export BER_XENC_TAG=llm BER_XENC_MODEL="$BER_LLM_MODEL" BER_XENC_BATCH="$LLM_BATCH" \
               BER_XENC_INF_BATCH="$LLM_INF_BATCH" BER_XENC_LR="$LLM_LR" BER_XENC_EPOCHS=1 \
               BER_XENC_MAXLEN=160 BER_XENC_P_LO="$LLM_P_LO" BER_XENC_P_HI="$LLM_P_HI" BER_GRAD_CKPT="$LLM_CKPT"
        for k in 0 1; do
            [ -f "$W/models/llm_fold$k/config.json" ] || $TORCHRUN gpu_xenc.py train $k
        done
        for s in train val test; do
            fresh "$W/xenc/llm_$s.parquet" "$W/models/llm_fold1/config.json" || $TORCHRUN gpu_xenc.py score $s
        done ;;
    stage2)
        python -u stage2.py fit
        python -u stage2.py predict
        python -u error_analysis.py stage2 || true
        cd "$BER_DATA_DIR/.."
        python utils/validate_submission.py --matching "$BER_OUT_DIR/matching_results.tsv" \
            --candidate "$BER_OUT_DIR/candidate_pairs.tsv" --test-dir "$BER_DATA_DIR/test" ;;
    *)
        echo "unknown stage $1"; exit 2 ;;
    esac
    touch "$W/.done_$1"
    echo "=== stage $1 done $(date)"
}

if [ "$STAGE" = "all" ]; then    # interactive: salloc --gres=gpu:h100:4 ... then run_stage.sh all
    for s in prep cand dense feat xenc llm stage2; do
        [ -f "$W/.done_$s" ] && { echo "skip $s (done)"; continue; }
        run $s
    done
else
    run "$STAGE"
fi
