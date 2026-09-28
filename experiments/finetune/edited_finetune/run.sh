#!/usr/bin/env bash
set -Eeuo pipefail
PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=/data1/anandkumar/amazon-edited-version/Edited_version
OUT=${1:-/data1/anandkumar/edited_finetune_run_01}
CPU_PY=${CPU_PY:-$ROOT/.venv/bin/python}
GPU_PY=${GPU_PY:-/data1/anandkumar/amazon-ce-improved/ce-fr-improved/.venv/bin/python}
export BER_N_JOBS=16 OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 RAYON_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
trap 'echo "FAILED line $LINENO. Preserve run directory; same command resumes completed preparation/training/inference stages." >&2' ERR
test -x "$CPU_PY"
test -x "$GPU_PY"
test -n "${CUDA_VISIBLE_DEVICES:-}"
mkdir -p "$OUT"
exec 9>"$OUT/.launcher.lock"
flock -n 9 || { echo "Run already active: $OUT" >&2; exit 1; }
date -Is
"$CPU_PY" -c 'import polars, numpy, sklearn, lightgbm; print("CPU_RUNTIME_PASS", flush=True)'
"$GPU_PY" "$PACKAGE/train_score.py" check --out "$OUT"
"$CPU_PY" "$PACKAGE/prepare.py" preflight --work "$ROOT/work" --out "$OUT"
"$CPU_PY" "$PACKAGE/prepare.py" prepare --work "$ROOT/work" --out "$OUT"
for FOLD in 0 1; do
  "$GPU_PY" "$PACKAGE/train_score.py" train --out "$OUT" --fold "$FOLD" --batch 16 --accum 4 --lr 0.000005
  "$GPU_PY" "$PACKAGE/train_score.py" score --out "$OUT" --fold "$FOLD" --split val --batch 16
done
"$CPU_PY" "$PACKAGE/evaluate.py" validation --out "$OUT" --code "$ROOT/code/business_entity_resolution"
RESULT=$("$CPU_PY" -c 'import json,sys; print("pass" if json.load(open(sys.argv[1]))["eligible_for_test_prediction"] else "reject")' "$OUT/comparison.json")
if [[ "$RESULT" == reject ]]; then
  echo "COMPLETE: fine-tuning evaluated; no accepted improvement; baseline retained."
  exit 0
fi
"$CPU_PY" "$PACKAGE/prepare.py" test --work "$ROOT/work" --out "$OUT"
for FOLD in 0 1; do
  "$GPU_PY" "$PACKAGE/train_score.py" score --out "$OUT" --fold "$FOLD" --split test --batch 16
done
"$CPU_PY" "$PACKAGE/evaluate.py" test --out "$OUT" --code "$ROOT/code/business_entity_resolution"
"$CPU_PY" /data1/anandkumar/student_resource/utils/validate_submission.py \
  --matching "$OUT/output/matching_results_finetuned.tsv" \
  --candidate "$OUT/output/candidate_pairs.tsv" \
  --test-dir /data1/anandkumar/student_resource/dataset/test --check-ids
sha256sum "$OUT/output/matching_results_finetuned.tsv"
date -Is
echo "COMPLETE: accepted fine-tuning candidate generated and officially validated."
