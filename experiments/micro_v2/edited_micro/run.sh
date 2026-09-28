#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data1/anandkumar/amazon-edited-version/Edited_version
PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:-/data1/anandkumar/edited_micro_run_02}
PY="$ROOT/.venv/bin/python"
export BER_DATA_DIR=/data1/anandkumar/student_resource/dataset
export BER_N_TRAIN_S1=300000 BER_N_VAL_S1=80000 BER_P_MIN=0.01
export BER_N_JOBS=16 BER_N_PROCS=16 OMP_NUM_THREADS=16
export POLARS_MAX_THREADS=16 RAYON_NUM_THREADS=16 PYTHONUNBUFFERED=1
trap 'echo "FAILED: line $LINENO; check log. Preserve partial output and use a new run directory." >&2' ERR
test -x "$PY"
test -s "$ROOT/work/val_pred_s2.parquet"
date -Is
"$PY" "$PACKAGE/experiment.py" preflight --code "$ROOT/code/business_entity_resolution" \
  --base-work "$ROOT/work" --out "$OUT"
"$PY" "$PACKAGE/experiment.py" fit --code "$ROOT/code/business_entity_resolution" \
  --base-work "$ROOT/work" --out "$OUT" --seeds 3
RESULT=$("$PY" -c 'import json,sys; print("pass" if json.load(open(sys.argv[1]))["any_candidate_passes_comparison"] else "reject")' "$OUT/comparison.json")
if [[ "$RESULT" == reject ]]; then
  echo "NO_ACCEPTED_IMPROVEMENT: generating the five requested candidates for review; comparison recommends baseline."
fi
"$PY" "$PACKAGE/experiment.py" predict --code "$ROOT/code/business_entity_resolution" \
  --base-work "$ROOT/work" --out "$OUT"
for NAME in 01_expected_f 02_threshold 03_missing_calibration 04_model_ensemble 05_missing_only_update; do
  TSV="$OUT/output/matching_results_$NAME.tsv"
  "$PY" /data1/anandkumar/student_resource/utils/validate_submission.py \
    --matching "$TSV" --candidate "$OUT/output/candidate_pairs.tsv" \
    --test-dir /data1/anandkumar/student_resource/dataset/test --check-ids
  sha256sum "$TSV"
  echo "VALIDATED_MICRO_TSV=$TSV"
done
date -Is
echo "COMPLETE: five candidates validated; inspect $OUT/comparison.json before choosing."
