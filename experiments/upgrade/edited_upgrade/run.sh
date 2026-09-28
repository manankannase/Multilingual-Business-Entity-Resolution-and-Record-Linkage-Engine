#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data1/anandkumar/amazon-edited-version/Edited_version
CODE=$ROOT/code/business_entity_resolution
EXPERIMENT=/data1/anandkumar/edited_upgrade
OUT=/data1/anandkumar/edited_upgrade_run_01
export BER_DATA_DIR=/data1/anandkumar/student_resource/dataset
export BER_WORK_DIR="$ROOT/work"
export BER_OUT_DIR="$OUT/output"
export BER_N_TRAIN_S1=300000 BER_N_VAL_S1=80000
export BER_N_JOBS=16 BER_N_PROCS=16
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 RAYON_NUM_THREADS=16
export PYTHONUNBUFFERED=1
PY=$ROOT/.venv/bin/python
test -x "$PY"
test -s "$ROOT/work/val_pred_s2.parquet"
test -s "$ROOT/work/decision.pkl"
"$PY" "$EXPERIMENT/upgrade.py" fit --code "$CODE" --base-work "$ROOT/work" --out "$OUT"
"$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["eligible_for_test_prediction"] else 3)' "$OUT/comparison.json" || {
  echo 'NO_ACCEPTED_IMPROVEMENT: comparison.json contains results; retain verified baseline.'
  exit 0
}
"$PY" "$EXPERIMENT/upgrade.py" predict --code "$CODE" --base-work "$ROOT/work" --out "$OUT"
"$PY" /data1/anandkumar/student_resource/utils/validate_submission.py \
  --matching "$OUT/output/matching_results.tsv" \
  --candidate "$OUT/output/candidate_pairs.tsv" \
  --test-dir /data1/anandkumar/student_resource/dataset/test --check-ids
echo "VALIDATED_UPGRADE_TSV=$OUT/output/matching_results.tsv"
