# Edited-version cross-encoder fine-tuning

Continues both saved `work/models/xenc_fold0` and `xenc_fold1` classifiers.
First experiment keeps the original `name_clean | addr_clean | state` input and
128-token budget. No architectural replacement or pseudo-label training.

## What runs

1. Check baseline caches, both classifiers, GPU runtime, and run identity.
2. Prepare at most 300,000 genuine labeled pairs per fold: positive/hard-negative/
   easy-negative budgets 50/35/15. Hard negatives have Stage-1 p>=0.2 or missing
   candidate addresses. Positive/hard strata have loss weight 1.5; easy negatives
   weight 1.0. Preserve original OOF fold membership. Additional training excludes
   all Source-1 validation IDs and all target IDs appearing in scored validation.
3. Continue each classifier for one fixed epoch, learning rate 5e-6, microbatch
   16, accumulation 4, AdamW, 6% warmup, gradient checkpointing, BF16 when supported.
4. Score a fixed validation repair scope using the mean of both models' logits:
   baseline p2 in [0.05,0.95], missing candidate address, or a close ownership
   contest (score gap<=0.15). All other baseline scores remain fixed before
   the common final calibration/selection layer.
5. Split validation into 40% calibration / 60% evaluation by whole country +
   normalized-name groups. Compare baseline-score calibration control versus
   fusion using the additional fine-tuned logit, and versus saved baseline.
   Selection uses existing entity expected-F0.5, including zero-candidate IDs.
6. Only if both comparisons gain >=0.0002, paired group-bootstrap lower bounds
   exceed zero, and neither country regresses, score the analogous test scope,
   generate one TSV, and run the official validator. Otherwise stop successfully
   with `NO_ACCEPTED_IMPROVEMENT` and retain baseline.

The calibration control isolates added score information, but is not an inference
pass from an unfine-tuned checkpoint on the same sampled examples. This run does
not claim to isolate every contribution of loss weighting versus sample mix.

## Server invocation

Unzip into a NEW package directory. Select one available GPU assigned to you;
do not automatically select GPU 4, which previously hosted other active jobs.

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
read -r -p "Assigned free GPU index: " GPU_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
nohup setsid bash /data1/anandkumar/edited_finetune/run.sh \
  /data1/anandkumar/edited_finetune_run_01 \
  </dev/null > /data1/anandkumar/edited_finetune_run_01.log 2>&1 &
tail -f /data1/anandkumar/edited_finetune_run_01.log
```

CPU Python defaults to Edited_version/.venv/bin/python. GPU Python defaults to
amazon-ce-improved/ce-fr-improved/.venv/bin/python, already used for CE-FR.
Override with `CPU_PY` / `GPU_PY` if necessary; no installation is performed.

At least 25 GB free disk is recommended for two retained optimizer checkpoints,
their atomic replacements, fine-tuned weights, and scoring shards. Baseline model
weights and cached data are read-only. A launcher lock prevents duplicate runs.

## Progress and recovery

Log prints fold, optimizer step, pairs/s, estimated remaining minutes for the
CURRENT stage, and checkpoint saves. Estimate excludes following stages.
Training checkpoints are written every 250 optimizer steps (~16,000 pairs),
and at completion. Inference writes atomic 10,000-pair shards and logs every
100 microbatches. Ctrl+C on tail stops the viewer only.

After a stopped/crashed job, rerun the identical launcher command and run directory.
Do not start a second writer. Completed stages/shards are reused; partial shards
are recomputed. No checkpoint is deleted. Code/input/settings changes require a
new run directory. Full optimizer state is loaded only from this run's locally
generated checkpoint; do not replace it with an untrusted file.

## Results

- `comparison.json`: entity F0.5 comparisons, uncertainty, country results, acceptance.
- `model_fold0/COMPLETE.json`, `model_fold1/COMPLETE.json`: completed fixed-epoch training.
- `output/matching_results_finetuned.tsv`: created only after acceptance.
- `TSV_CREATED.json`: file hash; official validation is confirmed by launcher log.

Existing validation and upstream checkpoints were previously examined/fit with
validation feedback. This is exploratory development evidence, not a pristine
independent holdout. Additional fine-tuning itself uses no validation labels;
calibration uses calibration labels. No genuine French labels, no guaranteed
public-score gain, and no trained models are included in this package.

Local tests verify data contracts, OOF isolation, score coverage, entity selection,
input drift rejection, and promotion gate. PyTorch classifier-logit/gradient checks
also run locally. Actual Hugging Face GPU training is not available locally and
must pass the server runtime/model checks before the full run.
