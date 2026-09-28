# CE-FR conservative revision

Based on the user-supplied modify_ml.zip. Original team authorship is retained;
the source archive contains no license file. Same supplied pairs and XLM-RoBERTa-large
backbone, same raw `name | address` serialization, 128 tokens, 1 epoch, learning
rate 1e-5, effective batch 64, AdamW weight decay .01 and 6% warm-up.

## What changed

- Deterministic shuffled training instead of relying on file order.
- France pseudo-label loss has weight .35 instead of 1. This is a conservative
  experimental default, not an optimized value. `--pseudo_weight 1` restores equal
  weighting; `--pseudo_weight 0` removes its loss contribution but still processes
  those rows. No new pseudo-labels or external business data are generated.
- True-label-only dev/audit, grouped by exact Source1 text. Remove training pairs
  sharing exact held-out Source1 or target text. Duplicate text pairs are removed;
  conflicting text labels are excluded. Entity IDs are absent, so these safeguards
  do NOT guarantee complete entity separation.
- Microbatch 16, accumulation 4, evaluation batch 16. BF16 on supported GPUs;
  FP16 with gradient scaling otherwise. Gradient checkpointing reduces memory.
- Atomic checkpoints retain optimizer/scheduler/scaler/RNG and exact shuffled
  progress. Resume requires unchanged input, code and arguments. A directory lock
  prevents duplicate writers. The final checkpoint is retained, never deleted.

The original file has 3,360,000 pairs: 2,400,000 true-labeled and 960,000 pseudo.
Its last 20,000 rows include 5,825 pseudo labels; 19,002 of those 20,000 rows share
Source1 text with the training prefix. It also contains 4,756 repeated exact pairs.
Those findings weaken the original pair validation, regardless of its reported
accuracy. The data audit found zero conflicting exact text pairs.

## Install on Linux

Use a NEW environment, separate from hybrid-v2. Python 3.11 works with these pins.
Run each step only after the previous one succeeds.

```bash
cd ~/ce-fr-improved
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
nvidia-smi
```

Select an available GPU assigned to you. The following uses physical GPU 4 as an
EXAMPLE; recheck availability rather than assuming it is still free.

```bash
export CUDA_VISIBLE_DEVICES=4
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

## Train and survive SSH disconnects

First test on a small sample. This downloads the backbone (~2.2 GB); it is not in
this ZIP. Large-model smoke runtime depends on GPU and network, not a fixed two minutes.

```bash
bash run.sh --out smoke_ce --limit 512
```

After it succeeds, start ONE full run:

```bash
nohup bash run.sh --out ce_xlmr_improved > train.log 2>&1 < /dev/null &
tail -f train.log
```

Ctrl+C while following `tail` stops the viewer only. Training continues. After
reconnecting, activate `.venv`, enter the project directory, and read `train.log`.
Resume an interrupted run using the IDENTICAL command, after confirming the old
process has stopped. `COMPLETE.json` confirms completed training and evaluation.
No H100/A40/V100 completion-time promise is made. Keep at least 25 GB free for
model download, weights, optimizer checkpoint and its temporary atomic replacement.

## Integration and meaningful comparison

This is a replacement cross-encoder training component, not a complete competition
pipeline. The supplied ZIP lacks candidate generation, entity IDs, original teacher,
threshold calibration, and submission assembly. Reuse the team's existing candidate
generator and scorer integration. Exported model/tokenizer load through the same
Hugging Face interfaces and use unchanged a,b serialization.

Alternatively score a parquet containing candidate a,b and optional entity IDs:

```bash
python score_pairs.py --model ce_xlmr_improved --pairs test_candidates.parquet --out scored_candidates.parquet
```

Do not use pair accuracy or a default .5 threshold to select the submission. Compare
the original and revised models on the SAME genuine labeled entity holdout and full
candidate set. Choose thresholds on separate calibration entities, then evaluate
macro F0.5 including singletons. Only then choose which model's predictions to submit.
The scores in COMPLETE.json are pair sanity checks, not that evaluation. France has
no genuine labels here; agreement with its pseudo-labels cannot establish accuracy.

## Research and limitations

[Ditto](https://arxiv.org/abs/2004.00584) supports sequence-pair fine-tuning for entity
matching. We retain that general cross-encoder approach; this is not a reproduction
of Ditto's augmentation or serialization. XLM-R-large's
[model card](https://huggingface.co/FacebookAI/xlm-roberta-large) lists MIT licensing.
Paper benchmark scores do not predict this competition's score. Whether test-data
pseudo-label training is permitted must follow the competition's transductive-use
rules; this archive inherits that data strategy from the supplied approach.

See verification.json for executed tests. Full GPU training and leaderboard gains
are unverified. No trained checkpoint or submission TSV is included.
