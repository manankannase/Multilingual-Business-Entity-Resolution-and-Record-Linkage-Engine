# Runbook

## Requirements

### Minimum software

- Linux; Bash for launch scripts.
- Python 3.10 or newer.
- CPU dependencies in `requirements/requirements-cpu.txt`.
- GPU dependencies in `requirements/requirements-gpu.txt`.
- CUDA-compatible PyTorch installed from the wheel index matching the server CUDA runtime.
- Dataset layout containing `train/`, `test/`, and the organizer's `utils/validate_submission.py`.

### Practical hardware

- CPU-only pipeline: 16+ cores, 64 GB RAM, about 25 GB scratch space.
- Full-data sparse feature generation benefits from 32–64 cores and 128 GB RAM.
- Dense/XENC stages: at least one 24 GB NVIDIA GPU; the recorded run used NVIDIA A40-class hardware. More GPUs reduce wall time through `torchrun`.
- Keep at least 100 GB free for full candidate, feature, model, score, and log caches. Fine-tuning alone recommends 25 GB free.

These are operational recommendations, not hard-coded limits. `config.py` exposes sample sizes and worker counts through environment variables.

## Install CPU environment

```bash
cd /path/Edited_Version_Portfolio/src/business_entity_resolution
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Install GPU environment

Example for CUDA 12.4:

```bash
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-gpu.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

`slurm/setup_env.sh` automates environment creation and model download on a cluster. Review `slurm/cluster.conf` before running it.

## CPU path

```bash
export BER_DATA_DIR=/data/student_resource/dataset
export BER_WORK_DIR=/data/ber-work
export BER_OUT_DIR=/data/ber-output
export BER_N_JOBS=32
export BER_N_PROCS=6

cd /path/Edited_Version_Portfolio/src/business_entity_resolution/src
python -u translit.py
python -u prep.py train test
python -u pipeline.py candidates train
python -u pipeline.py candidates test

while ! python -u pipeline.py features train; do
  test $? -eq 3 || exit $?
done
while ! python -u pipeline.py features test; do
  test $? -eq 3 || exit $?
done

python -u pipeline.py fit
python -u pipeline.py predict
```

Feature generation returns status 3 when another chunk remains. The supplied shell runners handle this automatically.

## Full SLURM path

```bash
cd /path/Edited_Version_Portfolio/src/business_entity_resolution/slurm
editor cluster.conf
bash setup_env.sh
DRY=1 bash submit_all.sh
bash submit_all.sh
```

Dependency chain:

```text
prep -> cand ------------------+
   +-> dense -----------------> feat -> xenc ---+
                                  +-> llm  -----> stage2 -> validator
```

Useful controls:

```bash
FROM=xenc bash submit_all.sh       # recompute from XENC onward
SKIP_DENSE=1 bash submit_all.sh    # omit dense retrieval
SKIP_LLM=1 bash submit_all.sh      # omit optional LLM judge
bash status.sh                     # inspect jobs and completion markers
```

Do not delete work caches before confirming the final TSV and checksum. Each successful stage writes `.done_<stage>`.

## Fine-tuning experiment

The included package continues both saved `xenc_fold0` and `xenc_fold1` checkpoints on genuine train pairs while excluding all validation Source-1 and target IDs. It hard-mines Stage-1 negatives and overweights positive/hard examples.

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
export CUDA_VISIBLE_DEVICES=7   # replace with GPU assigned to you

nohup setsid bash experiments/finetune/edited_finetune/run.sh \
  /data/edited_finetune_run_01 \
  </dev/null > /data/edited_finetune_run_01.log 2>&1 &

tail -f /data/edited_finetune_run_01.log
```

The run deliberately creates a test TSV only when all promotion conditions pass. The recorded experiment did not pass, so absence of a fine-tuned TSV is correct.

## Micro-candidate experiment

```bash
nohup setsid bash experiments/micro_v2/edited_micro/run.sh \
  /data/edited_micro_run_02 \
  </dev/null > /data/edited_micro_run_02.log 2>&1 &

tail -f /data/edited_micro_run_02.log
```

This experiment generates five exploratory TSV variants even when no candidate passes promotion. `comparison.json` and `validation_ranking.json` retain the decision evidence.

## Validate artifact integrity

From package root:

```bash
shasum -a 256 -c artifacts/FINAL_TSV.sha256
wc -l artifacts/final_xenc_stage2_20260927-122208.tsv
head -n 2 artifacts/final_xenc_stage2_20260927-122208.tsv
```

Expected line count: `1732545`. Run the organizer validator against your licensed dataset:

```bash
python /path/student_resource/utils/validate_submission.py \
  --matching artifacts/final_xenc_stage2_20260927-122208.tsv \
  --candidate /path/candidate_pairs.tsv \
  --test-dir /path/student_resource/dataset/test \
  --check-ids
```

