# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: normalise → blocking (TF-IDF char-3gram top-k, per country/state, forward + reverse + orphan legs)
→ pairwise features (rapidfuzz, token/number agreement) → LightGBM (MIT) → one-to-one assignment + F0.5-tuned threshold.
No external data or lookups; Indic→Latin dictionary is learned from the training pairs only.

## Run (from `src/`)
```
set BER_DATA_DIR=<path>/student_resource/dataset   # train/ and test/ TSVs
set BER_WORK_DIR=<scratch dir>                      # ~20 GB
set BER_OUT_DIR=<output dir>
python translit.py                  # learn Indic token dictionary from train pairs
python prep.py train test           # normalise all records
python pipeline.py candidates train # train/val candidates (whole-state subsets)
python pipeline.py candidates test
python pipeline.py features train
python pipeline.py features test
python pipeline.py fit              # LightGBM + threshold sweep on val (macro F0.5)
python pipeline.py predict          # writes matching_results.tsv and candidate_pairs.tsv
```

## GPU cluster (SLURM, H100) — full pipeline incl. dense blocking leg + cross-encoder re-ranker
```
cd slurm
vi cluster.conf              # usually only ACCOUNT / MODULES; everything else is auto-detected
bash setup_env.sh            # once, on the login node: venv + CUDA torch + model download (HF_HOME)
DRY=1 bash submit_all.sh     # shows detected partition / GPUs / CPUs / RAM and the sbatch commands
bash submit_all.sh           # submits the dependency chain below
```
```
prep ─► cand (TF-IDF blocking, CPU) ─────────────┐
  └───► dense (bi-encoder train + GPU top-k) ────┴─► feat (merge_dense, features, LightGBM, stage-1 OOF)
                                                     ├─► xenc (e5-base cross-encoder, 2 state folds) ─┐
                                                     └─► llm  (Qwen2.5-1.5B judge, uncertain band)  ──┴─► stage2 ─► validate
```
Stage 2 = stage-1 p + graph context (rank / competition between S1s) + group consistency (agreement with the S1's
other top candidates) + GPU model scores and their ranks/margins → 3 seed-bagged LightGBMs → decision layer
(`decision.py`: global threshold vs singleton gate vs exact expected-F0.5 per S1, picked on validation).
- `submit_all.sh` sizes each job from `sinfo`: the H100 partition and GRES type, GPUs per job (all GPUs of a node, max 4),
  and CPUs/RAM in proportion. `env.sh` reads the actual allocation at job start and sets worker processes,
  batch sizes, learning rate, contrastive pair count and the train/val S1 sample sizes (more RAM → more training S1).
- Resumable: every stage leaves `WORK_DIR/.done_<stage>`; re-running `submit_all.sh` skips finished stages.
  `FROM=xenc bash submit_all.sh` re-runs from a stage; `SKIP_DENSE=1` / `SKIP_XENC=1` leave a GPU stage out.
- Interactive alternative: `salloc --gres=gpu:h100:4 ...` then `BER_SLURM_DIR=$PWD bash run_stage.sh all`.
- Logs: `WORK_DIR/logs/ber_<stage>_<jobid>.out`. `error_analysis.py` output there shows where F0.5 is lost.
- GPU models: `intfloat/multilingual-e5-small` (dense) and `intfloat/multilingual-e5-base` (cross-encoder), both MIT.
