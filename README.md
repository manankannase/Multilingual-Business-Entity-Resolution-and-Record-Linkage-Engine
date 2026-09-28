# Multilingual Business Entity Resolution — Edited Version

Production and research package for the Amazon ML Challenge 2026 entity-resolution system. The pipeline links noisy Source-1 business records to Source-2/Source-3 records across India, the United States, and France, then emits one row per Source-1 entity.

## Result

- **Public leaderboard macro F0.5: 0.98076** (displayed as **0.981** on the submission list).
- **Validated final file:** `artifacts/final_xenc_stage2_20260927-122208.tsv`.
- **Rows:** 1,732,544 data rows plus one header.
- **SHA-256:** `86e708e2cee3c34ca1841423a4a55af3a772a9ed4c310f01ce3d0f46570d98da`.
- **Official validator:** PASS; no blocking issues.

The final score improved the observed submission path from roughly **0.92** to **0.98076**. Later micro-feature and cross-encoder fine-tuning experiments were evaluated, but did not satisfy the promotion gate. They are included as reproducible research, not represented as leaderboard improvements.

## Résumé summary

> Built a multilingual entity-resolution pipeline for 1.73M query businesses against 9.97M candidate entities, improving public macro F0.5 from 0.929 to 0.98076 through hybrid sparse/dense retrieval, 55+ lexical and address features, LightGBM ranking, multilingual cross-encoder reranking, graph/context features, calibrated expected-F0.5 decoding, and one-to-one ownership constraints.

## System design

```text
raw TSV records
    -> learned transliteration + normalization
    -> sparse TF-IDF blocking (name/address, forward/reverse/orphan)
       + optional multilingual dense retrieval
    -> pair features + Stage-1 LightGBM
    -> two-fold multilingual E5 cross-encoder
       + optional uncertain-band language-model judge
    -> graph, rank, margin, competition, and group-consistency context
    -> three-seed Stage-2 LightGBM ensemble
    -> probability calibration + expected-F0.5 decision
    -> one-to-one ownership enforcement
    -> matching_results.tsv + official validation
```

### 1. Normalization and transliteration

`src/business_entity_resolution/src/text.py`, `prep.py`, and `translit.py` normalize Unicode, case, punctuation, legal suffixes, address tokens, numbers, and state information. Indic-to-Latin token mappings are learned only from supplied training pairs. This reduces script and formatting variation without external entity data.

### 2. Candidate generation

`blocking.py` limits an otherwise intractable Cartesian product. Character 3-gram TF-IDF retrieval operates on multiple name/address views, country/state partitions, forward and reverse directions, plus an orphan recovery leg. `gpu_dense.py` optionally trains a multilingual E5 bi-encoder and performs GPU top-k retrieval. Multiple retrieval channels increase recall while retaining provenance columns for later models.

### 3. Pairwise evidence

`features.py` computes exact, fuzzy, token-set, containment, number, address, state, and blocking-score features. `pipeline.py` builds features in resumable chunks and trains Stage-1 LightGBM. Trees model nonlinear interactions such as strong name similarity becoming trustworthy only when address or numeric evidence agrees.

### 4. Multilingual cross-encoder

`gpu_xenc.py` fine-tunes two state-fold classifiers initialized from `intfloat/multilingual-e5-base`. Each candidate pair is jointly encoded, allowing token-level interaction beyond independent embeddings. Two folds reduce dependence on one geography split. The same runner can score an optional Qwen judge only in the uncertain probability band.

### 5. Contextual Stage 2

`context.py` and `stage2.py` add evidence unavailable to independent pair classification: rank within an entity, winner/runner-up margins, competition from other Source-1 entities for the same target, score distributions, candidate-group agreement, and model-score consistency. Three LightGBM seeds are averaged.

### 6. Metric-aware decoding

`decision.py` compares a global threshold, an empty-entity gate, and exact expected-F0.5 selection. The selected rule operates per Source-1 entity after calibration. Ownership logic prevents the same target entity from being assigned incompatibly. This matters because F0.5 values precision twice as strongly as recall.

### 7. Validation and safety

The run writes `matching_results.tsv` and `candidate_pairs.tsv`, then invokes the supplied official validator. Expensive stages are resumable through cached Parquet shards and `.done_<stage>` markers. Output hashes make submitted artifacts identifiable.

## Score and experiment history

| Stage | Evidence | Outcome |
|---|---:|---|
| Early edited submissions | 0.929 | Starting point visible in submission history |
| Intermediate edited submission | 0.973 | Large gain before final Stage-2/XENC run |
| Final XENC Stage 2 | **0.98076** | Verified public score; production artifact included |
| Micro candidate 05 | 0.98076 | Same public score; baseline retained |
| Continued XENC fine-tuning | +0.000056 offline vs control | CI crossed zero and US slice regressed; not promoted |

These values describe separate measurements. Public leaderboard score and offline validation F0.5 are not directly interchangeable. See `docs/SCORE_AND_EXPERIMENTS.md`.

## Repository map

```text
Edited_Version_Portfolio/
├── README.md
├── src/business_entity_resolution/       production pipeline
├── requirements/                         CPU/GPU dependencies and hardware notes
├── docs/
│   ├── RUNBOOK.md                         install, run, monitor, validate
│   ├── SCORE_AND_EXPERIMENTS.md           score lineage and promotion decisions
│   ├── REFERENCES.md                      technical references and rationale
│   ├── PROVENANCE.md                      verified predecessor lineage
│   └── evidence/leaderboard_score_0.98076.jpeg
├── experiments/
│   ├── upgrade/                           context/dropout experiment; rejected
│   ├── micro_v2/                          five controlled decision candidates
│   ├── finetune/                          continued cross-encoder training
│   └── ce_fr_xlmr/                        unpromoted full-pair XLM-R experiment
├── analysis/                              error audits and decision reports
├── provenance/                            exact Edited_version predecessor archive
└── artifacts/                             final TSV, manifest, checksums
```

Model weights, datasets, caches, and generated training Parquet files are excluded. They are large, competition-controlled, or reproducible from the supplied dataset. The final TSV is included only in the complete package.

## Quick start

Linux and Python 3.10+ are required. Full GPU stages need CUDA-compatible PyTorch and one or more modern NVIDIA GPUs. Commands below run the CPU pipeline; full cluster commands are in `docs/RUNBOOK.md`.

```bash
cd src/business_entity_resolution
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export BER_DATA_DIR=/absolute/path/student_resource/dataset
export BER_WORK_DIR=/absolute/path/work
export BER_OUT_DIR=/absolute/path/output

cd src
python translit.py
python prep.py train test
python pipeline.py candidates train
python pipeline.py candidates test
python pipeline.py features train
python pipeline.py features test
python pipeline.py fit
python pipeline.py predict
```

For the full sparse+dense+XENC+Stage-2 run, configure `src/business_entity_resolution/slurm/cluster.conf`, run `slurm/setup_env.sh`, inspect a dry run, then submit:

```bash
cd src/business_entity_resolution/slurm
bash setup_env.sh
DRY=1 bash submit_all.sh
bash submit_all.sh
```

## Reproducibility boundary

- Competition datasets are not redistributed.
- Hugging Face checkpoints are downloaded by model ID; derived weights are not included.
- Validation reports are retained even when negative.
- Fine-tuning uses no validation labels for gradient updates, but its selection still uses reused validation. It is exploratory evidence.
- No experiment in this package guarantees a higher private or future leaderboard score.

## Attribution and licensing

This result was built from the user-provided `Edited_version.zip`; an exact immutable copy and hash are under `provenance/`. File-by-file comparison confirmed that the audited production source matches that archive. `TEJAS_v3.zip` was a separate later comparison whose `stage2.py` and `error_analysis.py` differ; it is not the source archive for the 0.98076 result and is not bundled. Dependency and model licenses remain with their owners. No project-wide license was present in the supplied material, so this package should not be presented as open source until the owners choose a license.
