# Score and experiment record

## Public result

Final public leaderboard score: **0.98076**. The leaderboard submission list rounded this to **0.981**. Evidence is stored in `evidence/leaderboard_score_0.98076.jpeg`.

Observed iteration path:

| Submission stage | Public score | Interpretation |
|---|---:|---|
| Early baseline/edited iteration | 0.929 | Starting point for the documented improvement path |
| Intermediate iteration | 0.973 | Improved feature/model pipeline |
| Final XENC Stage 2 | **0.98076** | Verified final production artifact |

The earlier submission page also contained failed/ablation runs near 0.68–0.69. They are not milestones and are not used in the résumé improvement claim.

## Final production run

The accepted final path combined Stage-1 LightGBM evidence, multilingual E5 cross-encoder scores, contextual Stage-2 features, and exact expected-F0.5 selection. Logged validation macro F0.5 was about 0.9902. The organizer validator reported 1,732,544 required Source-1 rows and PASS.

## Context/dropout upgrade

`experiments/upgrade/` evaluated additional context with an untouched grouped holdout relative to the proposed update. Result:

- Baseline holdout F0.5: 0.9903359742.
- Upgrade holdout F0.5: 0.9900530744.
- Delta: -0.0002828999.
- Decision: reject; baseline retained.

This experiment demonstrates a key research discipline: more context features do not automatically improve transfer.

## Micro-feature and decoder study

`experiments/micro_v2/` introduced address-availability, weighted-name, rare-token, competitor-relative, and deduplicated-corroboration signals. It generated five prespecified decision variants:

1. Expected-F selection.
2. Threshold selection.
3. Missing-address calibration.
4. Model ensemble.
5. Missing-only update.

Candidate 05 ranked highest offline with +0.00003848 against the saved baseline, but its confidence interval crossed zero. It was nevertheless submitted as an exploratory test and received **0.98076**, identical to the verified baseline. `comparison.json` and `validation_ranking.json` contain the exact results and hashes.

## Continued cross-encoder fine-tuning

`experiments/finetune/` continued both saved multilingual E5 cross-encoder folds for one epoch:

- At most 300,000 labeled pairs per fold.
- 50/35/15 positive/hard-negative/easy-negative sampling.
- Validation IDs excluded from gradient updates.
- Learning rate 5e-6; microbatch 16; gradient accumulation 4.
- BF16 when supported; checkpoints every 250 steps.
- Repair-scope scoring only for uncertain, missing-address, or close-competition pairs.

Outcome:

- Control evaluation F0.5: 0.9901109493.
- Fine-tuned evaluation F0.5: 0.9901671446.
- Delta versus control: +0.0000561954.
- 95% grouped bootstrap CI: [-0.00003184, +0.00015879].
- India: +0.00021771; US: -0.00005470.
- `eligible_for_test_prediction`: false.

The mean gain was too small, uncertainty included zero, and the US slice regressed. The test-scoring gate correctly stopped; no fine-tuned TSV was promoted.

## CE-FR/XLM-R experiment

`experiments/ce_fr_xlmr/` contains a separate full-pair XLM-R training/scoring route designed to strengthen French matching. Validation checks revealed no reliable country field in the constructed pair text, and full test scoring covered 99,247,308 pairs. The route remained unfinished/unpromoted at entity-decision level and did not produce the verified final submission.

## Orphan ownership experiment

`analysis/stage2_orphan_experiment.json` tested reassignment among losing ownership edges. Calibration did not select a nonzero margin, and evaluation delta was exactly zero. No production change was made.

## Promotion rule

A candidate was promoted only when it showed a material mean gain, a positive grouped-bootstrap lower bound, no labeled-country regression, valid ID coverage, unique ownership, and official validator PASS. This protected the 0.98076 baseline from leaderboard-driven overfitting.

