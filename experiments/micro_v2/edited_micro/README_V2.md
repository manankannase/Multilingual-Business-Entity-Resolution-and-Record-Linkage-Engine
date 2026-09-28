# Edited micro V2: balanced split and five TSV candidates

The original stopping set was only 165 businesses (US nv 162 + India sk 3).
India's cached validation is dominated by state hr: 35,778 of 35,800 entities.
Balanced whole-state stopping, calibration, and evaluation is therefore impossible
for India. V2 uses whole states where counts can balance; otherwise it keeps whole
normalized business-name groups together. Selection uses sizes only, never labels
or validation performance. Target proportions are 20% stopping, 30% calibration,
50% evaluation, subject to group sizes. Each country must allocate at least 10%
to stopping, 10% to calibration, and 25% to evaluation. An impossible allocation
fails before training. Calibration cross-fitting uses two balanced name-group folds.

The same partition is used for unchanged control and enhanced variant. Both train
three LightGBM seeds on original training pairs without competition dropout.
The enhanced feature block adds raw address availability, weighted name agreement,
rare shared tokens, competitor-relative name evidence, and deduplicated name
corroboration. Token frequencies use training Source-1 names only. Labels never
enter new feature construction. Existing cross-encoder scores are reused.

Five decision mechanisms are fixed before evaluating their results:

| File suffix | Mechanism |
|---|---|
| 01_expected_f | Enhanced model; globally calibrated expected-F selection |
| 02_threshold | Enhanced model; macro-F0.5 threshold tuned on calibration only |
| 03_missing_calibration | Enhanced model; separate calibration for blank/nonblank candidate addresses, then expected-F selection |
| 04_model_ensemble | Fixed 75% enhanced + 25% saved baseline scores, recalibrated before expected-F selection |
| 05_missing_only_update | Enhanced scores for blank-address candidates, saved baseline scores otherwise, recalibrated before expected-F selection |

The request is to generate all five candidates, so a failed improvement comparison
does not stop TSV generation. Each candidate has an explicit passes_comparison flag.
Passing requires >=0.0002 mean gain over both matched control and saved baseline,
positive group-bootstrap intervals, and no decline in either labeled country.
If none passes, recommendation is retain_verified_baseline. Nothing is uploaded
to the challenge automatically. File hashes identify identical outputs; distinct
mechanisms can legitimately yield the same predictions.

This remains an exploratory comparison on reused validation. Original Stage-1
early stopping used validation; saved baseline used all validation. New splitting
does not undo upstream exposure. India name grouping tests unseen names within
the same state, not geographic transfer. Few US state groups make uncertainty
estimates approximate. Ranking five candidates reuses the evaluation set; there
is no pristine confirmation set or France ground truth. No leaderboard target is
guaranteed. The code does not copy validation labels into test predictions.

## Install and run on the server

The ZIP contains an edited_micro directory. Extract into a new parent to preserve
the original package and interrupted run:

```bash
unzip -n /data1/anandkumar/edited_micro_v2.zip -d /data1/anandkumar/edited_micro_v2
nohup setsid bash /data1/anandkumar/edited_micro_v2/edited_micro/run.sh \
  /data1/anandkumar/edited_micro_run_02 \
  </dev/null > /data1/anandkumar/edited_micro_v2/experiment.log 2>&1 &
```

The runner first performs preflight, then trains, predicts all five files, and runs
the official validator on each. It uses the existing baseline virtual environment
and 16 LightGBM threads. No new GPU scoring or dependency install is required.
Test scores use the retained Stage-2 shortlist, not the separate CE-FR full-pair job.

Monitor and inspect results:

```bash
tail -f /data1/anandkumar/edited_micro_v2/experiment.log
cat /data1/anandkumar/edited_micro_run_02/comparison.json
ls -lh /data1/anandkumar/edited_micro_run_02/output/matching_results_*.tsv
```

The output also contains candidate_pairs.tsv for validation and
validation_ranking.json with recommendation, pass/fail flags, and file hashes.
Per-candidate validation scores are in comparison.json.

Training and prediction refuse to reuse a started output directory. After an
interruption retain its files and use a fresh run name. Locks prevent duplicates
on the same output directory. Baseline files are read as inputs and preserved.
Progress is printed every 250,000 added-feature rows, every 100 boosting rounds,
and every test shard. Server completion time needs measured stage throughput.

## Checks

Run python test_features.py and python test_partition.py. The split regression
includes one dominant state and tiny states. test_integration.py additionally
uses temporary synthetic caches with the real baseline modules to exercise
preflight, both three-seed ensembles, calibration, all five TSV outputs, unique
ownership, unchanged baseline hashes, and overwrite guards. Synthetic scores
are software checks, not evidence of actual improvement.
