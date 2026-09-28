# Edited_version competition experiment

TEJAS_v3 differs from the supplied Edited_version code only in `stage2.py` and
`error_analysis.py`. The principal change is training under simulated missing
Source-1 owners. This package tests that idea using cached XENC scores.

Original and two dropout contexts are mixed with balanced weights. The source
records removed from a copy lose all their candidate pairs; remaining pairs keep
their original features and OOF probabilities. Competition and XENC context are
recomputed. Dropout rates 10% and 20% are stress conditions, not estimates of test
orphan rates. No new neural training or test XENC scoring is needed.

Outputs use a separate directory. Baseline models, scores and submissions are
read as inputs. The new learner uses half the validation states per country for
early stopping and decision calibration. Remaining groups are used for comparison.
Countries with fewer than four validation states use normalized-name groups.
The baseline already used those states earlier, so its score is not an independent
estimate. There are no France ground-truth labels.

Test prediction is enabled only if the new holdout gain is at least 0.0002,
the group-cluster bootstrap interval is positive, neither labeled country loses,
and the dropped-owner stress score is no worse than baseline. This is an experiment
acceptance criterion, not a guarantee of a higher leaderboard score. Rejecting an
experiment is a valid result; the script stops without generating a new submission.

`run.sh` is configured for the server paths and 300000/80000 sampling settings seen
in the original logs. Confirm these match the baseline run. It supports either
`CODE/src/stage2.py` or `CODE/stage2.py`. Use a new output directory after a failed
or interrupted fit rather than mixing partial model files with a rerun.

After validation the printed `VALIDATED_UPGRADE_TSV` identifies the new file.
Its candidate TSV reports the actual Stage-2 input shortlist.

The Python and shell syntax checks pass; full cached-data training and
prediction must run on the server. Estimated duration requires actual CPU timing.
