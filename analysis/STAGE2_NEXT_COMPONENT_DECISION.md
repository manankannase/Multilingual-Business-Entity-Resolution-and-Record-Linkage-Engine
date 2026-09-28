# Next experiment after Micro V2

Keep verified baseline: public score 0.98076. Screenshot says Overall Score;
check the submission-specific row before treating it as candidate 05's result.

## New measurements

Semantic comparison of local downloaded submissions (match sets, ignoring order):

- Candidate 05 changes 11,225 of 1,732,544 Source-1 entities (0.6479%).
- Adds 9,978 pairs and removes 1,415.
- Changed entities: France 4,875, US 3,580, India 2,770.
- It is not a copy of baseline.
- Candidate 05 reused-validation holdout delta against saved baseline: +0.0000384840,
  bootstrap interval [-0.0000694923, +0.0002420736]. No reliable advantage.
- Feature variant blank-address diagnostics: 59 more selected true pairs,
  19 more selected false pairs. These refer to feature variant, not candidate 05.
- Within scored validation pairs, no casefolded raw name/address/country pair
  signature had conflicting positive and negative labels. This does not establish
  separability under normalized text, or rule out intrinsic ambiguity elsewhere.

## Priority 1: ownership and match selection together

Existing code assigns each S2/S3 candidate to the S1 with largest Stage-2 score,
then decides which candidates each S1 accepts. No reassignment follows rejection.

Validation diagnostics:

- 956 true pairs lost by the highest-score ownership step.
- 934 of those have a winning pair which is ALSO rejected by final selection.
- These 934 missed pairs affect 903 entities.

This is evidence of a decision-order limitation, not proof the runner-up is right.
Other rejected edges can be false matches. Never assign every orphan to runner-up.

First experiment: hold model scores fixed; compare existing decoder with joint
selection over small competing-owner components. Score a change by its estimated
gain in the sum of per-entity expected F0.5. Permit leaving candidates unassigned.
Keep each candidate assigned to at most one S1, while allowing many candidates per
S1. Standard one-to-one Hungarian matching imposes the wrong S1 capacity.

Calibrate candidate alternatives on separate calibration groups: original
calibration was fitted only after highest-score ownership. Applying it unchanged
to runner-up edges is a distribution change. Select hyperparameters on calibration
groups; freeze before evaluation. Reused existing validation is development
evidence; full-pipeline training must exclude any new confirmation holdout.

## Priority 2: targeted contextual cross-encoder fine-tuning

The biggest validation loss remains final rejection: 5,949 true pairs, including
4,854 blank-address candidates. Extra name features alone have not reliably
improved saved baseline. Improve discrimination rather than simply lower cutoff.

Controlled experiments, one change at a time:

1. Explicit name/address/country fields and missing-address marker; compare raw
   and normalized inputs. Do not concatenate arbitrarily long text without a
   measured truncation audit.
2. Mine difficult labeled training negatives: same/similar names, competing S1
   owners, and plausible address decoys. Existing baseline already uses filtered
   hard candidate pairs; new work should target residual failure types. Never
   assume an arbitrary unlabeled pair is a negative.
3. Add up to two independent high-confidence peer records as contextual evidence.
   Derive peers from out-of-fold model scores, not truth. Exclude target itself;
   deduplicate copied records and measure wrong-anchor error propagation.
4. Test mild address masking only in a separate ablation. Recompute affected
   inputs/scores; retaining scores from unmasked text makes the experiment invalid.

Train on actual training labels, retain easy examples, and generate out-of-fold
scores for downstream fitting. Start evaluation on existing validation candidate
scope, then score only the prespecified test repair scope if the experiment helps.
Do not require another 99M-pair inference run for a targeted repair.

## Priority 3: retrieval and prefilter escape route

Retrieval misses 1,043 true validation pairs; p>=0.01 prefilter removes another
586. Test a bounded rare-name/character route for missing addresses and a small
name-based prefilter bypass. Score all added pairs properly; zero placeholders
are not model predictions. Keep strongest-name exact collisions competitive,
not unconditional matches.

## Domain transfer

France has no supplied labels in the evaluated split. 4,875 changed French
entities does not establish whether those edits helped. Validation-public gap
cannot be attributed to France from aggregate scores alone. Compare missingness,
normalization collisions, truncation, score distributions, and candidate counts
by country. French pseudo-label training needs its own ablation; greater
agreement with the original predictor is not evidence of higher true accuracy.

## Research basis

- Ditto: domain highlighting and difficult-example augmentation for entity
  matching. https://arxiv.org/abs/2004.00584
- HierMatcher: token-, attribute-, and entity-level matching for dirty and
  heterogeneous records. https://www.ijcai.org/proceedings/2020/507
- F-measure decision theory: optimizing decomposable pair loss does not generally
  optimize structured F-measure; distribution assumptions matter.
  https://arxiv.org/abs/1310.4849

These papers motivate experiments. They do not predict leaderboard gains here.

## Scale of requested target

0.99266666 - 0.98076 = 0.01190666, or 1.190666 percentage points. Reaching this
would remove approximately 61.88% of current score deficit to 1.0. Current
Micro V2 gain is not evidence that threshold variations can bridge that gap.

Audit script: stage2_next_component_audit.py.
Machine-readable findings: stage2_next_component_audit.json.
No production model, remote process, or submission was modified.
