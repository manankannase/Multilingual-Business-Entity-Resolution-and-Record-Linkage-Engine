# Edited_version: targeted improvement plan

## Decision

Keep verified public baseline **0.98076**. The older display 0.981 is rounding.
The competition-dropout upgrade is not accepted: holdout delta -0.000282900,
group-bootstrap interval [-0.000663030, -0.000134593], with regressions in India,
US, and synthetic stress. This does not isolate dropout as the cause: training,
calibration, and decision selection also differ. Baseline used all validation,
so the comparison is not a pristine independent test for baseline.

Next experiment: **missing-candidate-address disambiguation**, retaining the
existing retrieval, cross-encoder, and verified model as controls.

## Measured evidence

Local audit reconstructs ground truth from the supplied student resource ZIP and
reproduces full baseline validation F0.5 **0.9902145364364596**, over **87,992**
Source 1 entities. All counts below concern validation, not hidden test labels.

| Failure stage | Missed true pairs | Candidate address blank | Share |
|---|---:|---:|---:|
| Retrieval | 1,043 | 864 | 82.8% |
| Prefilter before Stage 2 | 586 | 369 | 63.0% |
| Candidate assigned to another Source 1 | 956 | 650 | 68.0% |
| Final selection rejects surviving true pair | 5,949 | 4,854 | 81.6% |
| Total false negatives | 8,534 | 6,737 | 78.9% |

There are 409 false-positive pairs; 151 also have a blank candidate address.
Of accepted true pairs, 6,627/295,570 (2.24%) have blank candidate addresses.
Thus missing addresses are strongly associated with errors, but are not proof
that the model mishandles missingness: they also remove real identifying evidence.
The same-name businesses can genuinely be indistinguishable without address data.

Perfectly recovering only the 4,854 final-selection blank-address false negatives,
without adding any false positives, would improve validation F0.5 by 0.00476629.
This is a label-aware diagnostic ceiling for that intervention, **not an expected
gain, deployable rule, or estimate of public-score improvement**.

The current raw scores do not show gross underconfidence for blank addresses:

| Raw score interval | Blank-address pairs | Actual match fraction |
|---|---:|---:|
| 0.2–0.4 | 4,945 | 26.1% |
| 0.4–0.6 | 2,683 | 48.2% |
| 0.6–0.8 | 1,324 | 65.7% |
| 0.9–1.0 | 6,360 | 99.0% |

These bins describe supplied validation scores before ownership; they are not
cross-fitted calibration results or valid thresholds for deployment.

Other findings:

- Maximum true match count is 11; no surviving true pair ranks beyond raw-score
  top 20. Increasing N_MAX is not supported by this audit. Calibrated ties remain
  a separate reproducibility check.
- Current parser extracts only six-digit postcodes and strips numeric leading
  zeros. That is unsuitable for US/French postcodes. However, a conservative US
  state-plus-ZIP detector finds only 23 final-rejection pairs with a ZIP on either
  side and none with ZIPs on both sides. This detector is incomplete; the audit
  does not establish postal normalization as a major source of remaining loss.
- First numeric tokens disagree for 42,410 accepted true pairs. Never impose a
  hard house-number mismatch veto: unit/floor/compound numbers can explain it.
- 47 false-positive pairs have p2 >= 0.99. High confidence is not a guarantee.
- Existing model already has name fuzziness, house-number, missing-street,
  name-frequency, competitor-margin, and collective-support features. Simply
  adding another generic string similarity is not the first priority.

## Experiment order

### 1. Establish a matched control

Use original training rows with dropout disabled. Give control and variant the
same model settings, training entities, early-stopping partition, calibration
partition, and untouched final evaluation groups. Split by whole entity/business
groups and geography where feasible; do not split duplicate pairs independently.
All learned stages must exclude final evaluation labels, including Stage 1 and
cross-encoder training. The existing validation set has been examined extensively;
use it for development diagnostics, not another claim of independent confirmation.

Report macro entity F0.5 over every Source 1 ID, including those with zero candidates.
Track India/US, blank/nonblank address, unmatched entities, and competing-owner cases.
There are no French labels, so country-transfer gains remain unverified.

### 2. Test a small missing-address-aware feature block

Do not replace the main model or regenerate 99 million cross-encoder scores.
Append features to the existing Stage-2 CPU model in an isolated experiment:

1. Explicit raw address availability on each side: distinguish missing evidence
   from disagreement. Existing parsed-street missingness is not identical.
2. Rare-name-token agreement and disagreement, using frequencies fitted without
   validation labels. Preserve distinctive tokens that generic token-set scores
   can hide. Existing exact-name counts do not capture partial-name rarity.
3. Competitor-relative name evidence: compare the candidate against its best and
   second-best Source 1 claimants, conditional on missing address. Preserve the
   one-owner-per-candidate constraint.
4. Deduplicated name corroboration from other high-confidence candidates of the
   same Source 1, excluding the candidate itself. Current t_support requires both
   name and address similarity >=80; a blank-address candidate cannot obtain this
   address-based support. Count support by independent source where possible;
   do not count repeated copies as independent confirmations.

Train using existing labeled training positives plus same-name competing-owner
hard negatives. Do not turn discovered validation errors into training examples.
First test the feature block alone; do not simultaneously change weights,
augmentation, model size, calibration, and decision rule.

Only if that experiment helps, compare mild address-masking augmentation on
training records. Mask an entity's address consistently across its training pairs,
recompute affected features and model inputs, and retain hard negatives. Do not
mask raw text while reusing cross-encoder scores from the original address.

### 3. Calibrate/select after improving discrimination

Compare pooled calibration with a regularized missing-address-aware alternative
on separate calibration data. Beta calibration is a small parametric option;
it is not known to beat isotonic here. Avoid many country-specific thresholds.

Evaluate final selection by macro F0.5, not pair AUC or log loss alone. Existing
expected-F selection assumes independent candidates; that is a modeling
assumption, not an unconditional guarantee of optimality. Repeated evidence from
duplicate records must not be mistaken for multiple independent observations.

### 4. Repair retrieval/prefilter selectively, afterward

For blank-address records only, test an extra bounded name-based retrieval route
using rare tokens/character similarity. Measure added true pairs against added
pair volume. For the prefilter, test retaining a small number of strongest
name-based candidates even when Stage-1 p < 0.01. Score every added pair properly;
do not replace unavailable model scores with zero or fabricated values.

### Promotion rule

Keep a variant only after consistent improvement against the matched control on
untouched grouped evaluation, with uncertainty reported and no material subgroup
regression. Generate a new test TSV only then; preserve baseline and its hash.
Do not tune against repeated leaderboard submissions.

## Research supporting the experiments

- [HierMatcher, IJCAI 2020](https://www.ijcai.org/proceedings/2020/507): attribute-aware
  matching addresses noisy/missing/misplaced data. Supports examining evidence by
  attribute; does not prove this proposed feature block improves this challenge.
- [Ditto, VLDB](https://arxiv.org/abs/2004.00584): domain-information highlighting
  and difficult-example augmentation can improve entity matching. Our selective
  address-masking proposal is an experiment inspired by that broader finding.
- [Beta calibration, AISTATS 2017](https://proceedings.mlr.press/v54/kull17a.html):
  small parametric calibration family including identity; motivates a controlled
  alternative to isotonic, not automatic replacement.
- [Bayes-optimal F-measure maximizers, JMLR](https://arxiv.org/abs/1310.4849):
  F-measure optimization depends on output-distribution assumptions; pairwise
  accuracy and independent-probability selection need not optimize the actual task.
- [Scikit-learn threshold tuning](https://scikit-learn.org/1.5/modules/classification_threshold.html):
  keep fitting and threshold-tuning data separate to avoid overfitting.

## Reproducibility

Scripts: `stage2_component_audit.py` (reuses `stage2_candidate_audit.py`) and
`stage2_address_audit.py`. Outputs: `stage2_component_audit_summary.json`,
`stage2_address_audit_summary.json`, and `stage2_entity_loss.parquet`.
Scripts contain local input paths; they are audit scripts, not a server upgrade.
No production model, remote process, or submission was modified.
