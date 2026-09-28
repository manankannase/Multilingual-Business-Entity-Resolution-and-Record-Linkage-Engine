"""Diagnostic oracle interventions; never produces submission predictions."""
import contextlib
import io
import json
import runpy
from pathlib import Path

import polars as pl

with contextlib.redirect_stdout(io.StringIO()):
    audit = runpy.run_path(str(Path(__file__).with_name('stage2_candidate_audit.py')))
entities, truth, pred, scored = (audit[k] for k in ('entities', 't', 'p', 'v'))
keys = ['s1', 'cand']
evaluate = audit['metrics']
base = evaluate(pred)['f05']
report = {'baseline_f05': base, 'entities': entities.height,
          'note': 'Label-aware diagnostics, NOT deployable improvements. Independent interventions overlap.'}

# Recover each FN class independently, leaving all other predictions unchanged.
interventions = {}
for label, variable in [('retrieval', 'retrieval_miss'), ('prefilter', 'filtered'),
                        ('ownership', 'ownership'), ('final_rejection', 'rejected')]:
    missing = audit[variable].select(keys)
    corrected = pl.concat([pred, missing]).unique()
    result = evaluate(corrected)
    interventions[label] = {'pairs': missing.height,
                            'affected_entities': missing['s1'].n_unique(),
                            'oracle_addition_delta': result['f05'] - base}
clean = pred.join(truth, on=keys, how='semi')
interventions['false_positives'] = {'pairs': audit['fp'].height,
    'affected_entities': audit['fp']['s1'].n_unique(),
    'oracle_removal_delta': evaluate(clean)['f05'] - base}
report['independent_oracle_interventions'] = interventions

agg = pred.join(truth.with_columns(pl.lit(1).alias('tp')), on=keys, how='left').group_by('s1').agg(
    pl.len().alias('n'), pl.col('tp').sum())
loss = entities.join(agg, on='s1', how='left').with_columns(pl.col('n', 'tp').fill_null(0))
loss = loss.with_columns(
    (pl.col('n') - pl.col('tp')).alias('fp'),
    (pl.col('truth_count') - pl.col('tp')).alias('fn'),
    pl.when((pl.col('truth_count') + pl.col('n')) == 0).then(1.)
      .otherwise(1.25 * pl.col('tp') / (.25 * pl.col('truth_count') + pl.col('n'))).alias('f05'))
loss = loss.with_columns((1 - pl.col('f05')).alias('loss'))
loss = loss.with_columns(pl.when(pl.col('loss') == 0).then(pl.lit('perfect'))
    .when(pl.col('truth_count') == 0).then(pl.lit('unmatched_entity_false_positive'))
    .when(pl.col('tp') == 0).then(pl.lit('matched_entity_zero_correct'))
    .when(pl.col('fp') > 0).then(pl.lit('some_correct_with_false_positive'))
    .otherwise(pl.lit('some_correct_missing_only')).alias('bucket'))
report['entity_loss_buckets'] = loss.group_by('bucket').agg(
    pl.len().alias('entities'), pl.col('fp', 'fn').sum(),
    (pl.col('loss').sum()/entities.height).alias('macro_score_loss'))
report['entity_loss_buckets'] = report['entity_loss_buckets'].sort('macro_score_loss', descending=True).to_dicts()
report['truth_size'] = {'max': entities['truth_count'].max(),
                       'entities_gt_20': entities.filter(pl.col('truth_count') > 20).height}
ranked = audit['winners'].sort(['s1', 'p2'], descending=[False, True]).with_columns(
    pl.int_range(pl.len()).over('s1').alias('rank'))
report['raw_score_top20'] = {'positive_pairs_beyond_cap': ranked.filter((pl.col('rank') >= 20) & (pl.col('y') == 1)).height,
    'note': 'Raw p2 order; calibrated ties can change ordering.'}

fp_scores = audit['fp'].join(scored, on=keys)
report['fp_score_counts'] = {str(t): fp_scores.filter(pl.col('p2') >= t).height for t in [.9, .95, .99]}
loss.sort('loss', descending=True).write_parquet('stage2_entity_loss.parquet')
Path('stage2_component_audit_summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
