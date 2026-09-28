import csv
import io
import json
import zipfile
from pathlib import Path

import polars as pl

D = Path('/Users/manankannasey/Downloads')
c = pl.read_parquet(D / 'edited_val_candidates.parquet', columns=['s1', 'cand'])
v = pl.read_parquet(D / 'val_scored_s2.parquet')
p = pl.read_parquet(D / 'val_pred_s2.parquet')
assert c.unique().height == c.height
assert v.select('s1', 'cand').join(c, on=['s1', 'cand'], how='anti').is_empty()
ids = set(c['s1'])
truth_rows, entity_rows = [], []
with zipfile.ZipFile(D / '6ab10eb3b23ba_student_resource.zip') as z:
    with z.open('student_resource/dataset/train/train_ground_truth.tsv') as f:
        for r in csv.DictReader(io.TextIOWrapper(f), delimiter='\t'):
            s = r['source1_entity_id']
            if s in ids:
                matches = list(filter(None, r['matched_entity_ids'].split(',')))
                entity_rows.append((s, len(matches)))
                truth_rows.extend((s, t) for t in matches)
    countries = []
    with z.open('student_resource/dataset/train/train_source1.tsv') as f:
        for r in csv.DictReader(io.TextIOWrapper(f), delimiter='\t'):
            if r['entity_id'] in ids:
                countries.append((r['entity_id'], r['country']))
t = pl.DataFrame(truth_rows, schema=['s1', 'cand'], orient='row')
entities = pl.DataFrame(entity_rows, schema=['s1', 'truth_count'], orient='row').join(
    pl.DataFrame(countries, schema=['s1', 'country'], orient='row'), on='s1', validate='1:1')
assert entities.height == len(ids)
keys = ['s1', 'cand']
retrieved = t.join(c, on=keys, how='semi')
retrieval_miss = t.join(c, on=keys, how='anti')
filtered = retrieved.join(v.select(keys), on=keys, how='anti')
scored_true = t.join(v, on=keys, how='inner')
assert scored_true.height == v.filter(pl.col('y') == 1).height
winners = v.sort('p2', descending=True).unique('cand', keep='first')
ownership = scored_true.join(winners.select(keys), on=keys, how='anti')
rejected = scored_true.join(winners.select(keys), on=keys, how='semi').join(p, on=keys, how='anti')
fp = p.join(t, on=keys, how='anti')
def metrics(pred):
    agg = pred.join(t.with_columns(pl.lit(1).alias('tp')), on=keys, how='left').group_by('s1').agg(
        pl.len().alias('n'), pl.col('tp').sum())
    a = entities.join(agg, on='s1', how='left').with_columns(pl.col('n', 'tp').fill_null(0))
    a = a.with_columns(pl.when((pl.col('truth_count') + pl.col('n')) == 0).then(1.)
        .otherwise(1.25 * pl.col('tp') / (.25 * pl.col('truth_count') + pl.col('n'))).alias('f05'))
    return {'f05': a['f05'].mean(), 'fp': int((a['n']-a['tp']).sum()),
        'fn': int((a['truth_count']-a['tp']).sum()),
        'countries': a.group_by('country').agg(pl.len(), pl.col('f05').mean()).to_dicts()}
summary = {'scope': 'All entities present in original validation candidate file',
    'candidate_pairs': c.height, 'entities': len(ids), 'scored_entities': v['s1'].n_unique(),
    'ground_truth_pairs': t.height, 'retrieved_true_pairs': retrieved.height,
    'retrieval_misses': retrieval_miss.height, 'filtered_true_pairs': filtered.height,
    'ownership_misses': ownership.height, 'decision_misses': rejected.height,
    'false_positives': fp.height, 'current': metrics(p),
    'retrieval_oracle': metrics(retrieved), 'scored_oracle': metrics(scored_true.select(keys))}
old = set(v['s1'])
summary['previous_scope'] = {'retrieval_misses': retrieval_miss.filter(pl.col('s1').is_in(list(old))).height,
    'filter_misses': filtered.filter(pl.col('s1').is_in(list(old))).height}
summary['error_countries'] = {}
for name, frame in [('retrieval', retrieval_miss), ('filter', filtered), ('ownership', ownership), ('decision', rejected), ('fp', fp)]:
    summary['error_countries'][name] = frame.join(entities.select('s1','country'),on='s1').group_by('country').len().to_dicts()
    frame.write_parquet(Path('stage2_' + name + '_errors.parquet'))
print(json.dumps(summary, indent=2))
Path('stage2_candidate_audit_summary.json').write_text(json.dumps(summary, indent=2))
