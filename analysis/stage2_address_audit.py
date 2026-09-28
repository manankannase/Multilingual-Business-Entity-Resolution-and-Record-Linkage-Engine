"""Read-only address-pattern audit; patterns are diagnostics, not match rules."""
import csv
import io
import json
import re
import zipfile
from pathlib import Path

import polars as pl

D = Path('/Users/manankannasey/Downloads')
scored = pl.read_parquet(D / 'val_scored_s2.parquet')
pred = pl.read_parquet(D / 'val_pred_s2.parquet')
frames = {'accepted_true': scored.filter(pl.col('y') == 1).join(pred, on=['s1', 'cand'], how='semi')}
for category in ['decision', 'ownership', 'filter', 'retrieval', 'fp']:
    frames[category] = pl.read_parquet(f'stage2_{category}_errors.parquet')
need = set().union(*(set(f['s1']) | set(f['cand']) for f in frames.values()))
need.update(scored['s1'])
need.update(scored['cand'])
records = {}
with zipfile.ZipFile(D / '6ab10eb3b23ba_student_resource.zip') as z:
    for source in [1, 2, 3]:
        with z.open(f'student_resource/dataset/train/train_source{source}.tsv') as f:
            for row in csv.DictReader(io.TextIOWrapper(f), delimiter='\t'):
                if row['entity_id'] in need:
                    records[row['entity_id']] = row
assert len(records) == len(need), (len(records), len(need))

digit = re.compile(r'\b\d+\b')
unit = re.compile(r'\b(?:unit|suite|ste|apt|apartment|flat|floor|fl)\b', re.I)
compound = re.compile(r'\b\d+[A-Za-z]?[-/]\d+(?:[-/]\w+)*\b')
zip_us = re.compile(r'\b[A-Z]{2}\s*,?\s*(\d{5})(?:-\d{4})?\b', re.I)
report = {}
for category, frame in frames.items():
    counts = {'pairs': frame.height, 'either_unit_marker': 0,
              'either_compound_number': 0, 'first_numeric_token_conflict': 0,
              'confident_us_zip_either': 0, 'confident_us_zip_both': 0,
              'confident_us_zip_conflict': 0, 'either_address_empty': 0,
              's1_address_empty': 0, 'candidate_address_empty': 0,
              'both_addresses_empty': 0, 'blank_address_name_exact_casefold': 0}
    for s, c in frame.select('s1', 'cand').iter_rows():
        a, b = records[s], records[c]
        x, y = a['business_address'], b['business_address']
        counts['either_unit_marker'] += bool(unit.search(x) or unit.search(y))
        counts['either_compound_number'] += bool(compound.search(x) or compound.search(y))
        nx, ny = digit.search(x), digit.search(y)
        counts['first_numeric_token_conflict'] += bool(nx and ny and int(nx[0]) != int(ny[0]))
        counts['either_address_empty'] += not bool(x.strip() and y.strip())
        counts['s1_address_empty'] += not bool(x.strip())
        counts['candidate_address_empty'] += not bool(y.strip())
        counts['both_addresses_empty'] += not bool(x.strip() or y.strip())
        counts['blank_address_name_exact_casefold'] += bool(
            not (x.strip() and y.strip()) and a['business_name'].strip()
            and a['business_name'].strip().casefold() == b['business_name'].strip().casefold())
        if a['country'] == 'US':
            zx, zy = zip_us.search(x), zip_us.search(y)
            counts['confident_us_zip_either'] += bool(zx or zy)
            counts['confident_us_zip_both'] += bool(zx and zy)
            counts['confident_us_zip_conflict'] += bool(zx and zy and zx[1] != zy[1])
    report[category] = counts
blank_ids = [s for s, r in records.items() if not r['business_address'].strip()]
blank_pair = pl.col('s1').is_in(blank_ids) | pl.col('cand').is_in(blank_ids)
loss = pl.read_parquet('stage2_entity_loss.parquet')
truth_sizes = dict(loss.select('s1', 'truth_count').iter_rows())
pred_counts = {
    s: (n, tp) for s, n, tp in loss.select('s1', 'n', 'tp').iter_rows()}
impact = {}
for category in ['decision', 'ownership', 'filter', 'retrieval']:
    missed = frames[category].filter(blank_pair)
    gain = 0.
    for s, additional in missed.group_by('s1').len().iter_rows():
        n, tp = pred_counts[s]
        den = .25 * truth_sizes[s] + n
        before = 1.25 * tp / den if den else 1.
        after = 1.25 * (tp + additional) / (den + additional)
        gain += after - before
    impact[category] = {'pairs': missed.height, 'entities': missed['s1'].n_unique(),
                        'oracle_addition_delta': gain / loss.height}
report['blank_address_oracle_interventions'] = impact
bins = (pl.when(pl.col('p2') < .2).then(pl.lit('[0,.2)'))
    .when(pl.col('p2') < .4).then(pl.lit('[.2,.4)'))
    .when(pl.col('p2') < .6).then(pl.lit('[.4,.6)'))
    .when(pl.col('p2') < .8).then(pl.lit('[.6,.8)'))
    .when(pl.col('p2') < .9).then(pl.lit('[.8,.9)'))
    .otherwise(pl.lit('[.9,1]')))
report['score_bins_exploratory'] = scored.with_columns(
    blank_pair.alias('blank_address'), bins.alias('score_bin')).group_by('blank_address', 'score_bin').agg(
    pl.len().alias('pairs'), pl.col('p2').mean().alias('mean_raw_score'),
    pl.col('y').mean().alias('true_match_fraction')).sort('blank_address', 'score_bin').to_dicts()
Path('stage2_address_audit_summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
