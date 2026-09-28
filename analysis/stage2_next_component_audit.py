"""Read-only submission comparison and validation ambiguity diagnostics."""
import csv
import io
import json
import zipfile
from pathlib import Path

import polars as pl

D = Path('/Users/manankannasey/Downloads')
KEYS = ['s1', 'cand']


def submission(path):
    frame = pl.read_csv(path, separator='\t', infer_schema=False,
                        missing_utf8_is_empty_string=True).rename(
        {'source1_entity_id': 's1', 'matched_entity_ids': 'matches'})
    if frame['s1'].n_unique() != frame.height:
        raise ValueError(f'Duplicate Source-1 ID: {path}')
    return frame.with_columns(
        pl.col('matches').fill_null('').str.split(',').list.eval(
            pl.element().filter(pl.element() != '')).list.unique().list.sort())


base = submission(D / 'final_xenc_stage2_20260927-122208.tsv')
report = {'scope': 'Read-only diagnostics; validation labels never produce predictions.',
          'submission_differences': {}}
changed_ids = set()
for path in sorted(D.glob('matching_results_0[1-5]_*.tsv')):
    variant = submission(path).rename({'matches': 'new'})
    if variant.height != base.height or variant.join(base, on='s1', how='anti').height:
        raise ValueError(f'Entity set differs: {path}')
    joined = base.join(variant, on='s1', validate='1:1')
    changed = joined.filter(pl.col('matches') != pl.col('new')).with_columns(
        pl.col('new').list.set_difference('matches').list.len().alias('added'),
        pl.col('matches').list.set_difference('new').list.len().alias('removed'))
    changed_ids.update(changed['s1'])
    report['submission_differences'][path.stem] = {
        'entities': base.height, 'changed_entities': changed.height,
        'changed_fraction': changed.height / base.height,
        'added_pairs': changed['added'].sum(), 'removed_pairs': changed['removed'].sum(),
        'empty_to_nonempty': changed.filter(
            (pl.col('matches').list.len() == 0) & (pl.col('new').list.len() > 0)).height,
        'nonempty_to_empty': changed.filter(
            (pl.col('matches').list.len() > 0) & (pl.col('new').list.len() == 0)).height,
    }
    print(path.name, report['submission_differences'][path.stem], flush=True)

scored = pl.read_parquet(D / 'val_scored_s2.parquet')
pred = pl.read_parquet(D / 'val_pred_s2.parquet')
need = set(scored['s1']) | set(scored['cand'])
records, countries = {}, {}
with zipfile.ZipFile(D / '6ab10eb3b23ba_student_resource.zip') as archive:
    with archive.open('student_resource/dataset/test/test_source1.tsv') as stream:
        for row in csv.DictReader(io.TextIOWrapper(stream), delimiter='\t'):
            if row['entity_id'] in changed_ids:
                countries[row['entity_id']] = row['country']
    for source in (1, 2, 3):
        with archive.open(f'student_resource/dataset/train/train_source{source}.tsv') as stream:
            for row in csv.DictReader(io.TextIOWrapper(stream), delimiter='\t'):
                if row['entity_id'] in need:
                    records[row['entity_id']] = row
        print(f'Read training source {source}; retained {len(records):,} records', flush=True)
if len(records) != len(need):
    raise ValueError('Missing validation attributes')
for name, summary in report['submission_differences'].items():
    variant = submission(D / f'{name}.tsv').rename({'matches': 'new'})
    changed = base.join(variant, on='s1').filter(pl.col('matches') != pl.col('new'))
    summary['changed_by_country'] = changed.with_columns(
        pl.col('s1').replace_strict(countries).alias('country')).group_by('country').len().to_dicts()


def text_key(record):
    return tuple(' '.join(record[column].casefold().split())
                 for column in ('business_name', 'business_address', 'country'))


attrs = pl.DataFrame([
    (entity, text_key(record)[0], not bool(record['business_address'].strip()),
     json.dumps(text_key(record), ensure_ascii=False))
    for entity, record in records.items()
], schema=['entity_id', 'name', 'blank', 'signature'], orient='row')
joined = scored.join(attrs.select(
    pl.col('entity_id').alias('s1'), pl.col('signature').alias('a_sig')), on='s1').join(
    attrs.rename({'entity_id': 'cand', 'signature': 'b_sig'}), on='cand')
conflicts = joined.group_by('a_sig', 'b_sig').agg(
    pl.col('y').n_unique().alias('labels')).filter(pl.col('labels') > 1)
ambiguous = joined.join(conflicts, on=['a_sig', 'b_sig'], how='semi')
missed = joined.filter(pl.col('y') == 1).join(pred, on=KEYS, how='anti')
report['observable_text_ambiguity'] = {
    'definition': 'Same casefolded name/address/country on both sides, opposite labels; IDs excluded.',
    'conflicting_text_groups': conflicts.height,
    'pairs_in_conflicting_groups': ambiguous.height,
    'true_pairs_in_conflicting_groups': ambiguous.filter(pl.col('y') == 1).height,
    'missed_true_pairs_in_conflicting_groups': missed.join(
        conflicts, on=['a_sig', 'b_sig'], how='semi').height,
    'missed_blank_true_pairs_in_conflicting_groups': missed.filter(pl.col('blank')).join(
        conflicts, on=['a_sig', 'b_sig'], how='semi').height,
    'limitation': 'Applies to a text-only pair classifier; candidate-graph context might distinguish some cases.',
}
winners = scored.sort(['p2', 's1', 'cand'], descending=[True, False, False]).unique('cand', keep='first')
ownership = scored.filter(pl.col('y') == 1).join(winners.select(KEYS), on=KEYS, how='anti')
unselected_candidates = winners.join(pred.select('cand'), on='cand', how='anti').select('cand')
stranded = ownership.join(unselected_candidates, on='cand', how='semi')
report['ownership_diagnostics'] = {
    'true_pairs_lost_by_highest_score_ownership': ownership.height,
    'lost_true_pairs_whose_winner_is_also_rejected': stranded.height,
    'affected_entities': stranded['s1'].n_unique(),
    'limitation': 'Label-aware opportunity count; assigning to runner-up is not a verified correction.',
}
Path('stage2_next_component_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
