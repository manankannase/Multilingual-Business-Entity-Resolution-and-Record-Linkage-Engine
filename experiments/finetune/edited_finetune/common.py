"""Shared data contracts; no writes to the baseline."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

KEYS = ['s1', 'cand']


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def atomic_parquet(frame, path):
    path = Path(path)
    tmp = path.with_suffix('.parquet.partial')
    frame.write_parquet(tmp, compression='zstd')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_digest():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def verify_pairs(frame, labels=False):
    required = KEYS + (['y'] if labels else [])
    if not set(required).issubset(frame.columns):
        raise ValueError(f'Missing columns {required}')
    if any(frame.select(required).null_count().row(0)):
        raise ValueError('Null identifiers or labels')
    if frame.select(KEYS).n_unique() != frame.height:
        raise ValueError('Duplicate candidate pairs')
    if labels and not set(frame['y'].unique()).issubset({0, 1}):
        raise ValueError('Labels must be binary')
    if 'p2' in frame.columns and (not frame['p2'].is_finite().all() or
                                  not frame['p2'].is_between(0., 1.).all()):
        raise ValueError('Baseline p2 must be finite probabilities')


def attrs(work, split, pairs):
    wanted = pl.concat([pairs['s1'], pairs['cand']]).unique()
    parts = []
    for source in (1, 2, 3):
        path = Path(work) / 'norm' / f'{split}_source{source}.parquet'
        parts.append(pl.scan_parquet(path).filter(pl.col('entity_id').is_in(wanted.implode()))
            .select('entity_id', 'name_clean', 'addr_clean', 'state', 'country',
                    pl.col('business_address').fill_null('').str.strip_chars().eq('').alias('blank'))
            .collect())
    result = pl.concat(parts).fill_null('')
    if result.height != wanted.len() or result['entity_id'].n_unique() != result.height:
        raise ValueError('Missing or duplicated normalized attributes')
    return result


def texts(pairs, attributes):
    # First experiment keeps original name | address | state representation.
    # This isolates additional training from changes in serialization/peer context.
    attributes = attributes.with_columns(pl.concat_str(
        ['name_clean', 'addr_clean', 'state'], separator=' | ').alias('text'))
    result = pairs.join(attributes.select(pl.col('entity_id').alias('s1'),
        pl.col('text').alias('a')), on='s1', how='left', validate='m:1').join(
        attributes.select(pl.col('entity_id').alias('cand'), pl.col('text').alias('b'),
                          pl.col('blank').alias('blank')), on='cand', how='left', validate='m:1')
    if result.height != pairs.height or any(result.select('a', 'b', 'blank').null_count().row(0)):
        raise ValueError('Pair-text join coverage mismatch')
    return result


def repair_scope(pairs):
    # Fixed before evaluation, includes both sides of close ownership contests.
    rival = pairs.group_by('cand').agg(pl.col('p2').max().alias('_max'),
        pl.col('p2').sort(descending=True).slice(1, 1).first().alias('_second'))
    return pairs.join(rival, on='cand', validate='m:1').with_columns(
        (((pl.col('p2') >= .05) & (pl.col('p2') <= .95)) | pl.col('blank') |
         ((pl.col('_max') - pl.col('_second') <= .15) &
          (pl.col('_max') - pl.col('p2') <= .15))).fill_null(False).alias('repair')
    ).drop('_max', '_second')


def owners(frame):
    return frame.sort(['p2', 's1', 'cand'], descending=[True, False, False]).unique('cand', keep='first')


def vector(pred, truth, ids):
    predicted, actual = {}, {}
    for s, c in pred.select(KEYS).iter_rows():
        predicted.setdefault(s, set()).add(c)
    for s, c in truth.select(KEYS).iter_rows():
        if c is not None:
            actual.setdefault(s, set()).add(c)
    values = []
    for s in ids:
        p, t = predicted.get(s, set()), actual.get(s, set())
        values.append(1.25 * len(p & t) / (.25 * len(t) + len(p)) if p or t else 1.)
    return np.asarray(values)
