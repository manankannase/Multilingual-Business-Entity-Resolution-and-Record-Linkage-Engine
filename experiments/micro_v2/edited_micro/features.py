"""Additional name evidence. Fits token frequencies on training Source-1 only."""
from collections import Counter
import math

import numpy as np
import polars as pl

KEYS = ['s1', 'cand']
EXTRA = ['ma_s1_blank', 'ma_cand_blank', 'ma_name_jacc', 'ma_name_cover_s1',
         'ma_name_cover_cand', 'ma_rare_shared', 'ma_name_margin',
         'ma_name_competitors', 'ma_peer_max', 'ma_peer_support', 'ma_peer_sources']


def read_attrs(base, split, pairs):
    ids = pl.concat([pairs['s1'], pairs['cand']]).unique()
    parts = []
    for source in (1, 2, 3):
        path = base / 'norm' / f'{split}_source{source}.parquet'
        parts.append(pl.scan_parquet(path).filter(pl.col('entity_id').is_in(ids.implode()))
            .select('entity_id', pl.col('name_core').fill_null(''),
                    (pl.col('business_address').fill_null('').str.strip_chars() == '').alias('blank'),
                    pl.lit(source, pl.Int8).alias('source')).collect())
    attrs = pl.concat(parts)
    if attrs['entity_id'].n_unique() != attrs.height or attrs.height != len(ids):
        raise ValueError('Missing or duplicate raw/normalized entity attributes')
    return attrs


def fit_weights(attrs, train_ids):
    names = attrs.filter(pl.col('entity_id').is_in(train_ids.implode()))['name_core']
    freq = Counter(t for name in names for t in set(name.split()))
    return {'n': len(names), 'idf': {t: math.log((len(names) + 1)/(n + 1)) + 1
                                   for t, n in freq.items()}}


def add_features(pairs, attrs, weights):
    if pairs.select(KEYS).n_unique() != pairs.height:
        raise ValueError('Duplicate input candidate pair')
    names = {name: frozenset(name.split()) for name in attrs['name_core'].unique()}
    records = {s: (name, names[name], bool(blank), source)
               for s, name, blank, source in attrs.iter_rows()}
    idf = weights['idf']
    unseen = math.log(weights['n'] + 1) + 1
    mass = {name: sum(idf.get(t, unseen) for t in tokens) for name, tokens in names.items()}

    def compare(a, b):
        common = a[1] & b[1]
        iw = sum(idf.get(t, unseen) for t in common)
        wa, wb = mass[a[0]], mass[b[0]]
        union = wa + wb - iw
        return (iw/union if union else 0., iw/wa if wa else 0., iw/wb if wb else 0.,
                max((idf.get(t, unseen) for t in common), default=0.))

    # Up to four distinct names per S1. Keep a second ID for self-exclusion.
    # Selection uses original OOF Stage-1 p only, never labels or new model scores.
    peers = {}
    ranked = pairs.select('s1', 'cand', 'p').sort(['s1', 'p', 'cand'], descending=[False, True, False])
    for s, c, p in ranked.iter_rows():
        name = records[c][0]
        if not name:
            continue
        group = peers.setdefault(s, {})
        if name not in group and len(group) >= 4:
            continue
        entries = group.setdefault(name, [])
        if len(entries) < 2:
            entries.append((c, float(p)))
    result = np.zeros((pairs.height, len(EXTRA)), dtype=np.float32)
    for i, (s, c) in enumerate(pairs.select(KEYS).iter_rows()):
        a, b = records[s], records[c]
        j, ca, cb, rare = compare(a, b)
        support, best, sources = 0., 0., set()
        for entries in peers.get(s, {}).values():
            other = next(((oid, p) for oid, p in entries if oid != c), None)
            if other is None:
                continue
            oid, p = other
            sim = compare(b, records[oid])[0]
            best = max(best, sim)
            support += p * sim
            if p >= .9 and sim >= .8:
                sources.add(records[oid][3])
        result[i] = (a[2], b[2], j, ca, cb, rare, 0., 0., best, support, len(sources))
        if (i+1) % 250000 == 0:
            print(f'  name features {i+1:,}/{pairs.height:,}', flush=True)
    extra = pairs.select(KEYS).with_columns([pl.Series(c, result[:,i]) for i,c in enumerate(EXTRA)])
    # Margin to the best OTHER Source-1 claimant; ties give zero, no rival gives 1.
    top = extra.sort(['cand', 'ma_name_jacc', 's1'], descending=[False, True, False]).group_by('cand').agg(
        pl.col('s1').first().alias('_owner'), pl.col('ma_name_jacc').first().alias('_best'),
        pl.col('ma_name_jacc').slice(1,1).first().alias('_second'), pl.len().alias('_claims'))
    extra = extra.join(top, on='cand', how='left', validate='m:1').with_columns(
        pl.when(pl.col('_claims') == 1).then(1.)
          .when(pl.col('s1') == pl.col('_owner')).then(pl.col('ma_name_jacc')-pl.col('_second'))
          .otherwise(pl.col('ma_name_jacc')-pl.col('_best')).cast(pl.Float32).alias('ma_name_margin'),
        (pl.col('_claims')-1).cast(pl.Float32).alias('ma_name_competitors')).select(KEYS+EXTRA)
    return pairs.join(extra, on=KEYS, how='left', validate='1:1')
