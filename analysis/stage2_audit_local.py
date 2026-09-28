import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

D = Path('/Users/manankannasey/Downloads')
v = pl.read_parquet(D / 'val_scored_s2.parquet')
pred = pl.read_parquet(D / 'val_pred_s2.parquet')
ids = sorted(set(v['s1']))
index = {s: i for i, s in enumerate(ids)}
truth = {}
country = {}
with zipfile.ZipFile(D / '6ab10eb3b23ba_student_resource.zip') as z:
    for filename, mode in [('train_ground_truth.tsv', 'truth'), ('train_source1.tsv', 'country')]:
        with z.open('student_resource/dataset/train/' + filename) as f:
            for row in csv.DictReader(io.TextIOWrapper(f), delimiter='\t'):
                s = row['source1_entity_id' if mode == 'truth' else 'entity_id']
                if s in index:
                    if mode == 'truth':
                        truth[s] = set(filter(None, row['matched_entity_ids'].split(',')))
                    else:
                        country[s] = row['country']
assert len(truth) == len(ids)
counts = np.array([len(truth[s]) for s in ids])
rows = list(v.select('s1', 'cand', 'y').iter_rows())
si = np.array([index[s] for s, c, y in rows])
labels = np.array([int(c in truth[s]) for s, c, y in rows])
assert np.array_equal(labels, v['y'].to_numpy())
pair_index = {(s, c): i for i, (s, c, y) in enumerate(rows)}
assert len(pair_index) == len(rows)
def metrics(selected):
    n = np.bincount(si[selected], minlength=len(ids))
    tp = np.bincount(si[selected], weights=labels[selected], minlength=len(ids))
    den = .25 * counts + n
    score = np.divide(1.25 * tp, den, out=np.ones(len(ids)), where=den != 0)
    return score, {'f05': float(score.mean()), 'fp': int((n-tp).sum()), 'fn': int((counts-tp).sum()), 'affected_entities': int((score < 1).sum())}
current = np.array([pair_index[(s,c)] for s,c in pred.iter_rows()])
baseline, report = metrics(current)
print('CURRENT observed entities only', report)
for c in sorted(set(country.values())):
    mask = np.array([country[s] == c for s in ids])
    print('COUNTRY', c, int(mask.sum()), float(baseline[mask].mean()))
available = np.bincount(si, weights=labels, minlength=len(ids))
print('TRUTH', int(counts.sum()), 'positive scored', int(labels.sum()), 'missing before stage2', int((counts-available).sum()))
print('ORACLE on supplied pairs', metrics(np.flatnonzero(labels))[1])
candidates = defaultdict(list)
for i, (_, c, _) in enumerate(rows): candidates[c].append(i)
print('competing candidates', sum(len(a)>1 for a in candidates.values()))
p, p2 = v['p'].to_numpy(), v['p2'].to_numpy()
best = []
for alpha in [0, .1, .25, .5, 1]:
    score = (1-alpha)*p2 + alpha*p
    winners = np.array([a[int(np.argmax(score[a]))] for a in candidates.values()])
    results=[]
    for t in np.arange(.4,.951,.01):
        sel=winners[score[winners]>=t]
        f, m=metrics(sel)
        results.append((m['f05'], float(t), m))
    b=max(results, key=lambda a:a[0]);best.append((alpha,b))
print('EXPLORATORY threshold/blend sweeps', json.dumps(best))
winners = np.array([a[int(np.argmax(p2[a]))] for a in candidates.values()])
print('ownership removes true pairs', int(labels.sum()-labels[winners].sum()))
print('THRESHOLD .72', metrics(winners[p2[winners]>=.72])[1])
selected=set(current)
fp=[i for i in current if not labels[i]]
fn=[i for i in range(len(rows)) if labels[i] and i not in selected]
for name, ix in [('false positives', fp),('scored false negatives',fn)]:
    print(name,len(ix),'p2 quantiles',np.quantile(p2[ix],[0,.1,.5,.9,1]).tolist())
Path('stage2_audit_summary.json').write_text(json.dumps({'scope':'entities present in scored file; missing validation entities excluded', 'current':report,'sweeps':best},indent=2))
