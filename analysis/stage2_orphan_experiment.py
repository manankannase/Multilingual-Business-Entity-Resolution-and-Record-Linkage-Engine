"""Exploratory validation replay: reassign unused candidates by expected-F gain."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent / 'edited_finetune'))
from common import KEYS, owners, vector
from evaluate import compare


def expected_f(probabilities, selected):
    if not selected:
        return float(np.prod(1 - probabilities))
    inside, outside = np.array([1.]), np.array([1.])
    for i, q in enumerate(probabilities):
        if i in selected:
            inside = np.convolve(inside, [1 - q, q])
        else:
            outside = np.convolve(outside, [1 - q, q])
    a = np.arange(len(inside))[:, None]
    b = np.arange(len(outside))[None, :]
    utility = 1.25 * a / (.25 * (a + b) + len(selected))
    return float((inside[:, None] * outside[None, :] * utility).sum())


def main():
    download = Path('/Users/manankannasey/Downloads')
    scored = pl.read_parquet(download / 'val_scored_s2.parquet')
    pred = pl.read_parquet(download / 'val_pred_s2.parquet')
    entities = pl.read_parquet('stage2_entity_loss.parquet')
    ids = entities['s1']
    with zipfile.ZipFile(download / '6ab10eb3b23ba_student_resource.zip') as archive:
        with archive.open('student_resource/dataset/train/train_source1.tsv') as stream:
            info = pl.read_csv(stream, separator='\t', quote_char=None, infer_schema=False,
                columns=['entity_id', 'business_name', 'country']).filter(pl.col('entity_id').is_in(ids.implode()))
    info = info.with_columns(pl.col('business_name').fill_null('').str.to_lowercase()
        .str.replace_all(r'\s+', ' ').str.strip_chars().alias('name_core')).sort('entity_id')
    mask = [int.from_bytes(hashlib.sha256((country + '|' + (name or entity)).encode()).digest()[:8], 'little') % 100 < 40
            for entity, country, name in info.select('entity_id', 'country', 'name_core').iter_rows()]
    cal_info, hold_info = info.filter(pl.Series(mask)), info.filter(~pl.Series(mask))
    cal_ids = cal_info['entity_id']
    truth = pl.concat([scored.filter(pl.col('y') == 1).select(KEYS)] +
        [pl.read_parquet(f'stage2_{stage}_errors.parquet').select(KEYS) for stage in ['retrieval', 'filter']]).unique()
    assert truth.height == 304104
    baseline = vector(pred, truth, info['entity_id'])
    assert abs(baseline.mean() - .9902145364364596) < 1e-12
    winners = owners(scored)
    losers = scored.join(winners.select(KEYS), on=KEYS, how='anti')
    pieces = []
    for frame in [winners, losers]:
        calibration = frame.filter(pl.col('s1').is_in(cal_ids.implode()))
        assert calibration['y'].n_unique() == 2
        iso = IsotonicRegression(out_of_bounds='clip').fit(calibration['p2'].to_numpy(), calibration['y'].to_numpy())
        pieces.append(frame.with_columns(pl.Series('q', iso.predict(frame['p2'].to_numpy()))))
    q = pl.concat(pieces).with_columns((pl.col('q') / pl.col('q').sum().over('cand').clip(lower_bound=1.)).alias('q'))
    # Keep the original selected set and rank remaining possible additions by q.
    ranked = q.sort(['s1', 'q', 'p2', 'cand'], descending=[False, True, True, False])
    selected_pairs = set(pred.iter_rows())
    records, selected, lookup = {}, {}, {}
    for s, c, probability in ranked.select('s1', 'cand', 'q').iter_rows():
        values = records.setdefault(s, [])
        if len(values) >= 20 and (s, c) not in selected_pairs:
            continue
        index = len(values)
        values.append(probability)
        lookup[s, c] = index
        if (s, c) in selected_pairs:
            selected.setdefault(s, set()).add(index)
    records = {s: np.array(values) for s, values in records.items()}
    orphan = losers.join(pred.select('cand'), on='cand', how='anti')
    proposals = orphan.select(KEYS).join(q.select(KEYS + ['q']), on=KEYS)
    candidates = {}
    for s, c, probability in proposals.iter_rows():
        if (s, c) in lookup and probability >= .05:
            candidates.setdefault(c, []).append((s, probability))
    candidates = sorted(candidates.items(), key=lambda x: (-max(p for _, p in x[1]), x[0]))
    variants, calibration_scores = {}, {}
    # Three margins fixed before evaluation; choose on calibration entities only.
    for margin in [0., .002, .01]:
        chosen = {s: set(v) for s, v in selected.items()}
        additional = []
        for c, alternatives in candidates:
            gains = []
            for s, _ in alternatives:
                current = chosen.get(s, set())
                gain = expected_f(records[s], current | {lookup[s, c]}) - expected_f(records[s], current)
                gains.append((gain, s))
            gain, s = max(gains)
            if gain > margin:
                chosen.setdefault(s, set()).add(lookup[s, c])
                additional.append((s, c))
        adds = pl.DataFrame(additional, schema=KEYS, orient='row') if additional else pred.head(0)
        result = pl.concat([pred, adds])
        assert result['cand'].n_unique() == result.height
        variants[margin] = result
        calibration_scores[margin] = float(vector(result, truth, cal_ids).mean())
        print(f'margin={margin} added={len(additional)} cal_F05={calibration_scores[margin]:.9f}', flush=True)
    best = max(calibration_scores, key=lambda m: (calibration_scores[m], m))
    original_cal = float(vector(pred, truth, cal_ids).mean())
    retained = calibration_scores[best] > original_cal
    result = variants[best] if retained else pred
    hold = hold_info['entity_id']
    report = dict(status='Exploratory reused-validation experiment; no production changes.',
        baseline_full_validation=float(baseline.mean()), calibration_entities=cal_info.height,
        evaluation_entities=hold_info.height, orphan_losing_edges=orphan.height,
        true_orphan_losing_edges=int(orphan['y'].sum()),
        calibrated_candidate_proposals=len(candidates), calibration_baseline=original_cal,
        calibration_scores=calibration_scores, selected_margin=best if retained else None,
        retained_on_calibration=retained,
        proposed_variant_evaluation=compare(vector(pred, truth, hold), vector(variants[best], truth, hold), hold_info),
        selected_policy_evaluation=compare(vector(pred, truth, hold), vector(result, truth, hold), hold_info),
        limitations=['Existing model and baseline previously used validation.',
            'Calibration/evaluation grouped by lowercase raw name and country; not a new untouched geography holdout.',
            'Expected-F assumes independent candidates within each entity; this is greedy orphan reassignment, not global optimization.'])
    Path('stage2_orphan_experiment.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    assert expected_f(np.array([1., 0.]), {0}) == 1.
    assert expected_f(np.array([1., 0.]), set()) == 0.
    assert abs(expected_f(np.array([.8]), {0}) - .8) < 1e-12
    main()
