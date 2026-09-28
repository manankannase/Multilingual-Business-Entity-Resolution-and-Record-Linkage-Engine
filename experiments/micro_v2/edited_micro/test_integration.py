"""Exercise real pipeline integration on temporary synthetic caches, not real data."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import polars as pl


def main():
    code = Path(sys.argv[1]).resolve()
    if (code/'src').exists():
        code = code/'src'
    root = Path(tempfile.mkdtemp(prefix='edited-micro-smoke-'))
    base, out = root/'base', root/'result'
    for folder in ['norm','raw','feat','feat/test_shards','xenc','cand']:
        (base/folder).mkdir(parents=True,exist_ok=True)
    os.environ.update(BER_WORK_DIR=str(base),BER_OUT_DIR=str(out/'output'),BER_N_JOBS='2',BER_P_MIN='0.01')
    import experiment as ex
    sys.path.insert(0,str(code))
    import stage2 as s2
    import lightgbm as lgb
    original_train = lgb.train

    def short_train(params, data, num_boost_round=100, **kwargs):
        params = dict(params,min_data_in_leaf=5,num_leaves=7)
        return original_train(params,data,min(num_boost_round,20),**kwargs)
    lgb.train = short_train
    rng = np.random.default_rng(5)
    records = {s:[] for s in (1,2,3)}
    parts, ground = {}, []

    def make(name, entities):
        rows = []
        for i in range(entities):
            sid = f'{name}_s{i}'
            country = 'India' if i % 2 == 0 else 'US'
            state = f'{country}{(i//2) % (5 if country=="India" else 6)}'
            entity_name = f'company{i} services'
            addr = f'{100+i} main road'
            records[1].append((sid,entity_name,addr,country,state,entity_name,addr,str(100+i)))
            for j in range(4):
                cid = f'{name}_c{i}_{j}'
                y = int(j<2)
                c_name = entity_name if y else f'decoy{i}_{j} services'
                c_addr = '' if j%2 else addr
                source = 2 if j%2==0 else 3
                records[source].append((cid,c_name,c_addr,country,state,c_name,c_addr,'' if not c_addr else str(100+i)))
                p = float(np.clip((.8 if y else .2)+rng.normal(0,.15),.02,.98))
                rows.append((sid,cid,y,p))
                if y and name!='test':
                    ground.append((sid,cid))
        return pl.DataFrame(rows,schema=['s1','cand','y','p'],orient='row').with_columns(pl.col('y').cast(pl.Int8),pl.col('p').cast(pl.Float32))

    for name,n in [('train',80),('val',220),('test',30)]:
        parts[name] = make(name,n)
    schema = ['entity_id','business_name','business_address','country','state','name_core','addr_clean','house']
    for source in records:
        frame = pl.DataFrame(records[source],schema=schema,orient='row')
        for split in ['train','test']:
            mask = pl.col('entity_id').str.starts_with('test_')
            frame.filter(mask if split=='test' else ~mask).write_parquet(base/'norm'/f'{split}_source{source}.parquet')
    pl.DataFrame(ground,schema=['source1_entity_id','match'],orient='row').write_parquet(base/'raw/train_ground_truth_long.parquet')
    cols = ['f']+['p']+s2.CTX_COLS+s2.GROUP_COLS+s2.tag_cols(['xenc'])
    (base/'lgb_s2.json').write_text(json.dumps({'cols':cols,'stage1_cols':['f'],'tags':['xenc'],'n_models':1}))
    for name,d in parts.items():
        d.select('s1','cand',pl.col('p').alias('xenc')).write_parquet(base/'xenc'/f'xenc_{name}.parquet')
        f = d.select('s1','cand',pl.col('p').alias('f'))
        if name=='test':
            d.drop('y').write_parquet(base/'test_scored.parquet')
            f.write_parquet(base/'feat/test_shards/0.parquet')
        else:
            f.write_parquet(base/'feat'/f'{name}.parquet')
            (d.with_columns(pl.lit(0,pl.Int8).alias('fold')) if name=='train' else d).write_parquet(base/f's2_{name}_p.parquet')
    val_ids = parts['val']['s1'].unique().sort().to_list()
    s2.s1_subset = lambda _: {'val':pl.Series(val_ids)}
    parts['val'].select('s1','cand').write_parquet(base/'cand/val.parquet')
    parts['val'].filter(pl.col('y')==1).select('s1','cand').write_parquet(base/'val_pred_s2.parquet')
    parts['val'].with_columns(pl.col('y').cast(pl.Float32).alias('p2')).write_parquet(base/'val_scored_s2.parquet')
    matrix = rng.random((80,len(cols)))
    original_train(dict(objective='binary',num_threads=2,verbose=-1,min_data_in_leaf=5),
        lgb.Dataset(matrix,(matrix[:,0]>.5).astype(int),feature_name=cols),20).save_model(str(base/'lgb_s2_0.txt'))
    hashes = {str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob('*') if p.is_file()}
    common = ['--code',str(code),'--base-work',str(base),'--out',str(out)]
    sys.argv = ['experiment.py','preflight',*common]
    ex.main()
    assert not (out/'RUN_STARTED.json').exists()
    sys.argv = ['experiment.py','fit',*common,'--seeds','3']
    ex.main()
    report = json.loads((out/'comparison.json').read_text())
    assert not report['any_candidate_passes_comparison']
    assert set(report['entities'])=={'stop','cal','hold'}
    assert sum(report['entities'].values())==220
    assert len(report['candidates'])==5
    assert report['recommended']=='retain_verified_baseline'
    sys.argv = ['experiment.py','predict',*common]
    ex.main()
    ids = set(parts['test']['cand'])
    assert len(list((out/'output').glob('matching_results_*.tsv')))==5
    for name in ex.CANDIDATES:
        final = pl.read_csv(out/'output'/f'matching_results_{name}.tsv',separator='\t',infer_schema=False)
        assert final.height==30
        assigned = []
        for matches in final['matched_entity_ids']:
            assigned.extend((matches or '').split(',') if matches else [])
        assert set(assigned).issubset(ids)
        assert len(assigned)==len(set(assigned))
    assert hashes=={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob('*') if p.is_file()}
    try:
        ex.main()
        raise AssertionError('Existing prediction overwritten')
    except FileExistsError:
        pass
    sys.argv = ['experiment.py','fit',*common,'--seeds','3']
    try:
        ex.main()
        raise AssertionError('Existing fit overwritten')
    except FileExistsError:
        pass
    print('INTEGRATION_PASS: preflight, train, calibrate, compare, five TSVs, unique ownership, baseline hashes, overwrite guards')
    print('Temporary test artifacts: '+str(root))


if __name__=='__main__':
    main()
