"""Regression for the actual 165-entity split failure and imbalanced states."""
from pathlib import Path
import sys
import unittest

import polars as pl

sys.path.insert(0,str(Path(__file__).parent))
from experiment import partition, calibration_folds


class PartitionTests(unittest.TestCase):
    def test_dominant_state_uses_names_and_balances_country(self):
        rows = []
        for country,counts in [('India',[1000,8,7,4,3]),('US',[500,300,200,150,90,3])]:
            for state,n in enumerate(counts):
                for i in range(n):
                    rows.append((f'{country}_{state}_{i}',country,str(state),f'{country}_business{i//2}_{state}'))
        info = pl.DataFrame(rows,schema=['entity_id','country','state','name_core'],orient='row')
        roles,groups = partition(info)
        self.assertTrue(all(g[1].startswith('name:') for s,g in groups.items() if g[0]=='India'))
        for country in ['India','US']:
            ids = info.filter(pl.col('country')==country)['entity_id']
            self.assertGreaterEqual(sum(roles[s]=='stop' for s in ids),.1*len(ids))
            self.assertGreaterEqual(sum(roles[s]=='cal' for s in ids),.1*len(ids))
            self.assertGreaterEqual(sum(roles[s]=='hold' for s in ids),.25*len(ids))
        group_roles = {}
        for s,g in groups.items():
            group_roles.setdefault(g,set()).add(roles[s])
        self.assertTrue(all(len(v)==1 for v in group_roles.values()))
        self.assertEqual((roles,groups),partition(info.reverse()))
        cal_ids = [s for s,r in roles.items() if r=='cal']
        folds = calibration_folds(info,cal_ids)
        names = {}
        for s,c,st,name in info.filter(pl.col('entity_id').is_in(cal_ids)).iter_rows():
            names.setdefault((c,name),set()).add(folds[s])
        self.assertTrue(all(len(v)==1 for v in names.values()))


if __name__=='__main__':
    unittest.main()
