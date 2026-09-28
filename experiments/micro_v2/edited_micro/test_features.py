"""Small regression checks for evidence leakage, ties, and metric accounting."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

spec = importlib.util.spec_from_file_location('micro',Path(__file__).with_name('features.py'))
micro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(micro)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.attrs = pl.DataFrame([
            ('s1','rare dental',False,1), ('s2','common dental',False,1),
            ('c1','rare dental',True,2), ('c2','rare dental',False,3),
            ('c3','common dental',False,2), ('c4','',True,3),
        ], schema=['entity_id','name_core','blank','source'], orient='row')
        self.weights = micro.fit_weights(self.attrs,pl.Series(['s1','s2']))
        self.pairs = pl.DataFrame([('s1','c1',.7),('s2','c1',.2),('s1','c2',.95),
                                  ('s2','c3',.9),('s1','c4',.1)],
                                 schema=['s1','cand','p'],orient='row')

    def test_missing_is_separate_and_no_self_support(self):
        single = self.pairs.head(1)
        d = micro.add_features(single,self.attrs,self.weights).row(0,named=True)
        self.assertEqual(d['ma_cand_blank'],1.)
        self.assertEqual(d['ma_s1_blank'],0.)
        self.assertEqual(d['ma_name_jacc'],1.)
        self.assertEqual(d['ma_peer_support'],0.)
        self.assertEqual(d['ma_peer_sources'],0.)

    def test_other_record_can_corroborate_blank_address(self):
        d = micro.add_features(self.pairs,self.attrs,self.weights)
        a = d.filter((pl.col('s1')=='s1') & (pl.col('cand')=='c1')).row(0,named=True)
        b = d.filter((pl.col('s1')=='s2') & (pl.col('cand')=='c1')).row(0,named=True)
        self.assertGreater(a['ma_peer_support'],.9)
        self.assertGreater(a['ma_name_margin'],0.)
        self.assertLess(b['ma_name_margin'],0.)
        self.assertTrue(np.isfinite(d.select(micro.EXTRA).to_numpy()).all())

    def test_order_and_label_invariance(self):
        a = micro.add_features(self.pairs,self.attrs,self.weights).select(micro.KEYS+micro.EXTRA).sort(micro.KEYS)
        b = micro.add_features(self.pairs.reverse().with_columns(pl.lit(1).alias('y')),
                               self.attrs.reverse(),self.weights).select(micro.KEYS+micro.EXTRA).sort(micro.KEYS)
        assert_frame_equal(a,b)

    def test_repeated_name_does_not_multiply_support(self):
        extra = pl.DataFrame([('c5','rare dental',False,3)],schema=self.attrs.schema,orient='row')
        pair = pl.DataFrame([('s1','c5',.94)],schema=self.pairs.schema,orient='row')
        def support(p,a):
            return micro.add_features(p,a,self.weights).filter((pl.col('s1')=='s1') & (pl.col('cand')=='c1'))['ma_peer_support'][0]
        self.assertEqual(support(self.pairs,self.attrs), support(pl.concat([self.pairs,pair]),pl.concat([self.attrs,extra])))

    def test_duplicate_pair_rejected(self):
        with self.assertRaises(ValueError):
            micro.add_features(pl.concat([self.pairs,self.pairs.head(1)]),self.attrs,self.weights)


if __name__ == '__main__':
    unittest.main()
