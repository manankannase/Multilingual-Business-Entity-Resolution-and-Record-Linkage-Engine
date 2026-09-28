"""Turn scored candidate pairs into final match lists and write the submission TSVs."""
import numpy as np
import polars as pl

from metric import entity_f05


def one_to_one(pairs: pl.DataFrame, score: str = "p") -> pl.DataFrame:
    """Each S2/S3 record belongs to at most one S1: keep only the pair where it scores highest."""
    return (pairs.sort(score, descending=True)
                 .unique(subset="cand", keep="first", maintain_order=False))


def select(pairs: pl.DataFrame, thr: float, score: str = "p", o2o: bool = True) -> pl.DataFrame:
    kept = one_to_one(pairs, score) if o2o else pairs
    return kept.filter(pl.col(score) >= thr).select("s1", "cand")


def evaluate(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids) -> float:
    """pred/truth: (s1, cand) pairs; truth may contain null cand for singletons."""
    P, T = {}, {}
    for s, c in pred.iter_rows():
        P.setdefault(s, set()).add(c)
    for s, c in truth.iter_rows():
        st = T.setdefault(s, set())
        if c is not None:
            st.add(c)
    ids = list(s1_ids)
    return float(np.mean([entity_f05(P.get(i, set()), T.get(i, set())) for i in ids]))


def sweep(pairs: pl.DataFrame, truth: pl.DataFrame, s1_ids, grid=None, score: str = "p", o2o: bool = True):
    grid = grid if grid is not None else np.round(np.arange(0.30, 0.96, 0.05), 2)
    kept = one_to_one(pairs, score) if o2o else pairs
    res = [(t, evaluate(kept.filter(pl.col(score) >= t).select("s1", "cand"), truth, s1_ids)) for t in grid]
    best = max(res, key=lambda x: x[1])
    return best, res


def write_lists(s1_ids, pairs: pl.DataFrame, path, id_col: str):
    """One row per S1 id, comma-joined unique candidate ids (empty when none)."""
    agg = pairs.group_by("s1").agg(pl.col("cand").unique().sort().str.join(",").alias(id_col))
    out = (pl.DataFrame({"source1_entity_id": list(s1_ids)})
             .join(agg.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left")
             .with_columns(pl.col(id_col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")
    return out
