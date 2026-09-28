"""Candidate generation: union of top-k TF-IDF cosine blockers, partitioned by country and state.

Per country, each view turns a record into a string and vectorises it with char n-gram TF-IDF (vocabulary fit
on a sample of the S2+S3 pool). Search runs inside (country, state) partitions, which keeps the sparse
posting lists short:
  forward : every S1 record -> top-k pool records of its state            (per view)
  reverse : every pool record -> top-k S1 records of its state            (captures "which S1 owns me")
  orphan  : pool records with no resolvable state (mostly empty address) -> top-k S1 of the whole country,
            by name.
Missing pool states are inferred from the city using a city->state table learned from the records themselves.
"""
import gc
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from config import N_JOBS, N_PROCS, WORK_DIR

VIEWS = {
    # view name -> (polars expression building the text, vectoriser kwargs)
    "name": (lambda: pl.col("name_core"),
             dict(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.05)),
    "addr": (lambda: pl.col("addr_clean"),
             dict(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.01)),
    "name_addr": (lambda: pl.concat_str(["name_core", "house", "street", "city"], separator=" "),
                  dict(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.01)),
}
FORWARD_K = {"name": 10, "addr": 15, "name_addr": 20}
REVERSE_K = {"name_addr": 3}
ORPHAN_K = {"name": 5}

BLOCK_COLS = ["entity_id", "country", "name_core", "addr_clean", "house", "street", "city", "state"]
STATE_GROUP = {"tg": "ap", "dc": "wa"}     # pairs that legitimately disagree in the data


def view_text(df: pl.DataFrame, view: str) -> list:
    fn, _ = VIEWS[view]
    return df.select(fn()).to_series().to_list()


def _transform(args):
    vec, texts = args
    return vec.transform(texts)


def transform_df(vec, df: pl.DataFrame, view: str, workers: Pool, step: int = 50_000) -> sparse.csr_matrix:
    """Vectorise `df` in slices; texts are built per slice so the full list of strings never exists at once."""
    if len(df) == 0:
        return sparse.csr_matrix((0, len(vec.vocabulary_)), dtype=np.float32)
    parts = []
    starts = list(range(0, len(df), step))
    for b in range(0, len(starts), N_PROCS * 2):
        batch = [(vec, view_text(df.slice(i, step), view)) for i in starts[b:b + N_PROCS * 2]]
        parts.extend(workers.map(_transform, batch))
        del batch
    m = sparse.vstack(parts, format="csr") if len(parts) > 1 else parts[0].tocsr()
    del parts
    return m


def fit_vectorizer(df: pl.DataFrame, view: str, kw: dict, sample: int = 1_000_000, seed: int = 0):
    """Fit vocabulary + IDF on a random sample of records (fast)."""
    vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kw)
    vec.fit(view_text(df.sample(min(sample, len(df)), seed=seed), view))
    return vec


def topk(query: sparse.csr_matrix, index: sparse.csr_matrix, k: int, chunk: int = 50000):
    """(query_row, index_row, cosine) of the top-k index rows for each query row."""
    if query.shape[0] == 0 or index.shape[0] == 0:
        return (np.empty(0, np.int32),) * 2 + (np.empty(0, np.float32),)
    index_T = index.T.tocsr()
    rows, cols, vals = [], [], []
    for s in range(0, query.shape[0], chunk):
        m = sp_matmul_topn(query[s:s + chunk], index_T, top_n=k, threshold=0.05,
                           n_threads=N_JOBS, sort=True).tocoo()
        rows.append(m.row.astype(np.int32) + s)
        cols.append(m.col.astype(np.int32))
        vals.append(m.data.astype(np.float32))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def state_keys(s1: pl.DataFrame, pool: pl.DataFrame):
    """Partition key per record: grouped state; pool records without state get it from their city."""
    grp = lambda c: pl.col(c).replace(STATE_GROUP)
    s1k = s1.select(grp("state").alias("k"))["k"]
    known = pl.concat([s1.select("city", grp("state").alias("k")),
                       pool.select("city", grp("state").alias("k"))]).filter(
        (pl.col("k") != "") & (pl.col("city") != ""))
    city_map = (known.group_by("city", "k").len()
                     .with_columns((pl.col("len") / pl.col("len").sum().over("city")).alias("share"),
                                   pl.col("len").sum().over("city").alias("n"))
                     .filter((pl.col("share") >= 0.8) & (pl.col("n") >= 3))
                     .select("city", pl.col("k").alias("k_city")))
    pk = (pool.select("city", grp("state").alias("k")).with_row_index("i")
              .join(city_map, on="city", how="left")
              .sort("i")
              .select(pl.when(pl.col("k") != "").then(pl.col("k"))
                        .otherwise(pl.col("k_city").fill_null("?")).alias("k")))["k"]
    return s1k.to_numpy(), pk.to_numpy()


def _collect(out: dict, view: str, direction: str, q_idx, p_idx, r, c, v, s1_is_query: bool):
    if len(r) == 0:
        return
    s1_rows = q_idx[r] if s1_is_query else p_idx[c]
    pool_rows = p_idx[c] if s1_is_query else q_idx[r]
    out.setdefault((view, direction), []).append(
        pl.DataFrame({"q": s1_rows.astype(np.int32), "c": pool_rows.astype(np.int32), "sim": v}))


def run_country(s1: pl.DataFrame, pool: pl.DataFrame, workers=None, keep_s1: np.ndarray = None,
                forward_k=None, reverse_k=None, orphan_k=None, log=print) -> pl.DataFrame:
    """All blockers for one country. Returns unique (q = s1 row, c = pool row) with per-blocker sims/ranks.
    keep_s1: optional boolean mask of S1 rows we need candidates for (all S1 rows still act as competitors)."""
    forward_k = FORWARD_K if forward_k is None else forward_k
    reverse_k = REVERSE_K if reverse_k is None else reverse_k
    orphan_k = ORPHAN_K if orphan_k is None else orphan_k
    keep_s1 = np.ones(len(s1), bool) if keep_s1 is None else keep_s1
    sk, pk = state_keys(s1, pool)
    parts_by_key = {k: (np.where((sk == k) & keep_s1)[0], np.where(sk == k)[0], np.where(pk == k)[0])
                    for k in np.unique(sk)}
    orphans = np.where(pk == "?")[0]
    log(f"    partitions {len(parts_by_key)}, orphan pool records {len(orphans):,} "
        f"({len(orphans) / max(len(pool), 1):.1%})")
    tmp = WORK_DIR / "tmp_blocking"
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.parquet"):
        f.unlink()
    spilled = []
    own = workers is None
    workers = workers or Pool(N_PROCS, maxtasksperchild=40)
    for view in sorted(set(forward_k) | set(reverse_k) | set(orphan_k)):
        t0 = time.time()
        out = {}
        vec = fit_vectorizer(pool, view, VIEWS[view][1])
        A = transform_df(vec, s1, view, workers)          # all S1 of the country (small vs pool)
        for k, (qi, si, pi) in parts_by_key.items():      # pool vectorised one partition at a time
            if len(pi) == 0 or len(qi) == 0:             # no query S1 here: nothing in it can be kept
                continue
            Bp = transform_df(vec, pool[pi], view, workers)
            if view in forward_k and len(qi):
                r, c, v = topk(A[qi], Bp, forward_k[view])
                _collect(out, view, "f", qi, pi, r, c, v, True)
            if view in reverse_k:
                r, c, v = topk(Bp, A[si], reverse_k[view])
                _collect(out, view, "r", pi, si, r, c, v, False)
            del Bp
        if view in orphan_k and len(orphans):
            Bo = transform_df(vec, pool[orphans], view, workers)
            r, c, v = topk(Bo, A, orphan_k[view])
            _collect(out, view, "o", orphans, np.arange(len(s1)), r, c, v, False)
            del Bo
        del A, vec
        # spill this view's pairs to disk (ranks computed before dropping non-query S1 rows)
        for (vw, d), lst in out.items():
            f = pl.concat(lst)
            col = f"{d}_{vw}"
            rank_over = "q" if d == "f" else "c"
            f = (f.with_columns(pl.col("sim").rank("ordinal", descending=True).over(rank_over).cast(pl.Int16)
                                .alias(f"rk_{col}"))
                  .rename({"sim": f"sim_{col}"})
                  .unique(subset=["q", "c"], keep="first"))
            if not keep_s1.all():
                f = f.filter(pl.Series(keep_s1[f["q"].to_numpy()]))
            path = tmp / f"{col}.parquet"
            f.write_parquet(path)
            spilled.append(path)
        del out
        gc.collect()
        log(f"    view {view:9s} done in {time.time() - t0:.0f}s")
    if own:
        workers.terminate()
    # streaming merge: one row per (q, c) with every blocker's sim/rank (null when that blocker missed it)
    merged = (pl.concat([pl.scan_parquet(f) for f in spilled], how="diagonal")
                .group_by("q", "c").agg(pl.all().max())
                .collect(engine="streaming"))
    for f in spilled:
        f.unlink()
    return merged
