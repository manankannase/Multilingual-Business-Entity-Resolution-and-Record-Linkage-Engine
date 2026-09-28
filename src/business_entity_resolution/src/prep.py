"""Normalise every source file once (multiprocessing) and cache as parquet in WORK_DIR/norm."""
import sys
import time
from multiprocessing import Pool

import polars as pl

import text
import translit
from config import N_JOBS, WORK_DIR


def scan_norm(split: str, src: int, cols=None, country: str = None) -> pl.DataFrame:
    """Read only the needed columns / country rows of a normalised file (memory-lean)."""
    lf = pl.scan_parquet(WORK_DIR / "norm" / f"{split}_source{src}.parquet")
    if country is not None:
        lf = lf.filter(pl.col("country") == country)
    if cols is not None:
        lf = lf.select(cols)
    return lf.collect()
from data import load_source

NAME_KEYS = ["name_clean", "name_core", "name_sorted", "name_concat", "legal", "is_domain"]
ADDR_KEYS = ["addr_clean", "addr_nums", "house", "street", "city", "state", "postcode"]
SCHEMA = {k: (pl.Int8 if k == "is_domain" else pl.Utf8) for k in NAME_KEYS + ADDR_KEYS}


def _init(d):
    text.set_indic_dict(d)


def _work(chunk):
    names, addrs, countries = chunk
    cols = {k: [] for k in NAME_KEYS + ADDR_KEYS}
    for n, a, c in zip(names, addrs, countries):
        r = text.norm_name(n)
        r.update(text.norm_address(a, c))
        for k, v in r.items():
            cols[k].append(v)
    return pl.DataFrame(cols, schema=SCHEMA)   # compact columnar result (dict rows cost GBs)


def normalise(split: str, src: int, force: bool = False) -> pl.DataFrame:
    out = WORK_DIR / "norm" / f"{split}_source{src}.parquet"
    if out.exists() and not force:
        return pl.read_parquet(out)
    t0 = time.time()
    df = load_source(split, src)
    names, addrs, ctry = (df[c].to_list() for c in ("business_name", "business_address", "country"))
    step = 20000
    chunks = [(names[i:i + step], addrs[i:i + step], ctry[i:i + step]) for i in range(0, len(names), step)]
    with Pool(N_JOBS, initializer=_init, initargs=(translit.load(),)) as pool:
        feats = pl.concat(list(pool.imap(_work, chunks, chunksize=1)))
    del chunks, names, addrs, ctry
    res = pl.concat([df, feats], how="horizontal")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.write_parquet(out)
    print(f"normalised {split} s{src}: {len(res):,} rows in {time.time() - t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    splits = sys.argv[1:] or ["train", "test"]
    for sp in splits:
        for s in (1, 2, 3):
            normalise(sp, s, force=True)
