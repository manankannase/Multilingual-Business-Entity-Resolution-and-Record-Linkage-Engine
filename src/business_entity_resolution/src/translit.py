"""Learn an Indic-script -> Latin token dictionary from the TRAINING ground truth only.

Names: Indic-script S2/S3 names are word-by-word renderings of the S1 name (token counts agree in >99.9% of
pairs), so aligned tokens give a direct dictionary. Addresses: non-Latin address tokens (mostly state / city
names) are mapped to the S1 address token they co-occur with most, normalised by that token's frequency.
"""
import json
import math
import re
from collections import Counter, defaultdict

import polars as pl

from config import WORK_DIR
from data import load_ground_truth, load_source

DICT_PATH = WORK_DIR / "indic_dict.json"
_SPLIT = re.compile(r"[\s,.;:()\-/]+")


def _toks(s: str):
    return [t for t in _SPLIT.split(s) if t]


def _nonlatin(t: str) -> bool:
    return any(ord(c) >= 0x250 for c in t)


def learn(min_count: int = 2) -> dict:
    gt = load_ground_truth().drop_nulls()
    s1 = load_source("train", 1).select("entity_id", "business_name", "business_address")
    other = pl.concat([load_source("train", 2), load_source("train", 3)]).select(
        pl.col("entity_id").alias("match"), pl.col("business_name").alias("n2"),
        pl.col("business_address").alias("a2"))
    pairs = (gt.join(s1, left_on="source1_entity_id", right_on="entity_id")
               .join(other, on="match"))
    nl = pairs.filter(pl.col("n2").str.contains(r"[^\x00-\x7F]") | pl.col("a2").str.contains(r"[^\x00-\x7F]"))

    name_cnt = defaultdict(Counter)
    co = defaultdict(Counter)
    s1_tok_freq = Counter()
    for n1, a1, n2, a2 in nl.select("business_name", "business_address", "n2", "a2").iter_rows():
        t1, t2 = n1.lower().split(), n2.split()
        if len(t1) == len(t2):
            for x, y in zip(t1, t2):
                if _nonlatin(y) and not _nonlatin(x):
                    name_cnt[y][x] += 1
        nl_addr = {t for t in _toks(a2) if _nonlatin(t)}
        if nl_addr:
            s1_toks = set(t.lower() for t in _toks(a1) if not t.isdigit())
            s1_tok_freq.update(s1_toks)
            for y in nl_addr:
                co[y].update(s1_toks)

    d = {}
    for y, c in co.items():
        n_y = sum(1 for _ in c) and max(c.values())
        best, score = None, 0.0
        for x, k in c.items():
            if k < min_count:
                continue
            sc = k / math.sqrt(s1_tok_freq[x])
            if sc > score:
                best, score = x, sc
        if best and n_y >= min_count:
            d[y] = best
    for y, c in name_cnt.items():          # name alignment is more reliable -> overrides
        x, k = c.most_common(1)[0]
        if k >= min_count:
            d[y] = x
    DICT_PATH.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return d


def load() -> dict:
    if DICT_PATH.exists():
        return json.loads(DICT_PATH.read_text(encoding="utf-8"))
    return learn()


if __name__ == "__main__":
    d = learn()
    print("entries", len(d))
    for k in list(d)[:15]:
        print(k, "->", d[k])
    for k in ["राजस्थान", "ગુજરાત", "दिल्ली", "प्राइवेट", "लिमिटेड"]:
        print(k, "->", d.get(k))
