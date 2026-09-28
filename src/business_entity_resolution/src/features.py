"""Pairwise features for (S1 record, candidate S2/S3 record). Country-agnostic by design."""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

STR_COLS = ["name_clean", "name_core", "name_sorted", "name_concat", "legal", "addr_clean", "addr_nums",
            "house", "street", "city", "state", "postcode"]


def _cp(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _tokset(arr):
    return [set(x.split()) if x else set() for x in arr]


def _jacc(A, B):
    out = np.empty(len(A), np.float32)
    for i, (a, b) in enumerate(zip(A, B)):
        u = len(a | b)
        out[i] = len(a & b) / u if u else np.nan
    return out


def _contain(A, B):
    """|A∩B| / |A|"""
    out = np.empty(len(A), np.float32)
    for i, (a, b) in enumerate(zip(A, B)):
        out[i] = len(a & b) / len(a) if a else np.nan
    return out


def _num_feats(h1, h2):
    """House-number agreement: equal, one absent, prefix relation, |diff|, digit edit distance."""
    n = len(h1)
    eq = np.zeros(n, np.int8); absent = np.zeros(n, np.int8); pref = np.zeros(n, np.int8)
    diff = np.full(n, np.nan, np.float32)
    for i, (a, b) in enumerate(zip(h1, h2)):
        if not a or not b:
            absent[i] = 1 if (a or b) else 2
            continue
        if a == b:
            eq[i] = 1
        elif a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a):
            pref[i] = 1
        if len(a) < 10 and len(b) < 10:
            diff[i] = abs(int(a) - int(b))
    return eq, absent, pref, diff


def pair_features(L: pl.DataFrame, R: pl.DataFrame) -> pl.DataFrame:
    """L, R: aligned frames (row i of L pairs with row i of R) holding the normalised columns."""
    a = {c: L[c].to_list() for c in STR_COLS}
    b = {c: R[c].to_list() for c in STR_COLS}
    f = {}
    # --- name
    f["n_ratio"] = _cp(a["name_core"], b["name_core"], fuzz.ratio)
    f["n_partial"] = _cp(a["name_core"], b["name_core"], fuzz.partial_ratio)
    f["n_tsort"] = _cp(a["name_core"], b["name_core"], fuzz.token_sort_ratio)
    f["n_tset"] = _cp(a["name_core"], b["name_core"], fuzz.token_set_ratio)
    f["n_wratio"] = _cp(a["name_clean"], b["name_clean"], fuzz.WRatio)
    f["n_concat_jw"] = _cp(a["name_concat"], b["name_concat"], JaroWinkler.normalized_similarity)
    f["n_concat_lev"] = _cp(a["name_concat"], b["name_concat"], Levenshtein.normalized_similarity)
    f["n_concat_partial"] = _cp(a["name_concat"], b["name_concat"], fuzz.partial_ratio)
    ta, tb = _tokset(a["name_core"]), _tokset(b["name_core"])
    f["n_jacc"] = _jacc(ta, tb)
    f["n_cont_ab"] = _contain(ta, tb)
    f["n_cont_ba"] = _contain(tb, ta)
    f["n_len_a"] = np.array([len(x) for x in a["name_core"]], np.float32)
    f["n_len_b"] = np.array([len(x) for x in b["name_core"]], np.float32)
    f["n_ntok_diff"] = np.array([len(x) - len(y) for x, y in zip(ta, tb)], np.float32)
    f["n_first_eq"] = np.array([x.split(" ", 1)[0] == y.split(" ", 1)[0] for x, y in
                                zip(a["name_core"], b["name_core"])], np.int8)
    f["n_sorted_eq"] = np.array([x == y for x, y in zip(a["name_sorted"], b["name_sorted"])], np.int8)
    la, lb = _tokset(a["legal"]), _tokset(b["legal"])
    f["legal_state"] = np.array([0 if not x and not y else 1 if not x or not y else 2 if x == y
                                 else 3 if x & y else 4 for x, y in zip(la, lb)], np.int8)
    # --- address
    f["a_ratio"] = _cp(a["addr_clean"], b["addr_clean"], fuzz.ratio)
    f["a_tset"] = _cp(a["addr_clean"], b["addr_clean"], fuzz.token_set_ratio)
    f["a_tsort"] = _cp(a["addr_clean"], b["addr_clean"], fuzz.token_sort_ratio)
    f["a_partial"] = _cp(a["addr_clean"], b["addr_clean"], fuzz.partial_ratio)
    aa, ab = _tokset(a["addr_clean"]), _tokset(b["addr_clean"])
    f["a_jacc"] = _jacc(aa, ab)
    f["a_cont_ba"] = _contain(ab, aa)
    f["a_len_b"] = np.array([len(x) for x in b["addr_clean"]], np.float32)
    na, nb = _tokset(a["addr_nums"]), _tokset(b["addr_nums"])
    f["num_jacc"] = _jacc(na, nb)
    f["num_cont_ba"] = _contain(nb, na)
    eq, absent, pref, diff = _num_feats(a["house"], b["house"])
    f["h_eq"], f["h_absent"], f["h_pref"], f["h_diff"] = eq, absent, pref, diff
    f["h_in_nums"] = np.array([1 if x and x in y else 0 for x, y in zip(a["house"], nb)], np.int8)
    f["st_jw"] = _cp(a["street"], b["street"], JaroWinkler.normalized_similarity)
    f["st_tset"] = _cp(a["street"], b["street"], fuzz.token_set_ratio)
    f["st_empty"] = np.array([(not x) + 2 * (not y) for x, y in zip(a["street"], b["street"])], np.int8)
    f["city_tset"] = _cp(a["city"], b["city"], fuzz.token_set_ratio)
    f["city_partial"] = _cp(a["city"], b["city"], fuzz.partial_ratio)
    f["state_state"] = np.array([0 if not x or not y else 1 if x == y else 2
                                 for x, y in zip(a["state"], b["state"])], np.int8)
    f["pin_state"] = np.array([0 if not x or not y else 1 if x == y else 2
                               for x, y in zip(a["postcode"], b["postcode"])], np.int8)
    # --- cross field: name found inside the other side's address (landmark / shuffled fields)
    f["x_name_in_addr"] = _cp(b["name_core"], a["addr_clean"], fuzz.partial_ratio)
    f["is_domain_b"] = R["is_domain"].to_numpy().astype(np.int8)
    return pl.DataFrame(f)


FEATURE_COLS = None  # filled by train.py after the first build
