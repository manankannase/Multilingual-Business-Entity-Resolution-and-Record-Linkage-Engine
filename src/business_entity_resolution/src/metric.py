"""Macro F0.5 per Source-1 entity, exactly as described in the challenge README."""
from typing import Dict, Iterable, Set


def entity_f05(pred: Set[str], true: Set[str], beta: float = 0.5) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(pred: Dict[str, Set[str]], true: Dict[str, Set[str]],
              ids: Iterable[str] = None) -> float:
    """Average entity_f05 over `ids` (default: all S1 ids in `true`)."""
    ids = list(true.keys() if ids is None else ids)
    return sum(entity_f05(pred.get(i, set()), true.get(i, set())) for i in ids) / max(len(ids), 1)


if __name__ == "__main__":
    s = entity_f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(s - 0.714) < 1e-3, s
    assert entity_f05(set(), set()) == 1.0 and entity_f05({"a"}, set()) == 0.0
    print("metric ok", round(s, 4))
