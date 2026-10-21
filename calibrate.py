"""Tune per-class logit offsets to maximise the official weighted F1.

Offsets are added to the classifier bias at export, so no custom code is needed at inference.
Honest gain is estimated with 2-fold cross-fitting before fitting on all validation clips.
Usage: python calibrate.py runs/E01/val_logits_best.npz [more.npz ...]  (several = averaged ensemble view)
"""
import json
import sys

import numpy as np

from common import LABELS, weighted_f1


def fit(logits, y, rounds=4, grid=np.linspace(-3, 3, 61), l2=0.02):
    b = np.zeros(logits.shape[1])
    for _ in range(rounds):
        for c in range(len(b)):
            best = (-1, 0)
            for g in grid:
                b2 = b.copy(); b2[c] = g
                s = weighted_f1(y, (logits + b2).argmax(1))[0] - l2 * abs(g) / 10
                if s > best[0]:
                    best = (s, g)
            b[c] = best[1]
    return b


def main(paths):
    ds = [np.load(p) for p in paths]
    logits = np.mean([d["logits"] - d["logits"].max(1, keepdims=True) for d in ds], 0)
    lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    y = ds[0]["y"]
    base = weighted_f1(y, lp.argmax(1))[0]
    rng = np.random.default_rng(0)
    gains = []
    for rep in range(3):
        idx = rng.permutation(len(y)); a, b = idx[: len(y) // 2], idx[len(y) // 2:]
        for tr, te in [(a, b), (b, a)]:
            off = fit(lp[tr], y[tr])
            gains.append(weighted_f1(y[te], (lp[te] + off).argmax(1))[0] - weighted_f1(y[te], lp[te].argmax(1))[0])
    off = fit(lp, y)
    tuned, per = weighted_f1(y, (lp + off).argmax(1))
    print(json.dumps({"base_wf1": round(base, 4), "fit_wf1_in_sample": round(tuned, 4),
                      "cv_gain_mean": round(float(np.mean(gains)), 4), "cv_gain_std": round(float(np.std(gains)), 4),
                      "offsets": {l: round(float(o), 2) for l, o in zip(LABELS, off)},
                      "per_class": {l: round(float(f), 3) for l, f in zip(LABELS, per)}}, indent=1))
    np.save(paths[0].replace(".npz", "_offsets.npy"), off)
    # confusion matrix for error analysis
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, (lp + off).argmax(1), labels=range(len(LABELS)))
    print("      " + " ".join(f"{l:>5}" for l in LABELS))
    for l, r in zip(LABELS, cm):
        print(f"{l:>5} " + " ".join(f"{v:5d}" for v in r))


if __name__ == "__main__":
    main(sys.argv[1:])
