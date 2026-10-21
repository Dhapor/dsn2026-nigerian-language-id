"""Estimate the efficiency score of an exported folder the way the organisers describe it:
load with Auto* classes, warm up, then time per clip over real validation clips (median of 3 passes).
Run in a fresh process so peak RSS is meaningful.
Usage: python bench.py export/E01 [--device cpu|mps] [--n 200]
"""
import argparse
import json
import os
import resource
import time

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from common import Split, efficiency, SR

p = argparse.ArgumentParser()
p.add_argument("path"); p.add_argument("--device", default="cpu"); p.add_argument("--n", type=int, default=200)
p.add_argument("--threads", type=int, default=0)
a = p.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)

va = Split("validation")
rng = np.random.default_rng(0)
idx = rng.choice(len(va), size=min(a.n, len(va)), replace=False)
clips = [va.wav(int(i), 0, min(int(va.lengths[i]), 30 * SR)) for i in idx]

dev = torch.device(a.device)
fe = AutoFeatureExtractor.from_pretrained(a.path)
model = AutoModelForAudioClassification.from_pretrained(a.path).to(dev).eval()
dtype = next(model.parameters()).dtype
if dev.type == "cpu" and dtype == torch.float16:  # what a CPU evaluator would likely have to do
    model = model.float(); dtype = torch.float32


def run():
    t0 = time.perf_counter()
    with torch.inference_mode():
        for x in clips:
            inp = fe(x, sampling_rate=SR, return_tensors="pt")
            inp = {k: v.to(dev, dtype) if v.is_floating_point() else v.to(dev) for k, v in inp.items()}
            model(**inp).logits.argmax(-1).item()
    if dev.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / len(clips)


run()  # warm-up
t = float(np.median([run() for _ in range(3)]))
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
params = sum(p.numel() for p in model.parameters())
repo = sum(os.path.getsize(os.path.join(a.path, f)) for f in os.listdir(a.path))
eff, parts = efficiency(params, repo, peak, t)
print(json.dumps({"path": a.path, "device": a.device, "params_M": round(params / 1e6, 1), "repo_GB": round(repo / 1e9, 3),
                  "peak_GB": round(peak / 1e9, 2), "sec_per_clip": round(t, 4), "avg_clip_s": round(np.mean([len(c) for c in clips]) / SR, 1),
                  "efficiency": round(eff, 3), **{k: round(v, 3) for k, v in parts.items()}}))
