"""Export a checkpoint as an fp16, rule-compliant HF repo folder; optionally bake in calibration offsets.

Usage: python export.py runs/E01/best export/E01 [--offsets runs/E01/val_logits_best_offsets.npy]
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from common import LABELS

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst")
p.add_argument("--offsets")
p.add_argument("--fp32", action="store_true")
a = p.parse_args()

model = AutoModelForAudioClassification.from_pretrained(a.src)
fe = AutoFeatureExtractor.from_pretrained(a.src)
if a.offsets:
    off = torch.tensor(np.load(a.offsets), dtype=torch.float32)
    model.classifier.bias.data += off
    print("baked offsets", off.numpy().round(2))
if not a.fp32:
    model = model.half()
    model.config.torch_dtype = "float16"
shutil.rmtree(a.dst, ignore_errors=True)
model.save_pretrained(a.dst, safe_serialization=True)
fe.save_pretrained(a.dst)

# ---- compliance checks (mirror the Submission page)
cfg = json.load(open(os.path.join(a.dst, "config.json")))
assert [cfg["id2label"][str(i)] for i in range(13)] == LABELS, cfg["id2label"]
assert "auto_map" not in cfg
bad = [f for f in os.listdir(a.dst) if f.endswith((".py", ".bin", ".pt", ".pkl", ".pth"))]
assert not bad, bad
size = sum(os.path.getsize(os.path.join(a.dst, f)) for f in os.listdir(a.dst))
n = sum(p.numel() for p in model.parameters())
print(f"OK {a.dst}: files={sorted(os.listdir(a.dst))} params={n/1e6:.1f}M size={size/1e9:.3f}GB")
