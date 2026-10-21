"""Pack an external (openly licensed, non-prohibited) speech dataset into the same memmap format as prep.py.

Usage: python prep_ext.py data/ext/efik_speech_dataset ext_efik efi --max_h 3
       python prep_ext.py data/ext/cv_ibb/cv-corpus-27.0-2026-09-11/ibb ext_cv_ibb ibb --max_h 4 --per_spk 100   (Common Voice folder)
Writes data/<name>.i16 + data/<name>.json; train with --ext <name>. Credit the source in the model card.
"""
import argparse
import glob
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from common import DATA, SR

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("name"); p.add_argument("label")
p.add_argument("--max_h", type=float, default=3.0, help="cap, so one external corpus cannot dominate its class")
p.add_argument("--max_s", type=float, default=30.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--per_spk", type=int, default=0, help="max clips per speaker (0 = no cap)")
a = p.parse_args()

rows = []
if os.path.exists(os.path.join(a.src, "validated.tsv")):  # Common Voice: validated + not-yet-voted clips, never invalidated
    import csv
    for t in ["validated.tsv", "other.tsv"]:
        for r in csv.DictReader(open(os.path.join(a.src, t)), delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append({"path": os.path.join(a.src, "clips", r["path"]), "speaker_id": r["client_id"][:12]})
else:
    for f in sorted(glob.glob(os.path.join(a.src, "data", "*.parquet"))):
        t = pq.read_table(f, columns=[c for c in ["audio", "speaker_id"] if c in pq.read_schema(f).names])
        rows += t.to_pylist()
rng = np.random.default_rng(a.seed)
rng.shuffle(rows)
if a.per_spk:
    from collections import Counter
    seen, kept = Counter(), []
    for r in rows:
        if seen[r.get("speaker_id", 0)] < a.per_spk:
            seen[r.get("speaker_id", 0)] += 1; kept.append(r)
    rows = kept

meta = {k: [] for k in ["id", "label", "speaker_id", "source", "duration", "offset", "length"]}
out = open(os.path.join(DATA, f"{a.name}.i16"), "wb")
off = 0
for k, row in enumerate(rows):
    if off >= a.max_h * 3600 * SR:
        break
    src = row["path"] if "path" in row else io.BytesIO(row["audio"]["bytes"])
    wav, sr = sf.read(src, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(SR, sr)
        wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
    wav = wav[: int(a.max_s * SR)]
    if len(wav) < SR:
        continue
    pcm = np.clip(wav * 32768, -32768, 32767).astype(np.int16)
    out.write(pcm.tobytes())
    meta["id"].append(f"{a.name}_{k}")
    meta["label"].append(a.label)
    meta["speaker_id"].append(f"{a.name}_{row.get('speaker_id', 0)}")
    meta["source"].append(a.name)
    meta["duration"].append(len(pcm) / SR)
    meta["offset"].append(off)
    meta["length"].append(len(pcm))
    off += len(pcm)
out.close()
json.dump(meta, open(os.path.join(DATA, f"{a.name}.json"), "w"))
print(a.name, "clips", len(meta["id"]), "hours", round(off / SR / 3600, 2), "speakers", sorted(set(meta["speaker_id"])))
