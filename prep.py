"""Download the parquet shards and pack audio into int16 memmaps for fast random crops."""
import glob
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download

from common import DATA, SR

os.makedirs(DATA, exist_ok=True)
src = snapshot_download("Data-Science-Nigeria/dsn-2026-lid-train", repo_type="dataset",
                        local_dir=os.path.join(DATA, "raw"))

for split in ["train", "validation"]:
    files = sorted(glob.glob(os.path.join(src, "data", f"{split}-*.parquet")))
    meta = {k: [] for k in ["id", "label", "speaker_id", "source", "duration", "offset", "length"]}
    out = open(os.path.join(DATA, f"{split}.i16"), "wb")
    off = 0
    for f in files:
        t = pq.read_table(f)
        cols = t.column_names
        for row in t.to_pylist():
            a = row["audio"]
            wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
            wav = wav.mean(1)
            if sr != SR:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            pcm = np.clip(wav * 32768, -32768, 32767).astype(np.int16)
            out.write(pcm.tobytes())
            for k in ["id", "label", "speaker_id", "source", "duration"]:
                meta[k].append(row.get(k) if k in cols else None)
            meta["offset"].append(off)
            meta["length"].append(len(pcm))
            off += len(pcm)
        print(split, f, len(meta["id"]), flush=True)
    out.close()
    meta["speaker_id"] = [str(s) for s in meta["speaker_id"]]
    json.dump(meta, open(os.path.join(DATA, f"{split}.json"), "w"))
    print(split, "clips", len(meta["id"]), "hours", off / SR / 3600)
