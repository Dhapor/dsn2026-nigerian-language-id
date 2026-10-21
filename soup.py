"""Model soup: average the weights of fine-tuned checkpoints that share one base model, optionally score it.

Build:  python soup.py build runs/S01 runs/F05/last runs/F07/last       (then export.py runs/S01/last export/S01)
Score:  python soup.py eval out.npz runs/A01/best runs/A03/best         (validation split; one path = no averaging)
The validation split has no Nupe clips, so compare soups only with single models scored the same way.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from common import LABELS, SR, Split, weighted_f1


def average(paths):
    ms = [AutoModelForAudioClassification.from_pretrained(p, dtype=torch.float32) for p in paths]
    sds = [m.state_dict() for m in ms]
    assert all(sd.keys() == sds[0].keys() for sd in sds), "checkpoints must share one architecture"
    ms[0].load_state_dict({k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]})
    return ms[0]


def build(out_dir, paths):
    os.makedirs(out_dir, exist_ok=True)
    average(paths).save_pretrained(os.path.join(out_dir, "last"), safe_serialization=True)
    AutoFeatureExtractor.from_pretrained(paths[-1]).save_pretrained(os.path.join(out_dir, "last"))
    # publish.py reads the training args; take the last member's and record the soup
    args = json.load(open(os.path.join(os.path.dirname(paths[-1]), "args.json")))
    args.update(exp=os.path.basename(out_dir), soup=paths)
    json.dump(args, open(os.path.join(out_dir, "args.json"), "w"), indent=1)
    print("soup of", paths, "->", out_dir)


def evaluate(out, paths):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = average(paths).to(dev).eval()
    fe = AutoFeatureExtractor.from_pretrained(paths[0])
    va = Split("validation")
    logits, t0 = [], time.time()
    with torch.no_grad():
        for i in range(len(va)):
            n = min(va.lengths[i], 12 * SR)
            n = n // SR * SR if n > SR else n  # whole seconds: few input shapes on MPS
            inp = {k: v.to(dev) for k, v in fe([va.wav(i, 0, n)], sampling_rate=SR, return_tensors="pt").items()}
            logits.append(m(**inp).logits.float().cpu().numpy()[0])
            if dev.type == "mps" and i % 500 == 0:
                torch.mps.empty_cache()
    logits = np.stack(logits)
    f1, per = weighted_f1(va.labels, logits.argmax(1))
    np.savez(out, logits=logits, y=va.labels, src=np.array(va.sources))
    print(paths, "wf1", round(f1, 4), {LABELS[c]: round(float(per[c]), 3) for c in range(len(LABELS))},
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmd, out, *paths = sys.argv[1:]
    {"build": build, "eval": evaluate}[cmd](out, paths)
