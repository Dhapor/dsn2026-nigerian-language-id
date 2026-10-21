"""Add model card + license to an export folder, push it to a NEW public HF repo, write submission.zip.

A new repo per submission, because a submitted repo must never change afterwards.
Usage: python publish.py export/W01 dsn2026-lid-w01 --exp W01 [--dry]
"""
import argparse
import json
import os
import shutil
import zipfile

from huggingface_hub import HfApi

from common import ROOT

p = argparse.ArgumentParser()
p.add_argument("folder"); p.add_argument("repo_name")
p.add_argument("--exp", required=True)
p.add_argument("--results", default="", help="markdown for the results section")
p.add_argument("--dry", action="store_true")
a = p.parse_args()

args = json.load(open(os.path.join(ROOT, "runs", a.exp, "args.json")))
cfg = json.load(open(os.path.join(a.folder, "config.json")))
api = HfApi()
user = api.whoami()["name"]
repo = f"{user}/{a.repo_name}"
whisper = "whisper" in args["base"]
from transformers import AutoModelForAudioClassification
n_params = sum(t.numel() for t in AutoModelForAudioClassification.from_pretrained(a.folder).parameters())

EXT = {  # openly licensed, not among the corpora in PROHIBITED_SOURCES.txt
    "ext_efik": "[offiongbassey/efik_speech_dataset](https://huggingface.co/datasets/offiongbassey/efik_speech_dataset) (CC-BY-4.0): "
                "Efik speech, 2 speakers; a random 3 h subset. From \"Towards Digital Preservation of Efik: TTS for a Low-Resource African Language\" (Interspeech 2026).",
    "ext_cv_ibb": "[Mozilla Common Voice Scripted Speech 27.0, Ibibio](https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk) (CC0-1.0): "
                  "validated and not-yet-reviewed clips (invalidated excluded), at most 100 clips per speaker, 19 speakers, 3.6 h.",
}


def ext_note(args):
    if not args.get("ext"):
        return ("**External datasets: none.** No audio outside the official training data was used. "
                f"The only external resource is the publicly released pretrained base model `{args['base']}`.")
    lines = "\n".join(f"- {EXT[e]}" for e in args["ext"])
    return ("**External datasets** (openly licensed; none of them is a corpus listed in the competition's PROHIBITED_SOURCES.txt), "
            f"used for training only:\n\n{lines}\n\nThe pretrained base model is `{args['base']}`.")


card = open(os.path.join(ROOT, "model_card.md")).read().format(
    base=args["base"], name=a.repo_name, params=f"{n_params/1e6:.1f}M",
    arch_note=(f"- **Input window:** {args['chunk']} s of log-mel features (positional embeddings truncated accordingly) for speed." if whisper else ""),
    speed_note=(", and speed perturbation (0.85-1.15x) for speaker robustness" if args.get("speed", 0) > 0 else ""),
    splits=("train and validation splits" if args.get("full") else "train split; validation split used for model selection and calibration"),
    results=a.results or "See the competition leaderboard.",
    ext_note=ext_note(args))
open(os.path.join(a.folder, "README.md"), "w").write(card)
shutil.copy(os.path.join(ROOT, "LICENSE"), os.path.join(a.folder, "LICENSE"))

allowed = {"config.json", "model.safetensors", "preprocessor_config.json", "README.md", "LICENSE"}
files = set(os.listdir(a.folder))
assert files <= allowed, f"unexpected files: {files - allowed}"
assert "auto_map" not in cfg
print("files", sorted(files), "->", repo)
if a.dry:
    raise SystemExit

api.create_repo(repo, private=False, exist_ok=False)
info = api.upload_folder(folder_path=a.folder, repo_id=repo, commit_message=f"{a.exp} submission")
sha = info.oid
sub_dir = os.path.join(ROOT, "submissions", a.repo_name)
os.makedirs(sub_dir, exist_ok=True)
json.dump({"repo": repo, "revision": sha}, open(os.path.join(sub_dir, "submission.json"), "w"))
with zipfile.ZipFile(os.path.join(sub_dir, "submission.zip"), "w") as z:
    z.write(os.path.join(sub_dir, "submission.json"), "submission.json")
print(json.dumps({"repo": repo, "revision": sha, "url": f"https://huggingface.co/{repo}",
                  "zip": os.path.join(sub_dir, "submission.zip")}))
