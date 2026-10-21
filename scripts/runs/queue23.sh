#!/bin/zsh
# A04 = A03 plain recipe + 3 h external Efik (offiongbassey/efik_speech_dataset, CC-BY-4.0). Waits for F06.
cd "$(dirname "$0")/../.."
while pgrep -f "train.py --exp F06" >/dev/null; do sleep 20; done
F='warn|shards|UNEXPECTED|MISSING|MISMATCH|Loading|LOAD REPORT|^Key|^---|^Notes|^\||^$'
.venv/bin/python train.py --exp A04 --ext ext_efik --base ajesujoba/AfriHuBERT --bs 16 --lr 5e-5 --max_crop 8 --crop_step 1 --epochs 3 --steps_per_epoch 1000 --speed 0.6 --head_lr 1e-3 --eval_max_s 12 --workers 6 2>&1 | grep --line-buffered -v -i -E "$F" > runs/A04.log
