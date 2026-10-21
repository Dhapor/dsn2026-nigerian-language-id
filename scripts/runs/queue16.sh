#!/bin/zsh
# A01 retry with periodic MPS cache clearing (first try ran out of memory at ~step 450). Waits for queue15 (W05) to finish.
cd "$(dirname "$0")/../.."
while pgrep -f queue15.sh >/dev/null; do sleep 30; done
F='warn|shards|UNEXPECTED|MISSING|MISMATCH|Loading|LOAD REPORT|^Key|^---|^Notes|^\||^$'
.venv/bin/python train.py --exp A01 --base ajesujoba/AfriHuBERT --bs 16 --lr 5e-5 --max_crop 8 --epochs 3 --steps_per_epoch 1000 --speed 0.6 --src_bal --adv 0.3 --chan 0.5 --head_lr 1e-3 --eval_max_s 12 --workers 6 2>&1 | grep --line-buffered -v -i -E "$F" > runs/A01.log
