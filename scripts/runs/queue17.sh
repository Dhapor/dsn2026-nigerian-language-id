#!/bin/zsh
# A01 retry 3: crop and eval lengths rounded to whole seconds (two earlier tries ran out of MPS memory at ~step 450-600).
cd "$(dirname "$0")/../.."
F='warn|shards|UNEXPECTED|MISSING|MISMATCH|Loading|LOAD REPORT|^Key|^---|^Notes|^\||^$'
.venv/bin/python train.py --exp A01 --base ajesujoba/AfriHuBERT --bs 16 --lr 5e-5 --max_crop 8 --crop_step 1 --epochs 3 --steps_per_epoch 1000 --speed 0.6 --src_bal --adv 0.3 --chan 0.5 --head_lr 1e-3 --eval_max_s 12 --workers 6 2>&1 | grep --line-buffered -v -i -E "$F" > runs/A01.log
