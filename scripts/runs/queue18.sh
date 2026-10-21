#!/bin/zsh
# F05 = final model: A01 recipe (AfriHuBERT, val wF1 0.669 at s2000) on train+validation, 3000 steps.
cd /Users/datapsalm/Desktop/dsn2026
F='warn|shards|UNEXPECTED|MISSING|MISMATCH|Loading|LOAD REPORT|^Key|^---|^Notes|^\||^$'
.venv/bin/python train.py --exp F05 --full --base ajesujoba/AfriHuBERT --bs 16 --lr 5e-5 --max_crop 8 --crop_step 1 --epochs 2 --steps_per_epoch 1500 --speed 0.6 --src_bal --adv 0.3 --chan 0.5 --head_lr 1e-3 --eval_max_s 12 --workers 6 2>&1 | grep --line-buffered -v -i -E "$F" > runs/F05.log
