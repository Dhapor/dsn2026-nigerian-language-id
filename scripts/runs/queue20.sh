#!/bin/zsh
# A02 = A01 + within-language speaker adversary 0.5 (W05 got Ibibio 0.22 early with it).
cd /Users/datapsalm/Desktop/dsn2026
F='warn|shards|UNEXPECTED|MISSING|MISMATCH|Loading|LOAD REPORT|^Key|^---|^Notes|^\||^$'
.venv/bin/python train.py --exp A02 --base ajesujoba/AfriHuBERT --bs 16 --lr 5e-5 --max_crop 8 --crop_step 1 --epochs 3 --steps_per_epoch 1000 --speed 0.6 --src_bal --adv 0.3 --chan 0.5 --spk_adv 0.5 --head_lr 1e-3 --eval_max_s 12 --workers 6 2>&1 | grep --line-buffered -v -i -E "$F" > runs/A02.log
