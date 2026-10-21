#!/bin/zsh
# A03 = plain AfriHuBERT (leader-like): no source adversary, no source balancing, no channel aug; keeps speed perturbation.
# F05 test showed our debiasing loses precision on urh/ego vs the leader.
cd "$(dirname "$0")/../.."
F='warn|shards|UNEXPECTED|MISSING|MISMATCH|Loading|LOAD REPORT|^Key|^---|^Notes|^\||^$'
.venv/bin/python train.py --exp A03 --base ajesujoba/AfriHuBERT --bs 16 --lr 5e-5 --max_crop 8 --crop_step 1 --epochs 3 --steps_per_epoch 1000 --speed 0.6 --head_lr 1e-3 --eval_max_s 12 --workers 6 2>&1 | grep --line-buffered -v -i -E "$F" > runs/A03.log
