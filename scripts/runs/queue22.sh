#!/bin/zsh
# F06 = final: A03 plain recipe on train+val, 3000 steps.
# F05 test showed our debiasing loses precision on urh/ego vs the leader.
cd "$(dirname "$0")/../.."
F='warn|shards|UNEXPECTED|MISSING|MISMATCH|Loading|LOAD REPORT|^Key|^---|^Notes|^\||^$'
.venv/bin/python train.py --exp F06 --full --base ajesujoba/AfriHuBERT --bs 16 --lr 5e-5 --max_crop 8 --crop_step 1 --epochs 2 --steps_per_epoch 1500 --speed 0.6 --head_lr 1e-3 --eval_max_s 12 --workers 6 2>&1 | grep --line-buffered -v -i -E "$F" > runs/F06.log
