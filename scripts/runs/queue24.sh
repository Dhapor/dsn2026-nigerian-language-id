#!/bin/zsh
# F07 = final: plain recipe (A03/F06) on train+val + external Efik (CC-BY-4.0, 2 spk, 3 h) + Common Voice Ibibio (CC0, 19 spk, 3.6 h), 4500 steps.
cd "$(dirname "$0")/../.."
F='warn|shards|UNEXPECTED|MISSING|MISMATCH|Loading|LOAD REPORT|^Key|^---|^Notes|^\||^$'
.venv/bin/python train.py --exp F07 --full --ext ext_efik ext_cv_ibb --base ajesujoba/AfriHuBERT --bs 16 --lr 5e-5 --max_crop 8 --crop_step 1 --epochs 3 --steps_per_epoch 1500 --speed 0.6 --head_lr 1e-3 --eval_max_s 12 --workers 6 2>&1 | grep --line-buffered -v -i -E "$F" > runs/F07.log
