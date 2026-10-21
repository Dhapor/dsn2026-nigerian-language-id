#!/bin/zsh
# F02 = W03 recipe (speed + source-balanced + bounded adversary) on train+validation, 6000 steps. Then H01 (mHuBERT, same recipe).
cd /Users/datapsalm/Desktop/ig/dsn2026
F='warn|shards|UNEXPECTED|MISSING|MISMATCH'
.venv/bin/python train.py --exp F02 --full --base openai/whisper-base --chunk 10 --epochs 4 --steps_per_epoch 1500 --bs 32 --lr 1e-4 --head_lr 1e-3 --max_crop 10 --speed 0.6 --src_bal --adv 0.3 2>&1 | grep --line-buffered -v -i -E "$F" > runs/F02.log
.venv/bin/python train.py --exp H01 --base utter-project/mHuBERT-147 --epochs 4 --steps_per_epoch 1500 --bs 16 --lr 5e-5 --head_lr 1e-3 --max_crop 8 --speed 0.6 --src_bal --adv 0.3 2>&1 | grep --line-buffered -v -i -E "$F" > runs/H01.log
