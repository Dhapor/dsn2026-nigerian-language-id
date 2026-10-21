#!/bin/zsh
# W03: Whisper-base + speed + source-balanced sampling + source-adversarial loss. H01: same recipe on mHuBERT-147.
cd "$(dirname "$0")/../.."
F='warn|shards|UNEXPECTED|MISSING|MISMATCH'
.venv/bin/python train.py --exp W03 --base openai/whisper-base --chunk 10 --epochs 4 --steps_per_epoch 1500 --bs 32 --lr 1e-4 --head_lr 1e-3 --max_crop 10 --speed 0.6 --src_bal --adv 0.3 2>&1 | grep --line-buffered -v -i -E "$F" > runs/W03.log
.venv/bin/python train.py --exp H01 --base utter-project/mHuBERT-147 --epochs 4 --steps_per_epoch 1500 --bs 16 --lr 5e-5 --head_lr 1e-3 --max_crop 8 --speed 0.6 --src_bal --adv 0.3 2>&1 | grep --line-buffered -v -i -E "$F" > runs/H01.log
