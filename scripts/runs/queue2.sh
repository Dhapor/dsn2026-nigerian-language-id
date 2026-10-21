#!/bin/zsh
# W02: W01 + speed perturbation (speaker robustness), 4 epochs. Then H01: mHuBERT-147 + speed.
cd /Users/datapsalm/Desktop/ig/dsn2026
.venv/bin/python train.py --exp W02 --base openai/whisper-base --chunk 10 --epochs 4 --steps_per_epoch 1500 --bs 32 --lr 1e-4 --head_lr 1e-3 --max_crop 10 --speed 0.6 2>&1 | grep -v -i -E "warn|shards" > runs/W02.log
.venv/bin/python train.py --exp H01 --base utter-project/mHuBERT-147 --epochs 4 --steps_per_epoch 1500 --bs 16 --lr 5e-5 --head_lr 1e-3 --max_crop 8 --speed 0.6 2>&1 | grep -v -i -E "warn|shards" > runs/H01.log
