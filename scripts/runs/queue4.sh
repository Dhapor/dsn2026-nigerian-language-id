#!/bin/zsh
# After W03: F01 = W02 recipe on train+validation (more speakers/sources), 3000 steps (best checkpoints were at 1500-4500).
# Then H01 = mHuBERT-147 with speed + source balancing + bounded adversary.
cd /Users/datapsalm/Desktop/ig/dsn2026
F='warn|shards|UNEXPECTED|MISSING|MISMATCH'
while pgrep -f "train.py --exp W03" >/dev/null; do sleep 20; done
.venv/bin/python train.py --exp F01 --full --base openai/whisper-base --chunk 10 --epochs 2 --steps_per_epoch 1500 --bs 32 --lr 1e-4 --head_lr 1e-3 --max_crop 10 --speed 0.6 2>&1 | grep --line-buffered -v -i -E "$F" > runs/F01.log
.venv/bin/python train.py --exp H01 --base utter-project/mHuBERT-147 --epochs 4 --steps_per_epoch 1500 --bs 16 --lr 5e-5 --head_lr 1e-3 --max_crop 8 --speed 0.6 --src_bal --adv 0.3 2>&1 | grep --line-buffered -v -i -E "$F" > runs/H01.log
