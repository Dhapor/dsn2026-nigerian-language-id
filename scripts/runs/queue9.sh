#!/bin/zsh
# H02 retry (mHuBERT-147, train+validation, 3000 steps) after F04 finishes; alone on the GPU.
cd /Users/datapsalm/Desktop/ig/dsn2026
F='warn|shards|UNEXPECTED|MISSING|MISMATCH'
while pgrep -f "queue8.sh" >/dev/null; do sleep 20; done
.venv/bin/python train.py --exp H02 --base utter-project/mHuBERT-147 --bs 16 --lr 5e-5 --max_crop 8 --full --epochs 2 --steps_per_epoch 1500 --speed 0.6 --src_bal --adv 0.3 --head_lr 1e-3 2>&1 | grep --line-buffered -v -i -E "$F" > runs/H02.log
.venv/bin/python export.py runs/H02/last export/H02 && .venv/bin/python publish.py export/H02 dsn2026-lid-h02 --exp H02 > runs/H02.pub 2>&1
