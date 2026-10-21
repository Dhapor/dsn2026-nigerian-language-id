#!/bin/zsh
# F04 retry: whisper-medium at batch 16 (batch 32 ran out of GPU memory), then H02 (mHuBERT-147). One run at a time.
cd "$(dirname "$0")/../.."
F='warn|shards|UNEXPECTED|MISSING|MISMATCH'
R='--full --epochs 2 --steps_per_epoch 1500 --speed 0.6 --src_bal --adv 0.3 --head_lr 1e-3'
.venv/bin/python train.py --exp F04 --base openai/whisper-medium --chunk 10 --bs 16 --lr 3e-5 --max_crop 10 ${=R} 2>&1 | grep --line-buffered -v -i -E "$F" > runs/F04.log
.venv/bin/python export.py runs/F04/last export/F04 && .venv/bin/python publish.py export/F04 dsn2026-lid-f04 --exp F04 > runs/F04.pub 2>&1
.venv/bin/python train.py --exp H02 --base utter-project/mHuBERT-147 --bs 16 --lr 5e-5 --max_crop 8 ${=R} 2>&1 | grep --line-buffered -v -i -E "$F" > runs/H02.log
.venv/bin/python export.py runs/H02/last export/H02 && .venv/bin/python publish.py export/H02 dsn2026-lid-h02 --exp H02 > runs/H02.pub 2>&1
