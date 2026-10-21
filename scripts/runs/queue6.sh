#!/bin/zsh
# After F02: two bigger base models in parallel, both on train+validation, 3000 steps (F02a's step-3000 checkpoint scored best).
# F03 = whisper-small encoder, H02 = mHuBERT-147. Each is exported and published when done.
cd "$(dirname "$0")/../.."
F='warn|shards|UNEXPECTED|MISSING|MISMATCH'
while pgrep -f "train.py --exp F02" >/dev/null; do sleep 20; done
R='--full --epochs 2 --steps_per_epoch 1500 --speed 0.6 --src_bal --adv 0.3 --head_lr 1e-3'
( .venv/bin/python train.py --exp F03 --base openai/whisper-small --chunk 10 --bs 32 --lr 5e-5 --max_crop 10 ${=R} 2>&1 | grep --line-buffered -v -i -E "$F" > runs/F03.log
  .venv/bin/python export.py runs/F03/last export/F03 && .venv/bin/python publish.py export/F03 dsn2026-lid-f03 --exp F03 ) > runs/F03.pub 2>&1 &
( .venv/bin/python train.py --exp H02 --base utter-project/mHuBERT-147 --bs 16 --lr 5e-5 --max_crop 8 ${=R} 2>&1 | grep --line-buffered -v -i -E "$F" > runs/H02.log
  .venv/bin/python export.py runs/H02/last export/H02 && .venv/bin/python publish.py export/H02 dsn2026-lid-h02 --exp H02 ) > runs/H02.pub 2>&1 &
wait
