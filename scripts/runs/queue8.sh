#!/bin/zsh
# F04 = whisper-medium encoder (~305M params, under the 350M limit), train+validation, 3000 steps. Runs after H02, alone on the GPU.
cd /Users/datapsalm/Desktop/ig/dsn2026
F='warn|shards|UNEXPECTED|MISSING|MISMATCH'
while pgrep -f "train.py --exp H02" >/dev/null || pgrep -f "queue7.sh" >/dev/null; do sleep 20; done
.venv/bin/python train.py --exp F04 --base openai/whisper-medium --chunk 10 --bs 32 --lr 3e-5 --max_crop 10 --full --epochs 2 --steps_per_epoch 1500 --speed 0.6 --src_bal --adv 0.3 --head_lr 1e-3 2>&1 | grep --line-buffered -v -i -E "$F" > runs/F04.log
.venv/bin/python export.py runs/F04/last export/F04 && .venv/bin/python publish.py export/F04 dsn2026-lid-f04 --exp F04 > runs/F04.pub 2>&1
