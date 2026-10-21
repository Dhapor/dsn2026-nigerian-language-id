#!/bin/zsh
# W04 = W03 + channel randomisation + wider speed range; W05 = W04 + within-language speaker adversary. Validation runs (not full), in parallel.
cd /Users/datapsalm/Desktop/dsn2026
F='warn|shards|UNEXPECTED|MISSING|MISMATCH|Loading|LOAD REPORT|^Key|^---|^Notes|^$'
R='--base openai/whisper-base --chunk 10 --bs 32 --lr 1e-4 --max_crop 10 --epochs 4 --steps_per_epoch 1500 --speed 0.6 --src_bal --adv 0.3 --workers 6'
.venv/bin/python train.py --exp W04 ${=R} --chan 0.7 --speed_lo 0.8 --speed_hi 1.25 2>&1 | grep --line-buffered -v -i -E "$F" > runs/W04.log &
.venv/bin/python train.py --exp W05 ${=R} --chan 0.7 --speed_lo 0.8 --speed_hi 1.25 --spk_adv 0.5 2>&1 | grep --line-buffered -v -i -E "$F" > runs/W05.log &
wait
