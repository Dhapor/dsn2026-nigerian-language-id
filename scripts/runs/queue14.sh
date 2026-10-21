#!/bin/zsh
# Resume plan after the laptop restart: one job at a time on the GPU.
# W05 = W04 recipe + within-language speaker adversary (whisper-base, validation run, ~35 min).
# A01 = AfriHuBERT (the leader's base) + debiasing recipe (validation run, ~2.5 h).
cd /Users/datapsalm/Desktop/dsn2026
F='warn|shards|UNEXPECTED|MISSING|MISMATCH|Loading|LOAD REPORT|^Key|^---|^Notes|^\||^$'
R='--base openai/whisper-base --chunk 10 --bs 32 --lr 1e-4 --max_crop 10 --epochs 4 --steps_per_epoch 1500 --speed 0.6 --src_bal --adv 0.3 --workers 6'
.venv/bin/python train.py --exp W05 ${=R} --chan 0.7 --speed_lo 0.8 --speed_hi 1.25 --spk_adv 0.5 2>&1 | grep --line-buffered -v -i -E "$F" > runs/W05.log
.venv/bin/python train.py --exp A01 --base ajesujoba/AfriHuBERT --bs 16 --lr 5e-5 --max_crop 8 --epochs 3 --steps_per_epoch 1000 --speed 0.6 --src_bal --adv 0.3 --chan 0.5 --head_lr 1e-3 --eval_max_s 12 --workers 6 2>&1 | grep --line-buffered -v -i -E "$F" > runs/A01.log
