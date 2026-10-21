#!/bin/zsh
# Sequential, one job on the GPU at a time: wait for W04, then W05, then A01.
cd "$(dirname "$0")/../.."
F='warn|shards|UNEXPECTED|MISSING|MISMATCH|Loading|LOAD REPORT|^Key|^---|^Notes|^\||^$'
while pgrep -f "exp W04" >/dev/null; do sleep 30; done
R='--base openai/whisper-base --chunk 10 --bs 32 --lr 1e-4 --max_crop 10 --epochs 4 --steps_per_epoch 1500 --speed 0.6 --src_bal --adv 0.3 --workers 6'
.venv/bin/python train.py --exp W05 ${=R} --chan 0.7 --speed_lo 0.8 --speed_hi 1.25 --spk_adv 0.5 2>&1 | grep --line-buffered -v -i -E "$F" > runs/W05.log
.venv/bin/python train.py --exp A01 --base ajesujoba/AfriHuBERT --bs 16 --lr 5e-5 --max_crop 8 --epochs 3 --steps_per_epoch 1000 --speed 0.6 --src_bal --adv 0.3 --chan 0.5 --head_lr 1e-3 --eval_max_s 12 --workers 6 2>&1 | grep --line-buffered -v -i -E "$F" > runs/A01.log
