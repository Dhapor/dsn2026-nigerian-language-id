# Find the Language I Am Speaking: DSN 2026 spoken language ID

Team **datapsalm** (Oludare Samuel, Victoria Fagbemiro, Sumayah Adegbite), DSN 2026 AI Bootcamp hackathon, ML track.

Given a 2–30 s audio clip, predict which of 12 Nigerian languages is spoken (Yoruba, Igbo, Hausa, Eggon, Annang, Efik, Ibibio, Idoma, Igala, Nupe, Tiv, Urhobo) or `other`. Scored on Codabench as
`language-weighted F1 × (0.5 + 0.5 × efficiency)`, where efficiency combines model size, peak memory and speed, and only F1 ≥ 0.60 is ranked.

**Best ranked submission:** [datapsalm/dsn2026-lid-f07](https://huggingface.co/datapsalm/dsn2026-lid-f07), F1 0.6247, efficiency 0.790, final score **0.5592** (3rd on the development leaderboard, 2026-10-21).

## Approach

1. **An African speech base.** [ajesujoba/AfriHuBERT](https://huggingface.co/ajesujoba/AfriHuBERT) (94.6M parameters), fine-tuned with a classification head. It out-scored Whisper-medium (F1 0.625 vs 0.613) at a third of the size.
2. **Robust training.** Noise, phone and reverb augmentation, speed perturbation, random 2–8 s crops (rounded to whole seconds so the Apple-GPU backend sees few input shapes). The final model trains on train + validation.
3. **More speakers, openly licensed.** Our weakest classes have 2–3 training speakers, so we added 19 Ibibio speakers and 2 Efik speakers (see Data).
4. **Model soup.** A uniform weight average of two fine-tunes from the same base (`soup.py`). On validation, averaging our two recipes scored F1 0.626 vs 0.608 and 0.594 for the single models, at no cost to size or speed.
5. **fp16 export.** 0.189 GB, 0.006 s per clip on the organisers' GPU. (An fp32 config made inference slower and lowered the score.)

## Results (Codabench test set)

| Submission | Model | Change | Test F1 | Final |
|---|---|---|---|---|
| W01 | Whisper-base encoder | Baseline | 0.523 | 0.493 (not ranked) |
| F04 | Whisper-medium encoder | Bigger model | 0.613 | 0.511 (ranked) |
| A01 | AfriHuBERT, train only | African base, source-debiased recipe | 0.529 | 0.474 (not ranked) |
| F05 | AfriHuBERT, train + val | All labelled data | 0.593 | 0.530 (not ranked) |
| **F07** | AfriHuBERT, train + val + external | Plain recipe, +21 speakers | **0.625** | **0.559 (ranked 3rd)** |
| S01 | Soup of F05 + F07 | Weight average | pending | pending |

The full log, including validation results and what each run taught us, is in [EXPERIMENTS.md](EXPERIMENTS.md).

## Data

- Official training data: [Data-Science-Nigeria/dsn-2026-lid-train](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train) (afrispeech_ipa CC-BY-4.0, hypa_voices Apache-2.0, omnilingual_asr CC-BY-4.0).
- External data (openly licensed; none of it is from the prohibited corpora Hypa-Voices, Omnilingual ASR or AfriSpeech-IPA that the test set was drawn from), listed on every model card that uses it:
  - [offiongbassey/efik_speech_dataset](https://huggingface.co/datasets/offiongbassey/efik_speech_dataset) (CC-BY-4.0): Efik, 2 speakers, a random 3 h subset. From "Towards Digital Preservation of Efik: TTS for a Low-Resource African Language" (Interspeech 2026).
  - [Mozilla Common Voice Scripted Speech 27.0, Ibibio](https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk) (CC0-1.0): validated and not-yet-reviewed clips (invalidated excluded), at most 100 clips per speaker, 19 speakers, 3.6 h. Download it from the Mozilla Data Collective (free account); it may not be re-hosted, so it is not in this repo.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python prep.py                                    # download the official data, pack into data/*.i16 memmaps
python prep_ext.py data/ext/efik_speech_dataset ext_efik efi --max_h 3
python prep_ext.py <cv-corpus-27.0>/ibb ext_cv_ibb ibb --max_h 4 --per_spk 100
# F07 (final): plain recipe, train + val + external data, 4500 steps
python train.py --exp F07 --full --ext ext_efik ext_cv_ibb --base ajesujoba/AfriHuBERT --bs 16 --lr 5e-5 \
  --max_crop 8 --crop_step 1 --epochs 3 --steps_per_epoch 1500 --speed 0.6 --head_lr 1e-3 --eval_max_s 12 --workers 6
python export.py runs/F07/last export/F07         # fp16, compliance checks
python publish.py export/F07 dsn2026-lid-f07 --exp F07   # new public HF repo + submission.zip
```

Every run's exact command is kept in `scripts/runs/` (one script per batch of runs, in order). Training ran on an Apple M4 Max (48 GB, MPS); an AfriHuBERT run takes about 25–40 minutes.

## Files

| File | What it does |
|---|---|
| `prep.py` | Downloads the official data and packs audio into int16 memmaps for fast random crops |
| `prep_ext.py` | Packs an external dataset (Hugging Face parquet or a Common Voice folder) the same way |
| `common.py` | Labels, the official weighted F1, the efficiency formula, the dataset and augmentation |
| `train.py` | Fine-tuning: class/source-balanced sampling, augmentation, optional adversaries, evaluation |
| `soup.py` | Builds and scores weight averages of checkpoints |
| `calibrate.py` | Fits per-class logit offsets on validation with cross-validation (tested; not used in the final models) |
| `export.py` | Exports a checkpoint as an fp16, rule-compliant Hugging Face folder |
| `bench.py` | Estimates the efficiency score locally |
| `publish.py` | Writes the model card, pushes a new public repo and writes `submission.zip` |
| `model_card.md` | Model card template |
| `EXPERIMENTS.md` | Every experiment and its result |

## License

Code: Apache-2.0 (`LICENSE`). Models inherit the licenses of their training data; see each model card.
