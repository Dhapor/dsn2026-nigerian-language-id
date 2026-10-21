---
license: apache-2.0
library_name: transformers
pipeline_tag: audio-classification
base_model: {base}
language: [yo, ig, ha, ego, anw, efi, ibb, idu, igl, nup, tiv, urh]
tags: [audio-classification, language-identification, nigerian-languages, dsn-2026]
---

# {name}: spoken language identification for 12 Nigerian languages

Submission to the DSN 2026 competition "Find the Language I Am Speaking". Given a speech clip, the model predicts one of 13 labels:
`yor ibo hau ego anw efi ibb idu igl nup tiv urh other`.

It is a standard `AutoModelForAudioClassification` checkpoint (fp16 safetensors), with no custom code.

## Team

- Oludare Samuel
- Fagbemiro Victoria
- Sumayah Adegbite

## Approach

- **Base model:** `{base}` (encoder only, {params} parameters), fine-tuned end to end with a weighted-layer-sum pooling head.
{arch_note}
- **Training:** random 2-10 s crops; class-rebalanced sampling (rare languages drawn more often); label smoothing 0.1.
- **Augmentation** (generated from the training audio itself, no external audio): coloured noise, babble from other training clips, simulated reverberation, telephone band-limiting with mu-law companding and clipping, random gain{speed_note}.
- **Calibration:** per-class logit offsets tuned on the validation split to maximise the competition's language-weighted F1. They are added to the classifier bias, so inference is a plain argmax.
- **Export:** fp16 weights.

## Training data

- [Data-Science-Nigeria/dsn-2026-lid-train](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train) ({splits}).

{ext_note}

### Attribution of the training data sources

The official training data combines these sources, each under its own license:

- afrispeech_ipa: CC-BY-4.0
- hypa_voices: Apache-2.0
- omnilingual_asr: CC-BY-4.0. If you use this portion in research, please cite: Omnilingual ASR Team et al., "Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages", arXiv:2511.09690, 2025.

## Validation results (local)

{results}

## Limitations

Several languages have very few training speakers (Nupe 1, Igala 2, Ibibio 2), so accuracy on unseen speakers of those languages is uncertain. Annang, Ibibio and Efik are closely related and can be confused.
