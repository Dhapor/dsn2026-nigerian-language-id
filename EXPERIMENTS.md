# Experiment log

Validation = official validation split + held-out Nupe block (Nupe has 1 speaker, so its F1 is optimistic).
Score = official language-weighted F1 x (0.5 + 0.5 x efficiency).
`xsrc_yor` = accuracy on the 90 validation Yoruba clips from Hypa-Voices (training has Yoruba from AfriSpeech only). It is our probe for source shortcutting.

| ID | Change | Model | Best val wF1 | Calibrated (CV) | xsrc_yor | Test (Codabench) | Decision |
|---|---|---|---|---|---|---|---|
| W01 | Baseline: noise/phone/reverb aug, alpha 0.5 | whisper-base enc, 10 s | 0.615 (step 4500) | ~0.672 | 0.078 | **final 0.493, F1 0.523, eff 0.886** (sub 1003865) | Baseline. Val overestimates test by ~0.15 |
| W02 | + speed perturbation 0.6 | whisper-base enc, 10 s | 0.631 (step 1500) | - | 0.056 | **final 0.498, F1 0.528, eff 0.887** (not ranked: F1 < 0.60) | KEEP speed (+0.07 at equal step) |
| W03 | + source-balanced sampling + source-adversarial (GRL 0.3) | whisper-base enc, 10 s | 0.648 (step 6000) | - | 0.089 | not scored: runner could not reach huggingface.co; resubmit | |
| H01 | W03 recipe on mHuBERT-147 | mHuBERT-147 | queued | | | | |

| F02a | W03 recipe, train+val, step 3000 | whisper-base | - | - | - | final 0.518, F1 0.549 (not ranked) | |
| F03 | W03 recipe, train+val, 3000 steps | whisper-small (87.6M) | - | - | - | final 0.539, F1 0.586, eff 0.839 (not ranked) | |
| F04 | W03 recipe, train+val, 3000 steps, bs16 | whisper-medium (306.5M) | - | - | - | **RANKED: final 0.511, F1 0.613, eff 0.667** (sub 1005236) | first ranked model |
| W04 | W03 + channel randomisation (EQ/band-limit, p 0.7) + speed 0.8-1.25 | whisper-base | 0.636 (s4500; s6000 0.613) | - | 0.033 | - | faster early (+0.04 at s1500), best-ever s4500, but noisy; efi/ibb still ~0 |
| W05 | W04 + within-language speaker adversary (0.5) | whisper-base | 0.643 (s1500); then 0.608/0.608/0.615 | - | 0.156 | - | ibb 0.216 at s1500 (first non-zero), lost later. Not worth a final run |
| A02 | A01 + speaker adversary 0.5 (finite mask; first try went NaN at s350) | AfriHuBERT | 0.646 (s3000); s1000 0.541, s2000 0.622 | - | 0.444 (s2000) | - | below A01; less source shortcut but ibb still 0. Not used |
| A01 | AfriHuBERT + W03 recipe + chan 0.5, crops rounded to 1 s (MPS OOM fix) | ajesujoba/AfriHuBERT (94.6M) | **0.669 (s2000)**; s1000 0.601, s3000 0.659 | +0.008 CV (not used) | **0.278** | - | BEST val + least source shortcut. ibb 0.0 (-> anw/efi), tiv -> hau |
| F05 | A01 recipe, train+val, 3000 steps | AfriHuBERT | - | - | - | zip ready 09:50 WAT (datapsalm/dsn2026-lid-f05 @ ba29b10) | final candidate |


## Findings

1. **Source shortcut.** Train label/source combinations are disjoint (Yoruba and Urhobo only from AfriSpeech; Efik, Ibibio, Igala and Nupe only from Hypa-Voices). The model learns the recording source. Hypa-Voices Yoruba gets 6-8% accuracy. Test Yoruba F1 0.38 vs val 0.85.
2. **Overfitting to speakers.** The best checkpoint is early (step 1500-4500); it degrades later.
3. **Efficiency on organiser hardware.** Speed is saturated (0.005 s/clip gives score 1.0). Memory 1.19 GB gives 0.70. Size 0.96. So bigger models cost only size and memory.
4. **Ranking needs F1 >= 0.60.** W01/W02 are NOT RANKED. On test, Tiv (140/242) and Efik (28/45) are predicted as Ibibio; Yoruba leaks to Idoma/Ibibio/Other. Ibibio and Idoma act as sink classes.
5. **Submission limits (Rules page): 3 per day, 20 total in development; final phase (from 2026-10-21) 3 per day, 5 total.** Deadline 2026-10-21. No tuning on the test set; never change a repo after submitting it.

6. **F05 NOT SCORED (sub 1007784, Development tab): 0 of 6659 clips in 1800 s.** Our fp16 config makes transformers 5 load the model in fp16; on CPU, HuBERT in fp16 takes ~8 s/clip (fp32: 0.077 s). The leader's repo behaves the same, so they must have run on a GPU. F05b = same fp16 weights (0.189 GB) with config dtype float32: loads as fp32, 0.078 s/clip on CPU, identical logits. Repo datapsalm/dsn2026-lid-f05b @ 4e16fb1.
7. **F05 test (sub 1008091, same zip re-run): F1 0.5925, eff 0.790, would-be 0.5304 (NOT RANKED).** The 09:51 failure was the runner, not fp16. vs leader: we win ibb 0.08/0.00, anw 0.77/0.72; lose urh 0.87/1.00 and ego 0.83/0.96 (both on precision), nup 0.47/0.59, efi 0.11/0.24. AfriHuBERT val->test gap ~0.08 (A01 val 0.669).
   Next: A01-best uploaded as is (train-only, val 0.669) to test whether full retraining helped; A03 = plain recipe (no adv/src_bal/chan).
8. **F05b (fp32 config) test: F1 0.5931, speed 0.015 s (score 0.904), mem 1.62 GB, eff 0.767, would-be 0.5241.** Runner is a GPU: fp16 config is better (F05 eff 0.790). Keep fp16 exports.
9. **A01 (train-only) test: F1 0.529 vs F05 (train+val) 0.593** -> full-data retraining is worth +0.06. A03 plain recipe val 0.671 (= A01). F06 = A03 on train+val (published, not submitted: kept the upload for F07).
10. **Other AfriHuBERT fine-tunes on the public board reach F1 ~0.68 at the same efficiency (0.79)**, strongest on anw / urh / ego / nup, so the gap is the recipe and data, not the base model.
11. **External data allowed** (open license, credited; not the PROHIBITED_SOURCES corpora). Added offiongbassey/efik_speech_dataset (CC-BY-4.0, 3 h, 2 spk) and Common Voice 27.0 Ibibio (CC0, 3.6 h, 19 spk, <=100 clips/spk).
    F07 = plain recipe + train+val + both external sets, 4500 steps. Final candidate.
    F07 published 14:20: datapsalm/dsn2026-lid-f07 @ 1ac06ea (model card credits both external sets). Awaiting test score.
12. **F07 test: F1 0.6247, final 0.5592 (ranked: F1 >= 0.60).** vs F05: tiv 0.43->0.63, nup 0.47->0.60, urh 0.87->1.00, ego 0.83->0.88, efi 0.11->0.25; anw 0.77->0.62 (sink: efi 25, tiv 45, ibb 11 -> anw), yor 0.80->0.65. Ibibio data did not transfer (2/23).
13. **Model soup** (no training): val split only (no nup), A01 0.594, A03 0.608, soup(A01,A03) 0.626; gains anw/igl/efi/yor. Calibration: A03 +0.025 CV, soup +0.007 CV (not used).
    S01 = soup(F05, F07) published as datapsalm/dsn2026-lid-s01.
