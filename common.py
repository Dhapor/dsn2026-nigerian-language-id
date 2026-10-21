"""Shared constants, the official metric, data access and augmentation."""
import json
import math
import os

import numpy as np

SOURCES = ["afrispeech_ipa", "hypa_voices", "omnilingual_asr", "ext_efik", "ext_cv_ibb"]  # ext_*: external data via prep_ext.py
LABELS = ["yor", "ibo", "hau", "ego", "anw", "efi", "ibb", "idu", "igl", "nup", "tiv", "urh", "other"]
L2I = {l: i for i, l in enumerate(LABELS)}
HIGH = {"yor", "ibo", "hau"}
# Official shares: high-resource 1x, the rest (incl. other) 2x; 3*1 + 10*2 = 23.
CLASS_W = np.array([1.0 if l in HIGH else 2.0 for l in LABELS]) / 23.0
SR = 16000
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DSN_DATA", os.path.join(ROOT, "data"))


def weighted_f1(y_true, y_pred):
    """Language-weighted F1 exactly as described on the Evaluation page."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = np.zeros(len(LABELS))
    for c in range(len(LABELS)):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1s[c] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return float(np.sum(f1s * CLASS_W)), f1s


def efficiency(params, repo_bytes, peak_bytes, sec_per_clip):
    size = ((1 - params / 350e6) + (1 - repo_bytes / 1.5e9)) / 2
    mem = 1 - peak_bytes / 4e9
    t = max(sec_per_clip, 0.01)
    speed = 1 - math.log(t / 0.01) / math.log(0.5 / 0.01)
    eff = (size + mem + speed) / 3
    return eff, dict(size=size, mem=mem, speed=speed)


def final_score(f1, eff):
    return f1 * (0.5 + 0.5 * eff)


class Split:
    """Audio packed as one int16 memmap plus offsets (built by prep.py)."""

    def __init__(self, name):
        meta = json.load(open(os.path.join(DATA, f"{name}.json")))
        self.ids = meta["id"]
        self.labels = np.array([L2I[l] for l in meta["label"]])
        self.speakers = meta["speaker_id"]
        self.sources = meta["source"]
        self.offsets = np.array(meta["offset"], dtype=np.int64)
        self.lengths = np.array(meta["length"], dtype=np.int64)
        self.path = os.path.join(DATA, f"{name}.i16")
        self._audio = None

    @property
    def audio(self):  # opened lazily so the object pickles cheaply into DataLoader workers
        if self._audio is None:
            self._audio = np.memmap(self.path, dtype=np.int16, mode="r")
        return self._audio

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_audio"] = None
        return d

    def __len__(self):
        return len(self.ids)

    def wav(self, i, start=0, n=None):
        n = self.lengths[i] - start if n is None else n
        o = self.offsets[i] + start
        return self.audio[o:o + n].astype(np.float32) / 32768.0


# ---------------------------------------------------------------- augmentation
def _colored_noise(n, rng):
    white = rng.standard_normal(n).astype(np.float32)
    beta = rng.uniform(0, 2)  # 0 white, 1 pink, 2 brown
    spec = np.fft.rfft(white)
    f = np.arange(len(spec), dtype=np.float32)
    f[0] = 1
    spec /= f ** (beta / 2)
    out = np.fft.irfft(spec, n).astype(np.float32)
    return out / (np.std(out) + 1e-8)


def _add_at_snr(x, noise, snr_db):
    px = np.mean(x ** 2) + 1e-8
    pn = np.mean(noise ** 2) + 1e-8
    return x + noise * np.sqrt(px / (pn * 10 ** (snr_db / 10)))


def _phone(x, rng):
    from scipy.signal import butter, sosfilt, resample_poly
    lo, hi = rng.uniform(200, 400), rng.uniform(3000, 3600)
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    y = sosfilt(sos, x).astype(np.float32)
    y = resample_poly(resample_poly(y, 1, 2), 2, 1).astype(np.float32)[: len(x)]
    if rng.random() < 0.5:  # mu-law 8 bit companding, like G.711
        mu = 255.0
        peak = np.max(np.abs(y)) + 1e-8
        z = np.sign(y / peak) * np.log1p(mu * np.abs(y / peak)) / np.log1p(mu)
        z = np.round(z * 127) / 127
        y = (np.sign(z) * ((1 + mu) ** np.abs(z) - 1) / mu * peak).astype(np.float32)
    if rng.random() < 0.3:  # soft clipping
        g = rng.uniform(2, 6)
        y = np.tanh(g * y) / g
    return y


def channel(x, rng):
    """Random recording channel (band limits, EQ tilt, peaking bands) so corpus/microphone cues stop being useful."""
    from scipy.signal import butter, sosfilt
    if rng.random() < 0.5:
        sos = butter(rng.integers(2, 6), rng.uniform(2800, 7600), btype="low", fs=SR, output="sos")
        x = sosfilt(sos, x)
    if rng.random() < 0.5:
        sos = butter(2, rng.uniform(40, 350), btype="high", fs=SR, output="sos")
        x = sosfilt(sos, x)
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.linspace(0, SR / 2, len(spec))
    lf = np.log2(np.maximum(f, 50) / 1000)
    g = rng.uniform(-6, 6) * lf / 3  # tilt in dB, about +-6 dB per 3 octaves
    for _ in range(rng.integers(0, 4)):  # peaking bands
        c, w, a = rng.uniform(-2.5, 2.5), rng.uniform(0.2, 1.0), rng.uniform(-9, 9)
        g = g + a * np.exp(-0.5 * ((lf - c) / w) ** 2)
    x = np.fft.irfft(spec * 10 ** (g / 20), n)
    return (x / (np.max(np.abs(x)) + 1e-8) * rng.uniform(0.1, 0.9)).astype(np.float32)


def augment(x, rng, babble_src=None):
    """Random noise / phone channel / gain, to mimic the noisy and phone test subsets."""
    if rng.random() < 0.35:
        x = _phone(x, rng)
    r = rng.random()
    if r < 0.35:
        x = _add_at_snr(x, _colored_noise(len(x), rng), rng.uniform(0, 25))
    elif r < 0.5 and babble_src is not None:
        x = _add_at_snr(x, babble_src(len(x)), rng.uniform(8, 25))
    if rng.random() < 0.2:  # reverb-ish: decaying random impulse response
        ir_len = int(SR * rng.uniform(0.05, 0.4))
        ir = rng.standard_normal(ir_len).astype(np.float32) * np.exp(-np.linspace(0, rng.uniform(4, 10), ir_len))
        ir[0] = 1.0
        x = np.convolve(x, ir)[: len(x)].astype(np.float32)
    x = x * 10 ** (rng.uniform(-12, 6) / 20)
    return np.clip(x, -1, 1).astype(np.float32)


class TrainDS:
    """Random crops + augmentation. Keys are (item index, crop samples, seed) from a batch sampler."""

    def __init__(self, splits, items, other_items, aug_p, speed_p=0.0, chan_p=0.0, speed_lo=0.85, speed_hi=1.15, spk=None):
        self.splits, self.items, self.other_items, self.aug_p = splits, items, other_items, aug_p
        self.speed_p, self.chan_p, self.speed_lo, self.speed_hi = speed_p, chan_p, speed_lo, speed_hi
        self.spk = spk  # item index -> speaker class (-1 = unknown)

    def __getitem__(self, key):
        import math
        k, crop_n, seed = key
        rng = np.random.default_rng(seed)
        s, i = self.items[k]
        sp = self.splits[s]
        n = sp.lengths[i]
        speed = rng.random() < self.speed_p
        if speed:  # Kaldi-style speed perturbation (pitch + tempo) so few speakers sound like many
            f = rng.uniform(self.speed_lo, self.speed_hi)
            want = int(crop_n * f) + 1
            if n >= want:
                x = sp.wav(i, rng.integers(0, n - want + 1), want)
            else:
                x = np.tile(sp.wav(i), int(math.ceil(want / n)))[:want]
            from scipy.signal import resample_poly
            x = resample_poly(x, 100, int(round(100 * f))).astype(np.float32)
            x = np.pad(x, (0, max(0, crop_n - len(x))), mode="wrap")[:crop_n]
        elif n >= crop_n:
            x = sp.wav(i, rng.integers(0, n - crop_n + 1), crop_n)
        else:  # short clip: tile to the batch length
            x = np.tile(sp.wav(i), int(math.ceil(crop_n / n)))[:crop_n]

        def babble(m):
            pool = self.other_items if rng.random() < 0.5 else self.items
            s2, j = pool[rng.integers(len(pool))]
            w = self.splits[s2].wav(j)
            return np.tile(w, int(math.ceil(m / len(w))))[:m]

        if rng.random() < self.chan_p:
            x = channel(x, rng)
        if rng.random() < self.aug_p:
            x = augment(x, rng, babble)
        spk = -1 if self.spk is None else int(self.spk[k])
        return x, int(sp.labels[i]), SOURCES.index(sp.sources[i]), spk
