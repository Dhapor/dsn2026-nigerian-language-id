"""Fine-tune an HF audio-classification model for 13-way LID, evaluated with the official metric.

Example:
  python train.py --exp E01 --base utter-project/mHuBERT-147 --epochs 6
  python train.py --exp W01 --base openai/whisper-base --chunk 10
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification

from common import LABELS, L2I, ROOT, SOURCES, SR, Split, TrainDS, weighted_f1

class Collate:
    def __init__(self, fe, whisper):
        self.fe, self.whisper = fe, whisper

    def __call__(self, b):
        wavs = [x for x, _, _, _ in b]
        out = self.fe(wavs, sampling_rate=SR, return_tensors="pt")
        key = "input_features" if self.whisper else "input_values"
        return ({key: out[key]}, torch.tensor([y for _, y, _, _ in b]), torch.tensor([s for _, _, s, _ in b]),
                torch.tensor([k for _, _, _, k in b]))


def main():
    global args
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--base", default="utter-project/mHuBERT-147")
    p.add_argument("--layers", type=int, default=0, help="keep only the first N transformer layers (0 = all)")
    p.add_argument("--chunk", type=int, default=30, help="whisper only: input window in seconds")
    p.add_argument("--epochs", type=float, default=6)
    p.add_argument("--steps_per_epoch", type=int, default=1500)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--min_crop", type=float, default=2.0)
    p.add_argument("--max_crop", type=float, default=8.0)
    p.add_argument("--crop_step", type=float, default=0, help="round crop/eval lengths to this many seconds (MPS: few shapes)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5, help="sampling weight = count^-alpha (0 natural, 1 balanced)")
    p.add_argument("--smooth", type=float, default=0.1)
    p.add_argument("--aug", type=float, default=0.8, help="probability a training clip is augmented")
    p.add_argument("--speed", type=float, default=0.0, help="probability of speed perturbation (speaker robustness)")
    p.add_argument("--adv", type=float, default=0.0, help="weight of the source-adversarial loss (gradient reversal)")
    p.add_argument("--chan", type=float, default=0.0, help="probability of random channel EQ / band-limiting")
    p.add_argument("--speed_lo", type=float, default=0.85)
    p.add_argument("--speed_hi", type=float, default=1.15)
    p.add_argument("--spk_adv", type=float, default=0.0, help="weight of the within-language speaker confusion loss")
    p.add_argument("--src_bal", action="store_true", help="within each language, sample recording sources equally")
    p.add_argument("--freeze_steps", type=int, default=300, help="train only the head at first")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ext", nargs="*", default=[], help="extra packed splits from prep_ext.py (training only)")
    p.add_argument("--full", action="store_true", help="train on train+validation (final model; no eval)")
    p.add_argument("--eval_max_s", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = os.path.join(ROOT, "runs", args.exp)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=1)

    # ---------------------------------------------------------------- data
    tr, va = Split("train"), Split("validation")
    # Nupe has no validation clips: hold out ~25% of Nupe train speakers so its F1 is measurable.
    train_items = [("t", i) for i in range(len(tr))]
    val_items = [("v", i) for i in range(len(va))]
    if not args.full:
        nup = [i for i in range(len(tr)) if tr.labels[i] == L2I["nup"]]
        spk = sorted({tr.speakers[i] for i in nup})
        rng = random.Random(123)
        rng.shuffle(spk)
        held, n = set(), 0
        for s in spk:
            if n >= 0.25 * len(nup):
                break
            held.add(s); n += sum(1 for i in nup if tr.speakers[i] == s)
        if len(spk) == 1:  # no speaker info: hold out a contiguous block
            held_idx = set(nup[: len(nup) // 4])
        else:
            held_idx = {i for i in nup if tr.speakers[i] in held}
        train_items = [("t", i) for i in range(len(tr)) if i not in held_idx]
        val_items += [("t", i) for i in sorted(held_idx)]
        print(f"nup holdout: {len(held_idx)} clips from {len(held)} speakers")
    else:
        train_items += val_items
        val_items = []

    splits = {"t": tr, "v": va}
    for j, name in enumerate(args.ext):
        splits[f"e{j}"] = Split(name)
        train_items += [(f"e{j}", i) for i in range(len(splits[f"e{j}"]))]
    lab = lambda it: splits[it[0]].labels[it[1]]
    counts = Counter(lab(it) for it in train_items)
    print("train counts", {LABELS[c]: counts[c] for c in range(len(LABELS))})
    src_of = lambda it: splits[it[0]].sources[it[1]]
    sw = np.array([counts[lab(it)] ** -args.alpha for it in train_items])
    if args.src_bal:  # each language's weight split equally over the sources it has
        ls = Counter((lab(it), src_of(it)) for it in train_items)
        nsrc = Counter(l for l, _ in ls)
        sw = sw * np.array([counts[lab(it)] / (nsrc[lab(it)] * ls[(lab(it), src_of(it))]) for it in train_items])
    sw /= sw.sum()
    other_items = [it for it in train_items if lab(it) == L2I["other"]]
    # Speaker classes for the speaker adversary: only sources with real speaker ids, only languages with >= 2 of them.
    spk_key = lambda it: (src_of(it), splits[it[0]].speakers[it[1]])
    by_lab = {}
    for it in train_items:
        if src_of(it) != "afrispeech_ipa":
            by_lab.setdefault(lab(it), set()).add(spk_key(it))
    spk_list = sorted({k for l, ks in by_lab.items() if len(ks) >= 2 for k in ks})
    spk_idx = {k: j for j, k in enumerate(spk_list)}
    item_spk = np.array([spk_idx.get(spk_key(it), -1) if src_of(it) != "afrispeech_ipa" else -1 for it in train_items])
    spk_mask = torch.full((len(LABELS), max(1, len(spk_list))), -1e4)  # finite: -inf gave NaN with HuBERT
    for it, j in zip(train_items, item_spk):
        if j >= 0:
            spk_mask[lab(it), j] = 0.0
    print("speaker classes", len(spk_list), {LABELS[l]: len(ks) for l, ks in by_lab.items()})

    # ---------------------------------------------------------------- model
    cfg_kw = dict(num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I)
    is_whisper = "whisper" in args.base
    if is_whisper:
        fe = AutoFeatureExtractor.from_pretrained(args.base, chunk_length=args.chunk)
        cfg = AutoConfig.from_pretrained(args.base, **cfg_kw, use_weighted_layer_sum=True)
        if args.layers:
            cfg.encoder_layers = args.layers
        full_pos = cfg.max_source_positions
        cfg.max_source_positions = args.chunk * 50  # 100 mel frames/s, conv stride 2
        model = AutoModelForAudioClassification.from_pretrained(args.base, config=cfg, ignore_mismatched_sizes=True)
        # Positional embeddings are sinusoidal-initialised and learned; keep the first chunk*50 rows.
        from transformers import WhisperForConditionalGeneration  # noqa
        import safetensors.torch
        from huggingface_hub import hf_hub_download
        sd = safetensors.torch.load_file(hf_hub_download(args.base, "model.safetensors"))
        key = [k for k in sd if k.endswith("encoder.embed_positions.weight")][0]
        model.encoder.embed_positions.weight.data.copy_(sd[key][: cfg.max_source_positions])
        backbone = model.encoder
        assert full_pos >= cfg.max_source_positions
    else:
        fe = AutoFeatureExtractor.from_pretrained(args.base)
        cfg = AutoConfig.from_pretrained(args.base, **cfg_kw, use_weighted_layer_sum=True, classifier_proj_size=256,
                                         final_dropout=0.1, layerdrop=0.05, mask_time_prob=0.05)
        if args.layers:
            cfg.num_hidden_layers = args.layers
        model = AutoModelForAudioClassification.from_pretrained(args.base, config=cfg)
        model.freeze_feature_encoder()
        backbone = getattr(model, cfg.model_type if hasattr(model, cfg.model_type) else "hubert", None) or model.base_model
    model.to(dev)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M")


    def featurize(wavs):
        """Apply the exact preprocessor that will ship in preprocessor_config.json."""
        if is_whisper:
            x = fe(wavs, sampling_rate=SR, return_tensors="pt").input_features
            return {"input_features": x}
        x = fe(wavs, sampling_rate=SR, return_tensors="pt").input_values
        return {"input_values": x}


    class BatchSampler:
        def __init__(self, steps):
            self.steps = steps

        def __len__(self):
            return self.steps

        def __iter__(self):
            rng = np.random.default_rng(args.seed + int(time.time()))
            for _ in range(self.steps):
                crop_s = rng.uniform(args.min_crop, args.max_crop)
                if args.crop_step:
                    crop_s = round(crop_s / args.crop_step) * args.crop_step
                crop_n = int(crop_s * SR)
                idx = rng.choice(len(train_items), size=args.bs, p=sw)
                yield [(int(k), crop_n, int(rng.integers(1 << 31))) for k in idx]


    collate = Collate(fe, is_whisper)


    total_steps = int(args.epochs * args.steps_per_epoch)
    dl = torch.utils.data.DataLoader(TrainDS(splits, train_items, other_items, args.aug, args.speed, args.chan, args.speed_lo, args.speed_hi, item_spk), batch_sampler=BatchSampler(total_steps), collate_fn=collate,
                                     num_workers=args.workers, persistent_workers=args.workers > 0, prefetch_factor=4)

    # Training-only source discriminator on the pooled projector output (dropped at export).
    feats = {}
    model.projector.register_forward_hook(lambda m, i, o: feats.__setitem__("h", o))
    src_head = torch.nn.Sequential(torch.nn.Linear(model.projector.out_features, 256), torch.nn.ReLU(),
                                   torch.nn.Linear(256, len(SOURCES))).to(dev)
    spk_head = torch.nn.Sequential(torch.nn.Linear(model.projector.out_features, 256), torch.nn.ReLU(),
                                   torch.nn.Linear(256, max(1, len(spk_list)))).to(dev)
    spk_mask = spk_mask.to(dev)

    head_params = [p for n, p in model.named_parameters() if not n.startswith(("hubert.", "wav2vec2.", "wavlm.", "encoder."))
                   or "layer_weights" in n]
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr},
                             {"params": list(src_head.parameters()) + list(spk_head.parameters()), "lr": args.head_lr}],
                            weight_decay=0.01)
    warm = int(0.08 * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total_steps))))


    @torch.no_grad()
    def evaluate(tag):
        model.eval()
        logits, ys, t0 = [], [], time.time()
        for s, i in val_items:
            sp = splits[s]
            n = min(sp.lengths[i], int(args.eval_max_s * SR))
            step_n = int(args.crop_step * SR)
            if step_n and n > step_n:
                n = n // step_n * step_n
            inp = {k: v.to(dev) for k, v in featurize([sp.wav(i, 0, n)]).items()}
            logits.append(model(**inp).logits.float().cpu().numpy()[0])
            ys.append(sp.labels[i])
        logits, ys = np.stack(logits), np.array(ys)
        f1, per = weighted_f1(ys, logits.argmax(1))
        acc = float((logits.argmax(1) == ys).mean())
        np.savez(os.path.join(out_dir, f"val_logits_{tag}.npz"), logits=logits, y=ys,
                 src=np.array([splits[s].sources[i] for s, i in val_items]))
        srcs = np.array([splits[s].sources[i] for s, i in val_items])
        probe = (ys == L2I["yor"]) & (srcs == "hypa_voices")  # language seen only from another source in training
        per_src = {s[:4]: round(float((logits.argmax(1)[srcs == s] == ys[srcs == s]).mean()), 3) for s in SOURCES if (srcs == s).any()}
        msg = {"tag": tag, "wf1": round(f1, 4), "acc": round(acc, 4), "xsrc_yor": round(float((logits.argmax(1)[probe] == ys[probe]).mean()), 3),
               "src_acc": per_src, "eval_s": round(time.time() - t0, 1),
               "per_class": {LABELS[c]: round(float(per[c]), 3) for c in range(len(LABELS))}}
        print(json.dumps(msg), flush=True)
        open(os.path.join(out_dir, "log.jsonl"), "a").write(json.dumps(msg) + "\n")
        model.train()
        return f1


    def set_body_trainable(flag):
        for p in body_params:
            p.requires_grad_(flag)


    model.train()
    set_body_trainable(False)
    best, t0, run_loss = -1, time.time(), 0.0
    for step, (inp, y, src, spk) in enumerate(dl):
        if step == args.freeze_steps:
            set_body_trainable(True)
        inp = {k: v.to(dev) for k, v in inp.items()}
        y, src, spk = y.to(dev), src.to(dev), spk.to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            logits = model(**inp).logits
        loss = F.cross_entropy(logits.float(), y, label_smoothing=args.smooth)
        if args.adv > 0:
            # Bounded "confusion" adversary: the head learns to spot the source on detached features, while the
            # encoder is pushed to make the (frozen-copy) head's prediction uniform. Unlike gradient reversal,
            # this loss is bounded below by log(3) and cannot blow up.
            lam = args.adv * min(1.0, step / (0.2 * total_steps))
            pooled = feats["h"].float().mean(1)
            disc_loss = F.cross_entropy(src_head(pooled.detach()), src)
            frozen = {k: v.detach() for k, v in src_head.named_parameters()}
            conf_logits = torch.func.functional_call(src_head, frozen, (pooled,))
            conf_loss = -F.log_softmax(conf_logits, -1).mean()
            loss = loss + disc_loss + lam * conf_loss
        if args.spk_adv > 0 and (spk >= 0).any():
            # Same idea within a language: the head tells apart the speakers of the clip's own language, the
            # encoder is pushed to make them indistinguishable (keeps language, drops speaker identity).
            lam = args.spk_adv * min(1.0, step / (0.2 * total_steps))
            m = spk >= 0
            pooled = feats["h"].float().mean(1)[m]
            mask = spk_mask[y[m]]
            disc = F.cross_entropy(spk_head(pooled.detach()) + mask, spk[m])
            frozen = {k: v.detach() for k, v in spk_head.named_parameters()}
            lp = F.log_softmax(torch.func.functional_call(spk_head, frozen, (pooled,)) + mask, -1)
            own = mask == 0
            conf = -(lp.masked_fill(~own, 0).sum(1) / own.sum(1)).mean()
            loss = loss + disc + lam * conf
        opt.zero_grad(set_to_none=True)
        ok = bool(torch.isfinite(loss))
        if ok:
            loss.backward()
            ok = bool(torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
        if ok:
            opt.step()
            run_loss += loss.item()
        else:  # one bad batch must not poison the weights
            print(f"step {step+1}: non-finite loss/grad, skipped", flush=True)
        sched.step()
        if dev.type == "mps" and (step + 1) % 25 == 0:
            # Raw-waveform models get a new batch length every step; MPS caches each shape and runs out of memory.
            torch.mps.empty_cache()
        if (step + 1) % 100 == 0:
            print(f"step {step+1}/{total_steps} loss {run_loss/100:.4f} {(time.time()-t0)/(step+1):.2f}s/step", flush=True)
            run_loss = 0.0
        if (step + 1) % args.steps_per_epoch == 0 or step + 1 == total_steps:
            tag = f"s{step+1}"
            if val_items:
                f1 = evaluate(tag)
                if f1 > best:
                    best = f1
                    model.save_pretrained(os.path.join(out_dir, "best"), safe_serialization=True)
                    fe.save_pretrained(os.path.join(out_dir, "best"))
                    os.replace(os.path.join(out_dir, f"val_logits_{tag}.npz"), os.path.join(out_dir, "val_logits_best.npz"))
            model.save_pretrained(os.path.join(out_dir, "last"), safe_serialization=True)
            fe.save_pretrained(os.path.join(out_dir, "last"))
    print("done best wf1", best, "minutes", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
