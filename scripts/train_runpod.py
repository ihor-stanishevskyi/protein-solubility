#!/usr/bin/env python3
"""Train the solubility model on a rented GPU and leave one archive behind.

Written for a RunPod pod after Colab ran out of free GPUs on 24.11.2026. It is
the Colab notebook with the Colab-only parts removed and every trap found that
day already closed:

  - onnxscript is installed, or torch.onnx.export dies after the training;
  - the exporter's external weight file is folded back in, so model.onnx is
    self-contained. A graph without its weights still loads where the weights
    happen to sit next to it, which is how that defect passed every check;
  - the three outputs are zipped into one file, so one download gets all of
    them and nothing is left behind.

Hyperparameters are frozen to the 2026-11-07 run. Changing any of them makes
the comparison with that run meaningless, which is the only reason we can tell
a good run from a broken one.

    pip install -q transformers datasets scikit-learn onnx onnxruntime onnxscript
    python train_runpod.py --out /workspace/out
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time

MODEL = "facebook/esm2_t12_35M_UR50D"
MAX_LEN, EPOCHS, BATCH, LR = 512, 2, 32, 2e-5
REFERENCE = {"mcc": 0.4754, "auroc": 0.8344}   # RTX 4090, 2026-11-07
DRIFT = 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/workspace/out"))
    args = ap.parse_args()
    out: pathlib.Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Explicit raises, not asserts: python -O removes asserts, and a guard an
    # interpreter flag can switch off is not a guard.
    if not torch.cuda.is_available():
        raise SystemExit("no GPU - this script is not worth running on a CPU")
    card = torch.cuda.get_device_name(0)
    print(f"GPU: {card}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB",
          flush=True)

    # ---- data: the same set as 2026-11-07, or the comparison is void --------
    ds = load_dataset("proteinea/solubility")

    def pick(split: str):
        d = ds[split]
        c = d.column_names
        sk = "sequences" if "sequences" in c else ("sequence" if "sequence" in c else c[0])
        lk = "labels" if "labels" in c else ("label" if "label" in c else c[-1])
        return list(d[sk]), [int(x) for x in d[lk]]

    tr_s, tr_y = pick("train")
    te_s, te_y = pick("test" if "test" in ds else "validation")
    print(f"train={len(tr_s)}  test={len(te_s)}", flush=True)
    if len(tr_s) <= 60000:
        raise SystemExit(f"train split is {len(tr_s)}, expected >60000 - "
                         "the comparison with 07.11 would be void")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).cuda()
    for n, p in model.named_parameters():
        # No gradient reaches it, and leaving it in makes distributed training fail.
        if "contact_head" in n:
            p.requires_grad = False
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f} M", flush=True)

    class Seqs(Dataset):
        def __init__(self, s, y):
            self.s, self.y = list(s), [int(v) for v in y]

        def __len__(self):
            return len(self.s)

        def __getitem__(self, i):
            return self.s[i], self.y[i]

    def collate(batch):
        seqs = [s[:MAX_LEN] for s, _ in batch]
        ys = torch.tensor([y for _, y in batch])
        enc = tok(seqs, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN)
        return enc, ys

    loader = DataLoader(Seqs(tr_s, tr_y), batch_size=BATCH, shuffle=True,
                        collate_fn=collate, num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")

    # ---- train --------------------------------------------------------------
    for ep in range(EPOCHS):
        model.train()
        t0, total = time.perf_counter(), 0.0
        for i, (enc, ys) in enumerate(loader):
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            ys = ys.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                res = model(**enc, labels=ys)
            scaler.scale(res.loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            total += res.loss.item()
            if i % 200 == 0:
                print(f"  epoch {ep+1}, batch {i}/{len(loader)}, loss {res.loss.item():.4f}",
                      flush=True)
        print(f"epoch {ep+1}/{EPOCHS}: {time.perf_counter()-t0:.0f}s, "
              f"mean loss {total/len(loader):.4f}", flush=True)
        torch.save(model.state_dict(), out / f"checkpoint_epoch{ep+1}.pt")

    # ---- measure ------------------------------------------------------------
    model.eval()
    probs = []
    ev = DataLoader(Seqs(te_s, te_y), batch_size=64, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for enc, _ in ev:
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                probs.append(torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu())
    probs = torch.cat(probs).numpy()
    pred = (probs > 0.5).astype(int)
    auroc = float(roc_auc_score(te_y, probs))
    mcc = float(matthews_corrcoef(te_y, pred))
    acc = float(accuracy_score(te_y, pred))

    print(f"\nMCC   {mcc:.4f}   (07.11: {REFERENCE['mcc']})")
    print(f"AUROC {auroc:.4f}   (07.11: {REFERENCE['auroc']})")
    print(f"accuracy {acc:.4f}")
    if abs(mcc - REFERENCE["mcc"]) > DRIFT:
        print("\nWARNING: further from the reference than run-to-run noise. "
              "Do not record this as a result until the gap is explained.", flush=True)
    else:
        print("\nWithin noise of the 07.11 run - reproduced.", flush=True)

    # ---- export, as one file ------------------------------------------------
    import onnx
    import onnxruntime as ort

    model_cpu = model.cpu().eval().float()
    sample = tok(["MKTVRQERLKSIVRILERSKEPVSGAQ"], return_tensors="pt", padding=True)
    path = str(out / "model.onnx")
    torch.onnx.export(
        model_cpu, (sample["input_ids"], sample["attention_mask"]), path,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
    )
    if os.path.exists(path + ".data"):
        # The weights went to a sidecar file. onnxruntime loads it silently
        # while it sits next to the graph, so every check here would pass and
        # the server - which gets only model.onnx - would hold an empty shell.
        onnx.save(onnx.load(path), path, save_as_external_data=False)
        os.remove(path + ".data")
        print("folded the weights back into model.onnx", flush=True)

    size_mb = os.path.getsize(path) / 1e6
    print(f"model.onnx: {size_mb:.1f} MB", flush=True)
    if size_mb <= 100:
        raise SystemExit(f"model.onnx is {size_mb:.1f} MB; 33.5 M parameters cannot "
                         "fit in that, so the file has no weights in it")

    # ---- prove the exported file answers like the trained one ---------------
    probe = te_s[:16]
    enc = tok([s[:MAX_LEN] for s in probe], return_tensors="pt",
              padding=True, truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        torch_logits = model_cpu(**enc).logits.numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(None, {"input_ids": enc["input_ids"].numpy(),
                                  "attention_mask": enc["attention_mask"].numpy()})[0]
    diff = float(np.abs(torch_logits - onnx_logits).max())
    print(f"largest PyTorch-vs-ONNX difference: {diff}", flush=True)
    if diff >= 1e-3:
        raise SystemExit(f"ONNX computes something else (largest difference {diff}) - "
                         "the export is broken, do not ship this file")

    # ---- what travels to the server -----------------------------------------
    (out / "vocab.json").write_text(json.dumps(
        {"vocab": tok.get_vocab(), "cls": tok.cls_token_id, "eos": tok.eos_token_id,
         "pad": tok.pad_token_id, "unk": tok.unk_token_id, "max_len": MAX_LEN},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({
        "model": MODEL, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "max_len": MAX_LEN,
        "trained_on": f"RunPod {card}, fp16 AMP",
        "n_train": len(tr_s), "n_test": len(te_s),
        "mcc": round(mcc, 4), "auroc": round(auroc, 4), "accuracy": round(acc, 4),
        "reference_2026_11_07": REFERENCE,
        "onnx_vs_torch_max_diff": diff,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # One archive, because three separate downloads is how a file gets left behind.
    staging = out / "for-s3"
    staging.mkdir(exist_ok=True)
    for n in ("model.onnx", "vocab.json", "metrics.json"):
        shutil.copy2(out / n, staging / n)
    archive = shutil.make_archive(str(out / "solubility-model"), "zip", str(staging))
    print(f"\nDOWNLOAD THIS ONE FILE: {archive} "
          f"({os.path.getsize(archive)/1e6:.1f} MB)", flush=True)
    print("It holds model.onnx, vocab.json and metrics.json - all three, nothing else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
