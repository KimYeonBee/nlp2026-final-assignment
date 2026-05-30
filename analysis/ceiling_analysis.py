#!/usr/bin/env python3
"""천장(label-noise) 증명용 분석 — 최종 모델로 dev logit 덤프 후:
  (1) Selective prediction: confidence 상위 coverage 별 정확도 (risk-coverage)
  (2) Confident-error 표집: 모델이 가장 confident 하게 틀린 dev 케이스 추출
GPU 는 dev forward 1패스에만 사용(나머지는 CPU). 결과를 CSV/표로 저장.
"""
import sys, os, csv, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader

import evaluation as E
E.TQDM_DISABLE = True
from evaluation import YES_TOKEN_ID, NO_TOKEN_ID
from paraphrase_detection import ParaphraseGPT
from datasets import ParaphraseDetectionDataset, load_paraphrase_data

CKPT = "quora_paws_raw_swap_bt_hardneg_train-10-...-paraphrase.pt"
DEVS = [
    ("Quora", "data/quora-dev.csv"),
    ("PAWS",  "data/paraphrase_extra_data/paws_dev.csv"),
    ("MRPC",  "data/paraphrase_extra_data/mrpc_traindev.csv"),
]
BATCH = 16
COVERAGES = [1.0, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
TOPK_ERR = 30
OUTDIR = "analysis"

@torch.no_grad()
def collect(model, raw, device):
    """dev 한 셋의 (yes_logit, no_logit) per-example 수집 (shuffle=False 로 raw 와 정렬)."""
    ds = ParaphraseDetectionDataset(raw, _ARGS)
    loader = DataLoader(ds, shuffle=False, batch_size=BATCH, collate_fn=ds.collate_fn)
    yes, no = [], []
    for batch in loader:
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        logits = model(b_ids, b_mask)
        yes.extend(logits[:, YES_TOKEN_ID].float().cpu().numpy().tolist())
        no.extend(logits[:, NO_TOKEN_ID].float().cpu().numpy().tolist())
    return np.array(yes), np.array(no)

def analyze(name, raw, yes, no):
    gold = np.array([int(r[2]) for r in raw])
    margin = yes - no                      # >0 → yes
    pred = (margin > 0).astype(int)
    # confidence = 두 토큰 softmax 의 예측확률
    m = np.maximum(yes, no)
    p_yes = np.exp(yes - m); p_no = np.exp(no - m)
    p_yes, p_no = p_yes/(p_yes+p_no), p_no/(p_yes+p_no)
    conf = np.maximum(p_yes, p_no)
    acc = (pred == gold).mean()

    # (1) selective prediction
    order = np.argsort(-conf)              # confident 순
    pred_s, gold_s = pred[order], gold[order]
    n = len(gold)
    rc = []
    for cov in COVERAGES:
        k = max(1, int(round(cov*n)))
        a = (pred_s[:k] == gold_s[:k]).mean()
        thr = conf[order][k-1]
        rc.append((cov, k, a, thr))

    # (2) confident errors
    wrong = np.where(pred != gold)[0]
    wrong_sorted = wrong[np.argsort(-conf[wrong])]
    errs = []
    for i in wrong_sorted[:TOPK_ERR]:
        errs.append((raw[i][3], int(gold[i]), int(pred[i]), float(conf[i]), raw[i][0], raw[i][1]))
    return acc, rc, errs, conf, pred, gold

def main():
    global _ARGS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CKPT, map_location=device)
    _ARGS = saved["args"]
    model = ParaphraseGPT(_ARGS); model.load_state_dict(saved["model"]); model = model.to(device); model.eval()
    print(f"[loaded] {CKPT}  size={_ARGS.model_size}  device={device}\n")

    for name, fp in DEVS:
        raw = load_paraphrase_data(fp)
        yes, no = collect(model, raw, device)
        acc, rc, errs, conf, pred, gold = analyze(name, raw, yes, no)
        print(f"================== {name} dev (n={len(raw)}) — overall acc {acc*100:.2f}% ==================")
        print("  [Selective prediction] coverage | n_kept | acc(%) | conf_threshold")
        for cov, k, a, thr in rc:
            print(f"    {cov*100:5.0f}% | {k:6d} | {a*100:6.2f} | conf≥{thr:.3f}")
        # confident error CSV
        out = os.path.join(OUTDIR, f"confident_errors_{name.lower()}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "gold", "pred", "confidence", "sentence1", "sentence2"])
            for row in errs:
                w.writerow(row)
        print(f"  [Confident errors] 상위 {TOPK_ERR}개 → {out}")
        for (sid, g, p, c, s1, s2) in errs[:8]:
            print(f"    conf={c:.3f} gold={g} pred={p} | {s1[:60]!r} <-> {s2[:60]!r}")
        print()

if __name__ == "__main__":
    main()
