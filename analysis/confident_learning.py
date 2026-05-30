#!/usr/bin/env python3
"""Confident learning — 라벨 노이즈 탐지·정제 (GPU1 트랙).
최종 모델로 Quora train 을 스코어링하여, 모델이 confident 하게 라벨과 불일치하는 예제를
'노이즈 후보'로 플래그하고, 임계값별로 제거한 정제 학습 CSV 를 생성한다.
이후 `paraphrase_detection.py --para_train <정제 CSV>` 로 재학습해 dev 변화를 본다.

주의: 최종 모델은 이 train 을 포함해 학습됐으므로 in-sample 스코어(메모리제이션 편향)다.
      confident 불일치는 강한 노이즈 신호지만, 엄밀히는 k-fold out-of-fold 예측이 더 정확하다.
"""
import sys, os, csv
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
TRAIN_CSV = "data/quora-train.csv"
THRESHOLDS = [0.90, 0.95, 0.99]
BATCH = 16
OUTDIR = "data/paraphrase_extra_data"

@torch.no_grad()
def score(model, raw, device):
    ds = ParaphraseDetectionDataset(raw, _ARGS)
    loader = DataLoader(ds, shuffle=False, batch_size=BATCH, collate_fn=ds.collate_fn)
    yes, no = [], []
    for batch in loader:
        logits = model(batch['token_ids'].to(device), batch['attention_mask'].to(device))
        yes.extend(logits[:, YES_TOKEN_ID].float().cpu().numpy().tolist())
        no.extend(logits[:, NO_TOKEN_ID].float().cpu().numpy().tolist())
    yes, no = np.array(yes), np.array(no)
    pred = (yes > no).astype(int)
    m = np.maximum(yes, no)
    py, pn = np.exp(yes - m), np.exp(no - m)
    conf = np.maximum(py, pn) / (py + pn)
    return pred, conf

def main():
    global _ARGS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CKPT, map_location=device)
    _ARGS = saved["args"]
    model = ParaphraseGPT(_ARGS); model.load_state_dict(saved["model"]); model = model.to(device); model.eval()

    raw = load_paraphrase_data(TRAIN_CSV)             # (s1,s2,label,id)
    gold = np.array([int(r[2]) for r in raw])
    pred, conf = score(model, raw, device)
    disagree = pred != gold
    print(f"train n={len(raw)}  model agree={100*(~disagree).mean():.2f}%  disagree={disagree.sum()}")

    # 플래그 = confident 불일치
    flagged_by_id = {}                                 # id_lower -> (gold,pred,conf)
    for i in np.where(disagree)[0]:
        flagged_by_id[raw[i][3]] = (int(gold[i]), int(pred[i]), float(conf[i]))

    # 검수용: confident 불일치 상위 목록
    insp = os.path.join("analysis", "noise_candidates_quora.csv")
    rows = sorted(((raw[i][3], int(gold[i]), int(pred[i]), float(conf[i]), raw[i][0], raw[i][1])
                   for i in np.where(disagree)[0]), key=lambda r: -r[3])
    with open(insp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id","gold","pred","confidence","sentence1","sentence2"])
        w.writerows(rows)
    print(f"[검수용] confident 불일치 {len(rows)}개 → {insp}")

    # 임계값별 정제 CSV (원본 포맷 유지, 플래그 id 제거)
    for thr in THRESHOLDS:
        drop = {sid for sid,(g,p,c) in flagged_by_id.items() if c >= thr}
        out = os.path.join(OUTDIR, f"quora_train_clean_p{int(thr*100)}.csv")
        kept = 0
        with open(TRAIN_CSV, encoding="utf-8-sig") as fin, open(out, "w", newline="", encoding="utf-8") as fout:
            r = csv.DictReader(fin, delimiter="\t"); w = csv.DictWriter(fout, fieldnames=r.fieldnames, delimiter="\t")
            w.writeheader()
            for rec in r:
                if rec["id"].strip().lower() in drop:
                    continue
                w.writerow(rec); kept += 1
        print(f"  thr={thr:.2f}: 제거 {len(drop):6d}  유지 {kept:6d}  → {out}")

    print("\n다음 단계: 정제 CSV 로 small 재학습 후 dev 비교 →")
    print("  python paraphrase_detection.py --use_gpu --model_size gpt2 --epochs 6 \\")
    print("    --para_train data/paraphrase_extra_data/quora_train_clean_p95.csv \\")
    print("    --para_dev data/quora-dev.csv --para_test data/quora-test-student.csv \\")
    print("    --para_dev_out predictions/para-dev-quora-clean95.csv --para_test_out predictions/para-test-quora-clean95.csv")

if __name__ == "__main__":
    main()
