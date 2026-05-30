#!/usr/bin/env python3
"""최종 Phase 2 체크포인트(재학습 없이)에 추론-time 기법만 적용해 dev 정확도 측정.
 - baseline / symmetric eval / prior calibration(도메인별) / symmetric+prior 4종을
   Quora·PAWS·MRPC dev 에 대해 평가하고 표로 출력.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

import evaluation as E
E.TQDM_DISABLE = True

from paraphrase_detection import ParaphraseGPT
from datasets import ParaphraseDetectionDataset, load_paraphrase_data
from evaluation import (
    model_eval_paraphrase, model_eval_paraphrase_symmetric,
    model_eval_paraphrase_calibrated, estimate_prior,
)

CKPT = "quora_paws_raw_swap_bt_hardneg_train-10-...-paraphrase.pt"
DEVS = [
    ("Quora", "data/quora-dev.csv"),
    ("PAWS",  "data/paraphrase_extra_data/paws_dev.csv"),
    ("MRPC",  "data/paraphrase_extra_data/mrpc_traindev.csv"),
]
BATCH = 16

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CKPT, map_location=device)
    args = saved["args"]
    model = ParaphraseGPT(args)
    model.load_state_dict(saved["model"])
    model = model.to(device)
    model.eval()
    print(f"[loaded] {CKPT}  model_size={args.model_size}  device={device}")

    rows = []
    for name, fp in DEVS:
        raw = load_paraphrase_data(fp)
        ds       = ParaphraseDetectionDataset(raw, args)
        ds_swap  = ParaphraseDetectionDataset(raw, args, swap=True)
        loader      = DataLoader(ds,      shuffle=False, batch_size=BATCH, collate_fn=ds.collate_fn)
        loader_swap = DataLoader(ds_swap, shuffle=False, batch_size=BATCH, collate_fn=ds_swap.collate_fn)

        base_acc, base_f1, *_ = model_eval_paraphrase(loader, model, device)
        sym_acc,  sym_f1,  *_ = model_eval_paraphrase_symmetric(loader, loader_swap, model, device)
        py, pn = estimate_prior(model, ds, device)
        pri_acc, pri_f1, *_ = model_eval_paraphrase_calibrated(loader, model, device, prior_yes=py, prior_no=pn)
        sp_acc,  sp_f1,  *_ = model_eval_paraphrase_symmetric(loader, loader_swap, model, device, prior_yes=py, prior_no=pn)

        rows.append((name, base_acc, base_f1, sym_acc, sym_f1, pri_acc, pri_f1, sp_acc, sp_f1, py - pn))
        print(f"[{name}] prior(yes-no)={py-pn:+.3f}  "
              f"base={base_acc*100:.1f}/{base_f1*100:.1f}  sym={sym_acc*100:.1f}/{sym_f1*100:.1f}  "
              f"prior={pri_acc*100:.1f}/{pri_f1*100:.1f}  sym+prior={sp_acc*100:.1f}/{sp_f1*100:.1f}")

    print("\n================ 최종 모델(epoch9) 추론-time 기법 — dev acc/F1 (%) ================")
    hdr = f"{'dev':6} | {'baseline':>13} | {'symmetric':>13} | {'prior(dom)':>13} | {'sym+prior':>13} | prior(y-n)"
    print(hdr); print("-"*len(hdr))
    for (name, ba,bf, sa,sf, pa,pf, spa,spf, pri) in rows:
        print(f"{name:6} | {ba*100:5.1f}/{bf*100:4.1f}    | {sa*100:5.1f}/{sf*100:4.1f}    | "
              f"{pa*100:5.1f}/{pf*100:4.1f}    | {spa*100:5.1f}/{spf*100:4.1f}    | {pri:+.3f}")

if __name__ == "__main__":
    main()
