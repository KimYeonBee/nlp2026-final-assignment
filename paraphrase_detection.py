'''
Paraphrase detection을 위한 시작 코드.

고려 사항:
 - ParaphraseGPT: 여러분이 구현한 GPT-2 분류 모델 .
 - train: Quora paraphrase detection 데이터셋에서 ParaphraseGPT를 훈련시키는 절차.
 - test: Test 절차. 프로젝트 결과 제출에 필요한 파일들을 생성함.

실행:
  `python paraphrase_detection.py --use_gpu`
ParaphraseGPT model을 훈련 및 평가하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import (
  model_eval_paraphrase,
  model_test_paraphrase,
  model_eval_paraphrase_calibrated,
  model_test_paraphrase_calibrated,
  estimate_prior,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

import wandb # 나중에 삭제
from pathlib import Path # 나중에 삭제

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """Paraphrase Detection을 위해 설계된 여러분의 GPT-2 Model."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection 의 출력은 두 가지: 1 (yes) or 0 (no).

    # 기본적으로, 전체 모델을 finetuning 한다.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    TODO: paraphrase_detection_head Linear layer를 사용하여 토큰의 레이블을 예측하시오.

    입력은 다음과 같은 구조를 갖는다:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    따라서, 문장의 끝에서 다음 토큰에 대한 예측을 해야 할 것이다. 
    훈련이 잘 되었다면, 패러프레이즈인 경우에는 토큰 "yes"(BPE index 8505)가, 
    패러프레이즈가 아닌 경우에는 토큰 "no" (BPE index 3919)가 될 것이다.
    """
    ### 완성시켜야 할 빈 코드 블록
    outputs = self.gpt(input_ids, attention_mask)
    last_token = outputs['last_token']
    logits = self.gpt.hidden_state_to_token(last_token)

    return logits
  

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def _split_csv_arg(s):
  """콤마 구분 문자열을 리스트로 분해 (공백 제거, 빈 토큰 제외)."""
  return [t.strip() for t in s.split(",") if t.strip()]


def train(args):
  """Quora 데이터셋에서 Paraphrase Detection을 위한 GPT-2 훈련."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  para_train_data = load_paraphrase_data(args.para_train)

  # --para_dev 는 콤마 구분 다중 파일을 허용. 첫 번째 dev 가 model selection 기준.
  dev_files = _split_csv_arg(args.para_dev)
  para_dev_loaders = []
  for fp in dev_files:
    ds = ParaphraseDetectionDataset(load_paraphrase_data(fp), args)
    loader = DataLoader(ds, shuffle=False, batch_size=args.batch_size,
                        collate_fn=ds.collate_fn)
    para_dev_loaders.append((fp, loader))

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  if args.balanced_sampler:
    labels = [int(x[2]) for x in para_train_data.dataset]
    cnt0 = sum(1 for l in labels if l == 0)
    cnt1 = sum(1 for l in labels if l == 1)
    # epoch 당 pos:neg ≈ 50:50 강제. num_samples 는 원본 크기 유지.
    w0 = 0.0 if cnt0 == 0 else 0.5 / cnt0
    w1 = 0.0 if cnt1 == 0 else 0.5 / cnt1
    weights = [w1 if l == 1 else w0 for l in labels]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    para_train_dataloader = DataLoader(para_train_data, sampler=sampler, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    print(f"balanced_sampler: pos={cnt1}, neg={cnt0} → epoch 당 50:50 sampling")
  else:
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)

  args = add_arguments(args)

  # 나중에 삭제
  config = {
      "task": "quora_paraphrase_detection",
      "model": "GPT-2",
      "train_file": args.para_train,
      "dev_file": args.para_dev,
      "test_file": args.para_test,
      "batch_size": args.batch_size,
      "epochs": args.epochs,
      "lr": args.lr,
      "weight_decay": args.weight_decay,
      "patience": args.patience,
      "use_gpu": args.use_gpu,
      "device": str(device),
      "dataset": 'quora-paws-swap-bt-hardneg(medium)'
  }

  if args.use_wandb:
    wandb.init(
        project='paraphrase_detection',
        # name=f'gpt2-{config["dataset"]}-lr{args.lr}-epoch{args.epochs}-patience{args.patience}-weight_decay{args.weight_decay}', # 기타 수정한 사항 name에 구분가게 표시
        # >>> ABLATION-ONLY: 실험 끝나면 group 줄 삭제 + name 줄을 위 주석으로 되돌리기 <<<
        group='Ablation',
        name=f'gpt2-ablation-{config["dataset"]}-epoch{args.epochs}',
        # >>> END ABLATION-ONLY <<<
        config=config
    )

  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
  best_dev_acc = 0
  best_epoch = -1
  no_improvement = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    train_correct = 0
    train_total = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1
      train_correct += (preds == labels).sum().item()
      train_total += labels.size(0)

    train_loss = train_loss / num_batches
    train_acc = train_correct / train_total

    # 다중 dev: 첫 번째 = primary (early stopping / 저장 기준), 나머지는 모니터링용
    dev_metrics = {}
    for fp, loader in para_dev_loaders:
      acc, f1, *_ = model_eval_paraphrase(loader, model, device)
      dev_metrics[fp] = (acc, f1)
      if fp != para_dev_loaders[0][0]:
        print(f"  aux dev [{fp}] acc :: {acc :.3f}, f1 :: {f1 :.3f}")
    dev_acc, dev_f1 = dev_metrics[para_dev_loaders[0][0]]

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      best_epoch = epoch
      no_improvement = 0
      save_model(model, optimizer, args, args.filepath)
    else:
      no_improvement += 1

    # 나중에 삭제
    # >>> ABLATION-ONLY: 아래 if 블록 전체가 ablation 전용. 실험 끝나면 이 블록 삭제 + 바로 아래 # 주석 블록 부활 <<<
    if args.use_wandb:
      ablation_payload = {
          "epoch": epoch,
          "train_loss": train_loss,
          "train_acc": train_acc,
          "best_dev_acc": best_dev_acc,
          "best_epoch": best_epoch,
          "lr": lr,
      }
      # dev 값을 source (quora / paws / mrpc) 별로 분리해서 기록
      for fp, (acc, f1) in dev_metrics.items():
        fp_low = fp.lower()
        if "quora" in fp_low:
          src = "quora"
        elif "paws" in fp_low:
          src = "paws"
        elif "mrpc" in fp_low:
          src = "mrpc"
        else:
          src = fp.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        ablation_payload[f"dev_acc_{src}"] = acc
        ablation_payload[f"dev_f1_{src}"] = f1
      wandb.log(ablation_payload)
    # >>> END ABLATION-ONLY <<<

    # 실험 종료 후 위 블록을 삭제하고, 아래 원본 블록의 # 만 제거하여 복원
    # if args.use_wandb:
    #   log_payload = {
    #       "epoch": epoch,
    #       "train_loss": train_loss,
    #       "train_acc": train_acc,
    #       "dev_acc": dev_acc,
    #       "dev_f1": dev_f1,
    #       "best_dev_acc": best_dev_acc,
    #       "best_epoch": best_epoch,
    #       "lr": lr,
    #   }
    #   # 보조 dev 메트릭은 파일명을 키 prefix 로 추가 (Quora 외 OOD 모니터링용)
    #   for fp, (acc, f1) in dev_metrics.items():
    #     if fp == para_dev_loaders[0][0]:
    #       continue
    #     # 파일명만 추출 (stdlib 추가 없이 string slicing)
    #     tag = fp.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    #     log_payload[f"dev_acc_{tag}"] = acc
    #     log_payload[f"dev_f1_{tag}"] = f1
    #   wandb.log(log_payload)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f} (best {best_dev_acc :.3f} @ epoch {best_epoch})")

    if args.patience is not None and no_improvement >= args.patience:
      print(f"Early stopping at epoch {epoch} (best dev acc {best_dev_acc :.3f} @ epoch {best_epoch})")
      break

@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  # 다중 dev / test: 콤마 구분 리스트 허용. 입력 - 출력 파일 수는 정확히 매칭되어야 함.
  dev_files = _split_csv_arg(args.para_dev)
  dev_out_files = _split_csv_arg(args.para_dev_out)
  test_files = _split_csv_arg(args.para_test)
  test_out_files = _split_csv_arg(args.para_test_out)
  assert len(dev_files) == len(dev_out_files), \
      f"--para_dev ({len(dev_files)}) 와 --para_dev_out ({len(dev_out_files)}) 길이가 다릅니다."
  assert len(test_files) == len(test_out_files), \
      f"--para_test ({len(test_files)}) 와 --para_test_out ({len(test_out_files)}) 길이가 다릅니다."

  # --prior_calibration: 첫 dev set 의 dataset 으로 dummy pair 한 번 forward → prior 추정.
  # prior_yes / prior_no 는 모든 dev/test 셋에 동일하게 차감됨.
  prior_yes, prior_no = 0.0, 0.0
  if args.prior_calibration:
    first_ds = ParaphraseDetectionDataset(load_paraphrase_data(dev_files[0]), args)
    prior_yes, prior_no = estimate_prior(model, first_ds, device)
    print(f"prior_calibration: prior_yes={prior_yes:.4f}, prior_no={prior_no:.4f} "
          f"(차이 {prior_yes - prior_no:+.4f} — yes 쪽으로 편향이면 양수)")

  for dev_fp, dev_out_fp in zip(dev_files, dev_out_files):
    data = load_paraphrase_data(dev_fp)
    ds = ParaphraseDetectionDataset(data, args)
    loader = DataLoader(ds, shuffle=False, batch_size=args.batch_size, collate_fn=ds.collate_fn)
    if args.prior_calibration:
      dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase_calibrated(
          loader, model, device, prior_yes=prior_yes, prior_no=prior_no)
    else:
      dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(loader, model, device)
    print(f"dev paraphrase acc [{dev_fp}] :: {dev_para_acc :.3f}")
    with open(dev_out_fp, "w+") as f:
      f.write(f"id \t Predicted_Is_Paraphrase \n")
      for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
        f.write(f"{p}, {s} \n")

  for test_fp, test_out_fp in zip(test_files, test_out_files):
    data = load_paraphrase_data(test_fp, split='test')
    ds = ParaphraseDetectionTestDataset(data, args)
    loader = DataLoader(ds, shuffle=True, batch_size=args.batch_size, collate_fn=ds.collate_fn)
    if args.prior_calibration:
      test_para_y_pred, test_para_sent_ids = model_test_paraphrase_calibrated(
          loader, model, device, prior_yes=prior_yes, prior_no=prior_no)
    else:
      test_para_y_pred, test_para_sent_ids = model_test_paraphrase(loader, model, device)
    print(f"test predictions saved [{test_fp}] -> {test_out_fp}")
    with open(test_out_fp, "w+") as f:
      f.write(f"id \t Predicted_Is_Paraphrase \n")
      for p, s in zip(test_para_sent_ids, test_para_y_pred):
        f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  # 나중에 삭제
  parser.add_argument('--use_wandb', action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--patience", type=int, default=None,
                      help="early stopping patience (epochs without dev improvement). 지정하지 않으면 비활성화")
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--balanced_sampler", action='store_true',
                      help="WeightedRandomSampler 로 epoch 당 pos:neg 50:50 강제 (학습 데이터 mix 가 한쪽으로 쏠릴 때 FPR/FNR 균형 회복)")
  parser.add_argument("--prior_calibration", action='store_true',
                      help="추론 시 빈 문장 페어로 prior (yes/no logit) 추정 후 실제 logit 에서 차감. 사전 편향 제거 — bt 셀처럼 yes 쪽으로 쏠린 모델 보정에 사용.")
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """모델 크기에 따라 결정되는 인수들을 추가."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  # args.filepath = f'{args.epochs}-{args.lr}-wd{args.weight_decay}-pat{args.patience}-paraphrase.pt'  # 경로명 저장.
  args.filepath = f'{Path(args.para_train).stem}-{args.epochs}-...-paraphrase.pt'
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train(args)
  test(args)

  # 나중에 삭제
  if args.use_wandb:
    wandb.finish()
