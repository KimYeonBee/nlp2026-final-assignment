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
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

import wandb # 나중에 삭제

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

  def forward(self, input_ids, attention_mask, return_lm_logits=False):
    """
    TODO: paraphrase_detection_head Linear layer를 사용하여 토큰의 레이블을 예측하시오.

    입력은 다음과 같은 구조를 갖는다:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    따라서, 문장의 끝에서 다음 토큰에 대한 예측을 해야 할 것이다.
    훈련이 잘 되었다면, 패러프레이즈인 경우에는 토큰 "yes"(BPE index 8505)가,
    패러프레이즈가 아닌 경우에는 토큰 "no" (BPE index 3919)가 될 것이다.

    return_lm_logits=True 일 때 (class_logits, lm_logits) 튜플 반환.
    lm_logits 는 전체 시퀀스에 대한 vocab logits [B, S, V] — LM loss 계산용.
    """
    ### 완성시켜야 할 빈 코드 블록
    outputs = self.gpt(input_ids, attention_mask)
    last_token = outputs['last_token']
    logits = self.gpt.hidden_state_to_token(last_token)

    if return_lm_logits:
      lm_logits = self.gpt.hidden_state_to_token(outputs['last_hidden_state'])
      return logits, lm_logits
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


def train(args):
  """Quora 데이터셋에서 Paraphrase Detection을 위한 GPT-2 훈련."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

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
      "lm_lambda": args.lm_lambda,
      "use_gpu": args.use_gpu,
      "device": str(device),
      "dataset": 'quora'
  }

  if args.use_wandb:
    wandb.init(
        project='paraphrase_detection',
        name=f'gpt2-{config["dataset"]}-lr{args.lr}-epoch{args.epochs}-patience{args.patience}-wd{args.weight_decay}-lmlam{args.lm_lambda}', # 기타 수정한 사항 name에 구분가게 표시
        config=config
    )

  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
  best_dev_acc = 0
  best_epoch = -1
  no_improvement = 0

  use_lm_loss = args.lm_lambda > 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    train_class_loss = 0
    train_lm_loss = 0
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
      if use_lm_loss:
        logits, lm_logits = model(b_ids, b_mask, return_lm_logits=True)
        # next-token prediction: position i predicts token i+1
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = b_ids[:, 1:].contiguous()
        shift_mask = b_mask[:, 1:].contiguous()
        # padding 위치는 ignore_index=-100 로 마스킹
        shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean',
        )
      else:
        logits = model(b_ids, b_mask)
        lm_loss = torch.tensor(0.0, device=device)

      class_loss = F.cross_entropy(logits, labels, reduction='mean')
      loss = class_loss + args.lm_lambda * lm_loss
      preds = torch.argmax(logits, dim=1)

      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      train_class_loss += class_loss.item()
      train_lm_loss += lm_loss.item()
      num_batches += 1
      train_correct += (preds == labels).sum().item()
      train_total += labels.size(0)

    train_loss = train_loss / num_batches
    train_class_loss = train_class_loss / num_batches
    train_lm_loss = train_lm_loss / num_batches
    train_acc = train_correct / train_total

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      best_epoch = epoch
      no_improvement = 0
      save_model(model, optimizer, args, args.filepath)
    else:
      no_improvement += 1

    # 나중에 삭제
    if args.use_wandb:
      wandb.log({
          "epoch": epoch,
          "train_loss": train_loss,
          "train_class_loss": train_class_loss,
          "train_lm_loss": train_lm_loss,
          "train_acc": train_acc,
          "dev_acc": dev_acc,
          "dev_f1": dev_f1,
          "best_dev_acc": best_dev_acc,
          "best_epoch": best_epoch,
          "lr": lr,
      })

    print(f"Epoch {epoch}: total loss :: {train_loss :.3f} (class {train_class_loss :.3f}, lm {train_lm_loss :.3f}), train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f} (best {best_dev_acc :.3f} @ epoch {best_epoch})")

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

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
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
  parser.add_argument("--lm_lambda", type=float, default=0.0,
                      help="classification loss + lambda * LM loss. 0이면 LM loss 비활성 (기본 동작)")
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
  args.filepath = f'{args.epochs}-{args.lr}-wd{args.weight_decay}-pat{args.patience}-lmlam{args.lm_lambda}-paraphrase.pt'  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train(args)
  test(args)

  # 나중에 삭제
  if args.use_wandb:
    wandb.finish()
