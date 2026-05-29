# !/usr/bin/env python3

"""
Quora paraphrase detection을 위한 평가.

model_eval_paraphrase: 레이블 정보가 있는 dev 및 train dataloader에 적합함.
model_test_paraphrase: 레이블 정보가 없는 test dataloader에 적합.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import (
  SonnetsDataset,
)

TQDM_DISABLE = False

YES_TOKEN_ID = 8505
NO_TOKEN_ID = 3919


@torch.no_grad()
def estimate_prior(model, dataset, device, dummy_pair=("", "")):
  """모델이 입력 문장과 무관하게 가지는 yes/no logit 편향을 추정.

  ParaphraseDetectionDataset 의 collate_fn 으로 dummy_pair 를 tokenize 후 한 번 forward.
  반환된 (prior_yes, prior_no) 는 추론 시 logit 에서 차감해서 사전 편향을 제거하는 데 사용.
  """
  model.eval()
  dummy = [(dummy_pair[0], dummy_pair[1], 0, "__prior__")]
  batch = dataset.collate_fn(dummy)
  b_ids = batch['token_ids'].to(device)
  b_mask = batch['attention_mask'].to(device)
  logits = model(b_ids, b_mask)[0]
  return float(logits[YES_TOKEN_ID].item()), float(logits[NO_TOKEN_ID].item())


def _calibrated_preds(logits, prior_yes, prior_no):
  """vocab logits 에서 yes/no 만 비교 (prior 차감). 기존 함수와 호환되도록 token id 반환."""
  yes_score = logits[:, YES_TOKEN_ID] - prior_yes
  no_score = logits[:, NO_TOKEN_ID] - prior_no
  preds = np.where(
      yes_score.cpu().numpy() > no_score.cpu().numpy(),
      YES_TOKEN_ID, NO_TOKEN_ID,
  ).astype(np.int64).flatten()
  return preds


@torch.no_grad()
def model_eval_paraphrase_calibrated(dataloader, model, device, prior_yes=0.0, prior_no=0.0):
  """model_eval_paraphrase 와 동일하지만 yes/no logit 에서 prior 를 차감 후 argmax."""
  model.eval()
  y_true, y_pred, sent_ids = [], [], []
  for batch in tqdm(dataloader, desc='eval-calibrated', disable=TQDM_DISABLE):
    b_ids = batch['token_ids'].to(device)
    b_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].flatten()
    b_sent_ids = batch['sent_ids']

    logits = model(b_ids, b_mask)
    preds = _calibrated_preds(logits, prior_yes, prior_no)

    y_true.extend(labels.cpu().numpy() if torch.is_tensor(labels) else labels)
    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)
  return acc, f1, y_pred, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase_calibrated(dataloader, model, device, prior_yes=0.0, prior_no=0.0):
  """레이블 없는 test split 용 calibrated 예측."""
  model.eval()
  y_pred, sent_ids = [], []
  for batch in tqdm(dataloader, desc='eval-calibrated', disable=TQDM_DISABLE):
    b_ids = batch['token_ids'].to(device)
    b_mask = batch['attention_mask'].to(device)
    b_sent_ids = batch['sent_ids']

    logits = model(b_ids, b_mask)
    preds = _calibrated_preds(logits, prior_yes, prior_no)

    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)
  return y_pred, sent_ids


@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids, labels = batch['token_ids'], batch['attention_mask'], batch['sent_ids'], batch[
      'labels'].flatten()

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_true.extend(labels)
    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  return y_pred, sent_ids


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()  # Character n-gram F-score

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)