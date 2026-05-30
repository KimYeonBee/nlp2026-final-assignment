#!/usr/bin/env bash
# LR warmup+decay sweep (GPU2 / 원격 서버 트랙).
# small(gpt2) · 전체 hardneg mix + balanced · 6 epoch 로 schedule 3종 비교.
# dev acc 는 각 로그 마지막의 "best ..." 줄에서 확인. 체크포인트는 config 간 덮어써짐(무시).
set -e
cd "$(dirname "$0")"

PY=${PY:-python}                     # 필요시 PY=~/miniconda3/envs/nlp_final/bin/python
TRAIN=data/paraphrase_extra_data/quora_paws_raw_swap_bt_hardneg_train.csv
DEV="data/quora-dev.csv,data/paraphrase_extra_data/paws_dev.csv,data/paraphrase_extra_data/mrpc_traindev.csv"
TST="data/quora-test-student.csv,data/paraphrase_extra_data/paws_test.csv,data/paraphrase_extra_data/mrpc_test.csv"
EPOCHS=6
SEED=${SEED:-11711}
mkdir -p logs predictions

run () {  # $1=tag  $2,$3 = extra flags
  local tag=$1; shift
  echo "==================== [LR-sweep] $tag (seed $SEED) ===================="
  local DO="predictions/para-dev-quora-lr-$tag.csv,predictions/para-dev-paws-lr-$tag.csv,predictions/para-dev-mrpc-lr-$tag.csv"
  local TO="predictions/para-test-quora-lr-$tag.csv,predictions/para-test-paws-lr-$tag.csv,predictions/para-test-mrpc-lr-$tag.csv"
  $PY paraphrase_detection.py --use_gpu \
    --model_size gpt2 --epochs $EPOCHS --seed $SEED \
    --balanced_sampler \
    --para_train "$TRAIN" --para_dev "$DEV" --para_test "$TST" \
    --para_dev_out "$DO" --para_test_out "$TO" \
    "$@" 2>&1 | tee "logs/lr_sweep_${tag}_seed${SEED}.log"
}

run fixed                                        # baseline: 고정 lr (기존 동작)
run warmup06-linear  --lr_schedule linear --warmup_ratio 0.06
run warmup10-cosine  --lr_schedule cosine --warmup_ratio 0.10

echo "==================== [LR-sweep] 완료 — best dev acc 비교 ===================="
grep -hE "Early stopping|best [0-9]" logs/lr_sweep_*_seed${SEED}.log | tail -20 || true
