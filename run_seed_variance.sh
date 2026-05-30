#!/usr/bin/env bash
# Seed variance (GPU1 트랙). best 구성(전체 hardneg mix + balanced)을 small 로 seed 만 바꿔 반복.
# 목적: dev acc 의 seed 노이즈 floor 추정 → ablation 차이(±0.5pp)가 유의한지 판단 기준.
set -e
cd "$(dirname "$0")"

PY=${PY:-python}
TRAIN=data/paraphrase_extra_data/quora_paws_raw_swap_bt_hardneg_train.csv
DEV="data/quora-dev.csv,data/paraphrase_extra_data/paws_dev.csv,data/paraphrase_extra_data/mrpc_traindev.csv"
TST="data/quora-test-student.csv,data/paraphrase_extra_data/paws_test.csv,data/paraphrase_extra_data/mrpc_test.csv"
EPOCHS=6
SEEDS=${SEEDS:-"11711 1234 2024"}
mkdir -p logs predictions

for S in $SEEDS; do
  echo "==================== [seed-var] seed $S ===================="
  DO="predictions/para-dev-quora-seed$S.csv,predictions/para-dev-paws-seed$S.csv,predictions/para-dev-mrpc-seed$S.csv"
  TO="predictions/para-test-quora-seed$S.csv,predictions/para-test-paws-seed$S.csv,predictions/para-test-mrpc-seed$S.csv"
  $PY paraphrase_detection.py --use_gpu \
    --model_size gpt2 --epochs $EPOCHS --seed $S \
    --balanced_sampler \
    --para_train "$TRAIN" --para_dev "$DEV" --para_test "$TST" \
    --para_dev_out "$DO" --para_test_out "$TO" \
    2>&1 | tee "logs/seed_var_${S}.log"
done

echo "==================== [seed-var] 완료 — seed 별 best dev acc ===================="
for S in $SEEDS; do
  echo -n "seed $S : "; grep -hoE "best [0-9.]+" "logs/seed_var_${S}.log" | tail -1
done
