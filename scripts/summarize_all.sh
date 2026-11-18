#!/bin/bash
# Resume os 8 cenarios de results/ (rodar no WSL, precisa de torch/datasets).
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python3}
for alpha in 0.5 0.1; do
  for attack in silent low_effort noisy fr50; do
    n=3; [ $attack = fr50 ] && n=5
    echo "=================== alpha=$alpha cenario=$attack ==================="
    $PY scripts/summarize_results.py --results-dir results/a${alpha/./}_$attack --alpha $alpha --num-free-riders $n
  done
done
