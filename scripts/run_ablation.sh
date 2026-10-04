#!/usr/bin/env bash
# Metre-alignment ablation + regularisation variants on one GPU (two runs at a time).
#
# Prerequisites, both produced outside this script:
#   python -m scripts.ablation_data          builds the three corpora in data/
#   the baseline "aligned" run               checkpoints/aligned.pt
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs reports

# Fail fast with a usable message. The previous version waited on data_prep.log,
# a file nothing in the repo creates, so a fresh run blocked forever.
missing=0
for corpus in data/melodies.jsonl data/melodies_legacy.jsonl data/melodies_legacy_duple.jsonl; do
  if [[ ! -s $corpus ]]; then
    echo "missing $corpus -- run: $PY -m scripts.ablation_data" >&2
    missing=1
  fi
done
if [[ ! -s checkpoints/aligned.pt ]]; then
  echo "missing checkpoints/aligned.pt -- train the baseline 'aligned' run first" >&2
  missing=1
fi
(( missing == 0 )) || exit 1

run() {  # name, extra --set args...
  local name=$1; shift
  $PY -m melodymorph.cli train --config configs/base.yaml --amp --epochs 12 --set patience=3 \
    --set checkpoint_path=checkpoints/$name.pt --set run_dir=runs/$name "$@" > logs/train_$name.log 2>&1
  $PY -m melodymorph.cli evaluate --checkpoint checkpoints/$name.pt --kn-orders \
    --report reports/$name.json > logs/eval_$name.log 2>&1
}

# CPU: n-gram baselines on both tokenizations of the same tunes
( $PY -m melodymorph.cli evaluate --checkpoint checkpoints/aligned.pt --kn-orders 3 6 9 \
    --report reports/kn_aligned.json > logs/kn_aligned.log 2>&1 ) &

run legacy_duple --set data_cache=data/melodies_legacy_duple.jsonl &
run aligned_s7 --set seed=7 &
wait
run legacy_duple_s7 --set data_cache=data/melodies_legacy_duple.jsonl --set seed=7 &
run legacy_full --set data_cache=data/melodies_legacy.jsonl &
wait
run aligned_drop02 --set dropout=0.2 &
run aligned_tr5 --set transpose_range=5 &
wait
echo ALL DONE > logs/ablation.done
