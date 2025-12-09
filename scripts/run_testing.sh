#!/bin/bash

# Setup
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$ROOT/.env/bin/activate"

# Run testing
python3 "$ROOT/evaluation/evaluate_avg.py" \
    --msg_len 4 \
    --data "wikitext-2" \
    --bptt 80 \
    --gen_model "WT2_mt_full_gen.pt" \
    --disc_model "WT2_mt_full_disc.pt" \
    --use_lm_loss 0 \
    --seed 200 \
    --samples_num 10 \
    --avg_cycle 2

echo -e "\n << * >> Completed testing stage."
