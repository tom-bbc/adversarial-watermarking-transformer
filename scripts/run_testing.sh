#!/bin/bash

# Setup
source .env/bin/activate
cd code/

# Run testing
python3 evaluate_avg.py \
    --msg_len 4 \
    --data wikitext-2/wikitext-2 \
    --bptt 80 \
    --gen_path models/WT2_mt_full_gen.pt \
    --disc_path models/WT2_mt_full_disc.pt \
    --use_lm_loss 0 \
    --seed 200 \
    --samples_num 10 \
    --avg_cycle 2

echo -e "\n << * >> Completed testing stage."
