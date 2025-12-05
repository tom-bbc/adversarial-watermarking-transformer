#!/bin/bash

# Setup
source .env/bin/activate
cd code/

# Run training
python3 main_train.py \
    --msg_len 4 \
    --data data/wikitext-2 \
    --batch_size 80  \
    --epochs 200 \
    --save WT2_mt_noft \
    --optimizer adam \
    --fixed_length 1 \
    --bptt 80 \
    --use_lm_loss 0 \
    --use_semantic_loss 0 \
    --discr_interval 1 \
    --msg_weight 5 \
    --gen_weight 1.5 \
    --reconst_weight 1.5 \
    --scheduler 1

echo -e "\n << * >> Completed pre-training stage."
