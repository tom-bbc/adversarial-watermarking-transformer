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
    --save WT2_mt_full \
    --optimizer adam \
    --fixed_length 0 \
    --bptt 80  \
    --discr_interval 3 \
    --msg_weight 6 \
    --gen_weight 1 \
    --reconst_weight 2 \
    --scheduler 1 \
    --shared_encoder 1 \
    --use_semantic_loss 1 \
    --sem_weight 6 \
    --resume WT2_mt_noft \
    --use_lm_loss 0 \
    --lm_weight 1.3

echo " << * >> Completed fine-tuning stage."
