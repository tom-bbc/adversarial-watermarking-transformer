#!/bin/bash

# Get root of project and module paths
current_path="$(pwd)"
repo_name="synthetic-text-watermarking"
REPO_ROOT_PATH="${current_path%$repo_name*}$repo_name"

AWT_PATH=$REPO_ROOT_PATH/synthetic_text_watermarking/adversarial_watermarking_transformer

# Run training
cd $AWT_PATH/code/
# touch WT2_mt_noft_gen.pt

python main_train.py \
    --cuda \
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
