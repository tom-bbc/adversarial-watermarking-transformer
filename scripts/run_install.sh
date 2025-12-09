#!/bin/bash

# Setup
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create python virtual environment
python3 -m venv $ROOT/.env
source "$ROOT/.env/bin/activate"
echo -e "\n << * >> Created virtual environment. \n"

# Install module dependencies
python3 -m pip install --upgrade pip
pip cache purge
pip install --no-cache-dir sentence-transformers
pip install -r "$ROOT/requirements.txt"

echo -e "\n << * >> Installed module dependencies. \n"

# Model checkpt of InferSent
mkdir -p "$ROOT/models/sent_encoder"
curl -Lo "$ROOT/models/sent_encoder/infersent2.pkl" https://dl.fbaipublicfiles.com/infersent/infersent2.pkl

echo -e "\n << * >> Downloaded InferSent encoder model. \n"

mkdir -p "$ROOT/models/sent_encoder/GloVe"
curl -Lo "$ROOT/models/sent_encoder/GloVe/glove.840B.300d.zip" http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip "$ROOT/models/sent_encoder/GloVe/glove.840B.300d.zip" -d "$ROOT/models/sent_encoder/GloVe/"

echo -e "\n << * >> Downloaded GloVe encoder model. \n"

# Model checkpt of AWD LSTM LM
pip install gdown
gdown 1S2-wmZK4JgJEIFpRp1Dy4SuzTqBcLKK7 -O "$ROOT/models/" # WT2_LM.pt
gdown 1q0OAKcHaWHkGvag5_g8tcJ5AF6G1V8s9 -O "$ROOT/models/"  # WT2_mt_full_gen.pt
gdown 1KiDbi3fZHNYbFwuuW19O2xuIr0e9029y -O "$ROOT/models/"  # WT2_mt_full_disc.pt
gdown 1XI2aZ-w5kMaq1MMzyAp38ruUgSo-6BXv -O "$ROOT/models/"  # model_dae_attack.pt
gdown 1zhzyi1S0w7PcFxp0ECw7L1RiTZHNfASp -O "$ROOT/models/"  # model2_dae_wm_pairs.pt
gdown 1tLBT08YxVFnEzQxhhmtA1sbFWLraOgBe -O "$ROOT/models/"  # WT2_classifier.pt

echo -e "\n << * >> Downloaded AWD LSTM language models. \n"

# WikiText-2 (WT2) dataset
mkdir -p "$ROOT/datasets/wikitext-2/"
curl -L \
    -o "$ROOT/datasets/wikitext-2/wikitext.zip" \
    https://www.kaggle.com/api/v1/datasets/download/rohitgr/wikitext

unzip "$ROOT/datasets/wikitext-2/wikitext.zip" -d "$ROOT/datasets/wikitext-2/"
rm "$ROOT/datasets/wikitext-2/wikitext.zip"

mv "$ROOT/datasets/wikitext-2/wikitext-2/wiki.train.tokens" "$ROOT/datasets/wikitext-2/train.txt"
mv "$ROOT/datasets/wikitext-2/wikitext-2/wiki.valid.tokens" "$ROOT/datasets/wikitext-2/valid.txt"
mv "$ROOT/datasets/wikitext-2/wikitext-2/wiki.test.tokens" "$ROOT/datasets/wikitext-2/test.txt"

# TO DO: switch over to HF method
# export HF_HUB_ENABLE_HF_TRANSFER=1
# hf download \
#     --local-dir datasets/ \
#     --repo-type dataset \
#     Salesforce/wikitext \
#     wikitext-2-v1/train-00000-of-00001.parquet wikitext-2-v1/validation-00000-of-00001.parquet wikitext-2-v1/test-00000-of-00001.parquet
# mv datasets/wikitext-2-v1/train-00000-of-00001.parquet datasets/wikitext-2/train.txt
# mv datasets/wikitext-2-v1/validation-00000-of-00001.parquet datasets/wikitext-2/valid.txt
# mv datasets/wikitext-2-v1/test-00000-of-00001.parquet datasets/wikitext-2/test.txt
# rm -r datasets/wikitext-2-v1

echo -e "\n << * >> Downloaded WikiText-2 dataset. \n"
