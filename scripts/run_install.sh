#!/bin/bash

# Create python virtual environment
python3 -m venv .env
source .env/bin/activate
echo " << * >> Created virtual environment."

# Install module dependencies
pip install -r requirements.txt
echo " << * >> Installed module dependencies."

# Model checkpt of InferSent
mkdir -p models/sent_encoder
curl -Lo models/sent_encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
echo "Downloaded InferSent encoder model."

mkdir -p models/sent_encoder/GloVe
curl -Lo models/sent_encoder/GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip models/sent_encoder/GloVe/glove.840B.300d.zip -d models/sent_encoder/GloVe/
echo " << * >> Downloaded GloVe encoder model."

# Model checkpt of AWD LSTM LM
pip install gdown
gdown 1S2-wmZK4JgJEIFpRp1Dy4SuzTqBcLKK7  # WT2_LM.pt
gdown 1q0OAKcHaWHkGvag5_g8tcJ5AF6G1V8s9  # WT2_mt_full_gen.pt
gdown 1KiDbi3fZHNYbFwuuW19O2xuIr0e9029y  # WT2_mt_full_disc.pt
gdown 1XI2aZ-w5kMaq1MMzyAp38ruUgSo-6BXv  # model_dae_attack.pt
gdown 1zhzyi1S0w7PcFxp0ECw7L1RiTZHNfASp  # model2_dae_wm_pairs.pt
gdown 1tLBT08YxVFnEzQxhhmtA1sbFWLraOgBe  # WT2_classifier.pt
mv *.pt models/
echo " << * >> Downloaded AWD LSTM language models."

# WikiText-2 (WT2) dataset
mkdir -p datasets/wikitext-2/
curl -L \
    -o datasets/wikitext-2/wikitext.zip \
    https://www.kaggle.com/api/v1/datasets/download/rohitgr/wikitext

unzip datasets/wikitext-2/wikitext.zip -d datasets/wikitext-2/
rm cdatasets/wikitext-2/wikitext.zip

mv datasets/wikitext-2/wikitext-2/wiki.train.tokens datasets/wikitext-2/train.txt
mv datasets/wikitext-2/wikitext-2/wiki.valid.tokens datasets/wikitext-2/valid.txt
mv datasets/wikitext-2/wikitext-2/wiki.test.tokens datasets/wikitext-2/test.txt

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

echo -e "\n << * >> Downloaded WikiText-2 dataset."
