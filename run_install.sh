#!/bin/bash

# Get root of project and module paths
current_path="$(pwd)"
repo_name="synthetic-text-watermarking"
REPO_ROOT_PATH="${current_path%$repo_name*}$repo_name"

AWT_PATH=$REPO_ROOT_PATH/synthetic_text_watermarking/adversarial_watermarking_transformer

# Create python virtual environment
python -m venv $REPO_ROOT_PATH/.env
source $REPO_ROOT_PATH/.env/bin/activate

# Install module dependencies
pip install -r $AWT_PATH/requirements.txt
echo "Installed module dependencies"

# Model checkpt of InferSent
mkdir $AWT_PATH/code/sent_encoder
curl -Lo $AWT_PATH/code/sent_encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
echo "Downloaded InferSent encoder model"

mkdir $AWT_PATH/code/sent_encoder/GloVe
curl -Lo $AWT_PATH/code/sent_encoder/GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip $AWT_PATH/code/sent_encoder/GloVe/glove.840B.300d.zip -d $AWT_PATH/code/sent_encoder/GloVe/
echo "Downloaded GloVe encoder model"

# Model checkpt of AWD LSTM LM
pip install gdown
gdown 1S2-wmZK4JgJEIFpRp1Dy4SuzTqBcLKK7  # WT2_LM.pt
gdown 1q0OAKcHaWHkGvag5_g8tcJ5AF6G1V8s9  # WT2_mt_full_gen.pt
gdown 1KiDbi3fZHNYbFwuuW19O2xuIr0e9029y  # WT2_mt_full_disc.pt
gdown 1XI2aZ-w5kMaq1MMzyAp38ruUgSo-6BXv  # model_dae_attack.pt
gdown 1zhzyi1S0w7PcFxp0ECw7L1RiTZHNfASp  # model2_dae_wm_pairs.pt
gdown 1tLBT08YxVFnEzQxhhmtA1sbFWLraOgBe  # WT2_classifier.pt
echo "Downloaded AWD LSTM language models"

# WikiText-2 (WT2) dataset
mkdir -p $AWT_PATH/code/data/wikitext-2/
curl -L \
    -o $AWT_PATH/code/data/wikitext-2/wikitext.zip \
    https://www.kaggle.com/api/v1/datasets/download/rohitgr/wikitext

unzip $AWT_PATH/code/data/wikitext-2/wikitext.zip -d $AWT_PATH/code/data/wikitext-2/
rm $AWT_PATH/code/data/wikitext-2/wikitext.zip

mv $AWT_PATH/code/data/wikitext-2/wikitext-2/wiki.train.tokens $AWT_PATH/code/data/wikitext-2/train.txt
mv $AWT_PATH/code/data/wikitext-2/wikitext-2/wiki.valid.tokens $AWT_PATH/code/data/wikitext-2/valid.txt
mv $AWT_PATH/code/data/wikitext-2/wikitext-2/wiki.test.tokens $AWT_PATH/code/data/wikitext-2/test.txt

# export HF_HUB_ENABLE_HF_TRANSFER=1
# hf download \
#     --local-dir $AWT_PATH/code/data/ \
#     --repo-type dataset \
#     Salesforce/wikitext \
#     wikitext-2-v1/train-00000-of-00001.parquet wikitext-2-v1/validation-00000-of-00001.parquet wikitext-2-v1/test-00000-of-00001.parquet
# mv $AWT_PATH/code/data/wikitext-2-v1/train-00000-of-00001.parquet $AWT_PATH/code/data/wikitext-2/train.txt
# mv $AWT_PATH/code/data/wikitext-2-v1/validation-00000-of-00001.parquet $AWT_PATH/code/data/wikitext-2/valid.txt
# mv $AWT_PATH/code/data/wikitext-2-v1/test-00000-of-00001.parquet $AWT_PATH/code/data/wikitext-2/test.txt
# rm -r $AWT_PATH/code/data/wikitext-2-v1

echo "Downloaded WikiText-2 dataset"
