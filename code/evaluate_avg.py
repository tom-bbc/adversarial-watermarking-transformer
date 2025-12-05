###############################################################################
# Imports
###############################################################################

import argparse
import json
import time

import data
import numpy as np
import torch
import torch.nn as nn
from lang_model import RNNModel
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils import batchify, generate_msgs, get_batch_different

###############################################################################
# Helper functions for test criteria
###############################################################################


def random_cut_sequence(sequence, limit=10):

    rand_cut_start = np.random.randint(low=0, high=limit)
    rand_cut_end = sequence.size(0) - np.random.randint(low=0, high=limit)

    sequence = sequence[rand_cut_start:rand_cut_end, :]
    # new_seq_len = rand_cut_end - rand_cut_start
    # print(sequence.size())

    return sequence


def compare_msg_whole(msgs, msg_out):

    correct = np.count_nonzero(
        np.sum(
            np.equal(
                msgs.detach().cpu().numpy().astype(int),
                msg_out.detach().cpu().numpy().astype(int),
            ),
            axis=1,
        )
        == args.msg_len
    )

    return correct


def compare_msg_bits(msgs, msg_out):

    correct = np.count_nonzero(
        np.equal(
            msgs.detach().cpu().numpy().astype(int),
            msg_out.detach().cpu().numpy().astype(int),
        )
        is True
    )

    return correct


def get_idx_from_logits(sequence, seq_len, bsz):

    m = nn.Softmax(dim=-1)
    sequence = sequence.view(seq_len, bsz, sequence.size(1))
    sequence = m(sequence)
    sequence_idx = torch.argmax(sequence, dim=-1)

    return sequence_idx


def noisy_sampling(sent_encoder_out, both_embeddings, data, model_gen):

    # Enable dropout
    candidates_emb = []
    candidates_one_hot = []
    candidates_soft_prob = []
    # model_gen.train()

    for i in range(0, args.samples_num):
        with torch.no_grad():
            sent_out_emb, sent_out_hot, sent_out_soft = model_gen.forward_sent_decoder(
                both_embeddings, data, sent_encoder_out, args.gumbel_temp
            )
            candidates_emb.append(sent_out_emb)
            candidates_one_hot.append(sent_out_hot)
            candidates_soft_prob.append(sent_out_soft)

    # model_gen.eval()
    return candidates_emb, candidates_one_hot, candidates_soft_prob


###############################################################################
# Run validation/testing of the models over a given dataset
###############################################################################


def evaluate(
    data_source,
    all_msgs,
    corpus,
    sbert_model,
    model_gen,
    model_disc,
    langModel,
    criterion_lm,
    out_file=None,
    batch_size=10,
):
    """Run validation/testing of the models over a given dataset"""

    # Turn on evaluation mode which disables dropout.
    model_gen.eval()
    model_disc.eval()

    if args.use_lm_loss:
        langModel.eval()

    total_loss_lm = 0
    # ntokens = len(corpus.dictionary)
    tot_count = 0
    correct_msg_count = 0
    tot_count_bits = 0
    correct_msg_count_bits = 0
    sig = nn.Sigmoid()
    batch_count = 0
    meteor_tot = 0
    l2_distances = 0
    bert_diff = [0 for i in range(0, args.samples_num)]
    prev_msgs = None
    prev_msg_out = 0

    same_avg_cycle = 0
    for i in tqdm(
        range(0, data_source.size(0) - args.bptt, args.bptt),
        desc="Evaluating test sample",
    ):
        data, msgs, targets = get_batch_different(
            data_source, i, args, all_msgs, evaluation=True
        )

        if i == 0 or same_avg_cycle == 0:
            prev_msgs = msgs

        if same_avg_cycle == 1:
            msgs = prev_msgs

        with torch.no_grad():
            both_embeddings, sent_encoder_out = model_gen.forward_sent_encoder(
                data, msgs, args.gumbel_temp
            )
            candidates_emb, candidates_one_hot, candidates_soft_prob = noisy_sampling(
                sent_encoder_out, both_embeddings, data, model_gen
            )

            output_text_beams = []
            word_idx_beams = []
            lm_loss_beams = []

            output_text = ""
            orig_text = ""

            for beam in range(0, args.samples_num):
                word_idx = get_idx_from_logits(
                    candidates_soft_prob[beam], data.size(0), batch_size
                )

                output_text = ""
                orig_text = ""

                for k in range(0, data.size(0)):
                    output_text = (
                        output_text + corpus.dictionary.idx2word[word_idx[k, 0]] + " "
                    )
                    orig_text = orig_text + corpus.dictionary.idx2word[data[k, 0]] + " "

                if orig_text == output_text:
                    continue

                output_text_beams.append(output_text)
                word_idx_beams.append(word_idx)

                if args.use_lm_loss:
                    hidden = langModel.init_hidden(batch_size)

                    lm_targets = word_idx[1 : data.size(0)]
                    lm_targets = lm_targets.view(
                        lm_targets.size(0) * lm_targets.size(1),
                    )
                    lm_inputs = word_idx[0 : data.size(0) - 1]
                    lm_out, hidden = langModel(lm_inputs, hidden, decode=True)
                    lm_loss = criterion_lm(lm_out, lm_targets)
                    lm_loss_beams.append(lm_loss.item())

            if len(lm_loss_beams) > 0:
                beam_argsort = np.argsort(np.asarray(lm_loss_beams))
                best_beam_idx = beam_argsort[0]
            else:
                output_text_beams.append(orig_text)
                word_idx_beams.append(data)
                best_beam_idx = 0

                if args.use_lm_loss:
                    hidden = langModel.init_hidden(batch_size)
                    lm_targets = data[1 : data.size(0)]
                    lm_targets = lm_targets.view(
                        lm_targets.size(0) * lm_targets.size(1),
                    )
                    lm_inputs = data[0 : data.size(0) - 1]
                    lm_out, hidden = langModel(lm_inputs, hidden, decode=True)
                    lm_loss = criterion_lm(lm_out, lm_targets)
                    lm_loss_beams.append(lm_loss.item())

            # calculate bert diff and meteor score for the selected sample
            sentences = [output_text_beams[best_beam_idx], orig_text]
            sbert_embs = sbert_model.encode(sentences)
            bert_diff_selected = np.linalg.norm(sbert_embs[0] - sbert_embs[1])

            references = [[orig_text]]
            hypothesis = [output_text_beams[best_beam_idx]]
            # print(f" << * >> References: {references}")
            # print(f" << * >> Hypothesis: {hypothesis}")

            meteor_score_selected = meteor_score(references, hypothesis)

            best_beam_data = word_idx_beams[best_beam_idx]
            best_beam_emb = model_gen.forward_sent(
                best_beam_data, msgs, args.gumbel_temp, only_embedding=True
            )
            msg_out = model_gen.forward_msg_decode(best_beam_emb)

            if bert_diff[best_beam_idx] < args.bert_threshold and out_file is not None:
                out_file.write("****" + "\n")
                out_file.write(str(batch_count) + "\n")
                out_file.write(str(meteor_score_selected) + "\n")
                out_file.write(str(bert_diff_selected) + "\n")
                out_file.write(orig_text + "\n")
                out_file.write(output_text_beams[best_beam_idx] + "\n")

                l2_distances = l2_distances + bert_diff_selected
                meteor_tot = meteor_tot + meteor_score_selected

                if args.use_lm_loss:
                    total_loss_lm += lm_loss_beams[best_beam_idx]
                    # lm_loss_selected = lm_loss_beams[best_beam_idx]

            else:
                bert_diff_selected = 0
                meteor_score_selected = 1
                meteor_tot = meteor_tot + meteor_score_selected
                # msg_out_random = model_gen.forward_msg_decode(data_emb)

                if args.use_lm_loss:
                    hidden = langModel.init_hidden(batch_size)
                    lm_targets = data[1 : data.size(0)]
                    lm_targets = lm_targets.view(
                        lm_targets.size(0) * lm_targets.size(1),
                    )
                    lm_inputs = data[0 : data.size(0) - 1]
                    lm_out, hidden = langModel(lm_inputs, hidden, decode=True)
                    lm_loss = criterion_lm(lm_out, lm_targets)
                    total_loss_lm += lm_loss.item()
                    # lm_loss_selected = lm_loss.item()

        if i == 0 or same_avg_cycle == 0:
            prev_msg_out = sig(msg_out)
        if same_avg_cycle:
            prev_msg_out = prev_msg_out + sig(msg_out)

        if i != 0 and (batch_count + 1) % args.avg_cycle == 0:
            prev_msg_out = prev_msg_out / args.avg_cycle
            msg_out_avg = torch.round(prev_msg_out)
            tot_count += msgs.shape[0]  # type: ignore
            tot_count_bits += msgs.shape[0] * msgs.shape[1]  # type: ignore
            same_avg_cycle = 0
            correct_msg_count += compare_msg_whole(msgs, msg_out_avg)
            correct_msg_count_bits += compare_msg_bits(msgs, msg_out_avg)
        else:
            same_avg_cycle = 1

        batch_count = batch_count + 1

    return (
        total_loss_lm / batch_count,
        correct_msg_count / tot_count,
        correct_msg_count_bits / tot_count_bits,
        meteor_tot / batch_count,
        l2_distances / batch_count,
    )


###############################################################################
# Main entrypoint of average evaluation script
###############################################################################


def main(args):
    """Main entrypoint of average evaluation script"""

    # Set random seeds for test reproducibility
    # =========================================================================

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print(
                " -- * -- WARNING: You have a CUDA device, "
                "so you should probably run with --cuda"
            )
        else:
            torch.cuda.manual_seed(args.seed)

    # Load validation and testing data
    # =========================================================================

    corpus = data.Corpus(args.data)

    eval_batch_size = 1
    test_batch_size = 1
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    # Instantiate loss criterions
    # =========================================================================

    ntokens = len(corpus.dictionary)

    map_location = "cpu" if args.cuda else "cuda"
    # criterion1 = nn.NLLLoss()
    # criterion2 = nn.BCEWithLogitsLoss()

    if args.use_lm_loss:
        criterion_lm = nn.CrossEntropyLoss()

        with open(args.lm_ckpt, "rb") as f:
            pretrained_lm, _, _ = torch.load(
                f, map_location=map_location, weights_only=False
            )
            langModel = RNNModel(
                args.model,
                ntokens,
                args.emsize_lm,
                args.nhid,
                args.nlayers,
                args.dropout,
                args.dropouth,
                args.dropouti_lm,
                args.dropoute_lm,
                args.wdrop,
                args.tied_lm,
                pretrained_lm,
            )

        del pretrained_lm

        if args.cuda:
            langModel = langModel.cuda()
            criterion_lm = criterion_lm.cuda()

    else:
        criterion_lm = None
        langModel = None

    # Generate random messages for message loss
    # =========================================================================

    all_msgs = generate_msgs(args)
    print(" << * >> Random messages:")
    print(all_msgs)

    if not args.given_msg == []:
        all_msgs = [int(i) for i in args.given_msg]
        all_msgs = np.asarray(all_msgs)
        all_msgs = all_msgs.reshape([1, args.msg_len])

    # Load trained and eval models
    # =========================================================================

    sbert_model = SentenceTransformer("bert-base-nli-mean-tokens")

    # Load the best saved model.
    with open(args.gen_path, "rb") as f:
        model_gen, _, _, _ = torch.load(
            f, map_location=map_location, weights_only=False
        )

    # Load the best saved model.
    with open(args.disc_path, "rb") as f:
        model_disc, _, _, _ = torch.load(
            f, map_location=map_location, weights_only=False
        )

    if args.cuda:
        model_gen.cuda()
        model_disc.cuda()

    # Run validation
    # =========================================================================

    f = open("val_out.txt", "w")
    # f_metrics = open("val_out_metrics.txt", "w")

    print("\n << * >> Running model validation.", end="\n\n")
    val_lm_loss, val_correct_msg, val_correct_bits_msg, val_meteor, val_l2_sbert = (
        evaluate(
            val_data,
            all_msgs,
            corpus,
            sbert_model=sbert_model,
            model_gen=model_gen,
            model_disc=model_disc,
            langModel=langModel,
            criterion_lm=criterion_lm,
            out_file=f,
            batch_size=eval_batch_size,
        )
    )

    results_log_statement = (
        f"| validation "
        f"| msg accuracy {val_correct_msg * 100:5.2f} "
        f"| msg bit accuracy {val_correct_bits_msg * 100:5.2f} "
        f"| meteor {val_meteor:5.4f} "
        f"| SentBert dist. {val_l2_sbert:5.4f}"
    )

    if args.use_lm_loss:
        results_log_statement += f"| lm loss {val_lm_loss:5.2f} "

    print("-" * 150)
    print(results_log_statement)
    print("-" * 150)
    f.close()

    # Run testing
    # =====================================================================

    f = open("test_out.txt", "w")
    # f_metrics = open("test_out_metrics.txt", "w")

    print("\n << * >> Running model testing.", end="\n\n")
    (
        test_lm_loss,
        test_correct_msg,
        test_correct_bits_msg,
        test_meteor,
        test_l2_sbert,
    ) = evaluate(
        test_data,
        all_msgs,
        corpus,
        sbert_model=sbert_model,
        model_gen=model_gen,
        model_disc=model_disc,
        criterion_lm=criterion_lm,
        out_file=f,
        batch_size=test_batch_size,
        langModel=langModel,
    )

    results_log_statement = (
        f"| validation "
        f"| msg accuracy {test_correct_msg * 100:5.2f} "
        f"| msg bit accuracy {test_correct_bits_msg * 100:5.2f} "
        f"| meteor {test_meteor:5.4f} "
        f"| SentBert dist. {test_l2_sbert:5.4f}"
    )

    if args.use_lm_loss:
        results_log_statement += f"| lm loss {test_lm_loss:5.2f} "

    print("-" * 150)
    print(results_log_statement)
    print("-" * 150)
    f.close()


if __name__ == "__main__":

    # Parse command-line arguments
    # =========================================================================

    parser = argparse.ArgumentParser(
        description="PyTorch PennTreeBank RNN/LSTM Language Model"
    )
    parser.add_argument(
        "--data", type=str, default="data/penn/", help="location of the data corpus"
    )

    parser.add_argument("--bptt", type=int, default=80, help="sequence length")

    parser.add_argument(
        "--given_msg", type=list, default=[], help="test against this msg only"
    )

    parser.add_argument("--seed", type=int, default=1111, help="random seed")

    parser.add_argument("--cuda", action="store_false", help="do not use CUDA")

    randomhash = "".join(str(time.time()).split("."))
    parser.add_argument(
        "--save",
        type=str,
        default=randomhash + ".pt",
        help="path to save the final model",
    )

    parser.add_argument(
        "--gen_path", type=str, default=randomhash + ".pt", help="path to the generator"
    )
    parser.add_argument(
        "--disc_path",
        type=str,
        default=randomhash + ".pt",
        help="path to the discriminator",
    )

    # Language model loss
    parser.add_argument(
        "--use_lm_loss", type=int, default=0, help="whether to use language model loss"
    )
    parser.add_argument(
        "--lm_ckpt",
        type=str,
        default="WT2_lm.pt",
        help="path to the fine tuned language model",
    )

    # Message arguments
    parser.add_argument(
        "--msg_len", type=int, default=4, help="The length of the binary message"
    )
    parser.add_argument(
        "--msgs_num",
        type=int,
        default=3,
        help="The number of messages encododed during training",
    )
    parser.add_argument(
        "--avg_cycle", type=int, default=2, help="Number of sentences to average"
    )

    # Model params.
    parser.add_argument(
        "--model",
        type=str,
        default="LSTM",
        help="type of recurrent net (LSTM, QRNN, GRU)",
    )
    parser.add_argument(
        "--emsize_lm", type=int, default=400, help="size of word embeddings"
    )
    parser.add_argument(
        "--nhid", type=int, default=1150, help="number of hidden units per layer"
    )
    parser.add_argument("--nlayers", type=int, default=3, help="number of layers")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument(
        "--dropouth",
        type=float,
        default=0.3,
        help="dropout for rnn layers (0 = no dropout)",
    )
    parser.add_argument(
        "--dropouti_lm",
        type=float,
        default=0.15,
        help="dropout for input embedding layers (0 = no dropout)",
    )
    parser.add_argument(
        "--dropoute_lm",
        type=float,
        default=0.05,
        help="dropout to remove words from embedding layer (0 = no dropout)",
    )
    parser.add_argument(
        "--wdrop",
        type=float,
        default=0,
        help="amount of weight dropout to apply to the RNN hidden to hidden matrix",
    )

    # Gumbel softmax arguments
    parser.add_argument(
        "--gumbel_temp", type=int, default=0.5, help="Gumbel softmax temprature"
    )
    parser.add_argument(
        "--gumbel_hard",
        type=bool,
        default=True,
        help="whether to use one hot encoding in the forward pass",
    )

    parser.add_argument(
        "--bert_threshold",
        type=float,
        default=2.5,
        help="Threshold on the bert distance",
    )

    parser.add_argument("--samples_num", type=int, default=10, help="Decoder beam size")

    args = parser.parse_args()
    args.tied_lm = True
    print(f" << * >> Args: {json.dumps(vars(args), indent=4)}", end="\n\n")

    # Run main testing process
    # =========================================================================

    main(args)
