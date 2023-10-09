import argparse
import os
import re
from collections import defaultdict
from nltk.stem.porter import *
import numpy as np

from tqdm import tqdm

import config

stemmer = PorterStemmer()


def update_score_dict(trg_kps, pred_kps, tag):
    if len(trg_kps) == 0:
        return

    if len(pred_kps) == 0:
        num_matches_at_k = 0
        num_matches_at_m = 0
    else:
        is_match_mask = np.zeros(len(pred_kps), dtype=bool)
        for pred_i, pred_kp in enumerate(pred_kps):
            pred_str = ' '.join(pred_kp)
            for _, trg_kp in enumerate(trg_kps):
                trg_str = ' '.join(trg_kp)
                if pred_str == trg_str:
                    is_match_mask[pred_i] = True
                    break

        num_matches = np.cumsum(is_match_mask)
        num_matches_at_k = num_matches[K-1] if len(pred_kps) >= K else num_matches[-1]
        num_matches_at_m = num_matches[-1]

    num_preds_at_k = min(K, len(pred_kps)) if args.meng_rui_precision else K
    num_trgs_at_k = min(K, len(trg_kps)) if args.choi_recall else len(trg_kps)

    precision_at_k = num_matches_at_k / num_preds_at_k
    recall_at_k = num_matches_at_k / num_trgs_at_k
    f1_at_k = (2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k + 1e-20))

    precision_at_m = num_matches_at_m / (len(pred_kps) + 1e-20)
    recall_at_m = num_matches_at_m / len(trg_kps)
    f1_at_m = (2 * precision_at_m * recall_at_m / (precision_at_m + recall_at_m + 1e-20))

    score_dict[f"P@{K}_{tag}"].append(precision_at_k)
    score_dict[f"R@{K}_{tag}"].append(recall_at_k)
    score_dict[f"F1@{K}_{tag}"].append(f1_at_k)
    score_dict[f"num_matches@{K}_{tag}"].append(num_matches_at_k)
    score_dict[f"num_preds@{K}_{tag}"].append(K)
    score_dict[f"num_trgs@{K}_{tag}"].append(num_trgs_at_k)
    score_dict[f"P@M_{tag}"].append(precision_at_m)
    score_dict[f"R@M_{tag}"].append(recall_at_m)    
    score_dict[f"F1@M_{tag}"].append(f1_at_m)
    score_dict[f"num_matches@M_{tag}"].append(num_matches_at_m)
    score_dict[f"num_preds@M_{tag}"].append(len(pred_kps))
    score_dict[f"num_trgs@M_{tag}"].append(len(trg_kps))


def separate_present_absent_by_source(src_words, kps):
    is_present_mask = np.zeros(len(kps), dtype=bool)
    present_kps, absent_kps = [], []
    stemmed_kp_2d_list = []

    for i, kp in enumerate(kps):
        match = False
        kp_str = re.sub("([!\"#$%&\'\(\)*+,-./:;<=>?@^+`{|}~])", r" \1 ", kp)
        kp_str = re.sub('\s{2,}', ' ', kp_str).strip()
        kp_words = [stemmer.stem(w.strip()) for w in kp_str.lower().split()]
        kp_words = list(filter(None, kp_words))
        if len(kp_words) == 0:
            continue
        for src_i in range(len(src_words) - len(kp_words) + 1):
            match = True
            for kp_i, kp_w in enumerate(kp_words):
                src_w = src_words[src_i + kp_i]
                if src_w != kp_w:
                    match = False
                    break
            if match:
                break
        is_present_mask[i] = True if match else False
        stemmed_kp_2d_list.append(kp_words)

    for kp, is_present in zip(stemmed_kp_2d_list, is_present_mask):
        if is_present:
            present_kps.append(kp)
        else:
            absent_kps.append(kp)

    return present_kps, absent_kps


def filter_keyphrases(kps, pred=True):
    is_unique_mask = np.ones(len(kps), dtype=bool)
    is_valid_mask = np.zeros(len(kps), dtype=bool)
    kp_set = set()
    
    for i, kp in enumerate(kps):
        # check duplicate keyphrases
        is_unique_mask[i] = False if '_'.join(kp) in kp_set else True
        kp_set.add('_'.join(kp))
        # check valid keyphrases
        if pred:
            keep_flag = False if len(kp) == 0 else True
            for w in kp:
                # TODO: invalidate non-alphanumeric keyphrases
                if w == ',' or w == '.':
                    keep_flag = False
            is_valid_mask[i] = keep_flag
        else:
            is_valid_mask[i] = True  # all trg kps are assumed to be valid
    
    _filter = is_unique_mask * is_valid_mask
    filtered_kps = [kp for kp, is_keep in zip(kps, _filter) if is_keep] 
    num_duplicates = len(kps) - np.sum(is_unique_mask)
    return filtered_kps, num_duplicates


def evaluate():
    total_num_src = 0
    total_num_predictions = 0
    total_num_unique_predictions = 0
    total_num_present_predictions = 0
    total_num_absent_predictions = 0
    total_num_targets = 0
    total_num_unique_targets = 0
    total_num_present_targets = 0
    total_num_absent_targets = 0
    max_unique_targets = 0

    ### Process & calculate per source / target / prediction line ###
    with open(src_filepath) as src_f, open(trg_filepath) as trg_f, \
         open(pre_pred_filepath) as pred_f1, open(abs_pred_filepath) as pred_f2:

        for src_l, trg_l, pred1_l, pred2_l in tqdm(zip(src_f, trg_f, pred_f1, pred_f2)):
            total_num_src += 1
            [title, context] = src_l.strip().split("<eos>")
            trg_kp_list = trg_l.strip().split(";")
            pre_pred_kp_list = pred1_l.strip().split(";")
            abs_pred_kp_list = pred2_l.strip().split(";")
            trg_kp_list = list(filter(None, trg_kp_list))
            pre_pred_kp_list = list(filter(None, pre_pred_kp_list))
            abs_pred_kp_list = list(filter(None, abs_pred_kp_list))

            src_str = title.lower().strip() + " " + context.lower().strip()
            src_str = re.sub("([!\"#$%&\'\(\)*+,-./:;<=>?@^+`{|}~])", r" \1 ", src_str)
            src_str = re.sub('\s{2,}', ' ', src_str).strip()
            src_words = [stemmer.stem(w.strip()) for w in src_str.split()]
            src_words = list(filter(None, src_words))

            ### Separate present/absent keyphrases ###
            present_pred_kps, _ = separate_present_absent_by_source(src_words, pre_pred_kp_list)
            _, absent_pred_kps = separate_present_absent_by_source(src_words, abs_pred_kp_list)
            present_trg_kps, absent_trg_kps = separate_present_absent_by_source(src_words, trg_kp_list)
            total_num_present_predictions += len(present_pred_kps)
            total_num_present_targets += len(present_trg_kps)
            total_num_absent_predictions += len(absent_pred_kps)
            total_num_absent_targets += len(absent_trg_kps)

            ### Filter duplicates & invalid keyphrases for PREDICTIONS ###
            num_predictions = len(present_pred_kps) + len(absent_pred_kps)
            present_pred_kps, num_pre_duplicates = filter_keyphrases(present_pred_kps, pred=False)
            absent_pred_kps, num_abs_duplicates = filter_keyphrases(absent_pred_kps, pred=False)
            total_num_unique_predictions += (num_predictions - num_pre_duplicates - num_abs_duplicates)
            total_num_predictions += num_predictions

            ### Filter duplicates for TARGETS ###
            num_targets = len(trg_kp_list)
            present_trg_kps, num_pre_duplicates = filter_keyphrases(present_trg_kps, pred=False)
            absent_trg_kps, num_abs_duplicates = filter_keyphrases(absent_trg_kps, pred=False)
            total_num_unique_targets += (num_targets - num_pre_duplicates - num_abs_duplicates)
            total_num_targets += num_targets
            if (len(present_trg_kps) + len(absent_trg_kps)) > max_unique_targets:
                max_unique_targets = len(present_trg_kps) + len(absent_trg_kps)

            ### Update score dict ###
            update_score_dict(present_trg_kps, present_pred_kps, "present")
            update_score_dict(absent_trg_kps, absent_pred_kps, "absent")

        ### Report stats ###
        f = open(res_filepath, "w")
        f.write(f"Total #samples: {total_num_src}\n" +
                f"Max unique targets: {max_unique_targets}\n" +
                f"Total #unique predictions: {total_num_unique_predictions}/{total_num_predictions}, " +
                f"dup ratio: {(total_num_predictions-total_num_unique_predictions)/total_num_predictions:.3f}\n\n")

        for tag in ["present", "absent"]:
            for topk in k_list:
                total_predictions_k = sum(score_dict[f"num_preds@{topk}_{tag}"])
                total_targets_k = sum(score_dict[f"num_trgs@{topk}_{tag}"])
                total_num_matches_k = sum(score_dict[f"num_matches@{topk}_{tag}"])
                macro_avg_precision_k = sum(score_dict[f"P@{topk}_{tag}"]) / len(score_dict[f"P@{topk}_{tag}"])
                macro_avg_recall_k = sum(score_dict[f"R@{topk}_{tag}"]) / len(score_dict[f"R@{topk}_{tag}"])
                macro_avg_f1_k = (2 * macro_avg_precision_k * macro_avg_recall_k) / (macro_avg_precision_k + macro_avg_recall_k + 1e-20)
                
                f.write(f"#target: {total_targets_k}, #pred: {total_predictions_k}, #match: {total_num_matches_k}\n" +
                        f"P@{topk}_{tag}: {macro_avg_precision_k:.5f}\n" +
                        f"R@{topk}_{tag}: {macro_avg_recall_k:.5f}\n" +
                        f"F1@{topk}_{tag}: {macro_avg_f1_k:.5f}\n")

        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.model_args(parser)
    config.data_args(parser)
    config.predict_args(parser)
    args = parser.parse_args()
    
    args.one2many = True if args.paradigm == "one2seq" else False
    model_arch = args.model_type.split("-")[0]

    # global variables
    src_filepath = os.path.join(args.data_dir, f"test_src_filtered_{model_arch}.txt")
    trg_filepath = os.path.join(args.data_dir, f"test_trg_filtered_{model_arch}.txt")
    pre_pred_filepath = os.path.join(args.test_output_dir, f"pre_kps.txt")
    abs_pred_filepath = os.path.join(args.test_output_dir2, f"B{args.beam_size}_{args.decoding_method}_abs_kps.txt")
    res_filepath = os.path.join(args.test_output_dir2, f"B{args.beam_size}_{args.decoding_method}_all_results.txt")

    score_dict = defaultdict(list)
    k_list = [5, 'M']
    K = 5
    evaluate()
