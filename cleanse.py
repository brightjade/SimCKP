import os
import os.path as osp
import json
import re
from collections import defaultdict

import fasttext
from datasketch import MinHash, MinHashLSH
from nltk.corpus import stopwords
from nltk.stem.porter import *

from logger import FileLogger

from tqdm import tqdm

import nltk
nltk.download('stopwords')


def filter_data(idx, ex, filtering_idxes, filtering_lines, dataset):
    keywords_str = ex["keyword"] if "kp20k" in dataset else ex["keywords"]

    # Filter empty examples
    if ex["title"] == "" or ex["abstract"] == "" or keywords_str == "":
        filtering_lines["empty"] += 1
        filtering_idxes.add(idx)

    # Filter invalid KP examples (cleanse only train data)
    if dataset == "kp20k_train" or dataset == "kp20k_valid":
        keywords = keywords_str.split(";")
        for keyword in keywords:
            if (re.search("[^A-Za-z0-9 '-]", keyword) or
                " -" in keyword):
                filtering_lines["invalid_kp"] += 1
                filtering_idxes.add(idx)
                break

    # Filter non-English examples
    if langdetect_model.predict(ex["abstract"])[0][0] != "__label__en":
        filtering_lines["nonenglish"] += 1
        filtering_idxes.add(idx)

    return filtering_idxes, filtering_lines


def minhash_abstract(minhash_dict, abstract, dataset, idx):
    nonstopwords = []
    m = MinHash(num_perm=128)
    for word in abstract.strip().split():
        if word not in stopword_dict:
            nonstopwords.append(word)
    for word in set(nonstopwords):
        m.update(word.encode('utf8'))
    minhash_dict[f"{dataset}-{idx}"] = m
    return m, minhash_dict


def main():
    ### Filter validation & test data ###
    for dataset in DATASET_LIST:
        filtering_idxes = set()
        filtering_lines = defaultdict(int)
        for idx, line in enumerate(tqdm(open(osp.join(RAW_DATA_DIR, f"{dataset}.json"), "r"))):
            example = json.loads(line)
            filtering_idxes, filtering_lines = filter_data(idx, example, filtering_idxes, filtering_lines, dataset)
            if idx not in filtering_idxes:
                dataset_dict[dataset].append(example)

        log.console(f"[{dataset}] # empty lines: {filtering_lines['empty']}")
        log.console(f"[{dataset}] # non-english lines: {filtering_lines['nonenglish']}")
        log.console(f"[{dataset}] # invalid KP lines: {filtering_lines['invalid_kp']}")
        log.console(f"[{dataset}] # total lines filtered: {len(filtering_idxes)}")

    ### MinHash abstracts for quick similarity comparison ###
    minhash_dict = {}
    for dataset in DATASET_LIST:
        i = 0
        for data in tqdm(dataset_dict[dataset], desc=f"Bucketing {dataset}"):
            _, minhash_dict = minhash_abstract(minhash_dict, data["abstract"], dataset, i)
            i += 1

    ### Create LSH index ###
    lsh = MinHashLSH(threshold=0.9, num_perm=128)
    for k, v in minhash_dict.items():
        lsh.insert(k, v)

    ### Filter training data ###
    dataset = "kp20k_train"
    filtering_idxes = set()
    filtering_lines = defaultdict(int)
    for idx, line in enumerate(tqdm(open(osp.join(RAW_DATA_DIR, f"{dataset}.json"), "r"))):
        example = json.loads(line)

        # remove empty, invalid KP, non-English examples
        filtering_idxes, filtering_lines = filter_data(idx, example, filtering_idxes, filtering_lines, dataset)
        
        # remove duplicates by finding similar documents using LSH
        m, minhash_dict = minhash_abstract(minhash_dict, example["abstract"], dataset, idx)
        sim_doc_list = lsh.query(m)
        if len(sim_doc_list) > 0:
            filtering_lines["duplicates"] += 1 
            filtering_idxes.add(idx)
        
        # add the current doc's minhash to the pool to find duplicates within the training dataset
        lsh.insert(f"{dataset}-{idx}", m)

        if idx not in filtering_idxes:
            dataset_dict[dataset].append(example)

    log.console(f"[{dataset}] # empty lines: {filtering_lines['empty']}")
    log.console(f"[{dataset}] # non-english lines: {filtering_lines['nonenglish']}")
    log.console(f"[{dataset}] # invalid KP lines: {filtering_lines['invalid_kp']}")
    log.console(f"[{dataset}] # duplicate lines: {filtering_lines['duplicates']}")
    log.console(f"[{dataset}] # total lines filtered: {len(filtering_idxes)}")                

    ### Write to file ###
    for dataset, examples in dataset_dict.items():
        dataset_name, split = dataset.split("_")
        os.makedirs(osp.join(DATA_DIR, dataset_name), exist_ok=True)
        src_file = open(osp.join(DATA_DIR, dataset_name, f"{split}_src.txt"), "w")
        trg_file = open(osp.join(DATA_DIR, dataset_name, f"{split}_trg.txt"), "w")
        for ex in tqdm(examples, desc=f"Writing {dataset}"):
            src_file.write(f"{ex['title']}. <eos> {ex['abstract']}\n")
            keywords_str = ex["keyword"] if "kp20k" in dataset else ex["keywords"]

            if dataset == "kp20k_train" or dataset == "kp20k_valid":
                src_str =  ex['title'].lower().strip() + " " + ex['abstract'].lower().strip()
                src_str = re.sub("([!\"#$%&\'\(\)*+,-./:;<=>?@^+`{|}~])", r" \1 ", src_str)
                src_str = re.sub('\s{2,}', ' ', src_str).strip()
                src_words = [stemmer.stem(w.strip()) for w in src_str.split()]
                src_words = list(filter(None, src_words))   # remove empty strings
                trg_list = keywords_str.split(";")
                pre_kps, abs_kps = [], []
                for trg in trg_list:
                    match = False
                    trg_str = re.sub("([!\"#$%&\'\(\)*+,-./:;<=>?@^+`{|}~])", r" \1 ", trg)
                    trg_str = re.sub('\s{2,}', ' ', trg_str).strip()
                    trg_words = [stemmer.stem(w.strip()) for w in trg_str.lower().split()]
                    trg_words = list(filter(None, trg_words))
                    if len(trg_words) == 0:
                        continue
                    for src_i in range(len(src_words) - len(trg_words) + 1):
                        match = True
                        for trg_i, trg_w in enumerate(trg_words):
                            src_w = src_words[src_i + trg_i]
                            if src_w != trg_w:
                                match = False
                                break
                        if match:
                            break
                    if match:
                        pre_kps.append(trg)
                    else:
                        abs_kps.append(trg)

                trg_line = ""
                for kp in pre_kps:
                    trg_line += kp + ";"
                trg_line += "<peos>;"
                for kp in abs_kps:
                    trg_line += kp + ";"
                
                trg_file.write(f"{trg_line[:-1]}\n")
            else:
                # remove unnecessary semicolons if exist
                keywords = [w.strip() for w in keywords_str.lower().split(";")]
                keywords = list(filter(None, keywords))
                trg_file.write(f"{';'.join(keywords)}\n")

        src_file.close()
        trg_file.close()


if __name__ == "__main__":
    DATA_DIR = "data"
    RAW_DATA_DIR = "data/raw_data"
    DATASET_LIST = ["inspec_test", "krapivin_test", "nus_test", "semeval_test", "kp20k_valid", "kp20k_test"]
    
    log = FileLogger(".logs/newdata/", is_master=True, is_rank0=True)

    langdetect_model = fasttext.load_model('data/lid.176.bin')
    stemmer = PorterStemmer()
    stopword_dict = set(stopwords.words('english')) 
    dataset_dict = defaultdict(list)

    main()
