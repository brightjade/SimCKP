import logging
import json
import os
import os.path as osp
import numpy as np
from collections import defaultdict
from nltk.stem.porter import *

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from evaluate import filter_keyphrases
from utils import seed_worker


def load_data(args, config, tokenizer, split="train"):
    dataset = KPRankingDataset(args, config, tokenizer, split)
    collate_fn = dataset.collate_fn_one2many if args.one2many else dataset.collate_fn_one2one
    train_sampler = None

    if split == "train":
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        dataloader = DataLoader(dataset,
                                batch_size=args.train_batch_size,
                                collate_fn=collate_fn,
                                worker_init_fn=seed_worker,
                                num_workers=args.num_workers,
                                sampler=train_sampler,
                                shuffle=(train_sampler is None),
                                drop_last=True,
                                pin_memory=True)
    elif split == "valid":
        dataloader = DataLoader(dataset,
                                batch_size=args.eval_batch_size,
                                collate_fn=collate_fn,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
    elif split =="test":
        dataloader = DataLoader(dataset,
                                batch_size=args.test_batch_size,
                                collate_fn=collate_fn,
                                shuffle=False,
                                drop_last=False)
    else:
        raise ValueError("Data split must be either train/valid/test.")
    
    return dataloader, train_sampler


class KPRankingDataset(Dataset):

    def __init__(self, args, config, tokenizer, split="train"):
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.model_arch = args.model_type.split("-")[0]
        self.prev_model_arch = args.prev_model_type.split("-")[0]
        self.stemmer = PorterStemmer()
        # test sets are always evaluated as one2many
        self.paradigm = "one2seq" if split == "test" else args.paradigm
        self.one2many = True if split == "test" else args.one2many
        self.save_file = osp.join(args.data_dir, f"{split}_{self.model_arch}_{self.prev_model_arch}_{self.paradigm}_N{args.max_ngram_length}_G{args.gamma}_S{args.seed}.json")

        if not osp.exists(self.save_file):
            self.__load_and_cache_examples()

        self.offset_dict = {}
        with open(self.save_file, "rb") as f:
            self.offset_dict[0] = 0
            for line, _ in enumerate(f, 1):
                offset = f.tell()
                self.offset_dict[line] = offset
        self.offset_dict.popitem()

    
    def __load_and_cache_examples(self):
        logging.info(f"Creating {self.paradigm} features to {self.save_file}")
        cls_token = self.tokenizer.cls_token    # [CLS] | <s>
        sep_token = self.tokenizer.sep_token    # [SEP] | </s>
        num_special_tokens = 2
        num_empty_abstrg_line = 0
        num_empty_positive_line = 0
        total_num_pre_trg_dups = 0
        total_num_abs_trg_dups = 0
        total_num_cand_dups = 0
        count = 0

        with open(osp.join(self.args.data_dir, f"{self.split}_src_filtered_{self.prev_model_arch}.txt")) as src_f, \
             open(osp.join(self.args.data_dir, f"{self.split}_trg_filtered_{self.prev_model_arch}.txt")) as trg_f, \
             open(osp.join(self.args.data_dir, f"{self.args.prev_model_type}_{self.paradigm}_N{self.args.max_ngram_length}",
                                               f"{self.split}_trg_B{self.args.beam_size}_{self.args.decoding_method}_G{self.args.gamma}_S{self.args.seed}.txt")) as trg_f2, \
             open(self.save_file, "w") as out_f:
            for i, (src_line, trg_line, cand_line) in enumerate(tqdm(zip(src_f, trg_f, trg_f2))):
                ############################## Process source line ##############################
                title_and_context = src_line.strip().split("<eos>")
                if len(title_and_context) == 1:  # no title
                    title = ""
                    [context] = title_and_context
                elif len(title_and_context) == 2:
                    [title, context] = title_and_context
                else:
                    raise ValueError("The source text contains more than one title")

                src_tokens = self.tokenizer.tokenize(title.lower().strip() + " " + context.lower().strip())
                src_tokens = src_tokens[:self.args.max_seq_length - num_special_tokens]
                src_tokens = [cls_token] + src_tokens + [sep_token]
                src_input_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
                assert len(src_input_ids) <= self.args.max_seq_length

                ############################## Process target line ##############################
                cand_kps = cand_line.strip().split("<sep>")
                cand_kps = list(filter(None, cand_kps))
                if self.split == "test":
                    candidate_ids = []
                    cand_labels = [1] if self.one2many else np.zeros(1, dtype=np.float32)
                    for i, cand_str in enumerate(cand_kps):
                        candidate_ids.append(self.tokenizer.encode(cand_str))
                else:
                    trg_list = trg_line.strip().split(";")
                    peos_idx = trg_list.index("<peos>")
                    pre_trgs = trg_list[:peos_idx]
                    abs_trgs = trg_list[peos_idx+1:]

                    # if absent kps do not exist, skip.
                    if len(abs_trgs) == 0:
                        num_empty_abstrg_line += 1
                        continue
                    
                    # Stem trgs
                    pre_trgs = [trg.lower().strip().split() for trg in pre_trgs]
                    pre_trgs = [[self.stemmer.stem(w.lower().strip()) for w in trg] for trg in pre_trgs]
                    abs_trgs = [trg.lower().strip().split() for trg in abs_trgs]
                    abs_trgs = [[self.stemmer.stem(w.lower().strip()) for w in trg] for trg in abs_trgs]

                    # Filter duplicates
                    pre_trgs, num_pre_trg_dups = filter_keyphrases(pre_trgs, pred=False)
                    abs_trgs, num_abs_trg_dups = filter_keyphrases(abs_trgs, pred=False)
                    total_num_pre_trg_dups += num_pre_trg_dups
                    total_num_abs_trg_dups += num_abs_trg_dups

                    # Create candidate ids & labels
                    if self.one2many:
                        candidate_ids = []
                        cand_labels = []
                        cand_set = defaultdict(int)
                        stem_map = {}
                        for i, cand_str in enumerate(cand_kps):
                            cand_kps_i = cand_str.strip().split(";")
                            cand_kps_i = list(filter(None, cand_kps_i))
                            for cand in cand_kps_i:
                                cand = cand.strip()
                                cand_s = ' '.join([self.stemmer.stem(w.lower().strip()) for w in cand.split()])
                                cand_set[cand_s] += 1
                                stem_map[cand_s] = cand

                        # Sort by freq for margin ranking loss (not required for ntxentloss)
                        for cand_s, freq in sorted(cand_set.items(), key=lambda x: x[1], reverse=True):
                            cand = stem_map[cand_s]
                            cand_id = self.tokenizer.encode(cand)
                            if len(cand_id) == 2:   # only contains special tokens
                                continue
                            candidate_ids.append(cand_id)
                            is_positive = False
                            for trg_kp in abs_trgs:
                                trg_s = ' '.join(trg_kp)
                                if cand_s == trg_s:
                                    is_positive = True
                                    break

                            if is_positive:
                                cand_labels.append(1)
                            else:
                                cand_labels.append(0)
                        
                        if sum(cand_labels) == 0:
                            num_empty_positive_line += 1
                            continue
                    else:
                        candidate_ids = []
                        # one2one has a fixed number of candidate labels (==beam_size)
                        cand_labels = np.zeros(len(cand_kps), dtype=np.float32)
                        for i, cand_str in enumerate(cand_kps):
                            # Convert candidate keyphrase string to input ids
                            candidate_ids.append(self.tokenizer.encode(cand_str))
                            # Create candidate labels
                            cand_kp = cand_str.lower().strip().split()
                            cand_kp = [self.stemmer.stem(w.lower().strip()) for w in cand_kp]
                            cand_s = ' '.join(cand_kp)
                            for trg_kp in abs_trgs:
                                trg_s = ' '.join(trg_kp)
                                if cand_s == trg_s:
                                    cand_labels[i] = 1
                                    break
                        
                        if np.all(cand_labels == 0):
                            num_empty_positive_line += 1
                            continue

                # Save features
                _dict = {
                    "src_input_ids": src_input_ids,
                    "candidate_ids": candidate_ids,
                    "cand_labels": cand_labels if self.one2many else cand_labels.tolist(),
                }

                json.dump(_dict, out_f)
                out_f.write("\n")
                count += 1

        logging.info(f"# empty absent KP lines filtered: {num_empty_abstrg_line}")
        logging.info(f"# entirely incorrect candidate KP lines filtered: {num_empty_positive_line}")
        logging.info(f"# present KP duplicates: {total_num_pre_trg_dups}")
        logging.info(f"# absent KP duplicates: {total_num_abs_trg_dups}")
        logging.info(f"# candidate KP duplicates: {total_num_cand_dups}")
        logging.info(f"# total features: {count}")


    def collate_fn_one2many(self, batches):
        PAD = self.config.pad_token_id

        max_src_len = max([len(b["src_input_ids"]) for b in batches])
        src_input_ids = [b["src_input_ids"] + [PAD] * (max_src_len - len(b["src_input_ids"])) for b in batches]

        candidate_ids = [b["candidate_ids"] for b in batches]
        max_cand_len = max([len(b["candidate_ids"]) for b in batches])
        max_cand_id_len = max([max([len(c) for c in x]) for x in candidate_ids])
        candidate_ids = [F.pad(torch.stack([torch.tensor(c + [PAD] * (max_cand_id_len - len(c)), dtype=torch.long) for c in x], dim=0),
                         (0,0,0,max_cand_len-len(x)), value=PAD) for x in candidate_ids]

        cand_labels = [b["cand_labels"] + [-100] * (max_cand_len - len(b["cand_labels"])) for b in batches]

        src_input_ids = torch.tensor(src_input_ids, dtype=torch.long)
        candidate_ids = torch.stack(candidate_ids, dim=0)
        cand_labels = torch.tensor(cand_labels)

        return {"src_input_ids": src_input_ids,
                "candidate_ids": candidate_ids,
                "cand_labels": cand_labels}


    def collate_fn_one2one(self, batches):
        PAD = self.config.pad_token_id

        max_src_len = max([len(b["src_input_ids"]) for b in batches])
        src_input_ids = [b["src_input_ids"] + [PAD] * (max_src_len - len(b["src_input_ids"])) for b in batches]

        candidate_ids = [b["candidate_ids"] for b in batches]
        max_cand_len = max([len(b["candidate_ids"]) for b in batches])
        max_cand_id_len = max([max([len(c) for c in x]) for x in candidate_ids])
        candidate_ids = [F.pad(torch.stack([torch.tensor(c + [PAD] * (max_cand_id_len - len(c)), dtype=torch.long) for c in x], dim=0),
                         (0,0,0,max_cand_len-len(x)), value=PAD) for x in candidate_ids]

        cand_labels = [b["cand_labels"] + [-100] * (max_cand_len - len(b["cand_labels"])) for b in batches]

        src_input_ids = torch.tensor(src_input_ids, dtype=torch.long)
        candidate_ids = torch.stack(candidate_ids, dim=0)
        cand_labels = torch.tensor(cand_labels)

        return {"src_input_ids": src_input_ids,
                "candidate_ids": candidate_ids,
                "cand_labels": cand_labels}


    def __len__(self):
        return len(self.offset_dict)

    def __getitem__(self, line):
        offset = self.offset_dict[line]
        with open(self.save_file) as f:
            f.seek(offset)
            return json.loads(f.readline())
