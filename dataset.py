import logging
import re
import json
import os.path as osp
from collections import defaultdict

import nltk
from nltk.stem.porter import *

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from utils import seed_worker


def load_data(args, config, tokenizer, split="train"):
    dataset = KeyphraseDataset(args, config, tokenizer, split)
    collate_fn = dataset.collate_fn_with_tags if args.extracting else dataset.collate_fn
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
                                shuffle=False if args.do_predict else (train_sampler is None),
                                drop_last=False if args.do_predict else True,
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


class KeyphraseDataset(Dataset):

    def __init__(self, args, config, tokenizer, split="train"):
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.model_arch = args.model_type.split("-")[0]

        if args.do_predict or split == "test":
            self.paradigm = "one2seq"
            self.one2many = True
        else:
            self.paradigm = args.paradigm
            self.one2many = True if args.paradigm == "one2seq" else False

        if args.extracting:
            self.save_file = osp.join(args.data_dir, f"{split}_{self.model_arch}_{self.paradigm}_N{args.max_ngram_length}.json")
        else:
            self.save_file = osp.join(args.data_dir, f"{split}_{self.model_arch}_{self.paradigm}.json")

        ### Stanford POS tagger ###
        if args.extracting:
            # Tag list: https://pythonalgos.com/natural-language-processing-part-of-speech-tagging/
            PHRASE_GRAMMAR = """
                PHRASE: {<IN|CD|DT|FW|GW|AFX|POS|HYPH|LS|ADD|:|NN.*|VB.*|JJ.*|RB.*>+<CC|RP|IN|CD|DT|FW|GW|AFX|POS|HYPH|LS|ADD|:|NN.*|VB.*|JJ.*|RB.*>*}
            """
            # CD: cardinal digit, FW: foreign word, GW: additional word, NN: noun, VB: verb, JJ: adj, RB: adv, ADD: email (for <digit>)
            self.indep_pos_set = {"CD", "FW", "GW", "NN", "VB", "JJ", "RB", "AD"}
            # DT: determiner, AF(AFX): affix, LS: list item marker
            self.end_dep_pos_set = {"DT", "AF", "LS"}           # can start but not end with these
            # CC: coordinating conjunction, PO(POS): possessive, HY(HYPH): hyphen, IN: subordinating conjunction or preposition
            self.dep_pos_set = {"CC", "PO", "HY", ":", "IN"}    # cannot start or end with these
            # RP: particle adverb (e.g., put it "back")
            self.start_dep_pos_set = {"RP"}                     # cannot start but end with these

            self.phrase_parser = nltk.RegexpParser(PHRASE_GRAMMAR)
            self.stemmer = PorterStemmer()

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
        cls_token = self.tokenizer.cls_token  # <s>
        eos_token = self.tokenizer.eos_token  # </s>
        custom_sep_token = "<sep>" if self.one2many else eos_token
        num_special_tokens = 2
        num_empty_title_line, num_empty_src_line, num_empty_trg_line, num_empty_abstrg_line = 0, 0, 0, 0
        long_seq_lines, misaligned_lines = [], []
        missed_pre_trgs = []
        total_pre_trg_counts = 0
        count, doc_count = 0, 0

        if self.args.overwrite_filtered_data:
            src_f = open(osp.join(self.args.data_dir, f"{self.split}_src_filtered_{self.model_arch}.txt"), "w")
            trg_f = open(osp.join(self.args.data_dir, f"{self.split}_trg_filtered_{self.model_arch}.txt"), "w")

        with open(osp.join(self.args.data_dir, f"{self.split}_postagged.json")) as f, \
             open(self.save_file, "w") as out_f:
            for i, line in enumerate(tqdm(f)):
                ex = json.loads(line)

                # Filter empty lines
                if ex["src"].strip() == "":
                    num_empty_src_line += 1
                    continue
                trg_list = ex["trg"].strip().split(";")
                trg_list = list(filter(None, trg_list))     # remove '' from list
                trg_list = [re.sub('\s{2,}', ' ', trg).strip() for trg in trg_list]
                if len(trg_list) == 0:
                    num_empty_trg_line += 1
                    continue

                ############################## Process source line ##############################
                title_and_context = ex["src"].strip().split("<eos>")
                if len(title_and_context) == 1:  # no title
                    title = ""
                    [context] = title_and_context
                elif len(title_and_context) == 2:
                    [title, context] = title_and_context
                else:
                    raise ValueError("The source text contains more than one title")

                eos_idx = ex["src_words"].index("<eos>")
                title_words = ex["src_words"][:eos_idx]
                context_words = ex["src_words"][eos_idx+1:]
                src_words = title_words + context_words
                src_tokens = [cls_token]
                src_spans = []
                candidate_kp_spans = defaultdict(list)
                candidate_kp_masks = []
                pre_labels_mat = []
                not_aligned = False
                j = 1   # pointer starts after cls index (j=0)

                tokens = self.tokenizer.tokenize(title.strip() + context)
                src_tokens += tokens
                j, src_spans, not_aligned = self.__align_token2word(j, src_spans, src_tokens, src_words)
                src_tokens += [eos_token]
                if not_aligned:
                    misaligned_lines.append((i, ex["src"]))
                    print(f"Not aligned: {i}, # {len(misaligned_lines)}")
                    continue

                # Skip long sequences for training (this follows other papers)
                if self.split != "test":
                    if len(tokens) > self.args.max_seq_length - num_special_tokens:
                        long_seq_lines.append((i, len(src_tokens)))
                        continue
                else:
                    # TODO: if test examples get truncated, extraction will be limited.
                    if len(tokens) > self.args.max_seq_length - num_special_tokens:
                        long_seq_lines.append((i, len(src_tokens)))
                    truncated_tokens = tokens[:self.args.max_seq_length - num_special_tokens]
                    src_tokens = [cls_token] + truncated_tokens + [eos_token]

                src_input_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
                assert len(src_input_ids) <= self.args.max_seq_length

                # Tag candidate keyphrases
                if self.args.extracting:
                    # json stores (word, pos tag) pair as a list -> convert to tuple to create a tree
                    src_word_tag_pairs = [tuple(x) for x in ex["src_word_tag_pairs"]]
                    phrase_tree = self.phrase_parser.parse(src_word_tag_pairs)
                    j = 0
                    for tree in phrase_tree:
                        if (isinstance(tree, nltk.tree.Tree) and tree._label == "PHRASE"):
                            # Extract all possible keyphrases within the current keyphrase
                            for k, (w1, t1) in enumerate(tree.leaves()):
                                # if POS tag not in the allowed list, skip it
                                if t1[:2] not in self.indep_pos_set and t1[:2] not in self.end_dep_pos_set:
                                    continue

                                word = w1.lower().replace(u'\xa0', u' ')
                                span = self.stemmer.stem(word)
                                orig_span = word
                                
                                # backtrack pointer j in case we point to wrong idx
                                # this happens b/c phrase parser parses more words than word tokenizer does
                                while j+k >= len(src_spans):
                                    j -= 1
                                start, end = src_spans[j+k]
                                curr_span = self.tokenizer.convert_tokens_to_string(src_tokens[start:end]).strip().lower()
                                st_curr_span = self.stemmer.stem(curr_span)
                                while (span not in st_curr_span and st_curr_span not in span and
                                    word not in curr_span and curr_span not in word):
                                    j -= 1
                                    if j+k < 0:
                                        not_aligned = True
                                        break
                                    start, end = src_spans[j+k]
                                    curr_span = self.tokenizer.convert_tokens_to_string(src_tokens[start:end]).strip().lower()
                                    st_curr_span = self.stemmer.stem(curr_span)

                                if not_aligned:
                                    break

                                # count the number of hyphens to correct ngram length for hyphenated phrases
                                num_hyphens = 0

                                # independent unigrams can be candidate kps (except subordinating conjunction)
                                if t1[:2] in self.indep_pos_set:
                                    num_hyphens += word.count("-")
                                    _word = word.replace("-", " ")
                                    span = " ".join([self.stemmer.stem(w) for w in _word.strip().split()])
                                    orig_span = word
                                    if t1[:2] != "IN":  # sub conj itself cannot be a candidate
                                        candidate_kp_spans[span].append((start, end))

                                # look for ngrams within the span
                                for kk, (w2, t2) in enumerate(tree.leaves()[k+1:], k+2):
                                    if len(span.split()) == (self.args.max_ngram_length + num_hyphens):
                                        break
                                    # if current word is a dependent or end-dependent POS, ngram cannot end with it. Continue the loop.
                                    # if hyphen or possessive, do not include it in span
                                    if t2[:2] in self.end_dep_pos_set or t2[:2] in self.dep_pos_set:
                                        if w2 == "-":
                                            num_hyphens += 1
                                        elif w2 == "'s":
                                            num_hyphens += 1
                                            span += f" {self.stemmer.stem(w2.lower())}"
                                            orig_span += f" {w2.lower()}"
                                        elif w2 == "'": # only for plural possessive
                                            continue
                                        else:
                                            span += f" {self.stemmer.stem(w2.lower())}"
                                            orig_span += f" {w2.lower()}"
                                        continue
                                    if t2[:2] in self.indep_pos_set or t2[:2] in self.start_dep_pos_set:
                                        num_hyphens += w2.count("-")
                                        _w2 = w2.replace("-", " ")
                                        span += f" {' '.join([self.stemmer.stem(w) for w in _w2.lower().strip().split()])}"
                                        orig_span += f" {w2}"
                                        span_list = src_spans[j+k:j+kk]
                                        start, end = span_list[0][0], span_list[-1][-1]
                                        candidate_kp_spans[span].append((start, end))

                            j += len(tree.leaves())
                        else:
                            j += 1

                        if not_aligned:
                            break

                    if not_aligned:
                        misaligned_lines.append((i, ex["src"]))
                        print(f"Not aligned: {i}, # {len(misaligned_lines)}")
                        continue

                    candidate_kp_masks = list(candidate_kp_spans.values())

                ############################## Process target line ##############################
                if self.split == "test":
                    trg_input_ids = self.tokenizer.convert_tokens_to_ids(["."])     # dummy
                    labels = [0]  # dummy
                else:
                    trg_tokens = []
                    if self.args.extracting:
                        abs_trg_tokens = []
                        curr_missed_pre_trgs = []
                        peos_idx = trg_list.index("<peos>")
                        pre_trgs = trg_list[:peos_idx]
                        abs_trgs = trg_list[peos_idx+1:]
                        
                        # if no absent kps found, skip. (skipnoabs on default)
                        if len(abs_trgs) == 0:
                            num_empty_abstrg_line += 1
                            continue

                        # Create present KP labels
                        candidate_kps = candidate_kp_spans.keys()
                        pre_labels = dict.fromkeys(candidate_kps, 0)
                        for trg in pre_trgs:
                            # src splits possessives (b/c of pos tagger) so trg should do the same
                            trg_str = trg.replace("'s", " 's").replace("s'", "s")
                            # src splits hyphens (b/c of pos tagger) so trg should do the same
                            trg_stem_str = " ".join([self.stemmer.stem(w.lower().strip()) for w in re.split("[ -]", trg_str.strip())])
                            if trg_stem_str in candidate_kps:
                                pre_labels[trg_stem_str] = 1
                            else:
                                missed_pre_trgs.append(trg)
                                curr_missed_pre_trgs.append(trg)
                            total_pre_trg_counts += 1
                        pre_labels_mat = list(pre_labels.values())  # (num_cand_kps,)

                        # if missed, move to absent KP labels
                        # abs_trgs += curr_missed_pre_trgs

                        # let the model be trained to generate present kps
                        # only when skipnoabs is False
                        # if len(abs_trgs) == 0:
                        #     abs_trgs = pre_trgs

                        # Create absent KP labels
                        if self.one2many:
                            for trg_idx, trg in enumerate(abs_trgs):
                                if trg_idx == len(abs_trgs) - 1:
                                    abs_trg_tokens += self.tokenizer.tokenize(trg) + [eos_token]
                                else:
                                    abs_trg_tokens += self.tokenizer.tokenize(trg) + [custom_sep_token]
                            trg_input_ids = self.tokenizer.convert_tokens_to_ids(abs_trg_tokens)
                        else:
                            for trg in abs_trgs:
                                abs_trg_tokens = self.tokenizer.tokenize(trg) + [eos_token]
                                trg_input_ids = self.tokenizer.convert_tokens_to_ids(abs_trg_tokens)
                                _dict = {
                                    "src_input_ids": src_input_ids,
                                    "candidate_kp_masks": candidate_kp_masks,
                                    "trg_input_ids": trg_input_ids,
                                    "pre_labels": pre_labels_mat,
                                }
                                json.dump(_dict, out_f)
                                out_f.write("\n")
                                count += 1
                                # if self.split == "valid": break     # valid one2one needs only one abs label

                    else:   ### not extracting, just generating ###
                        if self.one2many:
                            for trg_idx, trg in enumerate(trg_list):
                                if trg == "<peos>":
                                    continue
                                if trg_idx == len(trg_list) - 1:
                                    trg_tokens += self.tokenizer.tokenize(trg) + [eos_token]
                                else:
                                    trg_tokens += self.tokenizer.tokenize(trg) + [custom_sep_token]
                            trg_input_ids = self.tokenizer.convert_tokens_to_ids(trg_tokens)
                        else:
                            for trg in trg_list:
                                if trg == "<peos>":
                                    continue
                                trg_tokens = self.tokenizer.tokenize(trg) + [eos_token]
                                trg_input_ids = self.tokenizer.convert_tokens_to_ids(trg_tokens)
                                _dict = {
                                    "src_input_ids": src_input_ids,
                                    "candidate_kp_masks": candidate_kp_masks,
                                    "trg_input_ids": trg_input_ids,
                                    "pre_labels": pre_labels_mat,
                                }
                                json.dump(_dict, out_f)
                                out_f.write("\n")
                                count += 1

                # save json for one2seq / test data needs to be one2seq
                if self.one2many or self.split == "test":
                    _dict = {
                        "src_input_ids": src_input_ids,
                        "candidate_kp_masks": candidate_kp_masks,
                        "trg_input_ids": trg_input_ids,
                        "pre_labels": pre_labels_mat,
                    }
                    json.dump(_dict, out_f)
                    out_f.write("\n")
                    count += 1

                if self.args.overwrite_filtered_data:
                    src_f.write(f"{ex['src'].strip()}\n")
                    trg_f.write(f"{';'.join(trg_list)}\n")

                doc_count += 1

        if self.args.overwrite_filtered_data:
            src_f.close()
            trg_f.close()

        logging.info(f"# empty title examples: {num_empty_title_line}")
        logging.info(f"# empty lines filtered: {num_empty_src_line + num_empty_trg_line}")
        logging.info(f"# empty absent KP lines filtered: {num_empty_abstrg_line}")
        logging.info(f"# misaligned lines filtered: {len(misaligned_lines)}")
        logging.info(f"# long sequences filtered: {len(long_seq_lines)}")
        logging.info(f"(idx, src_line) of misaligned sequences: {misaligned_lines}")
        logging.info(f"(idx, #tokens) of long sequences: {long_seq_lines}")
        logging.info(f"# missed present KP ratio: {len(missed_pre_trgs)}/{total_pre_trg_counts}")
        logging.info(f"List of missed present KPs: {missed_pre_trgs}")
        logging.info(f"# total examples #(unique doc)/#(o2o doc): {doc_count}/{count}")


    def __align_token2word(self, j, src_spans, src_tokens, words):
        """
        Get idx mappings of subword tokens for each word.
        """
        for idx, word in enumerate(words):
            k = 1
            not_aligned = False
            while True:
                span = self.tokenizer.convert_tokens_to_string(src_tokens[j:j+k]).strip()
                if span == word:    # exact match
                    src_spans.append((j,j+k))
                    j += k
                    break
                if word in span:    # a subword may contain two or more words
                    src_spans.append((j,j+k))
                    next_word = words[idx+1] if idx < len(words)-1 else ""
                    j += k if word + next_word not in span and span not in word + next_word else 0
                    break
                if span in word:    # a word may contain two or more subwords (found partially, continue to find span)
                    next_span = self.tokenizer.convert_tokens_to_string(src_tokens[j:j+k+1]).strip() if j+k+1 <= len(src_tokens) else "!@#"
                    if next_span in word:
                        k += 1
                    else:
                        src_spans.append((j,j+k))
                        j += k
                        break
                else:               # move to the next pointer
                    j += 1
                    k = 1
                if j == len(src_tokens):
                    not_aligned = True
                    break
            if not_aligned:
                break
        return j, src_spans, not_aligned


    def collate_fn_with_tags(self, batches):
        PAD = self.config.pad_token_id

        max_src_len = max([len(b["src_input_ids"]) for b in batches])
        input_ids = [b["src_input_ids"] + [PAD] * (max_src_len - len(b["src_input_ids"])) for b in batches]
        attention_mask = [[1] * len(b["src_input_ids"]) + [0] * (max_src_len - len(b["src_input_ids"])) for b in batches]

        max_cand_kp_len = max([len(b["candidate_kp_masks"]) for b in batches])
        candidate_kp_masks = []
        for b in batches:
            sample_mask = []
            for x in b["candidate_kp_masks"]:
                span_mask = torch.zeros(max_src_len)
                for start, end in x:
                    span_mask[start:end] = 1
                sample_mask.append(span_mask)
            sample_mask = torch.stack(sample_mask, dim=0)
            sample_mask = F.pad(sample_mask, (0,0,0,max_cand_kp_len-sample_mask.shape[0]))
            candidate_kp_masks.append(sample_mask)

        max_pre_trg_len = max([len(b["pre_labels"]) for b in batches])
        pre_kp_labels = [b["pre_labels"] + [-100] * (max_pre_trg_len - len(b["pre_labels"])) for b in batches]

        max_abs_trg_len = max([len(b["trg_input_ids"]) for b in batches])
        abs_kp_labels = [b["trg_input_ids"] + [-100] * (max_abs_trg_len - len(b["trg_input_ids"])) for b in batches]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        candidate_kp_masks = torch.stack(candidate_kp_masks, dim=0)
        pre_kp_labels = torch.tensor(pre_kp_labels, dtype=torch.long)
        abs_kp_labels = torch.tensor(abs_kp_labels, dtype=torch.long)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "candidate_kp_masks": candidate_kp_masks,
                "pre_kp_labels": pre_kp_labels,
                "labels": abs_kp_labels}


    def collate_fn(self, batches):
        PAD = self.config.pad_token_id

        max_src_len = max([len(b["src_input_ids"]) for b in batches])
        input_ids = [b["src_input_ids"] + [PAD] * (max_src_len - len(b["src_input_ids"])) for b in batches]
        attention_mask = [[1] * len(b["src_input_ids"]) + [0] * (max_src_len - len(b["src_input_ids"])) for b in batches]

        max_trg_len = max([len(b["trg_input_ids"]) for b in batches])
        labels = [b["trg_input_ids"] + [-100] * (max_trg_len - len(b["trg_input_ids"])) for b in batches]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


    def __len__(self):
        return len(self.offset_dict)

    def __getitem__(self, line):
        offset = self.offset_dict[line]
        with open(self.save_file) as f:
            f.seek(offset)
            return json.loads(f.readline())
