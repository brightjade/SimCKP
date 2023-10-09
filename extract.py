import argparse
import os
import os.path as osp
import re
from nltk.stem.porter import *

import torch
from tqdm import tqdm

import config
from dataset import load_data
from models.utils import load_config, load_tokenizer, load_model
from logger import FileLogger


class Extractor:

    def __init__(self):
        ### Load config / tokenizer / model ###
        self.config = load_config(args)
        self.tokenizer = load_tokenizer(args)
        self.model = load_model(args, self.config, self.tokenizer)

        self.config.semicolon_token_id = self.tokenizer.convert_tokens_to_ids(";")
        self.stemmer = PorterStemmer()

        ### Load data ###
        if args.do_extract or args.do_generate or args.do_generate_abs_candidates:
            self.test_loader, _ = load_data(args, self.config, self.tokenizer, split="test")
        if args.do_generate_abs_candidates and dataset == "kp20k":
            self.train_loader, _ = load_data(args, self.config, self.tokenizer, split="train")
            self.valid_loader, _ = load_data(args, self.config, self.tokenizer, split="valid")

        self.theta = args.theta

        ### Load trained parameter weights ###
        if osp.exists(ckpt_model_path):
            log.console(f"Loading model checkpoint from {ckpt_model_path}...")
            ckpt = torch.load(ckpt_model_path)
            log.console(f"Validation loss was {ckpt['loss']:.4f}")
            log.console(f"Validation avg theta was {ckpt['theta']:.4f}")
            log.console(f"Validation avg topk was {ckpt['topk']:.4f}")
            log.console(f"Validation F1@5 was {ckpt['f1_at_k']:.4f}")
            log.console(f"Validation F1@M was {ckpt['f1_at_m']:.4f}")
            self.theta = ckpt['theta']
            pretrained_dict = {key.replace("module.", ""): value for key, value in ckpt['model_state_dict'].items()}
            self.model.load_state_dict(pretrained_dict)
        else:
            log.event("Predicting with untrained model!")


    @torch.no_grad()
    def extract(self):
        """
        Extracts present keyphrases.
        """
        total = len(self.test_loader)
        f = open(pre_filepath, "w")

        with tqdm(desc="Extracting", total=total, ncols=100) as pbar:
            for step, inputs in enumerate(self.test_loader, 1):
                for k, v in inputs.items():
                    inputs[k] = v.cuda(args.gpu, non_blocking=True)

                # Extract present KPs
                outputs = self.model.extract(**inputs, theta=self.theta)
                pre_kp_list = self.tokenizer.batch_decode(outputs)
                self.write_pred_to_file(pre_kp_list, f)
                
                pbar.update(1)

        f.close()


    @torch.no_grad()
    def generate(self):
        """
        Generates all keyphrases.
        """
        total = len(self.test_loader)
        f = open(pred_filepath, "w")

        with tqdm(desc="Generating", total=total, ncols=100) as pbar:
            for step, inputs in enumerate(self.test_loader, 1):
                for k, v in inputs.items():
                    inputs[k] = v.cuda(args.gpu, non_blocking=True)

                # Beam Search for fair comparison
                outputs = self.model.generate(inputs["input_ids"],
                                              num_beams=args.beam_size,
                                              max_new_tokens=100,
                                              no_repeat_ngram_size=2)

                pred_kp_list = self.tokenizer.batch_decode(outputs)
                pred_kp_list = self.remove_special_tokens(pred_kp_list)

                for b in range(len(pred_kp_list)):
                    start, end = b, b+1
                    gen_cand_list = pred_kp_list[start:end]
                    gen_cand_line = ";".join(gen_cand_list)
                    f.write(f"{gen_cand_line.lower().strip()}\n")

                pbar.update(1)
                del outputs

        f.close()


    @torch.no_grad()
    def generate_candidates(self, split):
        """
        Generates candidates for absent keyphrases.
        """
        if split == "train":
            loader = self.train_loader
        elif split == "valid":
            loader = self.valid_loader
        else:
            loader = self.test_loader
        total = len(loader)
        f = open(pred_filepath, "w")

        with tqdm(desc="Generating candidate absent keyphrases", total=total, ncols=100) as pbar:
            for step, inputs in enumerate(loader, 1):
                for k, v in inputs.items():
                    inputs[k] = v.cuda(args.gpu, non_blocking=True)

                # Perform Beam Search
                if args.decoding_method == "beam":
                    outputs = self.model.generate(inputs["input_ids"],
                                                  max_new_tokens=100,
                                                  num_beams=args.beam_size,
                                                  no_repeat_ngram_size=2,
                                                  num_return_sequences=args.num_return_sequences)
                elif args.decoding_method == "dbs":
                    outputs = self.model.generate(inputs["input_ids"],
                                                  max_new_tokens=100,
                                                  num_beams=args.beam_size,
                                                  num_beam_groups=args.num_beam_groups,
                                                  diversity_penalty=args.diversity_penalty,
                                                  no_repeat_ngram_size=2,
                                                  num_return_sequences=args.num_return_sequences)
                elif args.decoding_method == "topk":
                    outputs = self.model.generate(inputs["input_ids"],
                                                  max_new_tokens=100,
                                                  do_sample=True,
                                                  top_k=args.top_k,
                                                  no_repeat_ngram_size=2,
                                                  num_return_sequences=args.num_return_sequences)
                elif args.decoding_method == "nucleus":
                    outputs = self.model.generate(inputs["input_ids"],
                                                  max_new_tokens=100,
                                                  do_sample=True,
                                                  top_k=0,
                                                  top_p=args.top_p,
                                                  no_repeat_ngram_size=2,
                                                  num_return_sequences=args.num_return_sequences)

                pred_kp_list = self.tokenizer.batch_decode(outputs)
                pred_kp_list = self.remove_special_tokens(pred_kp_list)

                for b in range(args.test_batch_size):
                    start, end = b*args.num_return_sequences, (b+1)*args.num_return_sequences
                    cand_abs_trg_list = pred_kp_list[start:end]
                    cand_abs_trg_line = "<sep>".join(cand_abs_trg_list)
                    f.write(f"{cand_abs_trg_line.lower().strip()}\n")

                pbar.update(1)
                del outputs

        f.close()

    def remove_special_tokens(self, pred_kp_list):
        for i in range(len(pred_kp_list)):
            pred_kp_list[i] = pred_kp_list[i].replace("<s>", "")
            pred_kp_list[i] = pred_kp_list[i].replace("</s>", "")
            pred_kp_list[i] = pred_kp_list[i].replace("<pad>", "")
            pred_kp_list[i] = pred_kp_list[i].replace("<sep>", ";")
            pred_kp_list[i] = re.sub('\s{2,}', ' ', pred_kp_list[i])
            pred_kp_list[i] = pred_kp_list[i].strip()
        return pred_kp_list


    def write_pred_to_file(self, pred_kp_list, f):
        for pred_kp_l in pred_kp_list:
            pred_kp_l = pred_kp_l.replace("<s>", "")
            pred_kp_l = pred_kp_l.replace("</s>", "")
            pred_kp_l = pred_kp_l.replace("<pad>", "")
            pred_kp_l = pred_kp_l.replace("<sep>", ";")
            pred_kp_l = re.sub('\s{2,}', ' ', pred_kp_l)
            f.write(f"{pred_kp_l.strip()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KP Stage 1 Inference")
    config.model_args(parser)
    config.data_args(parser)
    config.predict_args(parser)
    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # other global variables
    ckpt_model_path = osp.join(args.train_output_dir, "best_valid_f1_at_m.pt")
    args.distributed = False
    args.train_batch_size = args.eval_batch_size = args.test_batch_size
    dataset = args.data_dir.split("/")[-1]

    os.makedirs(args.test_output_dir, exist_ok=True)

    log = FileLogger(args.test_output_dir, is_master=True, is_rank0=True, log_to_file=args.log_to_file)
    log.console(args)

    extractor = Extractor()
    if args.do_generate:
        log.console("Generate keyphrases...")
        pred_filepath = osp.join(args.test_output_dir, f"pred_kps.txt")
        if osp.exists(pred_filepath):
            raise Exception("Prediction files already exist!")
        extractor.generate()

    if args.do_extract:
        log.console("Extract present keyphrases...")
        pre_filepath = osp.join(args.test_output_dir, f"pre_kps.txt")
        if osp.exists(pre_filepath):
            raise Exception("Prediction files already exist!")
        extractor.extract()

    if args.do_generate_abs_candidates:
        log.console("Generate candidate absent keyphrases...")
        output_dir = osp.join(args.data_dir, f"{args.model_type}_{args.paradigm}_N{args.max_ngram_length}")
        os.makedirs(output_dir, exist_ok=True)
        splits = ["train", "valid", "test"] if dataset == "kp20k" else ["test"]
        for split in splits:
            pred_filepath = osp.join(output_dir, f"{split}_trg_B{args.num_return_sequences}_{args.decoding_method}_G{args.gamma}_S{args.seed}.txt")
            if osp.exists(pred_filepath):
                raise Exception("Prediction files already exist!")
            extractor.generate_candidates(split=split)
