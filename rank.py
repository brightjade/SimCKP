import argparse
import re
import os
import os.path as osp

import torch
from tqdm import tqdm

import config
from dataset2 import load_data
from models.utils import load_config, load_tokenizer, load_model
from logger import FileLogger


class Ranker:

    def __init__(self):
        ### Load config / tokenizer / model ###
        self.config = load_config(args)
        self.tokenizer = load_tokenizer(args)
        self.model = load_model(args, self.config, self.tokenizer)
        self.config.semicolon_token_id = self.tokenizer.convert_tokens_to_ids(";")

        ### Load data ###
        self.test_loader, _ = load_data(args, self.config, self.tokenizer, split="test")

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
            pretrained_dict = {key.replace("module.", ""): value for key, value in ckpt['model_state_dict'].items()}
            self.model.load_state_dict(pretrained_dict)
            self.theta = ckpt['theta']
        else:
            log.event("Predicting with untrained model!")


    @torch.no_grad()
    def rank(self):
        total = len(self.test_loader)
        f = open(abs_filepath, "w")
        with tqdm(desc="Ranking", total=total, ncols=100) as pbar:
            for step, inputs in enumerate(self.test_loader, 1):
                for k, v in inputs.items():
                    inputs[k] = v.to(args.device)

                # Rank absent KPs
                outputs = self.model.rank(**inputs, theta=self.theta)
                
                pred_kp_list = self.tokenizer.batch_decode(outputs)
                self.write_pred_to_file(pred_kp_list, f)

                pbar.update(1)
        
        f.close()


    def write_pred_to_file(self, pred_kp_list, _f):
        for pred_kp_l in pred_kp_list:
            if args.model_arch == "roberta":
                pred_kp_l = pred_kp_l.replace("<s>", "")
                pred_kp_l = pred_kp_l.replace("<pad>", "")
                pred_kp_l = pred_kp_l.replace("</s>", ";")
            else:
                pred_kp_l = pred_kp_l.replace("[CLS]", "")
                pred_kp_l = pred_kp_l.replace("[PAD]", "")
                pred_kp_l = pred_kp_l.replace("[SEP]", ";")
            pred_kp_l = re.sub('\s{2,}', ' ', pred_kp_l).strip()
            pred_kp_l = pred_kp_l.replace(" - ", "-")
            _f.write(f"{pred_kp_l.strip()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KP Stage 2 Inference")
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
    args.one2many = True if args.paradigm == "one2seq" else False
    args.model_arch = args.model_type.split("-")[0]

    os.makedirs(args.test_output_dir, exist_ok=True)

    log = FileLogger(args.test_output_dir, is_master=True, is_rank0=True, log_to_file=args.log_to_file)
    log.console(args)

    ranker = Ranker()
    log.console("Rank absent keyphrases")
    abs_filepath = osp.join(args.test_output_dir, f"B{args.beam_size}_{args.decoding_method}_abs_kps.txt")
    if osp.exists(abs_filepath):
        raise Exception("Prediction files already exist!")
    ranker.rank()
