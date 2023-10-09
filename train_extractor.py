import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
import os.path as osp
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
import wandb

import config
from dataset import load_data
from models.utils import load_config, load_tokenizer, load_model
from models.extractor import pad_fn
from logger import FileLogger
from utils import *


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.n_gpu = ngpus_per_node
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_rank0 = args.rank == 0

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            args.wandb_on = args.wandb_on if args.rank == 0 else False
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    is_master = (not args.distributed) or (args.rank == 0)

    os.makedirs(args.train_output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    global log
    log = FileLogger(args.train_output_dir, is_master=is_master, is_rank0=is_rank0, log_to_file=args.log_to_file)
    log.console(args)
    if args.wandb_on:
        wandb.init(project="SimCKP-S1", name="/".join(args.train_output_dir.split("/")[1:]))

    trainer = Trainer(args)
    start_time = time.time()
    trainer.train()
    log.console(f"Time for training: {time.time() - start_time:.1f} seconds")


class Trainer:

    def __init__(self, args):
        self.args = args

        ### Load config / tokenizer / model ###
        self.config = load_config(args)
        self.tokenizer = load_tokenizer(args)

        ### Load data ###
        self.train_loader, self.train_sampler = load_data(args, self.config, self.tokenizer, split="train")
        self.valid_loader, _ = load_data(args, self.config, self.tokenizer, split="valid")

        self.model = load_model(args, self.config, self.tokenizer)

        ### Calculate steps ###
        args.total_steps = int(len(self.train_loader) * args.epochs // args.gradient_accumulation_steps)
        args.warmup_steps = int(args.total_steps * args.warmup_ratio)
        log.console(f"warmup steps: {args.warmup_steps}, total steps: {args.total_steps}")

        ### scaler / optimizer / scheduler ###
        self.scaler = init_scaler(args)
        self.optimizer = init_optimizer(args, self.model)
        self.scheduler = init_scheduler(args, self.optimizer)

        self.best_valid_loss = float("inf")
        self.best_valid_f1_at_k = float("-inf")
        self.best_valid_f1_at_m = float("-inf")
        self.start_epoch = 0
        self.tolerance = 0
        self.global_step = 0

        ### Resume training ###
        ckpt_model_path = osp.join(args.train_output_dir, "best_valid_f1_at_m.pt")
        if args.resume and osp.exists(ckpt_model_path):
            log.console(f"Resuming {args.paradigm} model checkpoint from {ckpt_model_path}...")
            ckpt = torch.load(ckpt_model_path)
            self.best_valid_loss = ckpt["loss"]
            self.best_valid_f1_at_k = ckpt["f1_at_k"]
            self.best_valid_f1_at_m = ckpt["f1_at_m"]
            self.start_epoch = ckpt["epoch"]
            self.global_step = ckpt["steps"]
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            log.console(f"Validation loss was {ckpt['loss']:.4f}")
            log.console(f"Validation avg theta was {ckpt['theta']:.4f}")
            log.console(f"Validation avg topk was {ckpt['topk']:.4f}")
            log.console(f"Validation F1@5 was {ckpt['f1_at_k']:.4f}")
            log.console(f"Validation F1@M was {ckpt['f1_at_m']:.4f}")
        else:
            log.console(f"Training {args.paradigm} model from scratch")


    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
            avg_train_loss = self.__epoch_train(epoch)
            # avg_valid_loss, valid_score_dict = self.__epoch_valid()
            if self.tolerance == self.args.max_tolerance: break
            log.console(f"epoch: {epoch+1}, " +
                        f"steps: {self.global_step}, " +
                        f"current lr: {self.optimizer.param_groups[0]['lr']:.8f}, " +
                        f"train loss: {avg_train_loss:.4f}")


    def __epoch_train(self, epoch):
        self.model.train()
        train_loss, train_ext_loss, train_gen_loss = 0., 0., 0.
        total = len(self.train_loader)
        no_ext_count = 0

        with tqdm(desc="Training", total=total, ncols=100, disable=self.args.hide_tqdm) as pbar:
            for step, inputs in enumerate(self.train_loader, 1):
                for k, v in inputs.items():
                    inputs[k] = v.cuda(self.args.gpu, non_blocking=True)

                ### Forward pass ###
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    if self.args.extracting:
                        outputs = self.model(**inputs)
                        ext_logits, _, ext_loss, gen_loss, loss = outputs
                    else:
                        outputs = self.model(**inputs)
                        loss = outputs.loss

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                        if self.args.extracting:
                            ext_loss = ext_loss / self.args.gradient_accumulation_steps
                            gen_loss = gen_loss / self.args.gradient_accumulation_steps

                if self.args.extracting:
                    if ext_logits is None:
                        no_ext_count += 1
                        continue
                    train_ext_loss += ext_loss.item()
                    train_gen_loss += gen_loss.item()
                    train_loss += loss.item()
                else:
                    train_loss += loss.item()

                ### Backward pass ###
                _step = step - no_ext_count
                if _step % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.global_step += 1

                    if self.global_step == 1 or self.global_step % self.args.logging_steps == 0:
                        curr_train_loss = train_loss / (_step / self.args.gradient_accumulation_steps)
                        curr_ext_loss = train_ext_loss / (_step / self.args.gradient_accumulation_steps)
                        curr_gen_loss = train_gen_loss / (_step / self.args.gradient_accumulation_steps)
                        log.console(f"current lr: {self.optimizer.param_groups[0]['lr']:.8f}, " +
                                    f"steps: {self.global_step}, " +
                                    f"train loss: {(curr_train_loss):.4f}, " +
                                    f"ext loss: {(curr_ext_loss):.4f}, " +
                                    f"gen loss: {(curr_gen_loss):.4f}")
                        if self.args.wandb_on:
                            wandb.log({"Train Loss": curr_train_loss,
                                       "Train Extraction Loss": curr_ext_loss,
                                       "Train Generation Loss": curr_gen_loss}, step=self.global_step)
                    
                    if self.global_step % self.args.evaluation_steps == 0:
                        avg_valid_loss, valid_score_dict = self.__epoch_valid()
                        if avg_valid_loss < self.best_valid_loss:
                            self.tolerance = 0
                            self.best_valid_loss = avg_valid_loss

                            if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                and self.args.rank % self.args.n_gpu == 0):
                                log.console(f"Saving lowest valid loss checkpoint to {self.args.train_output_dir}...")
                                torch.save({'epoch': epoch,
                                            'steps': self.global_step,
                                            'loss': avg_valid_loss,
                                            'theta': valid_score_dict['theta'] if self.args.extracting else 0.,
                                            'topk': valid_score_dict['topk'] if self.args.extracting else 0.,
                                            'f1_at_k': valid_score_dict['F1@5'] if self.args.extracting else 0.,
                                            'f1_at_m': valid_score_dict['F1@M'] if self.args.extracting else 0.,
                                            'model_state_dict': self.model.state_dict(),
                                            'optimizer_state_dict': self.optimizer.state_dict(),
                                            'scheduler_state_dict': self.scheduler.state_dict()
                                            }, osp.join(self.args.train_output_dir, "lowest_valid_loss.pt"))
                                
                                with open(osp.join(args.train_output_dir, "best_loss_results.txt"), "w") as f:
                                    f.write(f"Epoch: {epoch}\n" +
                                            f"Total Steps: {self.global_step}\n" +
                                            f"Valid Loss: {avg_valid_loss}\n" +
                                            f"Theta: {valid_score_dict['theta'] if self.args.extracting else 0.}\n" +
                                            f"Top K: {valid_score_dict['topk'] if self.args.extracting else 0.}\n" +
                                            f"F1@5: {valid_score_dict['F1@5'] if self.args.extracting else 0.}\n" +
                                            f"F1@M: {valid_score_dict['F1@M'] if self.args.extracting else 0.}")
                        else:
                            self.tolerance += 1
                            log.console(f"Valid loss does not drop, patience: {self.tolerance}/{self.args.max_tolerance}")
                            # self.scheduler.step(avg_valid_loss)

                        if self.args.extracting:
                            if valid_score_dict['F1@M'] > self.best_valid_f1_at_m:
                                self.tolerance = 0
                                self.best_valid_f1_at_m = valid_score_dict['F1@M']
                                if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                    and self.args.rank % self.args.n_gpu == 0):
                                    log.console(f"Saving best valid F1@M checkpoint to {self.args.train_output_dir}...")
                                    torch.save({'epoch': epoch,
                                                'steps': self.global_step,
                                                'loss': avg_valid_loss,
                                                'theta': valid_score_dict['theta'],
                                                'topk': valid_score_dict['topk'],
                                                'f1_at_k': valid_score_dict['F1@5'],
                                                'f1_at_m': valid_score_dict['F1@M'],
                                                'model_state_dict': self.model.state_dict(),
                                                'optimizer_state_dict': self.optimizer.state_dict(),
                                                'scheduler_state_dict': self.scheduler.state_dict()
                                                }, osp.join(self.args.train_output_dir, "best_valid_f1_at_m.pt"))
                                    
                                    with open(osp.join(args.train_output_dir, "best_f1_results.txt"), "w") as f:
                                        f.write(f"Epoch: {epoch}\n" +
                                                f"Total Steps: {self.global_step}\n" +
                                                f"Valid Loss: {avg_valid_loss}\n" +
                                                f"Theta: {valid_score_dict['theta']}\n" +
                                                f"Top K: {valid_score_dict['topk']}\n" +
                                                f"F1@5: {valid_score_dict['F1@5']}\n" +
                                                f"F1@M: {valid_score_dict['F1@M']}")
                            # else:
                            #     self.tolerance += 1
                            #     log.console(f"F1@M does not improve, patience: {self.tolerance}/{self.args.max_tolerance}")

                        # Switch back to train mode!
                        self.model.train()

                if self.tolerance == self.args.max_tolerance:
                    log.console(f"Has not increased for {self.tolerance} checkpoints, early stop training.")
                    break

                pbar.update(1)
                del outputs, loss

        return train_loss / (total - no_ext_count)


    @torch.no_grad()
    def __epoch_valid(self):
        self.model.eval()
        valid_loss, valid_ext_loss, valid_gen_loss = 0., 0., 0.
        valid_logits, valid_labels = [], []
        score_dict = {}
        total = len(self.valid_loader)
        no_ext_count = 0

        with tqdm(desc="Validating", total=total, ncols=100, disable=self.args.hide_tqdm) as pbar:
            for step, inputs in enumerate(self.valid_loader, 1):
                for k, v in inputs.items():
                    inputs[k] = v.cuda(self.args.gpu, non_blocking=True)

                ### Forward pass ###
                if self.args.extracting:
                    outputs = self.model(**inputs)
                    ext_logits, ext_labels, ext_loss, gen_loss, loss = outputs

                    if ext_logits is None:
                        no_ext_count += 1
                        continue

                    valid_logits.append(ext_logits)
                    valid_labels.append(ext_labels)
                    valid_ext_loss += ext_loss.item()
                    valid_gen_loss += gen_loss.item()
                else:
                    outputs = self.model(**inputs)
                    loss = outputs.loss

                valid_loss += loss.item()

                pbar.update(1)
                del outputs, loss
                
        _total = total - no_ext_count
        
        log.console(f"steps: {self.global_step}, " +
                    f"valid loss: {(valid_loss / _total):.4f}, " +
                    f"ext loss: {(valid_ext_loss / _total):.4f}, " +
                    f"gen loss: {(valid_gen_loss / _total):.4f}, " +
                    f"best valid loss: {self.best_valid_loss:.4f}")
        if self.args.wandb_on:
            wandb.log({"Valid Loss": valid_loss / _total,
                       "Valid Extraction Loss": valid_ext_loss / _total,
                       "Valid Generation Loss": valid_gen_loss / _total}, step=self.global_step)
        
        if self.args.extracting:
            valid_logits = pad_fn(valid_logits, padding=float("-inf"))
            valid_labels = pad_fn(valid_labels, padding=-100)
            score_dict = self.calculate_scores(valid_logits, valid_labels)

            log.console(f"P@5 ({score_dict['num_matches@5']}/{score_dict['num_preds@5']}): {score_dict['P@5']:.5f}, " +
                        f"R@5 ({score_dict['num_matches@5']}/{score_dict['num_trgs@5']}): {score_dict['R@5']:.5f}, " +
                        f"F1@5: {score_dict['F1@5']:.5f}")
            log.console(f"P@M ({score_dict['num_matches@M']}/{score_dict['num_preds@M']}): {score_dict['P@M']:.5f}, " +
                        f"R@M ({score_dict['num_matches@M']}/{score_dict['num_trgs@M']}): {score_dict['R@M']:.5f}, " +
                        f"F1@M: {score_dict['F1@M']:.5f}")

            if self.args.wandb_on:
                wandb.log({"P@5": score_dict['P@5'], "R@5": score_dict['R@5'], "F1@5": score_dict['F1@5'],
                           "P@M": score_dict['P@M'], "R@M": score_dict['R@M'], "F1@M": score_dict['F1@M']}, step=self.global_step)

        return valid_loss / _total, score_dict


    def calculate_scores(self, logits, labels):
        if logits is None or labels is None:
            return None

        k = 5
        score_dict = {}

        sorted_logits, sorted_idxes = logits.sort(descending=True)
        preds = torch.zeros_like(sorted_logits).to(sorted_logits)
        preds[labels != -100] = 1.              # assume we predict all values except padding
        num_preds = (preds == 1).cumsum(1)
        num_trgs = (labels == 1).sum(1)
        num_preds_at_k = torch.minimum(torch.tensor(k, dtype=torch.float).to(num_preds), num_preds) if self.args.meng_rui_precision else k
        num_preds_at_m = num_preds
        num_trgs_at_k = torch.minimum(torch.tensor(k, dtype=torch.float).to(num_trgs), num_trgs) if self.args.choi_recall else num_trgs
        num_trgs_at_m = num_trgs.unsqueeze(1).expand((-1, preds.shape[1]))

        sorted_labels = torch.gather(labels, dim=1, index=sorted_idxes)
        num_matches_at_k = sorted_labels[:, :k].sum(1)
        num_matches_at_m = ((preds == 1) * (sorted_labels == 1)).cumsum(1)
    
        # Calculate @k metrics
        precision_at_k = num_matches_at_k / (num_preds_at_k + 1e-20)
        recall_at_k = num_matches_at_k / (num_trgs_at_k + 1e-20)
        f1_at_k = (2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k + 1e-20))

        # Calculate @M metrics
        precision_at_m = num_matches_at_m / (num_preds_at_m + 1e-20)
        recall_at_m = num_matches_at_m / (num_trgs_at_m + 1e-20)
        f1_at_m = (2 * precision_at_m * recall_at_m / (precision_at_m + recall_at_m + 1e-20))

        # find global threshold that maximizes F1
        f1_at_m[labels == -100] = 0.        # zero out padding values
        best_f1_at_m, best_f1_pos = f1_at_m.max(1)
        precision_at_m = precision_at_m[torch.arange(preds.shape[0]), best_f1_pos]
        recall_at_m = recall_at_m[torch.arange(preds.shape[0]), best_f1_pos]
        num_matches_at_m = num_matches_at_m[torch.arange(preds.shape[0]), best_f1_pos]
        num_preds = num_preds[torch.arange(preds.shape[0]), best_f1_pos]
        thetas = sorted_logits[torch.arange(preds.shape[0]), best_f1_pos]
        topks = best_f1_pos + 1
        f1_at_m = best_f1_at_m

        score_dict[f"P@{k}"] = precision_at_k.mean().item()
        score_dict[f"R@{k}"] = recall_at_k.mean().item()
        score_dict[f"F1@{k}"] = f1_at_k.mean().item()
        score_dict[f"num_matches@{k}"] = num_matches_at_k.sum().long().item()
        score_dict[f"num_preds@{k}"] = k * preds.shape[0]
        score_dict[f"num_trgs@{k}"] = num_trgs_at_k.sum().long().item()
        score_dict[f"P@M"] = precision_at_m.mean().item()
        score_dict[f"R@M"] = recall_at_m.mean().item()
        score_dict[f"F1@M"] = f1_at_m.mean().item()
        score_dict[f"num_matches@M"] = num_matches_at_m.sum().long().item()
        score_dict[f"num_preds@M"] = num_preds.sum().long().item()
        score_dict[f"num_trgs@M"] = num_trgs.sum().long().item()
        score_dict[f"theta"] = thetas.mean().item()
        score_dict[f"topk"] = topks.float().mean().item()

        return score_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KP Stage 1 Training")
    config.model_args(parser)
    config.data_args(parser)
    config.train_args(parser)
    args = parser.parse_args()
    main(args)
