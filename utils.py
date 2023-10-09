import random
import numpy as np

import torch
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_scaler(args):
    return torch.cuda.amp.GradScaler(enabled=args.use_amp)


def init_optimizer(args, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    return optimizer


def init_scheduler(args, optimizer):
    if args.warmup_ratio > 0:
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.total_steps)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=args.warmup_steps)
    return scheduler
