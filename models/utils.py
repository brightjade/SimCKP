import logging

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

from .modeling_bart import BartForConditionalGeneration
from .extractor import KPExtractor
from .ranker import KPRanker
from .dist_utils import DDP


def load_config(args):
    """
    Load model configurations.
    """
    config = AutoConfig.from_pretrained(
                args.config_name if args.config_name else args.model_name_or_path,
                n_positions=args.max_seq_length,
                cache_dir=args.cache_dir if args.cache_dir else None)
    return config


def load_tokenizer(args):
    """
    Load tokenizer.
    """
    num_added_tokens = 0
    special_tokens_dict = {'additional_special_tokens': []}
    tokenizer = AutoTokenizer.from_pretrained(
                    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                    cache_dir=args.cache_dir if args.cache_dir else None)

    if not args.stage_two and args.paradigm == "one2seq":
        # custom sep token is needed to create trg seq labels
        special_tokens_dict['additional_special_tokens'].extend(['<digit>', '<sep>'])
    else:
        special_tokens_dict['additional_special_tokens'].append("<digit>")

    num_added_tokens += tokenizer.add_special_tokens(special_tokens_dict)
    logging.info(f"# added tokens: {num_added_tokens}")
    return tokenizer


def load_model(args, config, tokenizer):
    """
    Load model.
    """
    if args.stage_two:
        model = AutoModel.from_pretrained(
                    args.model_name_or_path,
                    config=config,
                    cache_dir=args.cache_dir if args.cache_dir else None)
        if args.share_params:
            model2 = model
        else:
            model2 = AutoModel.from_pretrained(
                        args.model_name_or_path,
                        config=config,
                        cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = BartForConditionalGeneration.from_pretrained(
                    args.model_name_or_path,
                    config=config,
                    cache_dir=args.cache_dir if args.cache_dir else None)

    if tokenizer is not None:
        token_embedding_size = len(tokenizer)

    # Resize token embeddings for added special tokens
    # Pretrained weights stay, while new ones are randomly initialized.
    # https://github.com/huggingface/tokenizers/issues/247
    model.resize_token_embeddings(token_embedding_size)
    if args.stage_two and not args.share_params:
        model2.resize_token_embeddings(token_embedding_size)

    # Load the proposed model.
    if args.stage_two:
        model = KPRanker(args, config, model, model2)
    else:
        if args.extracting:
            model = KPExtractor(args, config, model)

    # Load the model to GPU.
    if not torch.cuda.is_available():
        logging.warn("Using CPU, this will be slow")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.train_batch_size = int(args.train_batch_size / args.n_gpu)
            args.eval_batch_size = int(args.eval_batch_size / args.n_gpu)
            args.num_workers = int((args.num_workers + args.n_gpu - 1) / args.n_gpu)
            model = DDP(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    elif args.n_gpu > 1:
        # Multi GPU
        model = torch.nn.DataParallel(model).cuda()
    else:
        # Single GPU
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            model.cuda()

    return model
