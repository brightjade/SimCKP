"""
Configurations.
"""

def model_args(parser):
    # pretrained model 
    parser.add_argument("--prev_model_type", type=str, default="bart-base")
    parser.add_argument("--model_type", type=str, default="bart-base")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/bart-base")
    parser.add_argument("--config_name", type=str, default="")
    parser.add_argument("--tokenizer_name", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=".cache/")
    # extractor
    parser.add_argument("--doc_pooler", type=str, default="cls", help="How to embed document: cls, avg, max, avg_first_last, avg_top2, eos (for decoder only).")
    parser.add_argument("--kp_pooler", type=str, default="sum")
    parser.add_argument("--dist_fn", type=str, default="cosine", help="Type of distance function: inner, cosine, euclidean, manhattan.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for ntxent loss.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Scale for extraction loss.")
    parser.add_argument("--stage_two", type=int, default=0)
    # ranker
    parser.add_argument("--share_params", type=int, default=1)


def data_args(parser):
    parser.add_argument('--paradigm', type=str, default="one2seq")
    parser.add_argument("--data_dir", type=str, default="data/kp20k")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--train_output_dir", type=str, default=".checkpoints/")
    parser.add_argument("--test_output_dir", type=str, default=".checkpoints/")
    parser.add_argument("--test_output_dir2", type=str, default=".checkpoints/")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--max_ngram_length", type=int, default=6)
    parser.add_argument("--extracting", type=int, default=1)
    parser.add_argument("--overwrite_filtered_data", type=int, default=1)
    parser.add_argument("--log_to_file", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    # inference + evaluation
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--meng_rui_precision", type=bool, default=False)
    parser.add_argument("--choi_recall", type=bool, default=False)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--decoding_method", type=str, default="beam")


def train_args(parser):
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=40)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1500)
    parser.add_argument("--evaluation_steps", type=int, default=5000)
    parser.add_argument("--max_tolerance", type=int, default=10)
    parser.add_argument("--use_amp", type=bool, default=False)
    parser.add_argument("--hide_tqdm", type=bool, default=False)
    parser.add_argument("--wandb_on", type=int, default=0)
    parser.add_argument("--multiprocessing_distributed", action="store_true")
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:29500')
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument("--resume", action="store_true")


def predict_args(parser):
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--num_return_sequences", type=int, default=50)
    parser.add_argument("--num_beam_groups", type=int, default=50)
    parser.add_argument("--diversity_penalty", type=float, default=2.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_extract", action="store_true")
    parser.add_argument("--do_rank", action="store_true")
    parser.add_argument("--do_generate", action="store_true")
    parser.add_argument("--do_generate_abs_candidates", action="store_true")
