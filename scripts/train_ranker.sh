#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_type="bert-base-uncased"
prev_model_type="bart-base"
model_name_or_path="${model_type}"
data_dir="data/kp20k"
cache_dir=".cache/"

paradigm="one2seq"      # one2one, one2seq
max_ngram_length=6
max_seq_length=512
dist_fn="cosine"
doc_pooler="cls"
kp_pooler="cls"
train_batch_size=8
eval_batch_size=40
gradient_accumulation_steps=1
temperature=0.1
warmup_ratio=0.1
lr=3e-5
seed=42
beam_size=50
decoding_method="beam"
logging_steps=1500
evaluation_steps=5000
extracting=1            # 0(false), 1(true)
stage_two=1             # "
share_params=1          # "
wandb_on=0              # "
log_to_file=1           # "
gamma=0.3

batch_size=$((train_batch_size * gradient_accumulation_steps))
if [ $extracting = 1 ] ; then ngram="_N${max_ngram_length}" ; else ngram="" ; fi
if [ $extracting = 1 ] ; then temp="_T${temperature}" ; else temp="" ; fi
if [ $share_params = 1 ] ; then shr="_shr" ; else shr="_sep" ; fi

exp="BS${batch_size}_LR${lr}_W${warmup_ratio}${ngram}${temp}_G${gamma}_S${seed}${shr}"
train_output_dir=".checkpoints/${model_type}/${prev_model_type}/${dist_fn}/${kp_pooler}/${doc_pooler}/${paradigm}/${exp}"

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python train_ranker.py \
    --prev_model_type ${prev_model_type} \
    --model_type ${model_type} \
    --model_name_or_path ${model_name_or_path} \
    --data_dir ${data_dir} \
    --cache_dir ${cache_dir} \
    --train_output_dir ${train_output_dir} \
    --paradigm ${paradigm} \
    --max_ngram_length ${max_ngram_length} \
    --max_seq_length ${max_seq_length} \
    --dist_fn ${dist_fn} \
    --doc_pooler ${doc_pooler} \
    --kp_pooler ${kp_pooler} \
    --temperature ${temperature} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --warmup_ratio ${warmup_ratio} \
    --lr ${lr} \
    --seed ${seed} \
    --beam_size ${beam_size} \
    --decoding_method ${decoding_method} \
    --logging_steps ${logging_steps} \
    --evaluation_steps ${evaluation_steps} \
    --extracting ${extracting} \
    --wandb_on ${wandb_on} \
    --log_to_file ${log_to_file} \
    --share_params ${share_params} \
    --stage_two ${stage_two} \
    --gamma ${gamma}
