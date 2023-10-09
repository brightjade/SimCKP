#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name_or_path="facebook/bart-base"
data_dir="data/kp20k"
cache_dir=".cache/"

paradigm="one2seq"          # one2one, one2seq
max_ngram_length=6
dist_fn="cosine"
doc_pooler="cls"
kp_pooler="sum"
train_batch_size=8
eval_batch_size=40
gradient_accumulation_steps=1
temperature=0.1
warmup_ratio=0.0
lr=5e-5
epochs=10
seed=42
logging_steps=1500
evaluation_steps=5000
extracting=1                # 0(false) or 1(true)
wandb_on=0                  # "
log_to_file=1               # "
overwrite_filtered_data=1   # "
gamma=0.3

batch_size=$((train_batch_size * gradient_accumulation_steps))
IFS='/' read -ra x <<< $model_name_or_path && model_type=${x[1]}    # model_name_or_path.split("/")[1]
if [ $extracting = 1 ] ; then ngram="_N${max_ngram_length}" ; else ngram="" ; fi
if [ $extracting = 1 ] ; then temp="_T${temperature}" ; else temp="" ; fi

exp="BS${batch_size}_LR${lr}_W${warmup_ratio}${ngram}${temp}_G${gamma}_S${seed}"
train_output_dir=".checkpoints/${model_type}/${dist_fn}/${kp_pooler}/${doc_pooler}/${paradigm}/${exp}"

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python train_extractor.py \
    --model_name_or_path ${model_name_or_path} \
    --model_type ${model_type} \
    --data_dir ${data_dir} \
    --cache_dir ${cache_dir} \
    --train_output_dir ${train_output_dir} \
    --paradigm ${paradigm} \
    --max_ngram_length ${max_ngram_length} \
    --dist_fn ${dist_fn} \
    --doc_pooler ${doc_pooler} \
    --kp_pooler ${kp_pooler} \
    --temperature ${temperature} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --warmup_ratio ${warmup_ratio} \
    --epochs ${epochs} \
    --lr ${lr} \
    --seed ${seed} \
    --logging_steps ${logging_steps} \
    --evaluation_steps ${evaluation_steps} \
    --extracting ${extracting} \
    --wandb_on ${wandb_on} \
    --log_to_file ${log_to_file} \
    --overwrite_filtered_data ${overwrite_filtered_data} \
    --gamma ${gamma}
