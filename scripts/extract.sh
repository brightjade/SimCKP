#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name_or_path="facebook/bart-base"
cache_dir=".cache/"

paradigm="one2seq"          # one2one, one2seq
max_ngram_length=6
dist_fn="cosine"
doc_pooler="cls"
kp_pooler="sum"
train_batch_size=8
test_batch_size=8
gradient_accumulation_steps=1
temperature=0.1
warmup_ratio=0.0
lr=5e-5
epochs=10
seed=42
extracting=1                # 0(false) or 1(true)
log_to_file=1               # "
overwrite_filtered_data=1   # "
gamma=0.3

decoding_method="beam"
beam_size=50
num_return_sequences=50
num_beam_groups=50
diversity_penalty=1.0
top_k=50
top_p=0.9

batch_size=$((train_batch_size * gradient_accumulation_steps))
IFS='/' read -ra x <<< $model_name_or_path && model_type=${x[1]}    # model_name_or_path.split("/")[1]
if [ $extracting = 1 ] ; then ngram="_N${max_ngram_length}" ; else ngram="" ; fi
if [ $extracting = 1 ] ; then temp="_T${temperature}" ; else temp="" ; fi

exp="BS${batch_size}_LR${lr}_W${warmup_ratio}${ngram}${temp}_G${gamma}_S${seed}"
train_output_dir=".checkpoints/${model_type}/${dist_fn}/${kp_pooler}/${doc_pooler}/${paradigm}/${exp}"

for data in "inspec" "krapivin" "nus" "semeval" "kp20k"
  do
    data_dir="data/${data}"
    test_output_dir="${train_output_dir}/${data}"

    # HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python extract.py \
        --model_name_or_path ${model_name_or_path} \
        --model_type ${model_type} \
        --data_dir ${data_dir} \
        --cache_dir ${cache_dir} \
        --train_output_dir ${train_output_dir} \
        --test_output_dir ${test_output_dir} \
        --paradigm ${paradigm} \
        --max_ngram_length ${max_ngram_length} \
        --dist_fn ${dist_fn} \
        --doc_pooler ${doc_pooler} \
        --kp_pooler ${kp_pooler} \
        --temperature ${temperature} \
        --test_batch_size ${test_batch_size} \
        --extracting ${extracting} \
        --log_to_file ${log_to_file} \
        --overwrite_filtered_data ${overwrite_filtered_data} \
        --gamma ${gamma} \
        --decoding_method ${decoding_method} \
        --beam_size ${beam_size} \
        --num_return_sequences ${num_return_sequences} \
        --num_beam_groups ${num_beam_groups} \
        --diversity_penalty ${diversity_penalty} \
        --top_k ${top_k} \
        --top_p ${top_p} \
        --seed ${seed} \
        --do_predict \
        --do_extract \
        --do_generate_abs_candidates

done
