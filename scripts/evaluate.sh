#!/bin/bash

model_type1="bart-base"
model_type2="bert-base-uncased"
train_data_dir="data/kp20k"

paradigm="one2seq"
max_ngram_length=6
dist_fn1="cosine"
dist_fn2="cosine"
doc_pooler1="cls"
doc_pooler2="cls"
kp_pooler1="sum"
kp_pooler2="cls"
temperature1=0.1
temperature2=0.1
train_batch_size1=8
train_batch_size2=8
gradient_accumulation_steps1=1
gradient_accumulation_steps2=1
warmup_ratio1=0.0
warmup_ratio2=0.1
lr1=5e-5
lr2=3e-5
seed1=42
seed2=42
decoding_method="beam"
beam_size=50
extracting=1            # 0(false), 1(true)
share_params=1          # "
gamma=0.3

batch_size1=$((train_batch_size1 * gradient_accumulation_steps1))
batch_size2=$((train_batch_size2 * gradient_accumulation_steps2))
if [ $extracting = 1 ] ; then ngram="_N${max_ngram_length}" ; else ngram="" ; fi
if [ $extracting = 1 ] ; then temp1="_T${temperature1}" ; else temp1="" ; fi
if [ $extracting = 1 ] ; then temp2="_T${temperature2}" ; else temp2="" ; fi
if [ $share_params = 1 ] ; then shr="_shr" ; else shr="_sep" ; fi

exp1="BS${batch_size1}_LR${lr1}_W${warmup_ratio1}${ngram}${temp1}_G${gamma}_S${seed1}"
exp2="BS${batch_size2}_LR${lr2}_W${warmup_ratio2}${ngram}${temp2}_G${gamma}_S${seed2}${shr}"

for data in "inspec" "krapivin" "nus" "semeval" "kp20k"
  do
    data_dir="data/${data}"
    test_output_dir1=".checkpoints/${model_type1}/${dist_fn1}/${kp_pooler1}/${doc_pooler1}/${paradigm}/${exp1}/${data}"
    test_output_dir2=".checkpoints/${model_type2}/${model_type1}/${dist_fn2}/${kp_pooler2}/${doc_pooler2}/${paradigm}/${exp2}/${data}"

    python evaluate.py \
        --model_type ${model_type1} \
        --data_dir ${data_dir} \
        --test_output_dir ${test_output_dir1} \
        --test_output_dir2 ${test_output_dir2} \
        --paradigm ${paradigm} \
        --beam_size ${beam_size} \
        --decoding_method ${decoding_method}

done
