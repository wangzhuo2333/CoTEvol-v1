#!/bin/bash
# shellcheck disable=SC2068
# read parameters
idx=0
for i in $@
do
  args[${idx}]=$i
  let "idx=${idx}+1"
done

device="0,1,2,3"
model_type="qwen-math"
dataset_path=${args[2]}
dataset_name=${args[3]}
model_name_or_path=${args[4]}

# bash ./scripts/run_cdpo_sweep.sh 0,1,2,3 qwen-math xx xx
python run_cdpo_sweep.py \
--checkpoint_file ${model_name_or_path} \
--dataset ${dataset_path} \
--dataset_name ${dataset_name} \
--model_type ${model_type} \
--device ${device}