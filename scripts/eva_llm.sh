set -ex

data_name=$1
model_type=$2
GPU_ID=$3
checkpoint_file=$4

# llama3.1-8B
# /extrahome0/download/Meta-Llama-3.1-8B-Base
# /userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/
# llama2-7B
# /userhome/Research_HUB/RLHF/trlx/examples/cl_rlhf/download/Llama-2-7b-hf

if [ "$model_type" = "llama3-base" ]; then
    model_name_or_path=/extrahome0/download/Meta-Llama-3.1-8B-Base
elif [ "$model_type" = "llama3-it" ]; then
    model_name_or_path=/userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/
elif [ "$model_type" = "qwen-math" ]; then
    model_name_or_path=/extrahome0/user/model/Qwen2.5-Math-1.5B/
elif [ "$model_type" = "qwen-math-it" ]; then
    model_name_or_path=/extrahome0/user/model/Qwen2.5-Math-Instruct-1.5B/
elif [ "$model_type" = "qwen-it" ]; then
    model_name_or_path=/extrahome0/HF_models/Qwen2.5-7B-Instruct/
else
    echo "Unknown model_type"
    model_name_or_path=""
fi

VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=${GPU_ID} python eva_llm.py \
    --model_name_or_path ${model_name_or_path} \
    --checkpoint_file ${checkpoint_file} \
    --model_max_length 2048 \
    --seed 42 \
    --model_type ${model_type} \
    --data_name ${data_name} \
    --gpu_id 0 \
    --pattern_name "checkpoint*" \
    --sample_data 10 \
    --half "all" \
    --temperature 0.6 \
    --reuse \
    --do_test \
    --generate_only True
    #--use_wandb True
    #--do_zst
    #--eval_data_path /extrahome0/user/data/gsm8k_test.jsonl \