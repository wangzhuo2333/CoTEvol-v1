# set -ex

# #data_name=$1
# GPU_ID=$1
# model_type=$2
# checkpoint_file=$3

# # llama3.1-8B
# # /extrahome0/download/Meta-Llama-3.1-8B-Base
# # /userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/
# # llama2-7B
# # /userhome/Research_HUB/RLHF/trlx/examples/cl_rlhf/download/Llama-2-7b-hf

# if [ "$model_type" = "llama3-base" ]; then
#     model_name_or_path=/extrahome0/download/Meta-Llama-3.1-8B-Base
# elif [ "$model_type" = "llama3-it" ]; then
#     model_name_or_path=/userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/
# elif [ "$model_type" = "qwen-math" ]; then
#     model_name_or_path=/extrahome0/user/model/Qwen2.5-Math-1.5B/
# elif [ "$model_type" = "qwen-math-it" ]; then
#     model_name_or_path=/extrahome0/user/model/Qwen2.5-Math-Instruct-1.5B/
# elif [ "$model_type" = "qwen-it" ]; then
# #    model_name_or_path=/extrahome0/HF_models/Qwen2.5-7B-Instruct/
#     model_name_or_path=/media/user/pretrained/Qwen2.5-7B-Instruct/
# elif [ "$model_type" = "qwq-32" ]; then
#     model_name_or_path=/extrahome0/HF_models/QwQ-32B/
# elif [ "$model_type" = "qwen-math-7-it" ]; then
#     model_name_or_path=/extrahome0/HF_models/Qwen2.5-Math-7B-Instruct/
# else
#     echo "Unknown model_type"
#     model_name_or_path=""
# fi

# #PROMPT_TYPE="qwen25-rl-cot"
# #MODEL_NAME_OR_PATH="/extrahome0/user/output/openr1/Qwen2.5-1.5B-ist-GRPO/"
# #OUTPUT_DIR="/extrahome0/user/output/openr1/Qwen2.5-1.5B-ist-GRPO/"

# # English open datasets
# DATA_NAME="gsm8k,math500,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23"
# # DATA_NAME="amc23"

# VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false \
# CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/run_math_eval.py \
#     --model_name_or_path ${model_name_or_path} \
#     --checkpoint_file ${checkpoint_file} \
#     --data_name ${DATA_NAME} \
#     --split "test" \
#     --prompt_type qwen25it-math-cot \
#     --num_test_sample -1 \
#     --max_tokens_per_call 16384 \
#     --seed 42 \
#     --temperature 0.0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --pattern_name "*/checkpoint*" \
#     --use_vllm \
#     --save_outputs \
#     --half "all" \
# #    --do_zst
# #    --half "right" \

set -ex

#data_name=$1
GPU_ID=$1
model_type=$2
checkpoint_file=$3

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
#    model_name_or_path=/extrahome0/HF_models/Qwen2.5-7B-Instruct/
#    model_name_or_path=/media/user/pretrained/Qwen2.5-7B-Instruct/
    model_name_or_path=/inspire/hdd/global_user/USER_ID/user/models/Qwen2.5-7B-Instruct
elif [ "$model_type" = "qwq-32" ]; then
    model_name_or_path=/extrahome0/HF_models/QwQ-32B/
elif [ "$model_type" = "qwen-math-7-it" ]; then
    model_name_or_path=/extrahome0/HF_models/Qwen2.5-Math-7B-Instruct/
    # model_name_or_path=/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/Qwen2.5-7B-Instruct
elif [ "$model_type" = "qwen-math-32-it" ]; then
    model_name_or_path=/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/DeepSeek-R1-Distill-Qwen-32B
elif [ "$model_type" = "qwen3-4b" ]; then
    model_name_or_path=/inspire/hdd/global_user/USER_ID/user/models/Qwen3-4B/
elif [ "$model_type" = "qwen3-8b" ]; then
    model_name_or_path=/inspire/hdd/global_user/USER_ID/user/models/Qwen3-8B/

else
    echo "Unknown model_type"
    model_name_or_path=""
fi

#PROMPT_TYPE="qwen25-rl-cot"
#MODEL_NAME_OR_PATH="/extrahome0/user/output/openr1/Qwen2.5-1.5B-ist-GRPO/"
#OUTPUT_DIR="/extrahome0/user/output/openr1/Qwen2.5-1.5B-ist-GRPO/"

# English open datasets
DATA_NAME="gsm8k,math500,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23"
# DATA_NAME="math500,minerva_math,gaokao2023en,olympiadbench,aime24,amc23"


VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=${GPU_ID:0:1} python eval/run_math_eval.py \
    --model_name_or_path ${model_name_or_path} \
    --checkpoint_file ${checkpoint_file} \
    --data_name ${DATA_NAME} \
    --split "test" \
    --prompt_type qwen25it-math-cot \
    --num_test_sample -1 \
    --max_tokens_per_call 3000 \
    --seed 42 \
    --temperature 0.0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --pattern_name "*/checkpoint*" \
    --use_vllm \
    --save_outputs \
    --half "all" \
    --reuse \
    --data_dir /inspire/hdd/global_user/USER_ID/user/zhang/data/srl_math_benchmark \
    # --do_z
    # --reuse
#    --half "right" \



