"""Grid Search for Tuning"""

import os
import sys
import itertools as it
from loguru import logger
from multiprocessing import Pool
import torch

torch.cuda.empty_cache()

def run_process(proc):
    os.system(proc)


train_batch_size=32
hyperparameter_grid = {
    "sft": {
        # "learning_rate": [7e-7, 5e-6, 1e-5],
        # "max_epochs": [1, 5],
        "learning_rate": [7e-7, 5e-6, 1e-5],
        # "learning_rate": [2e-7, 5e-7],
        "max_epochs": [1, 3],
    },
    "dpo": {
        "learning_rate": [7e-7, 1e-6, 5e-6, 1e-5],
        "beta": [0.2, 0.5, 1.0],
        "max_epochs": [1, 5],
    },
    "lora": {
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "lora_rank": [16, 32],  # lora_alpha  lora_r
    },
}

device = sys.argv[1]
model_type = sys.argv[2]
dataset_name = sys.argv[3] # prm_sft/prm_dpo/math_sft
tuning_type = sys.argv[4] # sft

if model_type == "llama3-base":
    model_name_or_path = "/extrahome0/download/Meta-Llama-3.1-8B-Base"
    micro_train_batch_size = 1
    zero_stage=2

elif model_type == "llama3-it":
    model_name_or_path = "/userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/"
    micro_train_batch_size = 1
    zero_stage=2

elif model_type == "qwen-math":
    model_name_or_path="/extrahome0/user/model/Qwen2.5-Math-1.5B/"
    micro_train_batch_size = 8
    zero_stage=1

elif model_type == "qwen-it":
    # model_name_or_path="/extrahome0/HF_models/Qwen2.5-7B-Instruct/"
    model_name_or_path="/inspire/hdd/global_user/USER_ID/user/models/Qwen2.5-7B-Instruct/"
    zero_stage=2
    
elif model_type == "qwen-math-it":
    model_name_or_path="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/Qwen2.5-Math-7B-Instruct/"
    micro_train_batch_size = 1
    zero_stage=1
elif model_type == "qwen-math-1.5b-it":
    model_name_or_path="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/Qwen2.5-Math-1.5B-Instruct/"
    micro_train_batch_size = 2
    zero_stage=2 
elif model_type == "deepseek-r1-distill-qwen1.5b":
    model_name_or_path="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepseek-r1-distill-qwen1.5b/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/Deepseek-R1-Distill-Qwen1.5B/"
    micro_train_batch_size = 2
    zero_stage=2 
elif model_type == "mistral-7b-it-v0.3":
    model_name_or_path="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/Mistral-7B-Instruct-v0.3/"
    micro_train_batch_size = 2
    zero_stage=2 
elif model_type == "mistral-8b-it":
    model_name_or_path="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/Ministral-8B-Instruct-2410/"
    micro_train_batch_size = 2
    zero_stage=2 
else:
    raise ValueError("Unknown model_type")

if dataset_name == "prm_r1":
    dataset="/userhome/Research_HUB/Reasoning/data_dir/prm800k_cut_train_valid_test.pt"
    max_len=2048
elif dataset_name == "prm_sft":
    dataset="/extrahome0/user/data/prm_sft/prm800k_train_eval.pt"
    max_len=2048
elif dataset_name == "gsm_sft":
    dataset="/extrahome0/user/data/gsm_sft.pt"
    max_len=512
elif dataset_name == "math_sft":
    dataset="/userhome/Research_HUB/Reasoning/prm800k/prm800k/math_train_valid_test/MATH_train_valid_test500.pt"
    max_len=1024

# 合并了limo和s1k数据的结果
elif dataset_name == "reason_base_long":
    dataset="/extrahome0/user/data/evol_baseline/s1k_limo_long.pt"
    max_len=16384
    zero_stage=2
elif dataset_name == "reason_base_short":
    dataset="/extrahome0/user/data/evol_baseline/s1k_limo_short.pt"
    max_len=3000
    zero_stage=2
elif dataset_name == "reason_base_evol":
    # dataset="/extrahome0/user/data/evol_baseline/s1k_limo_evol.pt"
    dataset="/extrahome0/user/data/evol_baseline/s1k_limo_evol.pt"
    max_len=3000
    zero_stage=2
elif dataset_name == "reason_base_bstn":
    dataset="/extrahome0/user/data/evol_baseline/s1k_limo_bstn.pt"
    max_len=3000
    zero_stage=2

# 单个数据数据的结果
elif dataset_name == "s1k_short":
    dataset="./evol/data/s1k_train_short.pt"
    max_len=2048
elif dataset_name == "s1k_long":
    dataset="./evol/data/s1k_train_long.pt"
    micro_train_batch_size = 1
    max_len=2048
elif dataset_name == "s1k_evol":
    micro_train_batch_size = 1
    # dataset="./evol/data/s1k_evol_merge_v1.pt"
    # dataset="./evol/data/s1k_evol_merge_1.5bit.pt"
    # dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main_new/ifdr-main/data/s1k_evol_merge_deepseek_all.pt"
    # dataset="./evol/data/s1k_evol_merge_mistral.pt"
    # dataset="./evol/data/s1k_evol_merge_mistral_8b.pt"
    dataset="./evol/data/s1k_7b_filter.pt"
    max_len=2048
elif dataset_name == "s1k_evol_iternum_1":
    micro_train_batch_size = 1
    dataset="./evol/data/s1k_iternum_1.pt"
    max_len=2048
elif dataset_name == "s1k_evol_iternum_2":
    micro_train_batch_size = 1
    dataset="./evol/data/s1k_iternum_2.pt"
    max_len=2048
    
    
elif dataset_name == "s1k_v1.1":
    dataset="./evol/data/long_cot_distill/s1k_v1.1_long.pt"
    micro_train_batch_size = 1
    max_len=2048
elif dataset_name == "s1k_ori":
    dataset="./evol/data/ori_cot/s1k_ori.pt"
    micro_train_batch_size = 1
    max_len=2048    
elif dataset_name == "r1-distil-qwen-32b-cot":
    dataset="./evol/data/long_cot_distill/r1-distl-qwen-32b-cot.pt"
    micro_train_batch_size = 1
    max_len=2048
elif dataset_name == "s1k-qwen-math-7b-ist-distil":
    dataset="./evol/data/long_cot_distill/qwen-math-7b-it.pt"
    micro_train_batch_size = 1
    max_len=2048 
elif dataset_name == "limo_ori":
    dataset="./evol/data/ori_cot/limo_ori.pt"
    micro_train_batch_size = 1
    max_len=2048
elif dataset_name == "limo-qwen-7b-it":
    dataset="./evol/data/long_cot_distill/limo-qwen-7b-it.pt"
    micro_train_batch_size = 1
    max_len=2048
elif dataset_name == "limo-qwen-math-7b-it":
    dataset="./evol/data/long_cot_distill/limo-qwen-math-7b-it.pt"
    micro_train_batch_size = 1
    max_len=2048         
elif dataset_name == "limo-qwen-math-72b-it":
    dataset="./evol/data/long_cot_distill/limo-qwen-math-72b-it.pt"
    micro_train_batch_size = 1
    max_len=2048 
elif dataset_name == "limo-r1-distl-qwen-32b":
    dataset="./evol/data/long_cot_distill/limo-r1-distl-qwen-32b.pt"
    micro_train_batch_size = 1
    max_len=2048
elif dataset_name == "limo-gemini-2.5-flash":
    dataset="./evol/data/limo_gemini_distil.pt"
    micro_train_batch_size = 1
    max_len=2048
elif dataset_name == 'limo_evol_2_iter':
    dataset='./evol/data/limo_2_iter.pt'
    micro_train_batch_size = 1
    max_len=2048 
elif dataset_name == 'limo_evol_1_iter':
    dataset='/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main-v2/ifdr-main-qwen1.5/evol/mix_limo_exp/evol_limo_m_1_iter.pt'  
    micro_train_batch_size = 1
    max_len=2048
    
elif dataset_name == "s1k_evol_cross":
    micro_train_batch_size = 1
    dataset="./evol/data/s1k_cross_filter.pt"
    max_len=2048
elif dataset_name == "s1k_evol_mutation":
    micro_train_batch_size = 1
    dataset="./evol/data/s1k_mutation_filter.pt"
    max_len=2048
elif dataset_name == "limo_long":
    dataset="/extrahome0/user/data/evol/limo_v1_long_cot.pt"
    max_len=16384
    zero_stage=2
elif dataset_name == "limo_evol":
    dataset="./evol/data/limo_evol.pt"
    micro_train_batch_size = 1
    max_len=3000
    zero_stage=2
elif dataset_name == "limo_short":
    dataset="/extrahome0/user/data/evol/limo_v1_short_cot.pt"
    max_len=3000
    zero_stage=2

# deepmath
elif dataset_name == "deepmath_10k_evol":
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_evol_10k.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
elif dataset_name == "deepmath_1k_evol":
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_evol_1k.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
elif dataset_name == "deepmath_3k_evol":
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_evol_3k.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
elif dataset_name == "deepmath_5k_evol":
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_evol_5k.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
elif dataset_name == "deepmath_7k_evol":
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_evol_7k.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
    
elif dataset_name == "deepmath_1k_long":
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_long_1k.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
elif dataset_name == "deepmath_3k_long":
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_long_3k.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
elif dataset_name == "deepmath_5k_long":
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_long_5k.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
elif dataset_name == "deepmath_7k_long":
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_long_7k.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
elif dataset_name == "deepmath_10k_long":
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_long_10k.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
    
elif dataset_name == "deepmath_1k_short":
    # dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_1k_short.pt"
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_8k/deepmath_1k_short.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
elif dataset_name == "deepmath_3k_short":
    # dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_3k_short.pt"
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_8k/deepmath_3k_short.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
elif dataset_name == "deepmath_5k_short":
    # dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_5k_short.pt"
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_8k/deepmath_5k_short.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
elif dataset_name == "deepmath_7k_short":
    # dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_7k_short.pt"
    dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_8k/deepmath_7k_short.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2
elif dataset_name == "deepmath_10k_short":
    # dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_10k_short.pt"
    # dataset="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deepmath_8k/deepmath_10k_short.pt"
    # dataset='/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deep_math_32b_distil.pt'
    dataset='/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/deepmath/deep_math_72b_distil_new.pt'
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 

elif dataset_name== "s1k_new_mutation":
    # dataset="/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/111_limo_new_mutation/evol_limo/v1.1/evol_limo_m.pt" 
    # dataset='/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/111_s1k_new_mutation_no_gt/evol_s1k/v1.1/evol_s1k_m.pt'
    # dataset="/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/111_deepmath_new_mutation/evol_deepmath/v1.1/evol_deepmath_m.pt"
    # dataset="/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/111_slk_new_mutation_0/evol_s1k/v1.1/evol_s1k_m.pt"
    # dataset="/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/111_limo_new_mutation_no_gt_v2/evol_limo/v1.1/evol_limo_m.pt"
    # dataset="/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/data/long_cot_distill/s1k_v1.1_long.pt"
    # dataset='/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/111_limo_new_mutation/evol_limo/v1.1/2026_02_17_evol_limo_m.pt'
    
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2 
    
# math7500
elif dataset_name == "math7500_10_short":
    dataset="./data_new/data_new/math7500_10/math7_5_10_short.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2
elif dataset_name == "math7500_10_long":
    dataset="./data_new/data_new/math7500_10/math7_5_10_long.pt"
    max_len = 8000
    micro_train_batch_size = 1
    zero_stage = 2
elif dataset_name == "math7500_10_evol":
    dataset="./data_new/data_new/math7500_10/math7_5_10_evol_filter.pt"
    max_len = 8000
    micro_train_batch_size = 1
    zero_stage = 2

elif dataset_name == "math7500_30_short":
    dataset="./data_new/data_new/math7500_30/math7_5_30_short.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2
elif dataset_name == "math7500_30_long":
    dataset="./data_new/data_new/math7500_30/math7_5_30_long.pt"
    max_len = 8000
    micro_train_batch_size = 1
    zero_stage = 2
elif dataset_name == "math7500_30_evol":
    dataset="./data_new/data_new/math7500_30/math7_5_30_evol_filter.pt"
    max_len = 8000
    micro_train_batch_size = 1
    zero_stage = 2
    
elif dataset_name == "math7500_50_short":
    dataset="./data_new/data_new/math7500_50/math7_5_50_short.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2
elif dataset_name == "math7500_50_long":
    dataset="./data_new/data_new/math7500_50/math7_5_50_long.pt"
    max_len = 8000
    micro_train_batch_size = 1
    zero_stage = 2
elif dataset_name == "math7500_50_evol":
    dataset="./data_new/data_new/math7500_50/math7_5_50_evol_filter.pt"
    max_len = 8000
    micro_train_batch_size = 1
    zero_stage = 2

elif dataset_name == "math7500_70_short":
    dataset="./data_new/data_new/math7500_70/math7_5_70_short.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2
elif dataset_name == "math7500_70_long":
    dataset="./data_new/data_new/math7500_70/math7_5_70_long.pt"
    max_len = 8000
    micro_train_batch_size = 1
    zero_stage = 2
elif dataset_name == "math7500_70_evol":
    dataset="./data_new/data_new/math7500_70/math7_5_70_evol_filter.pt"
    max_len = 8000
    micro_train_batch_size = 1
    zero_stage = 2
    
elif dataset_name == "math7500_100_short":
    dataset="./evol/data/math7500_short.pt"
    max_len=1024
    micro_train_batch_size = 2
    zero_stage=2
elif dataset_name == "math7500_100_long":
    dataset="./evol/data/math7500_long.pt"
    max_len = 8000
    micro_train_batch_size = 1
    zero_stage = 2
elif dataset_name == "math7500_100_evol":
    dataset="./evol/data/math7500_5c0_filter.pt"
    max_len = 8000
    micro_train_batch_size = 1
    zero_stage = 2
    
# mix_data实验
elif dataset_name == "s1k_evol_delete":
    micro_train_batch_size = 1
    dataset="./evol/mix_s1k_exp/evol_s1k/v1.1/evol_s1k_m_2_iter_delete.pt"
    max_len=2048
elif dataset_name == "s1k_evol_72b_buchong":
    micro_train_batch_size = 1
    dataset="./evol/mix_s1k_exp/evol_s1k/v1.1/evol_s1k_m_2_iter_traditional_72b.pt"
    max_len=2048
elif dataset_name == "s1k_evol_long_buchong":
    micro_train_batch_size = 1
    dataset="./evol/mix_s1k_exp/evol_s1k/v1.1/evol_s1k_m_2_iter_long.pt"
    max_len=2048
elif dataset_name == "s1k_evol_short_buchong":
    micro_train_batch_size = 1
    dataset="./evol/mix_s1k_exp/evol_s1k/v1.1/evol_s1k_short.pt"
    max_len=2048
elif dataset_name == 'limo_evol_short_buchong':
    micro_train_batch_size = 1
    dataset="./evol/mix_limo_exp/evol_limo_short.pt"
    max_len=2048
elif dataset_name == 'limo_evol_r1_buchong':
    micro_train_batch_size = 1
    dataset="./evol/mix_limo_exp/evol_limo_r1.pt"
    max_len=2048
elif dataset_name == 'limo_bstn':
    micro_train_batch_size = 1
    dataset='/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main-v2/ifdr-main-qwen1.5/data_new/data_new/limo_bstn.pt'
    max_len=2048
# 思路模型的训练
elif dataset_name == "rf_sft15k":
    dataset="/extrahome0/user/data/reasonflux/rf_sft15k.pt"
    max_len=2048
    zero_stage=2
    train_batch_size = 32
else:
    raise ValueError("Unknown dataset_name")


cmds = []
hyper_parameter = hyperparameter_grid[tuning_type]
for parameter in it.product(*list(hyper_parameter.values())):
    specific_parameter_dict = {key: parameter[list(hyper_parameter.keys()).index(key)]
                               for key in list(hyper_parameter.keys())}
    print('$$$$$$$$$',specific_parameter_dict)
    if "lora_rank" in specific_parameter_dict:
        specific_parameter_dict["lora_alpha"] = 2*specific_parameter_dict["lora_rank"]

    cmd = f'deepspeed --include localhost:{device} --master_port 10009 --module run_{tuning_type} '
    # print('#################',cmd)
    options = [
        "--save_path", "./wz/output_limo_2026_02_17_multi_solution/",
        "--save_steps", f"-1",
        "--logging_steps", "-1",
        "--eval_steps", "-1",
        "--train_batch_size", f"{train_batch_size}",
        "--micro_train_batch_size", f"{micro_train_batch_size}",
        "--pretrain", f"{model_name_or_path}",
        "--bf16",
        # "--max_epochs", f"{max_epochs}",
        "--max_len", f"{max_len}",
        "--zero_stage", f"{zero_stage}",
        "--dataset", f"{dataset}",
        "--dataset_name", f"{dataset_name}",
        "--method_name", f"{tuning_type}",
        "--model_type", f"{model_type}",
        "--dataset_sample", "0",
        "--gradient_checkpointing",
        # "--use_wandb", "True",
        # "--wandb_project", "evol_cot",
        "--flash_attn",
        # "--adam_offload"
        # "learning_rate"
    ]
    for key, value in specific_parameter_dict.items():
        options.extend(["--" + key, str(value)])

    one_cmd = cmd + " ".join(options)
    one_cmd += " & wait"
    cmds.append(one_cmd)

run_process("sleep 2s")
logger.warning(f"run {len(cmds)} grid-search tasks for {model_type}_{dataset_name}_{tuning_type}")

# run_process(cmds[0])  # debug
pool = Pool(processes=1)
pool.map(run_process, cmds)

# for cmd in cmds:
#     run_process(cmd)
