import torch
import pickle
from transformers import AutoTokenizer
import sys
sys.path.append("/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2")
from evol.fitness import (
    extract_answer, cosine_scaled_reward,
    accuracy_reward, format_reward, cosine_lang_reward
)

def calculate_rwd(solutions, gt_ans, tokenizer):
    len_rwd = cosine_scaled_reward(gt_ans, solutions, tokenizer)
    acc_rwd = accuracy_reward(gt_ans, solutions)
    format_rwd = format_reward(gt_ans, solutions)
    lang_rwd = cosine_lang_reward(gt_ans, solutions, tokenizer)
    return len_rwd, acc_rwd, format_rwd, lang_rwd

def calculate_fitness(solutions, gt_ans, tokenizer):
    len_rwd, acc_rwd, format_rwd, lang_rwd = calculate_rwd(solutions, gt_ans, tokenizer)

    rwd = []
    details_info = []
    for i in range(len(acc_rwd)):
        rwd.append(float(len_rwd[i] + acc_rwd[i] + format_rwd[i] + lang_rwd[i]))
        details_info.append(
            {"rwd": rwd[i], "len_rwd": len_rwd[i], "acc_rwd": acc_rwd[i],
             "format_rwd": format_rwd[i], "lang_rwd": lang_rwd[i]}
        )
    return rwd, details_info

def read_data(data_path):
    with open(data_path, "rb") as file:
        dps = pickle.load(file)
    for dp in dps:
        all_solutions = dp["all_solutions"]    
        new_all_solutions = []
        for solution in all_solutions:
#             if len(solution.split("#")) > 1:
#                 # 删除多余的阐述语句
#                 solution = "#".join(solution.split("#")[1:-1]).replace("Refined", "")
            new_all_solutions.append(solution)
        dp["all_solutions"] = new_all_solutions
    return dps


def get_wrong_ids(dps, tokenizer):
    wrong_ids = []
    for dp in dps:
        all_solutions = dp["all_solutions"]
        # init_solutions = all_solutions[0:-6]
        # evol_solutions = all_solutions[-6:]
        init_solutions = all_solutions[0:4]
        evol_solutions = all_solutions[4:]
        if 'attempt' in dp:
            gt_answer = extract_answer(dp['attempt'])
        elif "answer" in dp:
            gt_answer = dp["answer"]

        init_rwd, init_details = calculate_fitness(init_solutions, gt_answer, tokenizer)
        init_right = False
        init_r_cnt = 0
        for init_info in init_details:
            if init_info["acc_rwd"] >= 0.5:
                init_r_cnt += 1
                init_right = True

        evol_rwd, evol_details = calculate_fitness(evol_solutions, gt_answer, tokenizer)
        evol_right = False
        evol_r_cnt = 0
        for evol_info in evol_details:
            if evol_info["acc_rwd"] >= 0.5:
                evol_r_cnt += 1
                evol_right = True

        if not evol_right and not init_right:
            wrong_ids.append(dp["idx"])
    return wrong_ids

def convert_id_dps(dps):
    new_dps = {}
    for dp in dps:
        idx = dp["idx"]
        new_dps[idx] = dp
    return new_dps



def merge(merged_list, value_dict):
    """
    合并两个列表，并根据 value_dict 中的 score_1 和 score_2 进行排序。
    :param list1: 第一个列表
    :param list2: 第二个列表
    :param value_dict: 包含元素价值的字典，格式为 {元素: {"score_1": x, "score_2": y}}
    :return: 排序后的合并列表
    """    
    # 按照 score_1 进行降序排序，如果 score_1 相同，则按 score_2 降序排序
    sorted_list = sorted(merged_list, key=lambda x: (value_dict[x]["acc_rwd"], value_dict[x]["all_rwd"]), reverse=True)
    
    return sorted_list


# 按正确性和score排序
def sorted_solutions(solutions, gt_answer, tokenizer):
    rwd, details = calculate_fitness(solutions, gt_answer, tokenizer)
    rwd_dict = {}
    for idx in range(len(rwd)):
        rwd_dict[solutions[idx]] = {"acc_rwd": details[idx]['acc_rwd'], "all_rwd": rwd[idx]}
    sorted_solutions = merge(solutions, rwd_dict)
    return sorted_solutions

# 两个列表互补得到高级进化的solution集合
def merge_dps(id_dps, id_dps_, tokenizer):
    new_dps = []
    for idx in id_dps:
        if idx and idx % 500 == 0:
            print(f"process {idx} examples, {(idx/len(id_dps)):.3f}")
        new_dp = {}
        dp = id_dps[idx]
        if idx not in id_dps_:
            new_dps.append(dp)
            continue
        dp_ = id_dps_[idx]
        if 'attempt' in dp:
            gt_answer = extract_answer(dp['attempt'])
        elif "answer" in dp:
            gt_answer = dp["answer"]
        else:
            continue
        # init_solutions = dp["all_solutions"][0:-6] + dp_["all_solutions"][0:-6]
        # evol_solutions = dp["all_solutions"][-6:] + dp_["all_solutions"][-6:]
        init_solutions = dp["all_solutions"][0:4] + dp_["all_solutions"][0:4]
        evol_solutions = dp["all_solutions"][4:] + dp_["all_solutions"][4:]
        
        sorted_init_solutions = sorted_solutions(init_solutions, gt_answer, tokenizer)
        sorted_evol_solutions = sorted_solutions(evol_solutions, gt_answer, tokenizer)

        all_solutions = sorted_init_solutions[0:4] + sorted_evol_solutions[0:6]
        
        for key in dp:
            new_dp[key] = dp[key]
        new_dp["all_solutions"] = all_solutions
        
        if isinstance(new_dp["evol_solution"], list):
            best_evol_solutions = dp["evol_solution"]
            all_sorted_evol_solutions = sorted_solutions(evol_solutions, gt_answer, tokenizer)
            new_dp["evol_solution"] = all_sorted_evol_solutions
        
        new_dps.append(new_dp)

    return new_dps

tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/global_user/USER_ID/user/models/Qwen2.5-7B-Instruct")

data_name = "limo"
data_type = "v1.1"

# data_path = f"./evol/wz_output_deepmath/evol_{data_name}/{data_type}/evol_{data_name}_+.pkl"
# data_path_ = f"./evol/wz_output_deepmath/evol_{data_name}/{data_type}/evol_{data_name}_-.pkl"
data_path = '/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/2026_02_16_aime_hard_no_gt/evol_limo/v1.1/evol_limo_-.pkl'
data_path_ = '/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/2026_02_16_aime_hard_no_gt/evol_limo/v1.1/evol_limo_+.pkl'

dps = read_data(data_path)
dps_ = read_data(data_path_)

id_dps = convert_id_dps(dps)
id_dps_ = convert_id_dps(dps_)

new_dps = merge_dps(id_dps, id_dps_, tokenizer)

# output_file = f"./evol/mix_limo_exp/evol_{data_name}/{data_type}/evol_{data_name}_m_1_iter.pkl"
# output_file = f"./evol/111_s1k_new_mutation/evol_s1k/v1.1/evol_{data_name}_m.pkl"
output_file = f"./evol/2026_02_16_aime_hard_no_gt/evol_limo/v1.1/evol_{data_name}_m.pkl"
with open(output_file, "wb") as file:
    pickle.dump(new_dps, file)
    
wrong = get_wrong_ids(dps, tokenizer)
wrong_ = get_wrong_ids(dps_, tokenizer)

print(len(wrong))
print(len(wrong_))

retD = list(set(wrong_).difference(set(wrong)))
len(retD)

