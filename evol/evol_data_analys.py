
from transformers import AutoTokenizer

import pickle
import sys
import numpy as np
import re
import torch
import copy

from fitness import (
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

def truncate_on_weird_char(text):
    match = re.match(r'^[\x00-\x7F\n\r\t\\\[\]\{\}_^a-zA-Z0-9\s\.\,\(\)\+\-\*/=<>]*', text)
    return match.group(0).rstrip() if match else ''

data_name = sys.argv[1]
data_type = sys.argv[2]
data_suffix = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/global_user/USER_ID/user/models/Qwen2.5-7B-Instruct")

# data_path = f"./wz_output_limo/evol_{data_name}/{data_type}/evol_{data_name}_{data_suffix}_2_iter.pkl"
# data_path = '/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main-v2/ifdr-main-qwen1.5/evol/wz_output_limo/evol_limo/v1.1/evol_limo_-.pkl'
# data_path = '/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main-v2/ifdr-main-qwen1.5/evol/wz_output_deepmath/evol_deepmath/v1.1/evol_deepmath_m.pkl'
# data_path = '/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main-v2/ifdr-main-qwen1.5/evol/wz_output_s1k_7b/evol_s1k/v1.1/evol_s1k_-.pkl'
# data_path = '/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/111_s1k_new_mutation/evol_s1k/v1.1/evol_s1k_m.pkl'
# data_path = '/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/2026_02_16_aime_hard_no_gt/evol_limo/v1.1/evol_limo_m.pkl'
data_path = '/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/111_s1k_new_mutation/evol_s1k/v1.1/evol_s1k_m.pkl'
with open(data_path, "rb") as file:
    dps = pickle.load(file)
for dp in dps:
    all_solutions = dp["all_solutions"]
    new_all_solutions = []

    if 'attempt' in dp:
        gt_answer = extract_answer(dp['attempt'])
    elif "answer" in dp:
        gt_answer = dp["answer"]
    else:
        continue

    for solution in all_solutions:
        if len(solution.split("#")) > 1:
            # 删除多余的阐述语句
            sol_snips = []
            for i, snip in enumerate(solution.split("#")):
                if i == 0 and gt_answer in snip:
                    # 修订refine陈述句
                    continue
                elif snip:
                    sol_snips.append(snip)
            solution = "#".join(sol_snips)
        solution = solution.replace("Refined", "Solving").replace("refined", "solving")
        new_all_solutions.append(solution)
    dp["all_solutions"] = new_all_solutions

w2r, w2w, r2w = 0, 0, 0
int_r, evol_r = 0, 0
init_r_ratio = []
evol_r_ratio = []

real_evol = 0
can_evol = 0
evol_fitness = []
init_fit = []
evol_fit = []

sel_evol = 0
evol_results = []
wrong_questions = []
evol_result = {
    'prompt': '',
    'response': ''
}
for dp in dps:
    all_solutions = dp["all_solutions"]
    init_solutions = all_solutions[0:-6]
    evol_solutions = all_solutions[-6:]
    # init_solutions = all_solutions[0:4]
    # evol_solutions = all_solutions[4:]
    # print(len(all_solutions))
    # evol_solutions = dp["evol_solution"]
    if 'attempt' in dp:
        gt_answer = extract_answer(dp['attempt'])
    elif "answer" in dp:
        gt_answer = dp['answer']
    else:
        continue
    if len(evol_solutions) != 0:
        sorted_evol_solutions = sorted_solutions(evol_solutions, gt_answer, tokenizer)
        sorted_init_solutions = sorted_solutions(evol_solutions, gt_answer, tokenizer)
        evol_result['prompt'] = dp['problem']
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # print(len(sorted_evol_solutions))
        # evol_result['response'] = truncate_on_weird_char(sorted_evol_solutions[0])
        evol_result['response'] = truncate_on_weird_char(sorted_evol_solutions[0])

            
        # evol_results.append(copy.deepcopy(evol_result))
        
        init_rwd, init_details = calculate_fitness(init_solutions, gt_answer, tokenizer)
        init_right = False
        init_r_cnt = 0
        for init_info in init_details:
            if init_info["acc_rwd"] >= 0.5:
                init_r_cnt += 1
                init_right = True

        evol_rwd, evol_details = calculate_fitness(sorted_evol_solutions, gt_answer, tokenizer)
        evol_right = False
        evol_r_cnt = 0
        for evol_info in evol_details:
            if evol_info["acc_rwd"] >= 0.5:
                evol_r_cnt += 1
                evol_right = True

        # 进化有多少的样本从不正确到正确
        if not init_right and evol_right:
            w2r += 1
        elif init_right and not evol_right:
            r2w += 1
        elif not init_right and not evol_right:
            w2w += 1

        if init_right:
            int_r += 1
        if evol_right:
            evol_r += 1
            evol_results.append(copy.deepcopy(evol_result))
        if not evol_right:
            wrong_questions.append(copy.deepcopy(evol_result['prompt']))
            

        # 进化前后的正确率算不算w2w
        # if not init_right and not evol_right:
        #     continue  # 没有进化出来正确答案
        # 进化前后的平均正确率
        init_r_ratio.append(init_r_cnt / len(init_solutions))
        evol_r_ratio.append(evol_r_cnt / len(sorted_evol_solutions))

        # 进化有多少的样本的fitness提升了
        init_fitness_vals = np.array(init_rwd)
        evol_fitness_vals = np.array(evol_rwd)
        # 找到当前代的最优解
        max_fitness_idx = np.argmax(init_fitness_vals)
        min_fitness_idx = np.argmin(evol_fitness_vals)

        if init_fitness_vals[max_fitness_idx] < evol_fitness_vals[min_fitness_idx]:
            real_evol += 1
        if np.mean(evol_rwd) > np.mean(init_rwd):
            can_evol += 1
        # evol_fitness.append(evol_fitness_vals[min_fitness_idx]-init_fitness_vals[max_fitness_idx])
        evol_fitness.append(np.mean(evol_rwd)-np.mean(init_rwd))
        init_fit.append(np.mean(init_rwd))
        evol_fit.append(np.mean(evol_rwd))

        # 多少最优样本选择进化的
        fitness_vals = np.array(init_rwd + evol_rwd)
        max_fit_idx = np.argmax(fitness_vals)
        if max_fit_idx >= len(all_solutions) - 6:
            sel_evol += 1

    # TODO 进化之后多样性变化
# torch.save(evol_results, "/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/2026_02_16_aime_hard_no_gt/evol_limo/v1.1/evol_limo_m.pt")
# torch.save(wrong_questions, "./evol/mix_s1k_exp/evol_s1k/v1.1/evol_s1k_wrong_question.pt")

init_solve = int_r/len(dps)
evol_solve = evol_r/len(dps)
all_solve = (w2r+int_r)/len(dps)

print(f"总数据量: {len(dps)}")
print(f"将错误进化到正确: {w2r/len(dps):.3f}, "
      f"将正确进化到错误: {r2w/len(dps):.3f}, "
      f"无法进化出正确的: {w2w/len(dps):.3f}")
print(f"原始平均正确率: {np.mean(init_r_ratio):.3f}, 平均正确率中位数: {np.median(init_r_ratio):.3f}\n"
      f"进化平均正确率: {np.mean(evol_r_ratio):.3f}, 平均正确率中位数: {np.median(evol_r_ratio):.3f}")
print(f"原始fitness: {np.mean(init_fit):.3f}, 进化fitness: {np.mean(evol_fit):.3f}\n"
      f"完全提升fitness率: {real_evol}/{len(dps)}, 能够提升fitness率: {can_evol/len(dps):.3f}, "
      f"平均提升fitness: {np.mean(evol_fitness):.3f}")
print(f"选择进化的回复率: {sel_evol/len(dps):.3f}")
print(f"原始解决问题率pass@1: {init_solve:.3f}, 进化解决问题率pass@1: {evol_solve:.3f}, "
      f"总解决问题率：{all_solve:.3f}, 解决问题提升率：{all_solve-init_solve:.3f}")
