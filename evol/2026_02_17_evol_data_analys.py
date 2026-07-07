from transformers import AutoTokenizer
import pickle
import sys
import numpy as np
import re
import torch
import copy
from rouge_score import rouge_scorer

from fitness import (
    extract_answer, cosine_scaled_reward,
    accuracy_reward, format_reward, cosine_lang_reward
)


# ========= ROUGE 计算 =========
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_l_score(a, b):
    score = scorer.score(a, b)["rougeL"].fmeasure
    return score


# ========= Fitness =========
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
        total = float(len_rwd[i] + acc_rwd[i] + format_rwd[i] + lang_rwd[i])
        rwd.append(total)
        details_info.append(
            {
                "rwd": total,
                "acc_rwd": acc_rwd[i],
            }
        )
    return rwd, details_info


def truncate_on_weird_char(text):
    match = re.match(
        r'^[\x00-\x7F\n\r\t\\\[\]\{\}_^a-zA-Z0-9\s\.\,\(\)\+\-\*/=<>]*',
        text,
    )
    return match.group(0).rstrip() if match else ''


# =======================
# 主程序
# =======================

data_name = sys.argv[1]
data_type = sys.argv[2]
data_suffix = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained(
    "/inspire/hdd/global_user/USER_ID/user/models/Qwen2.5-7B-Instruct"
)

data_path = "/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/111_limo_new_mutation/evol_limo/v1.1/evol_limo_m.pkl"

with open(data_path, "rb") as file:
    dps = pickle.load(file)

evol_results = []
question_solution_counts = []

ROUGE_THRESHOLD = 0.8

for dp in dps:
    all_solutions = dp["all_solutions"]

    init_solutions = all_solutions[0:4]
    evol_solutions = all_solutions[4:]

    if "attempt" in dp:
        gt_answer = extract_answer(dp["attempt"])
    elif "answer" in dp:
        gt_answer = dp["answer"]
    else:
        continue

    if len(evol_solutions) == 0:
        continue

    # 计算fitness并排序
    rwd, details = calculate_fitness(evol_solutions, gt_answer, tokenizer)

    sorted_indices = sorted(
        range(len(rwd)),
        key=lambda i: (details[i]["acc_rwd"], rwd[i]),
        reverse=True
    )

    selected_responses = []

    # 先挑正确解
    for idx in sorted_indices:
        if details[idx]["acc_rwd"] < 0.5:
            continue

        candidate = truncate_on_weird_char(evol_solutions[idx])

        # ROUGE 去重
        keep = True
        for existing in selected_responses:
            if rouge_l_score(candidate, existing) > ROUGE_THRESHOLD:
                keep = False
                break

        if keep:
            selected_responses.append(candidate)


    # 如果没有正确解
    if len(selected_responses) == 0:
        best_idx = sorted_indices[0]
        selected_responses.append(
            truncate_on_weird_char(evol_solutions[best_idx])
        )

    question_solution_counts.append(len(selected_responses))

    for resp in selected_responses:
        evol_results.append(
            {
                "prompt": dp["problem"],
                "response": resp,
            }
        )

# =======================
# 保存
# =======================

output_path = "/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/evol/111_limo_new_mutation/evol_limo/v1.1/2026_02_17_evol_limo_m.pt"

torch.save(evol_results, output_path)

# =======================
# 统计
# =======================

avg_solutions = np.mean(question_solution_counts)

print(f"总问题数量: {len(question_solution_counts)}")
print(f"总保存解法数量: {len(evol_results)}")
print(f"每个 question 平均保留解法数量: {avg_solutions:.3f}")
