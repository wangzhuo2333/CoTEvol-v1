
import os
import re
import random
import pickle
import json
import time

import numpy as np
import pandas as pd
from openai import OpenAI
from collections import Counter
from loguru import logger

from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_api_result(result):
    to_return = []
    for idx, g in enumerate(result.choices):
        text = g.text
        logprob = sum(g.logprobs.token_logprobs)
        to_return.append((text, logprob))
    to_return = sorted(to_return, key=lambda tup: tup[1], reverse=True)
    to_return = [r[0] for r in to_return]
    return to_return


def parse_response(text):
    # 检测<think>...</think>的内容
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if not think_match:
        return "Think Error"
    think_content = think_match.group(1).strip()

    # 检测JSON格式的内容
    json_match = re.search(r'</think>(\{.*\})', text, re.DOTALL)
    if not json_match:
        return "Json Error"

    json_content = json_match.group(1).strip()

    try:
        knowledge_dict = json.loads(json_content)
    except json.JSONDecodeError:
        return "Json Error"

    # 确保JSON包含reason_flow键
    if "reason_flow" not in knowledge_dict:
        return "Reason Flow Error"

    try:
        reason_flow = " ".join(knowledge_dict["reason_flow"])
    except:
        return "Reason Flow Error"

    return {"think": think_content, "json": knowledge_dict}


def remove_dup_trajectory(path_infos, threshold=0.8):
    if not path_infos:
        return []
    knowledges = [path_info["knowledge"] for path_info in path_infos]
    paragraphs = [" ".join(knowledge["reason_flow"]) for knowledge in knowledges]
    # 创建 TF-IDF 向量化器
    vectorizer = TfidfVectorizer().fit_transform(paragraphs)

    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(vectorizer)

    # 创建一个布尔列表，标记哪些段落需要保留
    to_keep = [True] * len(paragraphs)

    for i in range(len(paragraphs)):
        if not to_keep[i]:  # 如果当前段落已经被标记为去除，则跳过
            continue

        for j in range(i + 1, len(paragraphs)):
            if similarity_matrix[i][j] > threshold:
                to_keep[j] = False  # 如果相似度超过阈值，则标记为删除

    # 根据标记的列表，返回不重复的段落
    return [path_infos[i] for i in range(len(path_infos)) if to_keep[i]]


nums = 4
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
model_path = "/extrahome0/user/output/rf_sft15k/qwen-it/20250324012330_qwen-it_lr5e-06_e1_bs32_ra0/checkpoint-549"
sys_prompt = (
    "Please construct a reasoning trajectory in JSON format based on the following problem. "
    "This reasoning trajectory should include the following: problem description, general knowledge category, "
    "specific direction, applied method, examined knowledge, and reason_flow. "
    "Please output according to the given format: {'Problem': Here describes the problem you constructed, 'General Knowledge Category': "
    "Here corresponds to the general category of mathematical knowledge to which the problem belongs, "
    "'Specific Direction': Here corresponds to the specific knowledge direction to which the problem belongs, "
    "'Applied Method': Here corresponds to the `template_name` of the input template, **please use its original name completely, do not refer, abbreviate or rewrite**, "
    "'Examined Knowledge': [Here is a list used to list the knowledge tags examined by this problem], "
    "'reason_flow': [This is a list, according to the `reason_flow` steps in the input template, "
    "to describe in detail the thinking process of solving the problem. Each step should be explained in conjunction with the specific situation of the problem, "
    "such as how to convert conditions, how to apply formulas, etc. But it should be noted that `reason_flow` is only a framework to guide students' thinking, "
    "and cannot directly give specific calculation results or answers, "
    "but should retain a certain degree of challenge for students to complete the specific calculations and derivations themselves.]}. "
    "Before providing a formal response, please carefully consider and analyze the question, and place your thoughts within <think></think> tags."
)

with open("/extrahome0/user/output/evol_s1k/v1/evol_s1k_v.pkl", "rb") as file:
    data = pickle.load(file)

client = OpenAI(
    base_url="http://localhost:8000/v1",  # vLLM服务的地址 端口1/2 8000/8012
    api_key="EMPTY",  # 如果设置了API密钥需要填写,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

output_path = "/code/Research_with_user/reasoning/GA/solution_path/"
output_file = os.path.join(output_path, f"solution_path.pkl")
if os.path.isfile(output_file):
    with open(output_file, "rb") as file:
        dps, unsolved_ids = pickle.load(file)
        reuse_idx = len(dps)
else:
    dps = []
    reuse_idx = 0
    unsolved_ids = []
logger.debug(f"Reuse {reuse_idx} data in {output_file}")

cost_times = []
for i, dp in enumerate(data):
    if i < reuse_idx:
        continue

    start_time = time.time()
    logger.info(f"The {i}-th question start")

    question = dp["question"]
    prompt = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"{question}"}]
    inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    cnt = 0
    temp = 0.6
    path_infos, error_info = [], []
    while len(path_infos) < nums:
        if cnt > 5:
            temp = temp + 0.1
            logger.debug(f"生成难度较大，提高温度{temp - 0.1:.1f}->{temp:.1f}")

        results = client.completions.create(
            model=model_path,  # 与启动时指定的模型名称一致
            prompt=inputs,
            max_tokens=3000,
            temperature=temp,
            top_p=0.95,
            n=nums,
            stop=stop_words,
            # logprobs=true
        )
        # responses = parse_api_result(results)
        # 过滤多余的样本和解析失败的样本
        for response in results.choices:
            sol_dict = parse_response(response.text)
            if isinstance(sol_dict, dict):
                path_infos.append(sol_dict)
            else:
                error_info.append(sol_dict)
        if cnt > 10:
            # path_infos = remove_dup_trajectory(path_infos)
        # else:
            logger.debug(f"难度很大，丢弃去重操作")
            break
        cnt += 1

    if len(path_infos) < 1:
        unsolved_ids.append(i)
        logger.warning(f"The {i}-th question failed, total unsolved: {len(unsolved_ids)}")

    dp["trajectory_knowledge"] = path_infos
    dps.append(dp)
    with open(output_file, "wb") as f:
        pickle.dump((dps, unsolved_ids), f)

    end_time = time.time()
    cost_times.append((end_time - start_time) / 60)
    mean_cost_time = sum(cost_times) / len(cost_times)
    remain_time = abs(len(data) - len(dps)) * mean_cost_time
    logger.debug(f"The {i}-th question end, cost {cost_times[-1]:.2f} min, "
                 f"remaining {remain_time:.2f} min ({remain_time/60:.2f} hrs)")

