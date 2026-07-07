import os
import time
import json
import random
import pickle
import numpy as np
from copy import deepcopy
from loguru import logger

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def build_problem_offset(dps):
    prob_offset = []
    new_dps = []
    # for idx, dp in enumerate(dps):
    #     for trajectory in dp["trajectory_knowledge"]:
    #         new_dp = {}
    #         new_dp["reason_flow"] = trajectory["knowledge"]["reason_flow"]
    #         new_dp["question"] = dp["question"]
    #         new_dp["problem_id"] = idx
    #         new_dps.append(new_dp)
    #         prob_offset.append(idx)
    for idx, dp in enumerate(dps):
        trajectory = dp["trajectory"]
        for reason_flow in trajectory["reason_flow"]:
            new_dp = {}
            new_dp["reason_flow"] = reason_flow
            new_dp["question"] = dp["question"]
            new_dp["problem_id"] = idx
            new_dps.append(new_dp)
            prob_offset.append(idx)

    # 按最大长度进行排序
    combined = sorted(zip(new_dps, prob_offset), key=lambda x: len(x[0]['reason_flow']), reverse=True)
    # 解压回两个列表
    sorted_dps, sorted_prob_offset = zip(*combined)

    new_dps, prob_offset = list(sorted_dps), list(sorted_prob_offset)

    return new_dps, prob_offset


def generate(messages, tokenizer, temp=0.6, max_tokens=1024):
    prompts = []
    for i, message in messages.items():
        prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=temp,
            top_p=0.95,
            max_tokens=max_tokens,
            n=1,
            stop=stop_words,
            stop_token_ids=(
                [151645, 151643]
            ),
        ),
        use_tqdm=False
    )
    outputs = sorted(
        outputs, key=lambda x: int(x.request_id)
    )  # sort outputs by request_id
    responses = [output.outputs[0].text for output in outputs]

    return responses


def build_step_messages(problems, step_insts, previous_instructions, previous_reasons):
    system_prompt = "Now you are a student who are interacting with your tutor, Your teacher will gradually guide you to solve a problem. " \
                    "Please follow the instructions and guidance given by the teacher to solve the problem step by step. " \
                    "Note that you only need to generate response less than 200 words under the teacher guideline. " \
                    "Before responding, you need to carefully consider the previous guidelines and reasoning steps."
    messages = {i: [] for i, step_inst in enumerate(step_insts) if step_inst != "-1"}
    for i in messages:
        system_prompt += '\nProblem: ' + problems[i]
        messages[i].append({"role": "system", "content": system_prompt})
        assert len(previous_instructions[i]) == len(previous_reasons[i])
        pre_ists = previous_instructions[i]
        pre_step = previous_reasons[i]
        for idx in range(len(pre_ists)):
            # Consider changing f-string here for consistency of step numbering
            messages[i].append({"role": "user", "content": f'Teacher Instruction for Step {idx + 1}:' + pre_ists[idx]})
            messages[i].append({"role": "assistant", "content": pre_step[idx]})
        messages[i].append(
            {"role": "user", "content": f'Teacher Instruction for Step {len(pre_ists) + 1}:' + step_insts[i]})
    return messages


def build_summy_messages(problems, reasons):
    summy_prompt = (
        "You are a math expert, you are now faced with a math problem and a reference solution, "
        "and you need to summarize the final answer based on the math problem and the reference solution.\n\n"
        "Note: (1) There may be errors in the reference solution, you need to pay attention to the errors inside, "
        "and take the initiative to correct;\n(2) You will put your answers in \\boxed{{}}.\n\n"
    )
    summy_user = (
        "[Math Promblem]\n{problem}\n"
        "[Reference Solution]\n{solution}\n"
    )
    solutions = ["\n\n".join(reason) for _, reason in reasons.items()]

    messages = {}
    cnt = 0
    for problem, solution in zip(problems, solutions):
        messages[cnt] = [{"role": "system", "content": summy_prompt},
                         {"role": "user", "content": summy_user.format(problem=problem, solution=solution)}]
        cnt += 1
    return messages


def step_generate(tokenizer, step_insts, problems,
                  previous_instructions, previous_reasons,
                  step_temp=0.6, len_step=2048):
    messages = build_step_messages(problems, step_insts, previous_instructions, previous_reasons)
    responses = generate(messages, tokenizer, temp=step_temp, max_tokens=len_step)

    solutions = {}
    for ix, i in enumerate(messages):
        solutions[i] = responses[ix]
    return solutions


def summy_steps(problems, reasons, summy_temp=0.6, len_summy=256):
    messages = build_summy_messages(problems, reasons)
    solutions = generate(messages, tokenizer, temp=summy_temp, max_tokens=len_summy)
    return solutions


def step_guid_generate(dps, tokenizer, step_temp=0.6,
                       len_step=2048, summy_temp=0.6, len_summy=256):
    max_steps = len(dps[0]["reason_flow"])
    current_instructions = []
    for dp in dps:
        reason_flow = dp["reason_flow"]
        new_reason_flow = []
        for step in reason_flow:
            # if step.startswith("Step"):
            #     step = ":".join(step.split(":")[1:])
            new_reason_flow.append(step)
        current_instructions.append(new_reason_flow)
    problems = [dp['question'] for dp in dps]

    previous_instructions = {i: [] for i in range(len(dps))}
    previous_reasons = {i: [] for i in range(len(dps))}  # 保存了每一步的推理步骤
    logger.info("|-- step guideline reason generate...")
    for step_idx in range(max_steps):
        # 当前步骤 batch
        step_insts = [cur_ists[step_idx] for cur_ists in current_instructions]
        # 只有步骤有insts的才有current_reasons {i: solution} i是数据点idx
        current_reasons = step_generate(tokenizer, step_insts, problems,
                                        previous_instructions, previous_reasons,
                                        step_temp=step_temp, len_step=len_step)
        for i in current_reasons:
            previous_instructions[i].append(step_insts[i])
            previous_reasons[i].append(current_reasons[i])
    logger.info("|-- summy step reason generate...")
    summy_reasons = summy_steps(problems, previous_reasons,
                                summy_temp=summy_temp, len_summy=len_summy)
    for i, reason in enumerate(summy_reasons):
        previous_reasons[i].append(reason)
    return previous_reasons

# 一些超参数
batch_size = 16
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
tokenizer = AutoTokenizer.from_pretrained("/extrahome0/HF_models/Qwen2.5-7B-Instruct")
step_temp, len_step = 0.6, 2048
summy_temp, len_summy = 0.6, 256
start = 0
end = -1

# data_path = "/code/Research_with_user/reasoning/GA/solution_path/solution_path.pkl"
data_path = "/code/Research_with_user/reasoning/GA/solution_path/gpt4o_v2.pkl"
output_path = "/code/Research_with_user/reasoning/GA/quality_score/"

# 加载调用vllm
available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
model_path = "/extrahome0/HF_models/Qwen2.5-7B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path)

llm = LLM(
    model=model_path,
    tensor_parallel_size=len(available_gpus),
    # pipeline_parallel_size=pipeline_parallel_size,
    trust_remote_code=True,
    gpu_memory_utilization=0.95
)

# 加载需要评估的数据
logger.info(f"Load data from {data_path}")
with open(data_path, "rb") as file:
    data = pickle.load(file)
dps = []
for dp in data:
    # if len(dp["trajectory_knowledge"]) >= 1:
    dps.append(dp)
if end > 0:
    dps = dps[start:end]
all_step_dps, offset = build_problem_offset(dps)
logger.info(f"Processed {len(all_step_dps)} dps")

# 输出文件的路径 以及加载和reuse先前生成的结果（断点保护）
output_file = os.path.join(output_path, f"gpt4o_step_guid_reason_s={start}_e={end}.pkl")
if os.path.isfile(output_file):
    with open(output_file, "rb") as file:
        results = pickle.load(file)
        reuse_idx = len(results)
else:
    results = []
    reuse_idx = -1

if len(results) > 0:
    remain_batch = (len(all_step_dps)-len(results))//batch_size
    if (len(all_step_dps)-len(results))%batch_size:
        remain_batch+=1
else:
    remain_batch = len(all_step_dps)//batch_size
    if len(all_step_dps)%batch_size:
        remain_batch+=1
logger.debug(f"Reuse {reuse_idx} data in {output_file}, remaining {remain_batch} batches.")

cost_times = []
for i in range(0, len(all_step_dps), batch_size):
    if i < reuse_idx:
        continue

    start_time = time.time()
    batch_input = all_step_dps[i:i + batch_size]
    batch_offset = offset[i:i + batch_size]
    logger.debug(f"This {i}-th question start")

    # pad max
    max_steps = max([len(dp["reason_flow"]) for dp in batch_input])
    for dp in batch_input:
        dp["reason_flow"].extend(["-1"] * (max_steps - len(dp["reason_flow"])))

    # all_step_solutions key是当前batch内部的idx，value是推测的每一步guideline得到的
    all_step_solutions = step_guid_generate(batch_input, tokenizer, step_temp=step_temp,
                       len_step=len_step, summy_temp=summy_temp, len_summy=len_summy)
    for ix, step_solutions in enumerate(all_step_solutions.items()):
        batch_input[ix]["step_solutions"] = step_solutions[-1]
        batch_input[ix]["reason_flow"] = [step for step in batch_input[ix]["reason_flow"] if step != "-1"]
        assert len(batch_input[ix]["step_solutions"]) == len(batch_input[ix]["reason_flow"]) + 1  # 最后有一个summy

    results.extend(batch_input)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    end_time = time.time()
    cost_times.append((end_time - start_time) / 60)
    mean_cost_time = sum(cost_times) / len(cost_times)
    remain_batch -= 1
    remain_time = remain_batch * mean_cost_time
    logger.debug(f"The {i}-th question end, cost {cost_times[-1]:.2f} min, "
          f"remaining about {len(all_step_dps) - len(results)} with {remain_time:.2f} min ({remain_time / 60:.2f} hrs)")

logger.warning(f"Total cost {sum(cost_times):.2f} min")
# 最后处理回原来的数据集中
for result in results:
    off_idx = result["problem_id"]
    original_dp = dps[off_idx]  # 将得到的step solution合并回原始数据
    if "step_reason_flow" not in original_dp:
        original_dp["step_reason_flow"] = [result["reason_flow"]]
    else:
        original_dp["step_reason_flow"].append(result["reason_flow"])
    if "step_solutions" not in original_dp:
        original_dp["step_solutions"] = [result["step_solutions"]]
    else:
        original_dp["step_solutions"].append(result["step_solutions"])

post_resulsts_file = os.path.join(output_path, f"gpt4o_step_guid_reason.pkl")
with open(post_resulsts_file, "wb") as f:
    pickle.dump(dps, f)
