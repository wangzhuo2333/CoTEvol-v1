
import os
import time
import json
import random
import pickle
import numpy as np

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def setup_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # maybe slower training


def generate(llm, prompts, temp=0.6, max_tokens=1024, model_path="qwen"):
    """
    Generates a response from the language model and extracts thought and solution.

    Args:
        messages (list): A list of message dictionaries for the language model.

    Returns:
        tuple: A tuple containing the extracted thought and solution from the response.
    """

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
                if "qwen" in model_path.lower()
                else None
            ),
        ),
    )
    outputs = sorted(
        outputs, key=lambda x: int(x.request_id)
    )  # sort outputs by request_id
    responses = [output.outputs[0].text for output in outputs]

    return responses


def interplay(tokenizer, client, instruction, problem, previous_instruction, previous_reasoning):
    """
    Simulates the interplay between a student and a tutor for problem-solving.

    Args:
        instruction (str): The current instruction from the tutor.
        problem (str): The problem description.
        previous_instruction (list): List of previous tutor instructions.
        previous_reasoning (list): List of student's reasoning for previous steps.

    Returns:
        tuple: A tuple containing the student's thought process and the solution.
    """
    system_prompt = "Now you are a student who are interacting with your tutor, Your teacher will gradually guide you to solve a problem. " \
                    "Please follow the instructions and guidance given by the teacher to solve the problem step by step. " \
                    "Note that you only need to generate response less than 200 words under the teacher guidline. " \
                    "Before responding, you need to carefully consider the previous guidlines and reasoning steps." + '\nProblem: ' + problem
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    assert len(previous_instruction) == len(previous_reasoning)
    for i in range(len(previous_instruction)):
        # Consider changing f-string here for consistency of step numbering
        messages.append({"role": "user",
                         "content": f'Teacher Instruction for Step {len(previous_instruction)}:' + previous_instruction[
                             i]})
        messages.append({"role": "assistant", "content": previous_reasoning[i]})
    # Consider changing f-string here for consistency of step numbering
    messages.append(
        {"role": "user", "content": f'Teacher Instruction for Step {len(previous_instruction) + 1}:' + instruction})
    solution = generate(messages, tokenizer, client)
    return solution

# 定义评估prompt，包括输出格式的要求等
summy_prompt = (
    "You are a math expert, you are now faced with a math problem and a reference solution, "
    "and you need to summarize the final answer based on the math problem and the reference solution.\n\n"
    "Note: (1) There may be errors in the reference solution, you need to pay attention to the errors inside, and take the initiative to correct;\n"
    "(2) You will put your answers in \\boxed{{}}.\n\n"
)
summy_user = (
    "[Math Promblem]\n{problem}\n"
    "[Reference Solution]\n{solution}\n"
)

temp=0.0 # 生成的温度
batch_size=32 # 多少样本保存一下
max_tokens=2048 # model最大的输出长度
seed=42
pipeline_parallel_size=4
# 如果使用Qwen model推荐使用，其他的可以使用None
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

setup_seed(seed)

# 加载调用vllm
# model_path = "/extrahome0/user/output/s1k_evol_sft/qwen-it/20250317090045_qwen-it_lr7e-07_e1_bs16_ra0/checkpoint-20"
model_path = "/extrahome0/HF_models/Qwen2.5-7B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(
    model=model_path,
    tensor_parallel_size=len(available_gpus),
    # pipeline_parallel_size=pipeline_parallel_size,
    trust_remote_code=True,
    gpu_memory_utilization=0.95
)

# 输出文件的路径 以及加载和reuse先前生成的结果（断点保护）
output_path = "/code/Research_with_user/reasoning/GA/quality_score/"
reuse_idx = []
# output_file = os.path.join(output_path, f"test_result.pkl")
output_file = os.path.join(output_path, f"test_result_gpt4o.pkl")
if os.path.isfile(output_file):
    with open(output_file, "rb") as file:
        dps = pickle.load(file)
        reuse_idx = len(dps)
else:
    dps = []
    reuse_idx = 0
print(f"Reuse {reuse_idx} data in {output_file}")

# 加载需要评估的数据
solution_path = "/code/Research_with_user/reasoning/GA/solution_path/evol_s1k_v_abstract_no_truth_v2.pkl"
data = []
with open(solution_path, "rb") as file:
    data = pickle.load(file)
# data = [dp for dp in data if len(dp["trajectory_knowledge"]) >= 1]
def get_remain_dps(data, reuse_idx=0):
    remain_dps = []
    for i, dp in enumerate(data):
        if i < reuse_idx:
            continue
        remain_dps.append(dp)
    return remain_dps
data = get_remain_dps(data, reuse_idx)
remain_batch = (len(data) // batch_size)
if len(data) % batch_size:
    remain_batch += 1

def format_input(dps, sys_prompt):
    prompts = []
    for i, dp in enumerate(dps):
        user_response = dp["problem"]  # 待评估的回复
        inputs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"{user_response}"}]
        prompt = tokenizer.apply_chat_template(
            inputs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts

# 开始评估
dps = []
cost_times = []
for i in range(0, len(data), batch_size):
    start_time = time.time()

    batch_input = data[i:i + batch_size]
    prompts = format_input(batch_input)

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
                if "qwen" in model_path.lower()
                else None
            ),
        ),
    )
    outputs = sorted(
        outputs, key=lambda x: int(x.request_id)
    )  # sort outputs by request_id
    responses = [output.outputs[0].text for output in outputs]

    for idx, dp in enumerate(batch_input):
        dp["output"] = responses[idx]
        dps.append(dp)
    with open(output_file, "wb") as f:
        pickle.dump(dps, f)

    end_time = time.time()
    cost_times.append((end_time - start_time) / 60)
    mean_cost_time = sum(cost_times) / len(cost_times)
    remain_batch -= 1
    remain_time = remain_batch * mean_cost_time
    print(f"The {i}-th question end, cost {cost_times[-1]:.2f} min, "
          f"remaining {len(data)-len(dps)} with {remain_time:.2f} min ({remain_time / 60:.2f} hrs)")

print(f"Total cost {sum(cost_times):.2f} min")