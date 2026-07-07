
import os
import re
import time
import json
import pickle

from openai import AzureOpenAI

def llama_call(input_text, sys_temp, model="o1-mini", max_tokens=4096):
    # Alternative: OpenAI-compatible endpoint (e.g. aihubmix)
    # client = OpenAI(
    #     api_key=os.environ["OPENAI_API_KEY"],
    #     base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    # )
    client = AzureOpenAI(
    api_key = os.environ["AZURE_OPENAI_API_KEY"],
    api_version = "2024-07-01-preview",
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
    )

    message_text = [{"role": "system", "content": sys_temp}]
    text = [{"role": "user", "content": f"math problem: {input_text}"}]
    input_information = message_text + text
    while True:
        try:
            result = client.chat.completions.create(model=model,
                                                    messages=input_information, max_tokens=max_tokens)
            result = result.choices[0].message.content
            return result
        except Exception as e:
            print(e)
            time.sleep(10)


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


def get_trajectory(dps, sys_prompt):
    results = []
    remain_dps = []
    for dp in dps:
        cnt = 0
        result = None
        while not isinstance(result, dict):
            question = dp['question']
            output = llama_call(question, sys_prompt)
            result = parse_response(output)
            cnt += 1
            if cnt > 10:
                break
        if isinstance(result, dict):
            dp["silu"] = result
            results.append(dp)
        else:
            remain_dps.append(dp)
    return results, remain_dps


sys_prompt = (
    "Please construct a reasoning trajectory in JSON format based on the following math problem. "
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

data_path = "/extrahome0/user/output/evol_s1k/v1/evol_s1k_v.pkl"
output_dir = "./evol_s1k_v_traj.pkl"

with open(data_path, "rb") as file:
    train_dps = pickle.load(file)

# TODO: debug
train_dps = train_dps[0:2]
results, remain_dps = get_trajectory(train_dps, sys_prompt)
print(f"success {len(results)}, fail {len(remain_dps)}")

with open(output_dir, "wb") as f:
    pickle.dump(results, f)
