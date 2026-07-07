
import torch
from datasets import Dataset

import json
import random
import pickle
from loguru import logger

from utils.prompts import GSM_COT_8_SHOT, math_message


def load_mix_polluted_dataset(data_name, data_split, tokenizer, is_train=True):
    data_path = "/userhome/Research_HUB/Reasoning/data_dir/Data_MIX/MATH5k_MATH500_gsm8k.pt"
    all_data = torch.load(data_path, weights_only=False)
    raw_data = all_data[data_name]
    if is_train:
        if data_split in ["train", "test"]:
            dps = raw_data[data_split]
        else:
            dps = raw_data["train"] + raw_data["test"]
    else:
        dps = raw_data["test"]

    examples = []
    for dp in dps:
        chosen = dp[1]
        rejected = ""
        if chosen is None or rejected is None:
            continue
        example = {
            "prompt": [{"role": "system", "content": math_message}, {"role": "user", "content": dp[0]}],
            "chosen": [{"role": "assistant", "content": chosen}, ],
            "rejected": [{"role": "assistant", "content": rejected}],
        }
        prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
        prompt_chosen = tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False)
        chosen = prompt_chosen[len(prompt):]
        examples.append(
            {"prompt": prompt, "response": chosen}
        )
    train_dataset = Dataset.from_list(examples)
    return train_dataset


def load_train_dev_dataset(data_path, dataset_sample, tokenizer, is_dpo=True, is_train=True):
    data_idx = 0 if is_train else 1
    raw_data = torch.load(data_path, weights_only=False)
    # print(len(raw_data))
    # print(raw_data[0].keys())
    examples = []
    # for dp in raw_data[data_idx]:
    for dp in raw_data:
        # print('$$$$$$$$$',dp)
        chosen = dp['chosen'] if 'chosen' in dp else dp['response']
        rejected = dp['rejected'] if 'rejected' in dp else ""
        if chosen is None or rejected is None:
            continue
        if "math_message" in dp:
            message = dp["math_message"]
        else:
            message = math_message
        example = {
            "prompt": [{"role": "system", "content": message}, {"role": "user", "content": dp['prompt']}],
            "chosen": [{"role": "assistant", "content": chosen}, ],
            "rejected": [{"role": "assistant", "content": rejected}],
        }
        prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
        prompt_chosen = tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False)
        chosen = prompt_chosen[len(prompt):]

        if is_dpo:
            prompt_rejected = tokenizer.apply_chat_template(example["prompt"] + example["rejected"], tokenize=False)
            rejected = prompt_rejected[len(prompt):]
            examples.append(
                {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            )
        else:
            examples.append(
                {"prompt": prompt, "response": chosen}
            )

    train_dataset = Dataset.from_list(examples)
    if dataset_sample > 0:
        num_sample = min(len(train_dataset), dataset_sample)
        train_dataset = train_dataset.select(range(num_sample))

    return train_dataset


def load_format_input(examples, tokenizer, prompt_key):
    # message = "You are a math teacher. Given a math problem, please use formal " \
    #           "mathematical expressions to provide the reasoning process step by step. The final aswer should be " \
    #           "formated as $\\boxed{xxx}$. "
    f_exps = []
    for example in examples:
        if "math_message" in example:
            message = example["math_message"]
        else:
            message = math_message
        prompt = [{"role": "system", "content": message}, {"role": "user", "content": example[prompt_key]}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        f_exps.append(prompt)
    return f_exps


def load_cot_input(examples, prompt_key, data_name="gsm"):
    if data_name == "gsm":
        cot_template = GSM_COT_8_SHOT
    else:
        cot_template = ""

    f_exps = []
    for example in examples:
        f_exp = cot_template.format_map({prompt_key: example[prompt_key]})
        f_exps.append(f_exp)
    return f_exps


def load_eval_data(data_name, tokenizer, sample_data=None, use_template=True, do_test=True):
    answer_key = "answer"
    if data_name == "gsm":
        eval_data_path = "/extrahome0/user/data/gsm8k_test.jsonl"
        prompt_key = "question"
    elif data_name == "prm":
        eval_data_path = "/userhome/Research_HUB/Reasoning/data_dir/prm800k_cut_train_valid_test.pt"
        prompt_key = "prompt"
    elif data_name == "math":
        eval_data_path = "/userhome/Research_HUB/Reasoning/prm800k/prm800k/math_train_valid_test/MATH_train_valid_test500.pt"
        prompt_key = "problem"
    elif data_name == "rf_sft15k":
        eval_data_path = "/extrahome0/user/data/reasonflux/rf_sft15k.pt"
        prompt_key = "prompt"
        answer_key = "response"
    else:
        raise ValueError

    data_idx = -1 if do_test else -2
    examples = []
    if eval_data_path.endswith('pkl'):
        with open(eval_data_path, "rb") as f:
            examples = pickle.load(f)
    elif eval_data_path.endswith('jsonl'):
        with open(eval_data_path) as f:
            for line in f:
                examples.append(json.loads(line))
    elif eval_data_path.endswith('json'):
        with open(eval_data_path) as f:
            examples = json.load(f)
    elif eval_data_path.endswith('pt'):
        examples = torch.load(eval_data_path)[data_idx]
    else:
        raise TypeError

    if sample_data:
        examples = random.sample(examples, sample_data)

    if use_template:
        logger.warning("using chat template")
        all_examples = load_format_input(examples, tokenizer, prompt_key)
    else:
        logger.warning(f"using {data_name}-8shot")
        all_examples = load_cot_input(examples, prompt_key, data_name)
    all_answers = []
    for example in examples:
        all_answers.append(example[answer_key])

    return all_examples, all_answers
