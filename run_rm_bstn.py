import argparse
import math
import json
import time

import os
import pickle
import torch
from transformers import AutoModel, AutoTokenizer
from loguru import logger

from utils.general import setup_seed
from utils.read_data import load_train_dev_dataset
from utils.prompts import math_message


message = "You are a math teacher. Given a math problem, please use formal " \
          "mathematical expressions to provide the reasoning process step by step. The final answer should be " \
          "formated as $\\boxed{{}}$. "


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/output/math_evol/limo_bstn/limo/qwen2.5-7b-instruct_n=10_t=0.6.pkl")
    parser.add_argument("--output_dir", type=str, default="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/output/math_evol/limo_bstn/limo/")
    parser.add_argument("--data_type", type=str, default="short")
    parser.add_argument("--data_name", type=str, default="limo")
    parser.add_argument("--model_name_or_path", type=str, default="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/Qwen2.5-Math-72B-Instruct/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"/extrahome0/user/data/{args.data_name}/{args.data_name}_{args.data_type}_cot.pt"
    return args


def read_data(data_path):
    with open(data_path, "rb") as file:
        data = pickle.load(file)
    dps = []
    for dp in data:
        # all_solutions = dp["all_solutions"][0:-6]
        all_solutions = dp["distil_output"]
        problem = dp["question"]
        dp["examples"] = []
        for solution in all_solutions:
            example = [
                {"role": "system", "content": math_message},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution}
            ]
            dp["examples"].append(example)
        dps.append(dp)
    return dps


def get_score(dps, tokenizer, model):
    scores = []
    for dp in dps:
        conversation_str = tokenizer.apply_chat_template(
            dp,
            tokenize=False,
            add_generation_prompt=False,
            max_length=6000
        )

        input_ids = tokenizer.encode(
            conversation_str,
            return_tensors="pt",
            add_special_tokens=False
        ).to(model.device)

        outputs = model(input_ids=input_ids)
        score = outputs[0].cpu().item()
        scores.append(score)

    return scores

def main():
    args = parse_args()
    setup_seed(args.seed)

    math_path = "/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main-v2/ifdr-main-qwen1.5/srl_math_benchmark/srl_math_benchmark/math500/test.jsonl"
    eval_data = []
    with open(math_path) as file:
        for line in file:
            eval_data.append(json.loads(line))

    eval_dps = []
    for dp in eval_data:
        eval_dp = {"prompt": dp["problem"], "response": dp["solution"]}
        eval_dps.append(eval_dp)

    device = "auto" # the device to load the model onto
    logger.info(f"loading dataset from {args.data_dir}")
    data = read_data(args.data_dir)

    logger.info(f"loading model and tokenizer from {args.model_name_or_path}")
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    dps = []
    start_time = time.time()
    for idx, dp in enumerate(data):
        if idx and idx % 100 == 0:
            logger.info(f"Processing {idx} / {len(data)}")
        examples = dp["examples"]
        response = ""
        best_score = -math.inf
        for ix, example in enumerate(examples):
            score = get_score([example], tokenizer, model)[0]
            if score > best_score:
                best_score = score
                response = dp["examples"][ix]
        bstn_dp = {
            "prompt": dp["question"],
            "idx": dp["idx"],
            "response": response,
        }
        dps.append(bstn_dp)
    end_time = time.time()

    output_file = os.path.join(args.output_dir, f"limo_bstn_s.pt")
    torch.save((dps,eval_dps), output_file)
    logger.success(f"Saved in {output_file}, Cost {(end_time - start_time)/60:.3f} minutes.")


if __name__ == "__main__":
    main()
