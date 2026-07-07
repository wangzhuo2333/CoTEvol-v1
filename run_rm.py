import argparse
import time

import os
import torch
from transformers import AutoModel, AutoTokenizer
from loguru import logger

from utils.general import setup_seed
from utils.read_data import load_train_dev_dataset
from utils.prompts import math_message


message = "You are a math teacher. Given a math problem, please use formal " \
          "mathematical expressions to provide the reasoning process step by step. The final aswer should be " \
          "formated as $\\boxed{{}}$. "


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/code/Research_with_user/reasoning/GA/reward_score/")
    parser.add_argument("--data_type", type=str, default="short")
    parser.add_argument("--data_name", type=str, default="s1k")
    parser.add_argument("--model_name_or_path", type=str, default="/extrahome0/HF_models/Qwen2.5-Math-RM-72B/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"/extrahome0/user/data/{args.data_name}/{args.data_name}_{args.data_type}_cot.pt"
    return args


def load_data(data_path):
    raw_data = torch.load(data_path, weights_only=False)
    data = []
    for dp in raw_data[0]:
        problem = dp["prompt"]
        solution = dp["response"]
        if "math_message" in dp:
            message = dp["math_message"]
        else:
            message = math_message
        example = [
            {"role": "system", "content": message},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ]
        data.append(
            {
                **dp, "example": [example]
            }
        )
    return data


def get_score(dps, tokenizer, model):
    scores = []
    for dp in dps:
        conversation_str = tokenizer.apply_chat_template(
            dp,
            tokenize=False,
            add_generation_prompt=False,
            max_length=3000
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

# def load_if_dataset(data_path):
#     dataset = torch.load(data_path, weights_only=False)
#     right, error = dataset[0], dataset[1]
#
#     def format_example(dp):
#         dp = list(dp)
#         problem, _ = dp[0]  # problem, gold_solution
#         dp.append([])  # dp[1] eval_solutions, dp[2] if-score, dp[3] format_example
#         for solution in dp[1]:
#             example = [
#                 {"role": "system", "content": math_message},
#                 {"role": "user", "content": problem},
#                 {"role": "assistant", "content": solution}
#             ]
#             dp[3].append(example)
#         return dp
#
#     new_right = []
#     for dp in right:
#         dp = format_example(dp)
#         new_right.append(dp)
#
#     new_error = []
#     for dp in error:
#         dp = format_example(dp)
#         new_error.append(dp)
#
#     return new_right, new_error
#     # return new_right[0:2], new_error[0:2]

def main():
    args = parse_args()
    setup_seed(args.seed)

    device = "auto" # the device to load the model onto
    logger.info(f"loading dataset from {args.data_dir}")
    # right, error = load_if_dataset(data_path)
    data = load_data(args.data_dir)

    logger.info(f"loading model and tokenizer from {args.model_name_or_path}")
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    start_time = time.time()
    for idx, dp in enumerate(data):
        if idx % 100 == 0:
            logger.info(f"Processing {idx} / {len(data)}")

        score = get_score(dp["example"], tokenizer, model)
        dp["score"] = score
    end_time = time.time()

    output_file = os.path.join(args.output_dir, f"s={args.seed}_{args.data_type}_score.pt")
    torch.save((data,), output_file)
    logger.success(f"Saved in {output_file}, Cost {(end_time - start_time)/60:.3f} minutes.")


if __name__ == "__main__":
    main()
