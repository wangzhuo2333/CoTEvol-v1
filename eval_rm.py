import argparse
import time

import os

import pickle
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from loguru import logger
from tabulate import tabulate
from glob import glob

from utils.general import setup_seed
from utils.read_data import load_train_dev_dataset
from utils.prompts import math_message, math_thought_message


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    # parser.add_argument("--output_dir", type=str,
    #                     default="/code/Research_with_user/reasoning/GA/reward_score/")
    parser.add_argument("--message_key", type=str, default="math_thought_message") # math_message
    parser.add_argument("--pattern_name", type=str, default="*tst_generated.pkl")
    parser.add_argument("--model_name_or_path", type=str,
                        default="/extrahome0/HF_models/Qwen2.5-Math-RM-72B/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--reuse', action="store_true", help="是否重用")
    args = parser.parse_args()

    return args


def load_data(data_path, message_key=None):
    # raw_data = torch.load(data_path, weights_only=False)
    with open(data_path, "rb") as f:
        raw_data = pickle.load(f)
    data = []
    for dp in raw_data:
        example = []
        problem = dp["prompt"]
        solution = dp["prd"]
        if message_key:
            if message_key in dp:
                message = dp[message_key]
            elif message_key == "math_thought_message":
                message = math_thought_message
            else:
                message = math_message
            example.append({"role": "system", "content": message})
        example.extend([
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ])
        data.append(
            {
                **dp, "example": example
            }
        )
    return data


def get_score(example, tokenizer, model):
    conversation_str = tokenizer.apply_chat_template(
        example,
        tokenize=False,
        add_generation_prompt=False,
        max_length=4000
    )
    input_ids = tokenizer.encode(
        conversation_str,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)

    outputs = model(input_ids=input_ids)
    score = outputs[0].cpu().item()

    return score


def main():
    args = parse_args()
    setup_seed(args.seed)

    device = "auto" # the device to load the model onto
    logger.info(f"loading model and tokenizer from {args.model_name_or_path}")
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    pattern = os.path.join(args.data_dir, args.pattern_name)
    data_files = sorted(glob(pattern, recursive=True), reverse=False)
    single_file = True if len(data_files) < 2 else False

    ckpt_metric = {}
    for idx, data_file in enumerate(data_files):
        if "/" in args.pattern_name:
            file_name = "_".join(data_file.split("/")[-3:]).replace("_tst_generated.pkl", "")
        else:
            file_name = data_file.split("/")[-2:]
        temp = file_name.split("_")[-1]
        output_dir = "/".join(data_file.split("/")[:-1])
        output_file = os.path.join(output_dir, f"s={args.seed}_{temp}_score.pkl")

        start_time = time.time()
        logger.debug(f"Eval {file_name} RM ({idx+1} / {len(data_files)})")
        data = load_data(data_file, message_key=args.message_key)
        scores = []
        if args.reuse and os.path.isfile(output_file):
            with open(output_file, "rb") as f:
                score_data = pickle.load(f)
            for i, dp in enumerate(score_data):
                scores.append(dp["score"])
                data[i]["score"] = dp["score"]
        for ix, dp in enumerate(data):
            if "score" in dp:
                continue
            if ix > 0 and ix % 100 == 0:
                logger.info(f"Processing {ix} / {len(data)}")
                with open(output_file, "wb") as f:
                    pickle.dump(data, f)
            score = get_score(dp["example"], tokenizer, model)
            dp["score"] = score
            scores.append(score)
        end_time = time.time()

        avg =  round(sum(scores) / len(scores), 3)
        ckpt_metric[file_name] = {"score": avg}
        logger.info(f"Eval {file_name} done, score: {avg}, "
                     f"cost time: {(end_time - start_time)/60:.3f} minutes")
        with open(output_file, "wb") as f:
            pickle.dump(data, f)

    if not single_file:
        metrics_df = pd.DataFrame.from_dict(ckpt_metric, orient='index')
        sorted_metrics_df = metrics_df.sort_values(by='score', ascending=False)
        sorted_metrics_df.reset_index(inplace=True)
        sorted_metrics_df.columns = ["steps"] + list(sorted_metrics_df.columns[1:])

        file_name = f"s={args.seed}_{temp}_score_metric.csv"
        metric_path = os.path.join(args.data_dir, file_name)
        sorted_metrics_df.to_csv(metric_path, sep="\t", index=False)
        logger.info(f"Metric outputs saved to {metric_path}")

        logger.info(f"Total Results\n{tabulate(sorted_metrics_df, headers='keys', tablefmt='pretty', showindex=False)}")
        logger.info(f"Copy Results")
        os.system(f"cat {metric_path}")
        logger.info(f"See in {metric_path}")

if __name__ == "__main__":
    main()