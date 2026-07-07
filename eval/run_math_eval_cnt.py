
import os
import time
import wandb
import random
import argparse
from glob import glob
import pandas as pd
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from datetime import datetime
from loguru import logger

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

# from tools.eval_mmbench import eval_result
from evaluate import evaluate
from utils import (
    setup_seed, clean_gpu, read_json, cleanup,
    load_jsonl, save_jsonl, construct_prompt, is_multi_choice
)
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="/extrahome0/user/data/srl_math_benchmark/", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    # parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )

    parser.add_argument('--use_wandb', type=str, default=None, help="是否使用wandb")
    parser.add_argument('--wandb_project', type=str, default="ifdr", help="wandb项目")
    parser.add_argument('--checkpoint_file', type=str, help="模型文件路径")
    parser.add_argument('--pattern_name', type=str, default="step*", help="模式名称")
    parser.add_argument('--half', type=str, default="all", help="半量or全量评估")
    parser.add_argument('--order', type=str, default="", help="正序还是倒序")
    parser.add_argument('--do_zst', action="store_true", help="是否进行基础模型测试")

    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)

    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if args.reuse:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def run_eval(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    data_list = [data_name for data_name in args.data_names.split(",") if data_name]
    need_eval_data_list = []
    results = []
    if args.reuse:
        for data_name in data_list:
            # args.output_dir 模型checkpoint的位置
            out_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
            out_file =  f"{args.output_dir}/{data_name}/{out_prefix}_s{args.start}_e{args.end}.jsonl"
            out_metric_json = out_file.replace(".jsonl", f"_metrics.json")
            
            if os.path.exists(out_metric_json):
                logger.info(f"Skipping {data_name} because {out_metric_json} already exists.")
                results.append(read_json(out_metric_json))
            else:
                need_eval_data_list.append(data_name)
    
        if len(need_eval_data_list) == 0:
            logger.info("All datasets already evaluated. Exiting.")
            data_list.append("avg")
            results.append(
                {
                    "acc": sum([result["acc"] for result in results]) / len(results),
                }
            )
            return data_list, results

        else:
            return need_eval_data_list, ["need_eval"]

    else:
        need_eval_data_list = data_list

    return data_list, results


def main():
    args = parse_args()
    setup_seed(args.seed)

    if not args.do_zst and args.use_wandb:
        wandb_dir = os.path.join(args.checkpoint_file, "wandb/")
        if not os.path.exists(wandb_dir):
            args.use_wandb = False
        else:
            run_id = None
            for file in os.listdir(wandb_dir):
                if file.startswith("run"):
                    run_id = file.split("-")[-1]
                    wandb.init(
                        entity="pcl-zh",
                        project=args.wandb_project,
                        # name=run_name,
                        resume='must',
                        id=run_id
                    )
                    logger.info(f"use wandb:{run_id}")
                    break
            if run_id is None:
                args.use_wandb = False

    checkpoint_file = args.checkpoint_file
    model_name_or_path = args.model_name_or_path
    # 模型的位置
    if args.do_zst:
        single_file = True
        output_dir = os.path.join(checkpoint_file, "zst/")
        checkpoint_files = [model_name_or_path]
    else:
        single_file = False
        pattern = os.path.join(checkpoint_file, args.pattern_name)
        checkpoint_files = sorted(glob(pattern, recursive=True),
                                  reverse=False)
        if len(checkpoint_files) == 0:
            # single checkpoint test
            logger.debug("Eval Single Checkpoint")
            checkpoint_files = [checkpoint_file]
            single_file = True
            output_dir = os.path.join(checkpoint_file, "../merge_model/")
        else:
            idx = len(checkpoint_files) // 2
            if args.half == "all":
                pass
            elif args.half == "right":
                checkpoint_files = checkpoint_files[0:idx]
            else:
                checkpoint_files = checkpoint_files[idx:]
            logger.debug(f"Eval {args.half} mode {len(checkpoint_files)} checkpoints")
            output_dir = os.path.join(checkpoint_file, f"{args.half}_merge_model/")


    idx = len(checkpoint_files) // 2
    if args.order == "-":
        checkpoint_files = checkpoint_files[idx:]
    elif args.order == "+":
        checkpoint_files = checkpoint_files[0:idx]

    ckpt_metric = {}
    # checkpoint_files = [checkpoint_files[0], checkpoint_files[-1]]
    cnt = 0
    need_eval = {}
    for checkpoint_idx, checkpoint_file in enumerate(checkpoint_files):
        logger.critical(f"{args.split} with {args.half} mode {checkpoint_idx + 1}/{len(checkpoint_files)}")

        if args.do_zst:
            file = 'zst'
        elif "/" in args.pattern_name:
            file = "_".join(checkpoint_file.split("/")[-2:])
        else:
            file = checkpoint_file.split("/")[-1]
        logger.debug(f"Eval {file} Start")

        if not args.do_zst:
            save_dir = checkpoint_file
        else:
            save_dir = output_dir
        args.output_dir = os.path.join(save_dir, "eval")
        os.makedirs(args.output_dir, exist_ok=True)
        args.model_name_or_path = checkpoint_file

        data_list, results = run_eval(args)

        if "need_eval" in results:
            logger.warning(f"{checkpoint_file} need test with {data_list}")
            cnt += 1
            need_eval[file] = data_list

    logger.debug(f"need test {cnt}")
    logger.info(f"{need_eval}")

if __name__ == "__main__":
    main()
    # test()
