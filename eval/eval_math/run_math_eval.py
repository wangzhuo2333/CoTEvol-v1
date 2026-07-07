import os.path
import sys
import time
import yaml
import pickle
import json
from openai import OpenAI
from typing import Optional
from loguru import logger
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field

from transformers import HfArgumentParser
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# from evolution import EvolOpt, BatchEvolOptV1, BatchEvolEvalOptV1
from evolution import EvolOpt, BatchEvolOptV1
from fitness import extract_answer

import sys
from pathlib import Path
path = f"{Path.cwd()}"
if "USER_ID" in path:
    sys.path.append("/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2")
else:
    sys.path.append("/extrahome0/user/code/reason")
from eval.evaluate import evaluate
from eval.parser import parse_ground_truth
from eval.parser import extract_answer as true_extract_answer
from eval.grader import math_equal
from eval.data_loader import load_data

from eval.parser import run_execute

evol_funcs = {
    "1.0": EvolOpt,
    "1.2": BatchEvolOptV1,
    # "eval_evol": BatchEvolEvalOptV1
}

# CUDA_VISIBLE_DEVICES=0,1 vllm serve /extrahome0/HF_models/Qwen2.5-7B-Instruct --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager --port 8001
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./evol/run_math_evol.py --config_path ./evol/prompt_eval_evol/evoluation+.yaml

def majority_true(flag):
    return flag.count(True) > flag.count(False)

@dataclass
class EvolutionArguments:
    config_path: Optional[str] = field(default="/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/eval/eval_math/evoluation_infer.yaml", metadata={"help": "The configuration file to use."})
    # order: Optional[str] = field(default="+", metadata={"help": "倒序还是正常顺序."})


def main():
    # 读取配置
    logger.info("loading configuration")
    parser = HfArgumentParser(EvolutionArguments)
    args = parser.parse_args_into_dataclasses()[0]
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    for k, v in config.items():
        setattr(args, k, v)

    if args.data_name in ["math500"]:
        problem_name = "problem"
    elif args.data_name in ["amc23", "aime24"]:
        problem_name = "question"
    # elif args.data_name == "o1j":
    #     rep_name = "longCot"
    # elif args.data_name == "aime":
    #     rep_name = "answer"
    # elif args.data_name in ["math", "hard_math"]:
    #     rep_name = "gt_solution"
    # # 测试集
    # elif args.data_name == "amc23":
    #     rep_name = "answer"
    else:
        raise f"can't find {args.data_name}"

    # 加载分词器
    logger.info("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 加载模型
    if args.data_type == "base":
        model_path = "/inspire/hdd/global_user/USER_ID/user/models/Qwen2.5-7B-Instruct"
    else:
        model_path = args.model_path
    logger.info("building models")
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=len(available_gpus),
        # pipeline_parallel_size=pipeline_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.95
    )

    # 定义Evolution Opt
    logger.info("loading Evolution Operations")
    evolOpt = evol_funcs[args.version]
    eval_opt = evolOpt(
        args=args, vllm=llm, tokenizer=tokenizer
    )

    # 加载Reuse
    args.output_path = os.path.join(args.output_path, f"eval_evol/")
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f"{args.data_name}_{args.data_type}_{args.order}.pkl")
    if os.path.isfile(output_file) and not args.overwrite:
        with open(output_file, "rb") as file:
            dps = pickle.load(file)
            reuse_idx = len(dps)
    else:
        dps = []
        reuse_idx = 0
    logger.debug(f"Reuse {reuse_idx} data in {output_file}")

    # 加载数据
    train_data = load_data(args.data_name, "test", args.problem_path)
    # if args.problem_path.endswith("pt"):
    #     train_data = load_dataset(args.problem_path)["train"]
    # elif args.problem_path.endswith("pkl"):
    #     with open(args.problem_path, "rb") as file:
    #         train_data = pickle.load(file)
    # elif args.problem_path.endswith("json"):
    #     with open(args.problem_path, "r") as file:
    #         train_data = json.load(file)
    # elif args.problem_path.endswith("jsonl"):
    #     train_data = []
    #     with open(args.problem_path, "r") as file:
    #         for line in file:
    #             train_data.append(json.loads(line))
    # else:
    #     raise NotImplementedError
    # train_data = train_data[0:args.batch_size]

    # if args.data_name == "aime":
    #     df = train_data.to_pandas()
    #     df["question"] = df["Question"]
    #     df["answer"] = df["Answer"]
    #     filtered_df = df[df["Year"].isin(list(range(2016, 2024)))]
    #     train_data = Dataset.from_pandas(filtered_df)

    if args.order == "-":
        idxs = list(reversed(list(range(len(train_data)))))
        min_valid_idx = len(train_data) - reuse_idx  # e.g., 100 - 14 = 86
    else:
        idxs = list(range(len(train_data)))

    cost_times = []
    remain_batch = ((len(train_data)-reuse_idx) // args.batch_size)
    if len(idxs) % args.batch_size:
        remain_batch += 1
    # for idx in idxs:
    for i in range(0, len(idxs), args.batch_size):
        start_time = time.time()
        batch = idxs[i:i + args.batch_size]

        if args.order == "-" and batch[0] > min_valid_idx:
            # 当前 batch 完全在已访问区域，跳过
            continue
        elif args.order == "+" and batch[-1] < reuse_idx:
            # 当前 batch 完全在已访问区域，跳过
            continue

        # 如果部分 batch 超过 reuse_idx，可以截断
        if args.order == "-":
            ids = [x for x in batch if x <= min_valid_idx]
        else:
            ids = [x for x in batch if x >= reuse_idx]

        # if args.order == "-" and idxs[idx] > reuse_idx:
        #     continue
        # elif args.order == "+" and idxs[idx] < reuse_idx:
        #     continue
        # batch_size = args.batch_size if idx+args.batch_size < len(idxs) else len(idxs)-len(dps)
        # # logger.info(f"{idx}, {batch_size}, {len(dps)}, idxs: {len(idxs)}, {idx + batch_size}")
        # logger.info(f"Processing problem {idx}+{batch_size}")
        # ids = idxs[idx:idxs[idx + batch_size]] if batch_size == args.batch_size else idxs[idx:]

        def get_batch_data(data, ids, data_name="", problem_name=""):
            problems, answers, use_ids, level = [], [], [], []
            for idx in ids:
                dp = data[idx]
                # print(dp.keys())
                problem = dp[problem_name]
                # if data_name == "amc23":
                gt_ans = parse_ground_truth(dp, data_name)[-1]
                # elif "answer" in dp:
                #     gt_ans = dp["answer"]
                # else:
                #     gt_ans = extract_answer(dp[rep_name])
                if not gt_ans:
                    continue
                if "level" in dp:
                    level.append(dp["level"])
                problems.append(problem)
                answers.append(gt_ans)
                use_ids.append(idx)
            return problems, answers, use_ids, level

        problems, answers, use_ids, level = get_batch_data(train_data, ids, data_name=args.data_name, problem_name=problem_name)

        # output = eval_opt.pipeline(problem, gt_ans)
        eval_solutions, all_solutions = eval_opt.pipeline(problems, answers)
        for ix, eval_solution in enumerate(eval_solutions):
            dp = {}
            dp["evol_solution"] = eval_solution
            dp["all_solutions"] = all_solutions[ix]
            dp["idx"] = use_ids[ix]
            dp["problem"] = problems[ix]
            dp["answer"] = answers[ix]
            dp["level"] = level[ix] if level else None
            # dp["pred"] = dp["all_solutions"] # 用于测试
            dps.append(dp)

        with open(output_file, "wb") as file:
            pickle.dump(dps, file)

        end_time = time.time()
        cost_times.append((end_time-start_time)/60)
        mean_cost_time = sum(cost_times) / len(cost_times)
        remain_batch -= 1
        remain_time = remain_batch * mean_cost_time
        logger.debug(f"{i+args.batch_size} 个问题处理完成，耗时 {(cost_times[-1]):.2f} mins, "
                     f"预计还需要 {remain_time:.2f} mins ({remain_time/60:.2f} hs)")

    slv_cnt = 0
    maj_cnt = 0
    acc_flgs = []
    avg_total = []
    for dp in dps:
        # print(dp['problem'])
        solutions = dp["all_solutions"]
        # print(len(solutions))
        # print(dp['all_solutions'])
        dp["pred"] = []
        # dp["pred"].append(true_extract_answer(solutions, args.data_name))
        for solution in solutions:
            dp["pred"].append(true_extract_answer(solution, args.data_name))
        # acc_flg = [math_equal(pred, parse_ground_truth(dp, args.data_name)[-1]) for pred in dp['pred']]
        acc_flg = [math_equal(pred, dp["answer"]) for pred in dp['pred']]
        # acc_flg = [math_equal(pred, dp["answer"]) for pred in dp['pred']]
        # print(acc_flg)
        # print(len(acc_flg))
        if any(acc_flg):
        # if acc_flg[0]:
            slv_cnt += 1
        # 每个问题的答案平均正确率
        acc_flgs.append(sum(acc_flg) / len(acc_flg))
        # 多数投票
        if majority_true(acc_flgs):
            maj_cnt += 1
    print(sum(acc_flgs))
    print(len(dps))
    logger.info(f"data_name: {args.data_name}, num_samples: {len(dps)}, "
                f"all avg acc: {sum(acc_flgs)/len(dps)*100:.1f}, pass@1: {slv_cnt / len(dps)*100:.1f}, "
                f"maj@10: {maj_cnt / len(dps)*100:.1f}")
    # all_samples, result_json = evaluate(args.data_name, None, dps, execute=True)


def test():
    with open("/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/wz/math_infer/eval_evol/math500_base_+.pkl", "rb") as file:
        data = pickle.load(file)
    cnt = 0
    acc_flgs = []
    for dp in data:
        solutions = dp["all_solutions"]
        dp["pred"] = []
        for solution in solutions:
            dp["pred"].append(true_extract_answer(solution, "math500"))
        acc_flg = [math_equal(pred, parse_ground_truth(dp, "math500")[-1]) for pred in dp['pred']]
        if any(acc_flg):
            cnt += 1
        acc_flgs.append(sum(acc_flg)/len(acc_flg))
    logger.info(f"pass@10: {cnt/len(data)}")
    # acc_list = [sum(acc_flg)/len() for acc_flg in acc_flgs]
    logger.info(f"all avg acc: {sum(acc_flgs)/len(acc_flgs)}")
    all_samples, result_json = evaluate("math500", None, data, execute=True)
    logger.info(f"official: {result_json['acc']}")

if __name__ == "__main__":
    main()
    # test()