
import time
import wandb

import argparse
from glob import glob
import pandas as pd

from tabulate import tabulate
from tqdm import tqdm
from datetime import datetime
from loguru import logger
# from openai import OpenAI
# from transformers import AutoTokenizer, AutoModelForCausalLM

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

from evolution import EvolOpt, BatchEvolOptV1
from fitness import extract_answer

evol_funcs = {
    "1.0": EvolOpt,
    "1.2": BatchEvolOptV1
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./srl_math_benchmark/srl_math_benchmark", type=str)
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
    
    parser.add_argument("--gen_responses_prompt", default="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main-v2/ifdr-main-qwen1.5/eval/evol_eval/mistral-7b-it-prompt/gen_resp.txt", type=str)
    parser.add_argument("--critic_prompt", default="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main-v2/ifdr-main-qwen1.5/eval/evol_eval/mistral-7b-it-prompt/critic.txt", type=str)
    parser.add_argument("--author_prompt", default="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main-v2/ifdr-main-qwen1.5/eval/evol_eval/mistral-7b-it-prompt/author.txt", type=str)
    parser.add_argument("--mutation_prompt", default="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main-v2/ifdr-main-qwen1.5/eval/evol_eval/mistral-7b-it-prompt/mutation.txt", type=str)


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
        need_eval_data_list = data_list

    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.95
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # infer & eval
    for data_name in need_eval_data_list:
        results.append(eval(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )
    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))

    # del llm.llm_engine.model_executor
    # del llm
    clean_gpu(llm)  # 只能单卡中使用
    # cleanup()
    # time.sleep(10)
    return data_list, results


def eval(llm, tokenizer, data_name, args):
    # 加载分词器
    logger.info("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # 定义Evolution Opt
    logger.info("loading Evolution Operations")
    evolOpt = evol_funcs["1.2"]
    eval_opt = evolOpt(
        args=args, llm=llm, tokenizer=tokenizer
    )
      
    examples, processed_samples, out_file = prepare_data(data_name, args)
    logger.info(f"data: {data_name}, remain samples: {len(examples)}")

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    # single_question
    # samples = []
    # count = 0
    # for example in tqdm(examples, total=len(examples)):
    #     idx = example["idx"]

    #     # parse question and answer
    #     example["question"] = parse_question(example, data_name)
    #     if example["question"] == "":
    #         continue
    #     gt_cot, gt_ans = parse_ground_truth(example, data_name)
    #     example["gt_ans"] = gt_ans
    #     # print(llm)
    #     _, all_solutions = eval_opt.pipeline(example['question'])
    #     print(len(all_solutions))
    #     eval_solutions = all_solutions[4:]
    #     # print(len(eval_solutions[0]))
    #     for eval_solution in eval_solutions:
    #         pred_ans = extract_answer(eval_solution)
    #         if pred_ans:
    #             if pred_ans == gt_ans:
    #                 count += 1
    #                 break
    #             elif gt_ans in eval_solution:
    #                 count += 1
    #                 break
    # print(count/len(examples))
    # return count/len(examples)
    
    # batch questions
    samples = []
    gt = []
    count = 0
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        samples.append(examples)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        gt.append(gt_ans)
    _, all_solutions = eval_opt.pipeline(samples)
    print(len(all_solutions))
    # all_solutions = sorted(all_solutions, key=lambda x: int(x.request_id))
    for preds, gold in zip(all_solutions, gt):
        print(len(preds))
        for pred in preds[4:]:
            pred_ans = extract_answer(pred)
            if pred_ans:
                if pred_ans == gold:
                    count += 1
                    # print('pred:',pred)
                    # print('$$$$$$$$$$$$$$$$$$$')
                    # print('gold:',gold)
                    break
                elif gt_ans in pred:
                    count += 1
                    # print('pred:',pred)
                    # print('$$$$$$$$$$$$$$$$$$$')
                    # print('gold:',gold)
                    break

    # print(len(all_solutions))
    # eval_solutions = all_solutions[4:]
    # # print(len(eval_solutions[0]))
    # for eval_solution in eval_solutions:
    #     pred_ans = extract_answer(eval_solution)
    #     if pred_ans:
    #         if pred_ans == gt_ans:
    #             count += 1
    #             break
    #         elif gt_ans in eval_solution:
    #             count += 1
    #             break
    print(count/len(examples))
    return count/len(examples)
    
    # break
    #     full_prompt = construct_prompt(example, data_name, args)

    #     if idx == args.start:
    #         print(full_prompt)

    #     sample = {
    #         "idx": idx,
    #         "question": example["question"],
    #         "gt_cot": gt_cot,
    #         "gt": gt_ans,
    #         "prompt": full_prompt,
    #     }

    #     # add remain fields
    #     for key in [
    #         "level",
    #         "type",
    #         "unit",
    #         "solution_type",
    #         "choices",
    #         "solution",
    #         "ques_type",
    #         "ans_type",
    #         "answer_type",
    #         "dataset",
    #         "subfield",
    #         "filed",
    #         "theorem",
    #         "answer",
    #     ]:
    #         if key in example:
    #             sample[key] = example[key]
    #     samples.append(sample)

    # # repeat n times
    # input_prompts = [
    #     sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    # ]
    # if args.apply_chat_template:
    #     input_prompts = [
    #         tokenizer.apply_chat_template(
    #             [{"role": "user", "content": prompt.strip()}],
    #             tokenize=False,
    #             add_generation_prompt=True,
    #         )
    #         for prompt in input_prompts
    #     ]
    # remain_prompts = input_prompts
    # remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    # end_prompts = []

    # max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    # stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    # if args.prompt_type in ["cot"]:
    #     stop_words.append("\n\nQuestion:")
    # if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
    #     stop_words.extend(["\n\n---", "```output"])
    # elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
    #     stop_words.extend(["Instruction", "Response"])
    # elif "jiuzhang" in args.prompt_type:
    #     stop_words.append("\n\n## Question")
    # elif "numina" in args.prompt_type:
    #     stop_words.append("\n### Problem")
    # elif "pure" in args.prompt_type:
    #     stop_words.append("\n\n\n")

    # # start inference
    # # measure time use
    # start_time = time.time()
    # for epoch in range(max_func_call):
    #     print("-" * 20, "Epoch", epoch)
    #     current_prompts = remain_prompts
    #     if len(current_prompts) == 0:
    #         break
    #     # assert False
    #     # if epoch > 0 :
    #     #     assert False
    #     # get all outputs
    #     prompts = [item[1] for item in current_prompts]
    #     if args.use_vllm:
    #         outputs = llm.generate(
    #             prompts,
    #             SamplingParams(
    #                 temperature=args.temperature,
    #                 top_p=args.top_p,
    #                 max_tokens=args.max_tokens_per_call,
    #                 n=1,
    #                 stop=stop_words,
    #                 stop_token_ids=(
    #                     [151645, 151643]
    #                     if "qwen2" in args.model_name_or_path.lower()
    #                     else None
    #                 ),
    #             ),
    #         )
    #         # eval_solution, all_solutions = eval_opt.pipeline(problem)
    #         outputs = sorted(
    #             outputs, key=lambda x: int(x.request_id)
    #         )  # sort outputs by request_id
    #         outputs = [output.outputs[0].text for output in outputs]
    #     else:
    #         outputs = generate_completions(
    #             model=llm,
    #             tokenizer=tokenizer,
    #             prompts=prompts,
    #             max_new_tokens=args.max_tokens_per_call,
    #             batch_size=16,
    #             stop_id_sequences=stop_words,
    #         )

    #     assert len(outputs) == len(current_prompts)

    #     # process all outputs
    #     remain_prompts = []
    #     remain_codes = []
    #     for (i, query), output in zip(current_prompts, outputs):
    #         output = output.rstrip()
    #         query += output
    #         if args.prompt_type == "pal":
    #             remain_prompts.append((i, query))
    #             if "```python" in output:
    #                 output = extract_program(query)
    #             remain_codes.append(output)
    #         elif args.prompt_type == "cot":
    #             end_prompts.append((i, query))
    #         # elif "boxed" not in output and output.endswith("```"): # disable code execution 
    #         #     program = extract_program(query)
    #         #     remain_prompts.append((i, query))
    #         #     remain_codes.append(program)
    #         else:
    #             end_prompts.append((i, query))

    #     # execute the remain prompts
    #     # assert len(remain_codes)==0
    #     remain_results = executor.batch_apply(remain_codes)
    #     for k in range(len(remain_prompts)):
    #         # assert False
    #         i, query = remain_prompts[k]
    #         res, report = remain_results[k]
    #         exec_result = res if res else report
    #         if "pal" in args.prompt_type:
    #             exec_result = "\\boxed{" + exec_result + "}"
    #         exec_result = f"\n```output\n{exec_result}\n```\n"
    #         query += exec_result
    #         # not end
    #         if epoch == max_func_call - 1:
    #             query += "\nReach max function call limit."
    #         remain_prompts[k] = (i, query)

    # # unsolved samples
    # logger.info(f"Unsolved samples: {len(remain_prompts)}")
    # end_prompts.extend(remain_prompts)
    # # sort by idx
    # end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # # remove input_prompt from end_prompt
    # codes = []
    # assert len(input_prompts) == len(end_prompts)
    # for i in range(len(input_prompts)):
    #     _, end_prompt = end_prompts[i]
    #     code = end_prompt.split(input_prompts[i])[-1].strip()
    #     for stop_word in stop_words:
    #         if stop_word in code:
    #             code = code.split(stop_word)[0].strip()
    #     codes.append(code)

    # # extract preds
    # results = [
    #     run_execute(executor, code, args.prompt_type, data_name) for code in codes
    # ]
    # time_use = time.time() - start_time

    # # put results back to examples
    # all_samples = []
    # for i, sample in enumerate(samples):
    #     code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
    #     result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
    #     preds = [item[0] for item in result]
    #     reports = [item[1] for item in result]
    #     for j in range(len(preds)):
    #         if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
    #             "A",
    #             "B",
    #             "C",
    #             "D",
    #             "E",
    #         ]:
    #             preds[j] = choice_answer_clean(code[j])
    #         else:
    #             if sample["gt"] is None or preds[j] is None:
    #                 continue
    #             if is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
    #                 # remove any non-choice char
    #                 preds[j] = "".join(
    #                     [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
    #                 )

    #     sample.pop("prompt")
    #     sample.update({"code": code, "pred": preds, "report": reports})
    #     all_samples.append(sample)

    # # add processed samples
    # all_samples.extend(processed_samples)
    # all_samples, result_json = evaluate(
    #     samples=all_samples,
    #     data_name=data_name,
    #     prompt_type=args.prompt_type,
    #     execute=True,
    # )

    # # save outputs
    # if len(processed_samples) < len(all_samples) and args.save_outputs:
    #     save_jsonl(all_samples, out_file)

    # result_json["time_use_in_second"] = time_use
    # result_json["time_use_in_minite"] = (
    #     f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    # )

    # metric_file = out_file.replace(".jsonl", f"_metrics.json")
    # logger.info(f"Eval {data_name} done, saved metrics to {metric_file}")
    # with open(
    #     metric_file, "w"
    # ) as f:
    #     json.dump(result_json, f, indent=4)
    # return result_json


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
        metric_results = {}
        for idx, data_name in enumerate(data_list):
            metric_results[data_name] = results[idx]["acc"]
        ckpt_metric[file] = metric_results

    if not single_file and args.half == "all":
        metrics_df = pd.DataFrame.from_dict(ckpt_metric, orient='index')
        sorted_metrics_df = metrics_df.sort_values(by='avg', ascending=False)
        sorted_metrics_df.reset_index(inplace=True)
        sorted_metrics_df.columns = ["steps"] + list(sorted_metrics_df.columns[1:])

        # 计算搜参model中最大的那个参数
        data_names = sorted_metrics_df.columns[1:-1]
        max_values = sorted_metrics_df[data_names].max()
        avg_values = max_values.mean()
        max_line = {
            "steps": "max",
            **max_values.to_dict(),
            "avg": avg_values
        }
        max_row = pd.DataFrame([max_line])
        sorted_metrics_df = pd.concat([max_row, sorted_metrics_df], ignore_index=True)

        file_name = f"s={args.seed}_t={args.temperature}_{args.split}_metric.csv"
        metric_path = os.path.join(args.checkpoint_file, file_name)
        sorted_metrics_df.to_csv(metric_path, sep="\t", index=False)
        logger.info(f"Metric outputs saved to {metric_path}")

        logger.info(f"Total Results\n{tabulate(sorted_metrics_df, headers='keys', tablefmt='pretty', showindex=False)}")
        logger.info(f"Copy Results")
        os.system(f"cat {metric_path}")
        logger.info(f"See in {metric_path}")

        if args.use_wandb:
            table = wandb.Table(dataframe=sorted_metrics_df)
            wandb.log({file_name: table})


def test():
    args = parse_args()
    setup_seed(args.seed)
    data_name = "math500"
    args.output_dir = "/extrahome0/user/output/test/"
    examples, processed_samples, out_file = prepare_data("math500", args)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]
        # parse question and answer
        if idx == args.start:
            example["question"] = parse_question(example, data_name)
            if example["question"] == "":
                continue
            gt_cot, gt_ans = parse_ground_truth(example, data_name)
            example["gt_ans"] = gt_ans
            full_prompt = construct_prompt(example, data_name, args)

            print(f"question: {example['question']}")
            print(f'gt_ans: {example["gt_ans"]}')
            print(f"full_prompt: {full_prompt}")


if __name__ == "__main__":
    main()
    # test()
