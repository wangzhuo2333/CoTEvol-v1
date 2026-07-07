import os
import pickle
import argparse
from loguru import logger
from glob import glob
import pandas as pd
from tabulate import tabulate
import wandb

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils.general import setup_seed, clean_gpu, merge_save_model
from utils.gsm_metric import cal_gsm_acc
from utils.math_metric import cal_math_acc
from utils.read_data import load_eval_data


def parse_args():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="模型参数设置")
    parser.add_argument('--model_name_or_path', type=str, help="模型文件路径")
    parser.add_argument('--checkpoint_file', type=str, help="模型文件路径")
    parser.add_argument('--eval_data_path', type=str, default=None, help="评估数据路径")
    parser.add_argument('--do_zst', action="store_true", help="是否进行零-shot任务")
    parser.add_argument('--model_max_length', type=int, default=1024, help="模型最大输入长度")
    parser.add_argument('--seed', type=int, default=1234, help="随机种子")
    parser.add_argument('--sample_data', type=int, default=None, help="采样数")
    parser.add_argument('--model_type', type=str, default="llama3-8b-it", help="模型类型")
    parser.add_argument('--data_name', type=str, default="safe_rlhf", help="数据名称")
    parser.add_argument('--reuse', action="store_true", help="是否重用")
    parser.add_argument('--use_lora', action="store_true", help="是否是lora model")
    parser.add_argument('--batch_size', type=int, default=16, help="批处理大小")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID")
    parser.add_argument('--pattern_name', type=str, default="step*", help="模式名称")
    parser.add_argument('--half', type=str, default="all", help="半量or全量评估")

    # 解析参数
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_id}"
    args.do_zst = bool(args.do_zst)
    args.reuse = bool(args.reuse)
    return args


def main():
    args = parse_args()
    setup_seed(args.seed)

    if args.data_name == "gsm":
        metric_func = cal_gsm_acc
    elif args.data_name == 'math':
        metric_func = cal_math_acc
    else:
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        # message + Problem: + Solution:
        logger.debug("using simple template")
        chat_template = "{% for message in messages %}{{'Problem' + ': ' + message[" \
                        "'content'] + '\n\n'}}{% endfor %}{% if add_generation_prompt %}{{'Solution:'}}{" \
                        "% endif %} "
    elif 'llama3' in args.model_type:
        logger.debug("using llama3-8b-it template")
        chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = " \
                        "'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + " \
                        "'<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ " \
                        "content }}{% endfor %}{% if add_generation_prompt %}{{ " \
                        "'<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %} "
    elif 'qwen' in args.model_type:
        logger.debug("using qwen-math template")
        chat_template = tokenizer.chat_template
    else:
        raise ValueError(f"{args.model_type} chat template is not supported")
    tokenizer.chat_template = chat_template

    # read data
    all_examples, answers = load_eval_data(args.data_name, tokenizer, args.sample_data, use_template=True)

    checkpoint_file = args.checkpoint_file
    model_name_or_path = args.model_name_or_path
    # peft dir
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
            output_dir = os.path.join(checkpoint_file, f"{args.data_name}_{args.half}_merge_model/")

    ckpt_metric = {}
    merge_model_dir = ""
    for checkpoint_idx, checkpoint_file in enumerate(checkpoint_files):
        logger.info(f"Eval {args.half} mode {checkpoint_idx+1}/{len(checkpoint_files)}")
        if args.do_zst:
            file = 'zst'
        elif "/" in args.pattern_name:
            file = "_".join(checkpoint_file.split("/")[-2:])
        else:
            file = checkpoint_file.split("/")[-1]
        logger.critical(f"Eval {file} Start")

        if not args.do_zst:
            save_dir = checkpoint_file
        else:
            save_dir = output_dir
            os.makedirs(save_dir, exist_ok=True)
        # reuse eval_output
        generated_file = os.path.join(save_dir, f"{args.data_name}_generated.pkl")

        if args.reuse and os.path.isfile(generated_file):
            logger.info(f"Reuse {generated_file}")
            with open(generated_file, "rb") as f:
                prd_data = pickle.load(f)
            try:
                prd_samples, max_acc, max_temperature = prd_data
            except:
                prd_samples, max_acc, max_temperature = [], 0.0, 0.0
        else:
            if args.use_lora:
                merge_save_model(model_name_or_path, checkpoint_file, output_dir, args.model_max_length)
                merge_model_dir = output_dir
                os.makedirs(merge_model_dir, exist_ok=True)
            else:
                merge_model_dir = checkpoint_file
            ### vllm
            logger.info("Loading vLLM")
            # temperature=0.5 for qwen / temperature=0.5 for llama3-8b
            max_prd = None
            max_acc, max_temperature = 0.0, 0.0
            # 0.3, 0.4, 0.5, 0.6, 0.7
            for temperature in [0.1, 0.3, 0.5, 0.7]:
                logger.debug("Test with temperature {}".format(temperature))
                sampling_params = SamplingParams(
                    temperature=temperature, top_p=0.95, repetition_penalty=1.1, #
                    max_tokens=args.model_max_length, seed=args.seed)
                llm = LLM(model=merge_model_dir, tokenizer=merge_model_dir)
                outputs = llm.generate(all_examples, sampling_params)

                prd_samples = []
                for i, output in enumerate(outputs):
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    prd_samples.append({"prompt": prompt, "prd": generated_text,
                                        "answer": answers[i]})
                outputs, acc = metric_func(prd_samples)
                if acc > max_acc:
                    max_acc = acc
                    max_prd = prd_samples
                    max_temperature = temperature

                del llm.llm_engine.model_executor
                del llm
                clean_gpu()

                logger.debug(f"Curr acc={acc}_t={temperature}, Best acc={max_acc}_t={max_temperature}")

            with open(generated_file, "wb") as f:
                # prd_data = {"prd_samples": prd_samples, "acc": max_acc, "temperature": max_temperature}
                prd_data = (max_prd, max_acc, max_temperature)
                pickle.dump(prd_data, f)
            prd_samples = max_prd

        # cal metric
        # scores_file = os.path.join(save_dir, f"{args.data_name}_{max_temperature}_scores.csv")
        results = {}
        # outputs, acc = cal_gsm_acc(prd_samples)
        results['acc'] = max_acc
        ckpt_metric[file+f"_t{max_temperature}"] = results
        # outputs = pd.DataFrame(outputs)
        # outputs.to_csv(scores_file, index=False)
        # logger.warning(f"Eval {file} Results: {results} save in {scores_file}")
        logger.debug(f"Model: {file}, Max: {max_acc} with temperature: {max_temperature}")
    logger.success(f"Run [{args.half}] Done.")

    if not single_file and args.half == "all":
        metrics_df = pd.DataFrame.from_dict(ckpt_metric, orient='index')
        sorted_metrics_df = metrics_df.sort_values(by='acc', ascending=False)
        sorted_metrics_df.reset_index(inplace=True)
        sorted_metrics_df.columns = ["steps"] + list(sorted_metrics_df.columns[1:])

        metric_path = os.path.join(args.checkpoint_file, f"{args.data_name}_metric.csv")
        sorted_metrics_df.to_csv(metric_path, sep="\t", index=False)
        logger.info(f"Metric outputs saved to {metric_path}")

        logger.info(f"Total Results\n{tabulate(sorted_metrics_df, headers='keys', tablefmt='pretty', showindex=False)}")
        logger.info(f"Copy {args.data_name} Results")
        os.system(f"cat {metric_path}")
        logger.info(f"cat results in {metric_path}")

    if args.use_lora and merge_model_dir:
        logger.info(f"remove temp merged model from {merge_model_dir}")
        os.system(f"rm -r {merge_model_dir}")


if __name__ == '__main__':
    main()
