import os.path
import sys
import time
import yaml
import pickle
from openai import OpenAI
from typing import Optional
from loguru import logger
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field

from transformers import HfArgumentParser
from transformers import AutoTokenizer

from evolution import EvolOpt, EvolOptV1_1
from fitness import extract_answer

evol_funcs = {
    "1.0": EvolOpt,
    "1.1": EvolOptV1_1
}

# CUDA_VISIBLE_DEVICES=0,1 vllm serve /extrahome0/HF_models/Qwen2.5-7B-Instruct --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager --port 8001
@dataclass
class EvolutionArguments:
    config_path: Optional[str] = field(default="./evoluation+.yaml", metadata={"help": "The configuration file to use."})
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

    if args.data_name in ["s1k", "s1k_v"]:
        rep_name = "attempt"
    elif args.data_name == "s1k_v1.1":
        rep_name = 'deepseek_attempt'
    elif args.data_name == "limo":
        rep_name = "solution"
    elif args.data_name == "o1j":
        rep_name = "longCot"
    elif args.data_name == "aime":
        rep_name = "answer"
    elif args.data_name == 'math7500':
        rep_name = "gt_answer"
    else:
        raise f"can't find {args.data_name}"

    # 加载分词器
    logger.info("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 加载客户端
    logger.info("building client")
    client = OpenAI(
        base_url=args.base_url,  # vLLM服务的地址 端口1/2 8000/8012
        api_key=args.api_key,  # 如果设置了API密钥需要填写,
        # repetition_penalty=1.1,
    )

    # 定义Evolution Opt
    logger.info("loading Evolution Operations")
    evolOpt = evol_funcs[args.version]
    eval_opt = evolOpt(
        args=args, client=client, tokenizer=tokenizer
    )

    # 加载预测数据
    args.output_path = os.path.join(args.output_path, f"evol_{args.data_name}/{args.data_type}/")
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f"evol_{args.data_name}_{args.order}.pkl")
    if os.path.isfile(output_file):
        with open(output_file, "rb") as file:
            dps = pickle.load(file)
            reuse_idx = len(dps)
    else:
        dps = []
        reuse_idx = 0
    logger.debug(f"Reuse {reuse_idx} data in {output_file}")

    try:
        train_data = load_dataset(args.problem_path)["train"]
    except:
        with open(args.problem_path, "rb") as file:
            train_data = pickle.load(file)
    if args.data_name == "aime":
        df = train_data.to_pandas()
        df["question"] = df["Question"]
        df["answer"] = df["Answer"]
        filtered_df = df[df["Year"].isin(list(range(2016, 2024)))]
        train_data = Dataset.from_pandas(filtered_df)

    if args.order == "-":
        idxs = list(reversed(list(range(len(train_data)))))
        reuse_idx = len(train_data) - reuse_idx
    else:
        idxs = list(range(len(train_data)))

    cost_times = []
    for idx in idxs:
        start_time = time.time()
        if args.order == "-" and idx > reuse_idx:
            continue
        elif args.order == "+" and idx < reuse_idx:
            continue

        # logger.debug(f"处理第 {idx} 个问题")
        dp = train_data[idx]
        problem = dp["question"]
        if "answer" in dp:
            gt_ans = dp["answer"]
        else:
            gt_ans = extract_answer(dp[rep_name])
        if not gt_ans:
            continue

        # output = eval_opt.pipeline(problem, gt_ans)
        eval_solution, all_solutions = eval_opt.pipeline(problem, gt_ans)
        dp["evol_solution"] = eval_solution
        dp["all_solutions"] = all_solutions
        dp["idx"] = idx
        dps.append(dp)

        with open(output_file, "wb") as file:
            pickle.dump(dps, file)

        end_time = time.time()
        cost_times.append((end_time-start_time)/60)
        mean_cost_time = sum(cost_times) / len(cost_times)
        remain_time = abs(len(train_data)-len(dps))*mean_cost_time
        logger.debug(f"第 {idx} 个问题处理完成，耗时 {(cost_times[-1]):.2f} mins, "
                     f"预计还需要 {remain_time:.2f} mins ({remain_time/24:.2f} hs)")


def test():
    # 读取配置
    logger.info("loading configuration")
    parser = HfArgumentParser(EvolutionArguments)
    args = parser.parse_args_into_dataclasses()[0]
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    for k, v in config.items():
        setattr(args, k, v)

    # 加载分词器
    logger.info("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 加载客户端
    logger.info("building client")
    client = OpenAI(
        base_url=args.base_url,  # vLLM服务的地址 端口1/2 8000/8012
        api_key=args.api_key,  # 如果设置了API密钥需要填写,
        # repetition_penalty=1.1,
    )

    # 定义Evolution Opt
    logger.info("loading Evolution Operations")
    evolOpt = evol_funcs[args.version]
    eval_opt = evolOpt(
        args=args, client=client, tokenizer=tokenizer
    )
    # test_problem
    problem = [("A square is inscribed in a right triangle with legs of lengths 6 and 8, such that the square shares "
               "the right angle with the triangle. Find the side length of the square."),
               ("Given a rational number, write it as a fraction in lowest terms and calculate the product of the "
                "resulting numerator and denominator. For how many rational numbers between 0 and 1 will $20_{}^{}!$ be the resulting product?")
               ]
    gt_ans = ['12.0', '128']
    eval_opt.pipeline(problem, gt_ans)

    # solutions = eval_opt.init_population(problem)
    # solution_1 = "To find the side length of the square inscribed in a right triangle with legs of lengths 6 and 8, " \
    #              "we start by visualizing the problem. Let's denote the right triangle as \\(\\triangle ABC\\) with " \
    #              "\\( \\angle C = 90^\\circ \\), \\(AC = 6\\), and \\(BC = 8\\). The square is inscribed such that " \
    #              "one of its vertices is at \\(C\\) and the other two vertices lie on \\(AC\\) and \\(BC\\). Let the " \
    #              "side length of the square be \\(s\\).\n\nWe can place the square such that one of its sides lies " \
    #              "along the legs of the triangle. This means that the square cuts off two smaller right triangles " \
    #              "from the original triangle. The legs of these smaller right triangles are \\(6-s\\) and \\(8-s\\), " \
    #              "and the hypotenuse is the same as the original hypotenuse minus the side of the square.\n\nThe area " \
    #              "of the original triangle can be calculated as:\n\\[\n\\text{Area} = \\frac{1}{2} \\times 6 \\times " \
    #              "8 = 24.\n\\]\n\nThe area can also be expressed as the sum of the area of the square and the areas " \
    #              "of the two smaller right triangles:\n\\[\n\\text{Area} = s^2 + \\frac{1}{2} \\times s \\times (6-s) " \
    #              "+ \\frac{1}{2} \\times s \\times (8-s).\n\\]\n\nSimplifying the right-hand side:\n\\[\n24 = s^2 + " \
    #              "\\frac{1}{2} s (6 + 8 - s) = s^2 + \\frac{1}{2} s (14 - s) = s^2 + 7s - \\frac{1}{2} s^2 = \\frac{" \
    #              "1}{2} s^2 + 7s.\n\\]\n\nMultiplying through by 2 to clear the fraction:\n\\[\n48 = s^2 + " \
    #              "14s.\n\\]\n\nRearranging the equation into standard quadratic form:\n\\[\ns^2 + 14s - 48 = " \
    #              "0.\n\\]\n\nWe solve this quadratic equation using the quadratic formula \\(s = \\frac{-b \\pm " \
    #              "\\sqrt{b^2 - 4ac}}{2a}\\), where \\(a = 1\\), \\(b = 14\\), and \\(c = -48\\):\n\\[\ns = \\frac{-14 " \
    #              "\\pm \\sqrt{14^2 - 4 \\cdot 1 \\cdot (-48)}}{2 \\cdot 1} = \\frac{-14 \\pm \\sqrt{196 + 192}}{2} = " \
    #              "\\frac{-14 \\pm \\sqrt{388}}{2} = \\frac{-14 \\pm 2\\sqrt{97}}{2} = -7 \\pm \\sqrt{" \
    #              "97}.\n\\]\n\nSince \\(s\\) must be a positive length, we take the positive root:\n\\[\ns = -7 + " \
    #              "\\sqrt{97}.\n\\]\n\nTo verify, we can use the geometric relationship. The side length \\(s\\) can " \
    #              "also be found by considering the similar triangles formed. The ratio of the sides of the smaller " \
    #              "triangles to the original triangle is the same as the ratio of the side of the square to the leg of " \
    #              "the triangle minus the side of the square. This gives us:\n\\[\n\\frac{s}{6-s} = \\frac{8-s}{" \
    #              "s}.\n\\]\n\nCross-multiplying gives:\n\\[\ns^2 = (6-s)(8-s) = 48 - 14s + s^2.\n\\]\n\nSimplifying, " \
    #              "we get:\n\\[\n0 = 48 - 14s \\implies s = \\frac{48}{14} = \\frac{24}{7}.\n\\]\n\nThus, " \
    #              "the side length of the square is:\n\\[\n\\boxed{\\frac{24}{7}}.\n\\] "
    # solution_2 = 'To find the side length of the square inscribed in a right triangle with legs of lengths 6 and 8, ' \
    #              'we start by visualizing the problem. Let the right triangle have vertices at \\( (0,0) \\), \\( (6,' \
    #              '0) \\), and \\( (0,8) \\). The square will have one vertex at the right angle of the triangle, ' \
    #              'and its sides will be parallel to the legs of the triangle.\n\nLet the side length of the square be ' \
    #              '\\( s \\). The square will touch the hypotenuse of the triangle. The coordinates of the vertices of ' \
    #              'the square can be described as follows:\n- One vertex at \\( (0,0) \\)\n- Another vertex at \\( (s,' \
    #              '0) \\)\n- Another vertex at \\( (0,s) \\)\n- The fourth vertex at \\( (s,s) \\)\n\nThe equation of ' \
    #              'the hypotenuse of the triangle can be found using the two points \\( (6,0) \\) and \\( (0,' \
    #              '8) \\). The slope of the hypotenuse is:\n\\[\n\\text{slope} = \\frac{8-0}{0-6} = -\\frac{4}{' \
    #              '3}\n\\]\nThe equation of the line in slope-intercept form is:\n\\[\ny = -\\frac{4}{3}x + ' \
    #              '8\n\\]\nSince the fourth vertex of the square \\( (s,s) \\) lies on this line, we substitute \\( x ' \
    #              '= s \\) and \\( y = s \\) into the equation:\n\\[\ns = -\\frac{4}{3}s + 8\n\\]\nTo solve for \\( s ' \
    #              '\\), we first eliminate the fraction by multiplying every term by 3:\n\\[\n3s = -4s + ' \
    #              '24\n\\]\nAdding \\( 4s \\) to both sides gives:\n\\[\n7s = 24\n\\]\nDividing both sides by 7, ' \
    #              'we get:\n\\[\ns = \\frac{24}{7}\n\\]\nThus, the side length of the square is \\(\\boxed{\\frac{24}{' \
    #              '7}}\\). '
    # new_solution = eval_opt.crossover(problem, solution_1, solution_2, gt_ans)
    # mutation = eval_opt.mutation(problem, new_solution, gt_ans)


if __name__ == "__main__":
    main()
    # test()