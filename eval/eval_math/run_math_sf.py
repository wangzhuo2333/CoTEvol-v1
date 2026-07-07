import os
import json
import torch
import numpy as np
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from loguru import logger
import tqdm
from parser import extract_answer
from grader import math_equal


# def get_gen_prompt(problem):
#     """第一步：生成初始答案"""
#     return [
#         {"role": "system", "content": "You are an expert mathematician. Provide a step-by-step reasoning process and end with the final answer in $\\boxed{answer}$."},
#         {"role": "user", "content": problem}
#     ]

# def get_critique_prompt(problem, initial_solution):
#     """第二步：自我批判（找错）"""
#     content = f"Problem: {problem}\n\nInitial Solution: {initial_solution}\n\n"
#     content += "Please check the solution above for any logical flaws, calculation errors, or missing steps. If there are errors, point them out clearly. If it is correct, confirm its correctness."
#     return [
#         {"role": "system", "content": "You are a critical math reviewer."},
#         {"role": "user", "content": content}
#     ]

# def get_refine_prompt(problem, initial_solution, critique):
#     """第三步：根据反馈修正答案"""
#     content = f"Problem: {problem}\n\nInitial Solution: {initial_solution}\n\nCritique: {critique}\n\n"
#     content += "Based on the critique, provide a final, corrected version of the solution. Ensure the final answer is in $\\boxed{answer}$."
#     return [
#         {"role": "system", "content": "You are an expert mathematician. Fix the solution based on the critique provided."},
#         {"role": "user", "content": content}
#     ]
import json

def load_math500(path):
    problems = []
    answers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            problems.append(item["problem"])
            answers.append(item["answer"])
    return problems, answers


def build_solve_prompt(problem):
    return [
        {"role": "system", "content": "You are a helpful math reasoning assistant."},
        {"role": "user", "content": problem}
    ]

def build_critique_prompt(problem, solution):
    return [
        {"role": "system", "content": "You are a critical math reviewer."},
        {"role": "user", "content": f"Problem:\n{problem}\n\nSolution:\n{solution}\n\nPlease critique the reasoning."}
    ]

def build_refine_prompt(problem, solution, critique):
    return [
        {"role": "system", "content": "You are a math expert improving a solution."},
        {"role": "user", "content": f"Problem:\n{problem}\n\nOriginal Solution:\n{solution}\n\nCritique:\n{critique}\n\nPlease provide a refined solution."}
    ]

# ==========================================
# 3. Self-Refine Pipeline
# ==========================================

# class SelfRefinePipeline:
#     def __init__(self, model_path, tp=2):
#         # 只需要加载一个生成模型，不需要 RM
#         logger.info(f"Loading Model for Self-Refine: {os.path.basename(model_path)}")
#         self.llm = LLM(
#             model=model_path,
#             tensor_parallel_size=tp,
#             gpu_memory_utilization=0.9, # 不用给 RM 留显存了，可以调高
#             trust_remote_code=True
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
#         # Self-Refine 通常使用较低的温度以保证逻辑严谨
#         self.sampling_params = SamplingParams(
#             temperature=0.2, 
#             top_p=0.95, 
#             max_tokens=2048
#         )

#     def run(self, data_path, batch_size=16):
#         with open(data_path, 'r', encoding='utf-8') as f:
#             dataset = [json.loads(line) for line in f]
        
#         results = []
#         correct_count = 0

#         for i in range(0, len(dataset), batch_size):
#             batch = dataset[i : i + batch_size]
#             problems = [item['problem'] for item in batch]
#             ground_truths = [item['answer'] for item in batch]

#             # --- Step 1: Initial Generation ---
#             logger.info(f"Step 1: Generating initial solutions for batch {i//batch_size}")
#             prompts_gen = [
#                 self.tokenizer.apply_chat_template(get_gen_prompt(p), tokenize=False, add_generation_prompt=True) 
#                 for p in problems
#             ]
#             outputs_gen = self.llm.generate(prompts_gen, self.sampling_params)
#             initial_solutions = [out.outputs[0].text for out in outputs_gen]

#             # --- Step 2: Self-Critique ---
#             logger.info(f"Step 2: Critiquing solutions for batch {i//batch_size}")
#             prompts_critique = [
#                 self.tokenizer.apply_chat_template(get_critique_prompt(problems[j], initial_solutions[j]), tokenize=False, add_generation_prompt=True) 
#                 for j in range(len(batch))
#             ]
#             outputs_critique = self.llm.generate(prompts_critique, self.sampling_params)
#             critiques = [out.outputs[0].text for out in outputs_critique]

#             # --- Step 3: Refined Generation ---
#             logger.info(f"Step 3: Refining solutions for batch {i//batch_size}")
#             prompts_refine = [
#                 self.tokenizer.apply_chat_template(get_refine_prompt(problems[j], initial_solutions[j], critiques[j]), tokenize=False, add_generation_prompt=True) 
#                 for j in range(len(batch))
#             ]
#             outputs_refine = self.llm.generate(prompts_refine, self.sampling_params)
#             refined_solutions = [out.outputs[0].text for out in outputs_refine]

#             # --- 统计与保存 ---
#             for j in range(len(batch)):
#                 is_correct = math_equal(extract_answer(refined_solutions[j], "math500"), ground_truths[j])
#                 if is_correct:
#                     correct_count += 1
                
#                 results.append({
#                     "problem": problems[j],
#                     "answer": ground_truths[j],
#                     "initial_solution": initial_solutions[j],
#                     "critique": critiques[j],
#                     "final_solution": refined_solutions[j],
#                     "is_correct": is_correct
#                 })

#             logger.info(f"Processed {i+len(batch)} | Current Acc: {correct_count / (i+len(batch)):.2%}")

#         final_acc = correct_count / len(dataset)
#         logger.success(f"Final Self-Refine Accuracy: {final_acc:.2%}")
        
#         with open("self_refine_results.json", "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=4, ensure_ascii=False)

class BestOfNSelfRefineEvaluator:

    def __init__(self, model_path: str):
        self.llm = LLM(model=model_path)

        # 一次生成 10 个解
        self.gen_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            n=10
        )

        # refine 阶段低温
        self.refine_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=2048
        )

    def evaluate(self, problems, answers):
        results = []

        # ===== Step 1: 生成 10 个初始解 =====
        prompts = [
            self.llm.get_tokenizer().apply_chat_template(
                build_solve_prompt(p),
                tokenize=False,
                add_generation_prompt=True
            )
            for p in problems
        ]

        gen_outputs = self.llm.generate(prompts, self.gen_params)

        total_pass1 = 0
        total_avg = 0

        # ===== Step 2: 对每个解单独 Self-Refine =====
        for idx, output in enumerate(gen_outputs):
            problem = problems[idx]
            gt = answers[idx]

            initial_solutions = [o.text for o in output.outputs]

            refined_solutions = []
            correctness = []
                # ===== Step 1: 批量构造 critique prompt =====
            critique_prompts = [
                self.llm.get_tokenizer().apply_chat_template(
                    build_critique_prompt(problem, sol),
                    tokenize=False,
                    add_generation_prompt=True
                )
                for sol in initial_solutions
            ]

            # ===== Step 2: 批量生成 critique =====
            critique_results = self.llm.generate(critique_prompts, self.refine_params)
            critiques = [res.outputs[0].text for res in critique_results]

            # ===== Step 3: 批量构造 refine prompt =====
            refine_prompts = [
                self.llm.get_tokenizer().apply_chat_template(
                    build_refine_prompt(problem, sol, crit),
                    tokenize=False,
                    add_generation_prompt=True
                )
                for sol, crit in zip(initial_solutions, critiques)
            ]

            # ===== Step 4: 批量生成 refine =====
            refined_results = self.llm.generate(refine_prompts, self.refine_params)
            refined_solutions = [res.outputs[0].text for res in refined_results]

            # ===== Step 5: 计算 correctness =====
            correctness = [
                math_equal(extract_answer(refined, 'math500'), gt)
                for refined in refined_solutions
            ]

            # ===== Step 6: 计算指标 =====
            pass1 = any(correctness)
            avg = sum(correctness) / len(correctness)

            total_pass1 += int(pass1)
            total_avg += avg

            results.append({
                "problem": problem,
                "answer": gt,
                "initial_solutions": initial_solutions,
                "refined_solutions": refined_solutions,
                "correctness": correctness,
                "pass@1": pass1,
                "avg": avg
            })

            print(f"Final Pass@1: {total_pass1 / len(results):.4f}")
            print(f"Final Avg: {total_avg / len(results):.4f}")


        final_pass1 = total_pass1 / len(problems)
        final_avg = total_avg / len(problems)

        print(f"Final Pass@1: {final_pass1:.4f}")
        print(f"Final Avg: {final_avg:.4f}")

        return results
# ==========================================
# 4. 执行入口
# ==========================================

# if __name__ == "__main__":
    # CONFIG = {
    #     # "model_path": "/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/wz/output_limo_new_v1/s1k_new_mutation_sft/qwen-it/20251225000855_qwen-it_lr1e-05_e1_bs64_ra0/checkpoint-4",
    #     "model_path": "/inspire/hdd/global_user/USER_ID/user/models/Qwen2.5-Math-7B-Instruct",
    #     "data_path": "/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/eval/data/math500/test.jsonl",
    #     "tp_size": 2
    # }

    # pipeline = SelfRefinePipeline(
    #     model_path=CONFIG["model_path"],
    #     tp=CONFIG["tp_size"]
    # )
    
    # # 因为没有 72B RM 的压力，batch_size 可以适当放大（如 16 或 32）
    # pipeline.run(CONFIG["data_path"], batch_size=16)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    CONFIG = {
        "model_path": "/inspire/hdd/global_user/USER_ID/user/models/Qwen2.5-Math-7B-Instruct",
        "data_path": "/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/eval/data/math500/test.jsonl",
    }

    # ===== 1. 读取数据 =====
    problems, answers = load_math500(CONFIG["data_path"])

    # ===== 2. 初始化 evaluator =====
    evaluator = BestOfNSelfRefineEvaluator(
        model_path=CONFIG["model_path"]
    )

    # ===== 3. 运行评测 =====
    results = evaluator.evaluate(problems, answers)

    # ===== 4. 保存结果（可选，但强烈推荐）=====
    with open("bestof10_self_refine_results_amc23.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
