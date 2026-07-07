
from parser import extract_answer

import os
import json
import torch
import numpy as np
from vllm import LLM, SamplingParams
# 换回 AutoModel，但我们会手动处理输出
from transformers import AutoModel, AutoTokenizer
from loguru import logger
import tqdm
from grader import math_equal

# --- 1. 基础工具函数 ---
# def math_equal(pred, gt):
#     # 修复之前的拼写错误：确保两边都是 'gaokao2023'
#     # 如果你的 parser.extract_answer 依赖这个字符串，请务必保持一致
#     pred_ans = extract_answer(str(pred), 'gaokao2023')
#     gt_ans = extract_answer(str(gt), 'gaokao2023')
#     return pred_ans == gt_ans



# ==========================================
# 2. Prompt 模板定义
# ==========================================
def get_gen_prompt(problem):
    return [
        {"role": "system", "content": "You are an expert mathematician. Provide a step-by-step reasoning process and end with the final answer in $\\boxed{answer}$."},
        {"role": "user", "content": problem}
    ]

def get_rm_prompt(problem, solution):
    return [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution}
    ]

# ==========================================
# 3. 核心 BoN Pipeline
# ==========================================
class BoNPipeline:
    def __init__(self, gen_path, rm_path, n=10, tp=2):
        self.n = n
        
        # --- A. 初始化生成模型 (vLLM) ---
        logger.info(f"Loading Gen Model: {os.path.basename(gen_path)}")
        self.llm = LLM(
            model=gen_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.4, 
            trust_remote_code=True
        )
        self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_path)
        self.sampling_params = SamplingParams(
            n=n, 
            temperature=0.7, 
            top_p=0.95, 
            max_tokens=4096 
        )

        # --- B. 初始化奖励模型 ---
        # logger.info(f"Loading RM Model: {os.path.basename(rm_path)}")
        # # 换回 AutoModel
        # self.rm_model = AutoModel.from_pretrained(
        #     rm_path,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        #     trust_remote_code=True
        # ).eval()
        
        # self.rm_tokenizer = AutoTokenizer.from_pretrained(rm_path)
        # if self.rm_tokenizer.pad_token is None:
        #     self.rm_tokenizer.pad_token = self.rm_tokenizer.eos_token
        
        # # 显式注入 pad_token_id 解决之前 ValueError 的核心
        # self.rm_model.config.pad_token_id = self.rm_tokenizer.pad_token_id
        # self.rm_tokenizer.padding_side = "right"
        # self.rm_tokenizer.truncation_side = 'left' # 优先保留结尾的答案

    def run(self, data_path, batch_size=1):
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        
        results = []
        correct_count = 0
        pass_at_n_count = 0 

        # 遍历数据集
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            # 确认你的 jsonl 里字段是 question 还是 problem
            problems = [item['problem'] for item in batch]
            ground_truths = [item['answer'] for item in batch]

            # 1. 生成
            prompts = [
                self.gen_tokenizer.apply_chat_template(get_gen_prompt(p), tokenize=False, add_generation_prompt=True) 
                for p in problems
            ]
            outputs = self.llm.generate(prompts, self.sampling_params)
            all_candidates = [[res.text for res in out.outputs] for out in outputs]
            
            avg_total = 0
            # 2. 对每个问题进行打分
            for idx, cands in enumerate(all_candidates):
                current_prob = problems[idx]
                gt = ground_truths[idx]
                                # ===== 计算每个 candidate 的正确性 =====
                correctness = []
                for sol in cands:
                    pred = extract_answer(sol, 'amc23')   # 你已有的函数
                    correctness.append(math_equal(pred, gt))

                # ===== 指标计算 =====
                pass_n = any(correctness)                 # Pass@N
                avg = sum(correctness) / len(correctness) # Avg
                acc = correctness[0]                      # Pass@1 / Acc

                # ===== 全局统计 =====
                correct_count += int(acc)
                pass_at_n_count += int(pass_n)
                avg_total += avg

                results.append({
                    "problem": current_prob,
                    "answer": gt,
                    "solutions": cands,
                    "correctness": correctness,
                    "acc": acc,
                    "pass@N": pass_n,
                    "avg": avg
                })
                print('avg', avg)
                
            #     # 诊断用：Pass@N
            #     any_correct = any([math_equal(str(extract_answer(c, 'amc23')), gt) for c in cands])
            #     if any_correct:
            #         pass_at_n_count += 1

            #     # 构造 RM 输入
            #     rm_inputs = [
            #         self.rm_tokenizer.apply_chat_template(get_rm_prompt(current_prob, c), tokenize=False, add_generation_prompt=False)
            #         for c in cands
            #     ]
                
                
            #     inputs = self.rm_tokenizer(rm_inputs, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.rm_model.device)
                
            #     with torch.no_grad():
            #         outputs = self.rm_model(**inputs, use_cache=False) 
                    
            #         # --- 重点：如何从 AutoModel 中安全提取分数 ---
            #         if hasattr(outputs, 'logits'):
            #             logits = outputs.logits
            #         elif isinstance(outputs, torch.Tensor):
            #             logits = outputs
            #         else:
            #             # 某些自定义 RM 会返回一个 SequenceClassifierOutput 对象
            #             logits = outputs[0]

            #         logits = logits.view(-1).float().cpu().numpy()
                
            #     # 3. 选出得分最高的
            #     best_idx = np.argmax(logits)
            #     best_solution = cands[best_idx]
                
            #     is_correct = math_equal(str(extract_answer(best_solution, 'amc23')), gt)
            #     if is_correct:
            #         correct_count += 1
                
            #     results.append({
            #         "problem": current_prob,
            #         "gt": gt,
            #         "best_solution": best_solution,
            #         "is_correct": is_correct,
            #         "pass_at_n": any_correct,
            #         "scores": logits.tolist()
            #     })

            # processed = i + len(batch)
            # logger.info(f"Processed: {processed} | BoN Acc: {correct_count/processed:.2%} | Pass@{self.n}: {pass_at_n_count/processed:.2%}")

        # 最终统计
        # final_acc = correct_count / len(dataset)
        # logger.success(f"Final Accuracy: {final_acc:.2%}")
        # return results
        total = len(dataset)
        print(f"Accuracy / Pass@1: {correct_count / total:.4f}")
        print(f"Pass@{self.n}: {pass_at_n_count / total:.4f}")
        print(f"Avg: {sum(r['avg'] for r in results) / total:.4f}")

        
        with open("bon_results_fixed.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        return results

# ==========================================
# 4. 执行入口
# ==========================================

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # 请根据实际环境修改以下路径
    CONFIG = {
        # "gen_model": "/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/wz/output_limo_new_v1/s1k_new_mutation_sft/qwen-it/20251225000855_qwen-it_lr1e-05_e1_bs64_ra0/checkpoint-4",
        "gen_model": "/inspire/hdd/global_user/USER_ID/user/models/Qwen2.5-Math-7B-Instruct",
        "rm_model": "/inspire/hdd/global_user/USER_ID/user/models/Qwen2.5-Math-RM-72B/models--Qwen--Qwen2.5-Math-RM-72B/snapshots/32aa4aae9021d3e2258694ec0ac6d3b4a68f013b",
        "data_path": "/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/eval/data/amc23/test.jsonl",
        "n_samples": 10,
        "tp_size": 2
    }

    pipeline = BoNPipeline(
        gen_path=CONFIG["gen_model"],
        rm_path=CONFIG["rm_model"],
        n=CONFIG["n_samples"],
        tp=CONFIG["tp_size"]
    )
    
    pipeline.run(CONFIG["data_path"], batch_size=10)