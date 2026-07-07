import json
import re
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ================= 1. 环境与路径配置 =================
# 建议在终端执行: export CUDA_VISIBLE_DEVICES=1,2,3
# 或者在此处指定（必须在所有导入前，但建议用终端 export 更靠谱）
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

DATA_PATH = "/inspire/hdd/global_user/USER_ID/user/data/gpqa_diamond_test.json" # 您保存的本地 JSON 文件
MODEL_PATH="/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/wz/output_limo_new_v1/s1k_new_mutation_sft/qwen-it/20251224233406_qwen-it_lr5e-06_e1_bs64_ra0/checkpoint-4"
OUTPUT_PATH = "qwen2.5_ours1_gpqa_diamond_results.jsonl"

BATCH_SIZE = 256 
SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,  # 评估使用贪婪解码
    max_tokens=2048,  # GPQA 题目复杂，推理过程较长
    stop=["<|endoftext|>", "<|im_end|>"]
)

# ================= 2. 工具函数 =================
def format_prompt(item):
    """
    由于您的数据中 question 字段已经包含了选项 A/B/C/D，
    我们只需要给它加上引导词和 CoT 引导。
    """
    question_text = item["question"]
    
    prompt = (
        "The following is a highly complex scientific multiple-choice question.\n\n"
        f"{question_text}\n\n"
        "Please reason through the problem step-by-step and provide your final answer letter "
        "within \\boxed{}. For example: \\boxed{A}."
    )
    return prompt

def extract_answer(text):
    """从生成文本中提取 \boxed{} 内的 A, B, C, D"""
    match = re.search(r"\\boxed\{([A-D])\}", text)
    return match.group(1) if match else None

# ================= 3. 主程序逻辑 =================
def main():
    # --- 加载标准 JSON 数据 (List of Dicts) ---
    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到文件 {DATA_PATH}")
        return
        
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"成功加载样本总数: {len(all_data)}")

    # --- 断点续传检查 ---
    done_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # 使用数据中的 question 内容作为唯一标识，如果没有 id 的话
                    done_ids.add(json.loads(line)["question_hash"])
                except: continue
    
    # 过滤待处理数据 (这里用 question 的前 50 个字符做简单哈希标识)
    todo_data = [item for item in all_data if item["question"][:50] not in done_ids]
    print(f"已完成: {len(done_ids)} | 待处理: {len(todo_data)}")

    if not todo_data:
        print("所有样本均已处理完成。")
        calculate_acc()
        return

    # --- 初始化 vLLM ---
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1, # 根据您的显卡配置调整
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )

    # --- 分批推理 ---
    for i in tqdm(range(0, len(todo_data), BATCH_SIZE), desc="GPQA Diamond Running"):
        batch_items = todo_data[i : i + BATCH_SIZE]
        batch_prompts = [format_prompt(item) for item in batch_items]
        
        outputs = llm.generate(batch_prompts, SAMPLING_PARAMS, use_tqdm=False)

        # 实时保存结果
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for j, output in enumerate(outputs):
                gen_text = output.outputs[0].text
                pred = extract_answer(gen_text)
                gold = batch_items[j]["answer"]
                
                res = {
                    "question_hash": batch_items[j]["question"][:50],
                    "gold": gold,
                    "pred": pred,
                    "is_correct": (pred == gold),
                    "model_output": gen_text
                }
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # --- 计算准确率 ---
    calculate_acc()

def calculate_acc():
    if not os.path.exists(OUTPUT_PATH):
        return
    correct, total = 0, 0
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            total += 1
            if item["is_correct"]:
                correct += 1
    
    if total > 0:
        print("\n" + "="*30)
        print(f"GPQA-Diamond 测试报告")
        print(f"总样本数: {total}")
        print(f"正确数: {correct}")
        print(f"总准确率: {correct/total:.2%}")
        print("="*30)

if __name__ == "__main__":
    main()