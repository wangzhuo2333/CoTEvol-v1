import json
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ================= 1. 核心配置 =================
# 请根据您的实际路径修改
DATA_PATH = "/inspire/hdd/global_user/USER_ID/user/data/arc-c/arc_challenge_json/test.jsonl" 
MODEL_PATH="/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/wz/output_limo_new_v1/s1k_new_mutation_sft/qwen-it/20251224233406_qwen-it_lr5e-06_e1_bs64_ra0/checkpoint-4"
OUTPUT_PATH = "qwen2.5_ours1_arc_c_results.jsonl"

BATCH_SIZE = 100 
GPU_MEMORY_UTILIZATION = 0.9

# 推理参数：设置为 0.0 保证学术评估的可复现性
SAMPLING_PARAMS = SamplingParams(
    temperature=0.6,
    max_tokens=1024,
    stop=["<|endoftext|>", "<|im_end|>"]
)

# ================= 2. 工具函数 =================
def format_prompt(item):
    """
    针对 ARC 数据格式进行解析：
    {"question": "...", "choices": {"text": ["A_text", "B_text"], "label": ["A", "B"]}}
    """
    question = item["question"]
    choices_dict = item["choices"]
    
    # 构建选项字符串，例如: (A) Option 1 \n (B) Option 2
    options_formatted = []
    for label, text in zip(choices_dict["label"], choices_dict["text"]):
        options_formatted.append(f"({label}) {text}")
    
    options_str = "\n".join(options_formatted)
    
    prompt = (
        "The following is a multiple-choice question about science.\n\n"
        f"Question: {question}\n"
        f"Options:\n{options_str}\n\n"
        "Please reason through the problem step-by-step and provide your final answer letter "
        "within \\boxed{}. For example: \\boxed{A}."
    )
    return prompt

def extract_answer(text):
    """提取 \boxed{字母}，ARC 的答案通常是 A, B, C, D, E 或数字 1, 2, 3, 4"""
    # 匹配 A-E 或 1-5
    pattern = r"\\boxed\{([A-E1-5])\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

# ================= 3. 主推理逻辑 =================
def main():
    # --- 加载数据 (JSONL 格式) ---
    all_data = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            all_data.append(json.loads(line))
    print(f"Total ARC samples loaded: {len(all_data)}")

    # --- 断点续传检查 ---
    done_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except: continue
    
    todo_data = [item for item in all_data if item["id"] not in done_ids]
    print(f"Already processed: {len(done_ids)}, Remaining: {len(todo_data)}")

    if not todo_data:
        print("All samples processed.")
        return

    # --- 初始化 vLLM ---
    # 注意：建议在命令行执行 export CUDA_VISIBLE_DEVICES=1,2,3
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1, # 根据显卡数量调整
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True
    )

    # --- 分 Batch 循环 ---
    for i in tqdm(range(0, len(todo_data), BATCH_SIZE), desc="ARC-c Inference"):
        batch_items = todo_data[i : i + BATCH_SIZE]
        batch_prompts = [format_prompt(item) for item in batch_items]
        
        # 批量生成
        outputs = llm.generate(batch_prompts, SAMPLING_PARAMS, use_tqdm=False)

        # 实时保存
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for j, output in enumerate(outputs):
                gen_text = output.outputs[0].text
                pred = extract_answer(gen_text)
                gold = batch_items[j]["answerKey"]
                
                res = {
                    "id": batch_items[j]["id"],
                    "gold": gold,
                    "pred": pred,
                    "is_correct": (str(pred) == str(gold)),
                    "model_output": gen_text
                }
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # --- 最终评估 ---
    calculate_acc()

def calculate_acc():
    correct, total = 0, 0
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            total += 1
            if item["is_correct"]: correct += 1
    if total > 0:
        print(f"\nFinal Accuracy on ARC-Challenge: {correct/total:.2%} ({correct}/{total})")

if __name__ == "__main__":
    main()