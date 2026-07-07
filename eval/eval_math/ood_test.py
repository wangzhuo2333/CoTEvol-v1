import json
import re
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ================= 配置区 =================
DATA_PATH = "/inspire/hdd/global_user/USER_ID/user/data/mmlu_pro/mmlu_pro_test.json"
MODEL_PATH="/inspire/hdd/global_user/USER_ID/user/code/ifdr-main-wz-v2/wz/output_s1k_new_long_cot/s1k_new_mutation_sft/qwen-it/20251230160854_qwen-it_lr7e-07_e1_bs32_ra0/checkpoint-20"
OUTPUT_PATH = "qwen2.5_dcot_mmlu_pro_results.jsonl"  # 结果保存路径
BATCH_SIZE = 512 # 每批处理多少条数据后保存一次
CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# vLLM 推理参数
SAMPLING_PARAMS = SamplingParams(
    temperature=0.6,  # 学术评估建议设为 0.0 以保证结果可复现
    max_tokens=2048,  # MMLU Pro 的 CoT 推理过程较长，建议给足空间
    stop=["<|endoftext|>", "<|im_end|>"]
)

# ================= 工具函数 =================
def format_prompt(item):
    """构建适合 Qwen2.5-Instruct 的推理提示词"""
    question = item["question"]
    options = item["options"]
    category = item.get("category", "academic knowledge")
    
    options_str = "\n".join([f"({CHOICES[i]}) {opt}" for i, opt in enumerate(options)])
    
    prompt = (
        f"The following is a multiple-choice question about {category}.\n\n"
        f"Question: {question}\n"
        f"Options:\n{options_str}\n\n"
        "Please reason through the problem step-by-step and provide your final answer letter "
        "within \\boxed{}. For example: \\boxed{A}."
    )
    return prompt

def extract_answer(text):
    """从生成文本中提取 \boxed{} 内的选项"""
    pattern = r"\\boxed\{([A-J])\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

# ================= 主程序 =================
def main():
    # 1. 加载完整原始数据
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"Loaded total samples: {len(all_data)}")

    # 2. 检查断点续传：读取已经处理过的 question_id
    done_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    done_ids.add(record["question_id"])
                except:
                    continue
    
    # 过滤待处理数据
    todo_data = [item for item in all_data if item["question_id"] not in done_ids]
    print(f"Done: {len(done_ids)}, Remaining: {len(todo_data)}")

    if not todo_data:
        print("All data has been processed.")
        return

    # 3. 初始化 vLLM 推理引擎
    # 根据您的路径配置，12k数据建议开启多显卡(如果显存不足)或调整 gpu_memory_utilization
    llm = LLM(
        model=MODEL_PATH, 
        tensor_parallel_size=1, # 如果有多张显卡，可以增加此数值
        gpu_memory_utilization=0.9, # 根据显存实际负载调整
        trust_remote_code=True
    )

    # 4. 分批推理循环
    for i in range(0, len(todo_data), BATCH_SIZE):
        chunk = todo_data[i : i + BATCH_SIZE]
        prompts = [format_prompt(item) for item in chunk]
        
        print(f"\n>>> Running Batch {i//BATCH_SIZE + 1} ({len(chunk)} samples)...")
        outputs = llm.generate(prompts, SAMPLING_PARAMS)

        # 5. 实时保存结果至 JSONL
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for j, output in enumerate(outputs):
                raw_text = output.outputs[0].text
                pred = extract_answer(raw_text)
                
                result = {
                    "question_id": chunk[j]["question_id"],
                    "category": chunk[j].get("category"),
                    "gold": chunk[j]["answer"],
                    "pred": pred,
                    "is_correct": (pred == chunk[j]["answer"]),
                    "generated_text": raw_text
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 6. 推理结束后自动计算总性能
    calculate_metrics()

def calculate_metrics():
    """读取保存的文件计算准确率"""
    if not os.path.exists(OUTPUT_PATH):
        return
        
    correct = 0
    total = 0
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            total += 1
            if item["is_correct"]:
                correct += 1
    
    if total > 0:
        print("\n" + "="*30)
        print(f"Final Report for MMLU Pro")
        print(f"Total Samples: {total}")
        print(f"Correct: {correct}")
        print(f"Overall Accuracy: {(correct/total)*100:.2f}%")
        print("="*30)

if __name__ == "__main__":
    main()