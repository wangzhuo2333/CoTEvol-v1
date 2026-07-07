
# import os
# import pickle
# import random
#
# from utils.math_metric import cal_math_acc, extract_boxed_answers, grade_answer
# from utils.general import setup_seed
#
# setup_seed(42)
#
# with open("/code/InfluenceFun/Research_with_user/reasoning/math_lr1e-5e1_51/math_qwen-math_n=20_t=0.9_merge.pkl", "rb") as file:
#     data = pickle.load(file)
#
# for dp in data:
#     is_right = []
#     tru_a = dp['answer']
#     for prd in dp['prd']:
#         prd_a = extract_boxed_answers(prd)
#         if prd_a and grade_answer(prd_a, tru_a):
#             is_right.append((prd, True))
#         else:
#             is_right.append((prd, False))
#     dp['prd'] = is_right
#
# with open("/code/InfluenceFun/Research_with_user/reasoning/math_lr1e-5e1_51/math_qwen-math_n=20_t=0.9_merge.pkl", "wb") as f:
#     pickle.dump(data, f)

from datasets import load_dataset
# ds = load_dataset("simplescaling/s1K")["train"]
# ds.to_json(f'/extrahome0/user/data/s1.json')

# from datasets import load_dataset, Dataset
ds = load_dataset("nishadsinghi/math7500_train_solutions_DeepSeek-R1-Distill-Qwen-7B_32K_tokens", split="train")
ds.to_json(f'/media/user/data/ga/math7500/math7500.json')
# train_data = load_dataset("GAIR/o1-journey")["train"]
# train_data = load_dataset("/extrahome0/user/data/limo/limo.json")
# ds = load_dataset("Gen-Verse/ReasonFlux_SFT_15k", split="train")
# 转换为 Pandas DataFrame
# df = ds.to_pandas()
# 过滤满足特定年份的样本
# filtered_df = df[df["Year"].isin(list(range(2016, 2024)))]
# print(filtered_df.head())
# print(len(filtered_df))
# filtered_dataset = Dataset.from_pandas(filtered_df)
# print(len(filtered_dataset))
# ds.to_json(f'/extrahome0/user/data/reasonflux/rf_sft15k.json')

# import json
# data = []
# with open('/extrahome0/user/data/limo/limo.json') as file:
#     for line in file:
#         data.append(json.loads(line))
# print(len(data))
# print(data[0].keys())
#
# for dp in data:
#     dp["attempt"] = dp["solution"]
# with open('/extrahome0/user/data/limo/limo.json', "w") as file:
#     json.dump(data, file)

# ds = load_dataset("simplescaling/s1K")["train"]
# cnt = 0
# gt_cnt = 0
# for idx in range(len(ds)):
#     data = ds[idx]["metadata"]
#     gt_ans = extract_answer(ds[idx]["attempt"])
#     if gt_ans:
#         gt_cnt += 1
#     else:
#         data = eval(ds[idx]["metadata"])
#         if "answer" in data:
#             cnt += 1
#             print(data["answer"])
# print(gt_cnt, cnt, len(ds))

