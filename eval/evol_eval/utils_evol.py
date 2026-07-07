
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


def read_prompt(pmt_path):
    with open(pmt_path, encoding='utf-8') as file:
        pmt = file.readlines()
    pmt = "".join([line for line in pmt if line != "\n"])
    return pmt


def format_input(pmt, tokenizer):
    system_prompt = (
        "Below is an instruction that describes a task."
        "Write a response that appropriately completes the request."
    )
    inputs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": pmt}]
    inputs = tokenizer.apply_chat_template(
        inputs, tokenize=False, add_generation_prompt=True
    )
    return inputs


def format_code_input(pmt, tokenizer):
    system_prompt = (
        "You are a powerful agent with broad math knowledge and great python programming skills. "
        "You need to use python to solve given math questions.\n\n"
        "!!!Remember:\n"
        "1. Use code solve the problem step by step. Before writing code, clearly identify key variables, "
        "constraints, and equations necessary to solve the problem. Store your result as a variable named 'ans'.\n "
        "2. All calculations should be done in python code. The python code is in ```python``` block. Provide concise "
        "reasoning and thinking in the comments of the code.\n "
        "3. The most related python packages include `math`, `sympy`, `scipy`, and `numpy`.\n"
        "4. Ensure your code can execute correctly and avoid undefined variables (NameError), unimported packages, "
        "or formatting errors (SyntaxError, TypeError).\n "
        "5. In the last step of the code, print the final answer i..e., print(ans).\n\n"
        #     "!!!Here are some demonstration:\n"
        #     "{examples}"
    )
    inputs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": pmt}]
    inputs = tokenizer.apply_chat_template(
        inputs, tokenize=False, add_generation_prompt=True
    )
    return inputs


def softmax(x):
    x = np.array(x)
    # 计算输入数列的指数
    exp_x = np.exp(x - np.max(x))  # 减去 np.max(x) 是为了防止溢出
    return exp_x / np.sum(exp_x)  # 将指数值除以总和进行归一化


def find_repeated_phrases(text, min_repeat=3):
    """
    检测文本中重复出现的片段，如果某个片段重复超过给定次数，则认为是复读机模式。
    :param text: 输入的文本
    :param min_repeat: 重复的最小次数，默认3次
    :return: 返回重复的片段及其出现次数
    """
    # 按行分割文本
    lines = text.split('Assistant：')
    # lines = text.split('\n')

    # 将所有的"Assistant："去除，只保留具体回答内容
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    # 使用Counter来统计每个片段的出现次数
    counter = Counter(cleaned_lines)

    # 找出重复出现超过min_repeat次数的片段
    repeated_phrases = {phrase: count for phrase, count in counter.items() if count >= min_repeat}

    return repeated_phrases, cleaned_lines


def correct_repeated_text(text, min_repeat=3):
    """
    矫正重复片段，截断重复的部分，只保留第一次出现的部分。
    :param text: 输入的文本
    :param min_repeat: 重复的最小次数，默认3次
    :return: 修正后的文本
    """
    repeated_phrases, cleaned_lines = find_repeated_phrases(text, min_repeat)

    # 如果没有发现重复片段，返回原文本
    if not repeated_phrases:
        return text

    # 修正文本，截断重复的部分
    corrected_lines = []
    seen = set()

    # 遍历每一行，如果是重复的片段，跳过
    for line in cleaned_lines:
        if line not in seen:
            corrected_lines.append(line)
            seen.add(line)
        else:
            break  # 一旦遇到重复的片段，停止追加

    # 重新构建文本
    corrected_text = 'Assistant：'.join(corrected_lines)
    return corrected_text


def remove_duplicates(text):
    # 假设文本按句子分割
    sentences = re.split(r'(?<=[.!?。]) +', text)  # 句子分割
    seen = set()  # 用于记录已出现的句子
    unique_sentences = []  # 用于存储去重后的句子

    for sentence in sentences:
        if sentence not in seen:  # 如果句子未出现过
            unique_sentences.append(sentence)
            seen.add(sentence)  # 添加到集合中，表示已经处理过

    return ' '.join(unique_sentences)


def clean_solution(text, min_repeat=3):
    repeated, _ = find_repeated_phrases(text, min_repeat=min_repeat)
    while repeated:
        text = correct_repeated_text(text, min_repeat=3)
        repeated, _ = find_repeated_phrases(text, min_repeat=3)
    text = remove_duplicates(text)
    return text


def remove_dup_solutions(paragraphs, threshold=0.8):
    # 创建 TF-IDF 向量化器
    vectorizer = TfidfVectorizer().fit_transform(paragraphs)

    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(vectorizer)

    # 创建一个布尔列表，标记哪些段落需要保留
    to_keep = [True] * len(paragraphs)

    for i in range(len(paragraphs)):
        if not to_keep[i]:  # 如果当前段落已经被标记为去除，则跳过
            continue

        for j in range(i + 1, len(paragraphs)):
            if similarity_matrix[i][j] > threshold:
                to_keep[j] = False  # 如果相似度超过阈值，则标记为删除

    # 根据标记的列表，返回不重复的段落
    return [paragraphs[i] for i in range(len(paragraphs)) if to_keep[i]]
