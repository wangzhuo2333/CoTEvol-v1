
import re
import math


def extract_answer(pred_str, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    else:  # use the last number
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""
    return pred


def accuracy_reward(gt_ans, solution, correct_value=1.0, format_value=0.5, wrong_value=-0.5):
    """Reward function that checks if the completion is the same as the ground truth."""
    rewards = []

    for sol in solution:
        pred_ans = extract_answer(sol)
        if pred_ans:
            if pred_ans == gt_ans:
                rewards.append(correct_value)
            elif gt_ans in sol:
                rewards.append(format_value)
            else:
                rewards.append(wrong_value)
        else:
            rewards.append(wrong_value)

    return rewards


def format_reward(gt_ans, solution, correct_format=0.2, wrong_format=-0.0):

    rewards = []

    for sol in solution:
        pred_ans = extract_answer(sol)
        # 如果没有抽取到答案那么说明格式没有匹配上
        if pred_ans:
            rewards.append(correct_format)
        else:
            rewards.append(wrong_format)

    return rewards


def cosine_scaled_reward(gt_ans, solution, tokenizer,
                         min_value_wrong=1.0, max_value_wrong=0.5, min_value_correct=0.5, max_value_correct=1.0, weight=0.5):
    """Reward function that scales based on completion length using a cosine schedule.
    This function is parameterized by the following arguments:
        min_value_wrong: Minimum reward for wrong answers
        max_value_wrong: Maximum reward for wrong answers
        min_value_correct: Minimum reward for correct answers
        max_value_correct: Maximum reward for correct answers
        max_len: Maximum length for scaling
    """
    rewards = []
    count = 0
    if len(solution) == 0:
        is_correct = False
        min_value = max_value_wrong
        max_value = min_value_wrong
        reward = min_value + 0.5 * (max_value - min_value)
        rewards.append(float(reward ) *weight)
        count+=1
        
    else:
        max_len = max([len(tokenizer.encode(sol)) for sol in solution])
        for sol in solution:
            pred_ans = extract_answer(sol)
            if pred_ans == gt_ans:
                is_correct = True
            elif gt_ans in sol:
                is_correct = True
            else:
                is_correct = False
            gen_len = len(sol)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward ) *weight)
    if count==1:
        print(count)
    return rewards


def is_english_token(token):
    # 正则匹配字母、数字和常见标点符号
    #     return bool(re.match(r'^[a-zA-Z0-9!.,?;:()_-]+$', token))
    #     pattern = r'^[a-zA-Z0-9!.,?;:()_-“”‘’[]{}|\\/\*+&^%$#@!~`<>]+$'
    pattern = r'^[a-zA-Z0-9!.,?;:()_\[\]{}|\\/*+&^%$#@!~`<>]+'
    return bool(re.match(pattern, token))


def calculate_language_ratio(text, tokenizer, target_language="en", thres=0.8):
    # 使用 Hugging Face tokenizer 对文本进行分词
    tokens = tokenizer.tokenize(text)
    # 统计目标语言的单词数
    target_language_count = 0
    total_tokens = len(tokens)

    target_words = []
    for token in tokens:
        token = token.replace("Ġ", "")
        # 检测每个 token 的语言，这里使用 langid 来判断语言
        #         lang = langid.classify(token)[0]  # langid 返回一个元组 (language, probability)
        if is_english_token(token):
            lang = target_language
        else:
            lang = "zh"
        # 如果 token 的语言与目标语言相同，则计数
        if lang == target_language:
            target_language_count += 1
            target_words.append(token)

    # 计算目标语言单词的比例
    if total_tokens == 0:
        return 0.0  # 防止除以0的情况

    ratio = target_language_count / total_tokens
    #     return 1.0 if ratio > thres else 0.0
    return ratio


def cosine_lang_reward(gt_ans, solution, tokenizer,
                       min_value_wrong=1.0, max_value_wrong=0.5, min_value_correct=0.5, max_value_correct=1.0, weight=1.0):
    # 目标语言一致性reward
    rewards = []
    lang_ratio = [calculate_language_ratio(sol, tokenizer) for sol in solution]
    #     lang_ratio = [ratio / sum(lang_ratio) for ratio in lang_ratio]
    for idx, sol in enumerate(solution):
        pred_ans = extract_answer(sol)
        if pred_ans == gt_ans:
            is_correct = True
        elif gt_ans in sol:
            is_correct = True
        else:
            is_correct = False
        gen_len = len(sol)

        # Apply cosine scaling based on length

        progress = 1.0 - lang_ratio[idx]
        cosine = math.cos(progress * math.pi)

        if is_correct:
            min_value = min_value_correct
            max_value = max_value_correct
        else:
            # Swap min/max for incorrect answers
            min_value = max_value_wrong
            max_value = min_value_wrong

        reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        rewards.append(float(reward ) *weight)

    return rewards
