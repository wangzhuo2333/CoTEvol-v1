
import re


def gsm_extract_last_num(text: str):
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return num_str
    else:
        return ""


def math_evaluate(gtr, prd):
    gtr = gsm_extract_last_num(gtr)
    if not gtr:
        return True
    m = re.search(r"\\boxed{\s*(?P<text>.+?)}", prd, re.DOTALL)
    if m:
        m = m.group("text")
        if gtr in m:
            return True
    if gtr in prd:
        return True


def cal_gsm_acc(gen_datas):
    correct = 0
    outputs = []
    for gen in gen_datas:
        result = dict(
            **gen,
            extract_true_num=gsm_extract_last_num(gen["answer"]),
            extract_pred_num=gsm_extract_last_num(gen["prd"]),
            is_correct=None,
        )
        # if abs(result["extract_true_num"] - result["extract_pred_num"]) < 1e-3:
        if math_evaluate(result["extract_true_num"], result["extract_pred_num"]):
            result["is_correct"] = True
            correct += 1
        else:
            result["is_correct"] = False
        outputs.append(result)
    acc = round(correct / len(outputs), 3)
    return outputs, acc
