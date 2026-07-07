from typing import Union, Any
from math import isclose
import func_timeout
from sympy.solvers import solve
from sympy import Symbol, Eq
import math
from sympy import simplify
import numpy as np
import cvxpy as cp
import statistics
import re


def floatify_ans(ans):
    if ans is None:
        return None
    elif type(ans) == dict:
        ans = list(ans.values())[0]
    elif type(ans) == bool:
        ans = ans
    elif type(ans) in [list, tuple]:
        if not ans:
            return None
        else:
            try:
                ans = float(ans[0])
            except Exception:
                ans = str(ans[0])
    else:
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    return ans


def solve_it(equation, variable):
    solution = solve(equation, variable, dict=True)
    if not solution:
        if isinstance(variable, list):
            solution = {v: None for v in variable}
        else:
            solution = {variable: None}
        return solution
    else:
        solution = solution[0]
        return solution


def safe_execute(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            if keys is None:
                return locals_.get('ans', None)
            else:
                return [locals_.get(k, None) for k in keys]
        except Exception:
            return None
    try:
        ans = func_timeout.func_timeout(5, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans


def synthesize_program(result: str, prefix: str) -> str:
    program = prefix
    for i, line in enumerate(result.split('\n')):
        if i == 0:
            program += line + '\n'
        else:
            if line.startswith('    '):
                program += line + '\n'
            else:
                break
    program += 'ans = solver()'
    return program


def extract_python_blocks(text):
    """
    从文本中提取 Python 代码块。
    :param text: 包含 Python 代码块的文本
    :return: 提取的 Python 代码块列表
    """
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def gen_code_post(solution, answer=None):
    # 提取 Python 代码块
    python_blocks = extract_python_blocks(solution)
    code_bolcks = []
    for i, block in enumerate(python_blocks):
        if "ans" not in block or len([_ for _ in block.split("\n") if _]) < 3:
            # 如果生成的code不符合标准以及太多但是太短code
            continue
        code_bolcks.append(block)

    # 筛选能够得到答案的code块
    ans = ""
    code = ""
    for code_bolck in code_bolcks:
        ans = safe_execute(code_bolck)
        ans = floatify_ans(ans)
        if ans:
            code = code_bolck
            break

    # 构建格式
    q_analys = solution.split("```python")[0]
    answer = f"So, the final answer is \\boxed{{ans}}."
    if code:
        solution = "\n".join([q_analys, f"```python\n{code}\n```", answer])
    # 否则没有正确的答案直接返回
    return solution


