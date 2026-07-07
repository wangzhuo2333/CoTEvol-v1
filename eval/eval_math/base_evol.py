import numpy as np
import re
from loguru import logger
from rouge_score import rouge_scorer

# from python_tool import gen_code_post, floatify_ans
from utils_evol import clean_solution, remove_dup_solutions, read_prompt, format_input, softmax, format_code_input
from fitness import cosine_scaled_reward, accuracy_reward, format_reward, cosine_lang_reward


stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
def compute_confidence(Logprobs):
    confs = []
    for token_logprobs in Logprobs:
        if token_logprobs:
            mean_probs = np.mean([lp.logprob for lp in token_logprobs.values()])
            confs.append(round(-mean_probs, 3))
    return confs
            

class EvolOpt(object):

    def __init__(self, args, client, tokenizer):
        """进化问题problem的solution
            client: 部署的vllm的client
            pop_size: 初始种群大小，也就是初次生成的个数；
            tokenizer：分词器用于进行fitness计算
        """
        self.args = args
        self.client = client
        self.tokenizer = tokenizer
        self.stoe_words = stop_words

    def init_population(self, problem, gt_ans):

        gen_pmt = read_prompt(self.args.gen_responses_prompt)
        gen_pmt = gen_pmt.format_map({"problem": problem, "occup": "<your answer>"})
        gen_inputs = format_input(gen_pmt, self.tokenizer)
        # logger.info(f"Formated input:\n {gen_inputs}")

        cnt = 0
        solutions = []
        gen_responses_temp = self.args.gen_responses_temp

        while len(solutions) < self.args.pop_size:
            # 生成多样性回复困难则提高温度
            if cnt > 5:
                gen_responses_temp = gen_responses_temp + 0.1
                logger.debug(f"多样性回复难度较大，提高温度{gen_responses_temp - 0.1:.1f}->{gen_responses_temp:.1f}")

            response = self.client.completions.create(
                model=self.args.model_path,  # 与启动时指定的模型名称一致
                prompt=gen_inputs,
                max_tokens=self.args.max_response_len,
                temperature=gen_responses_temp,
                top_p=0.95,
                # repetition_penalty=1.1,
                n=self.args.pop_size,
                stop=stop_words,
            )
            for chose in response.choices:
                solutions.append(clean_solution(chose.text))

            # 如果生成的多样性回复难度太大 那么不需要在去重了
            if cnt < 10:
                solutions = remove_dup_solutions(solutions)
                cnt += 1
            else:
                logger.debug(f"难度很大，丢弃去重操作")
                break

        return solutions

    def calculate_fitness(self, solutions, gt_ans):
        len_rwd = cosine_scaled_reward(gt_ans, solutions, self.tokenizer)

        acc_rwd = accuracy_reward(gt_ans, solutions)

        format_rwd = format_reward(gt_ans, solutions)

        lang_rwd = cosine_lang_reward(gt_ans, solutions, self.tokenizer)

        rwd = []
        details_info = []
        for i in range(len(acc_rwd)):
            rwd.append(float(len_rwd[i] + acc_rwd[i] + format_rwd[i] + lang_rwd[i]))
            details_info.append(
                {"solutions": solutions[i], "rwd": rwd[i], "len_rwd": len_rwd[i], "acc_rwd": acc_rwd[i],
                 "format_rwd": format_rwd[i], "lang_rwd": lang_rwd[i]}
            )
        # sel_p = softmax(rwd)
        # return sel_p, details_info
        return rwd

    def select(self, solutions, gt_ans):
        fitness_vals = self.calculate_fitness(solutions, gt_ans)
        prob = softmax(fitness_vals) # 计算每个个体被选择的概率
        solutions = np.array(solutions)
        selected_idx = np.random.choice(range(len(solutions)), size=2, p=prob)
        return list(solutions[selected_idx])
    
    def split_solution_into_steps(self, solution: str) -> list:
        """将 solution 按步骤切割，按照 'step' 关键词分割"""
        # 按 'step' 关键词（不区分大小写）分割
        # 匹配 "Step 1", "step 1", "STEP 1", "Step1" 等模式，并保留分隔符
        step_pattern = r'(?i)(?:^|\n)(\s*step\s*\d+[.:]?\s*.*?)(?=\n\s*step\s*\d+[.:]?\s*|$)'
        matches = re.findall(step_pattern, solution, re.DOTALL)
        
        if matches:
            # 如果找到 step 标记，使用匹配的结果
            steps = [match.strip() for match in matches if match.strip()]
        else:
            # 如果没有找到 step 关键词，尝试按行分割
            steps = [step.strip() for step in solution.split('\n') if step.strip()]
        
        return steps if steps else [solution]  # 如果分割后为空，返回原 solution

    def crossover(self, problem, solution_1, solution_2, gt_ans, tokenizer):
        critic_pmt = read_prompt(self.args.critic_prompt)
        critic_pmt = critic_pmt.format_map(
            {"problem": problem, "answer": gt_ans, "solution_1": solution_1,
             "solution_2": solution_2})
        critic_inputs = format_input(critic_pmt, self.tokenizer)

        response = self.client.completions.create(
            model=self.args.model_path,  # 与启动时指定的模型名称一致
            prompt=critic_inputs,
            max_tokens=self.args.max_critic_len,
            temperature=self.args.gen_critic_temp,
            top_p=0.95,
            n=1,
            stop=stop_words,
        )
        critic = response.choices[0].text
        # print(critic)

        author_pmt = read_prompt(self.args.author_prompt)
        author_pmt = author_pmt.format_map(
            {"problem": problem, "solution_1": solution_1,
             "solution_2": solution_2, "critic_feedback": critic})
        author_inputs = format_input(author_pmt, self.tokenizer)

        response = self.client.completions.create(
            model=self.args.model_path,  # 与启动时指定的模型名称一致
            prompt=author_inputs,
            max_tokens=self.args.max_author_len,
            temperature=self.args.gen_author_temp,
            top_p=0.95,
            logprobs=5,
            n=1,
            stop=stop_words,
        )
        generated_text = response.choices[0].text
        logprobs = response.choices[0].logprobs
        # 计算每个 token 的置信度
        all_confs = compute_confidence(logprobs)
                # 使用原始 solution 来定位每个步骤的 token 位置
        # 将原始 solution tokenize，找到每个步骤对应的 token 位置范围
        solution_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
        
        # 找到每个步骤在 solution 中的字符位置，然后转换为 token 位置
        step_confidences = []
        current_pos = 0
        steps = self.split_solution_into_steps(generated_text)
        
        for step_idx, step in enumerate(steps):
            # 找到当前步骤在 solution 中的起始位置
            step_start_pos = generated_text.find(step, current_pos)
            if step_start_pos == -1:
                logger.warning(f"Step {step_idx} not found in solution")
                step_confidences.append({
                    'step_text': step,
                    'step_index': step_idx,
                    'confidence': 0.0,
                    'token_start': 0,
                    'token_end': 0
                })
                continue
            
            step_end_pos = step_start_pos + len(step)
            
            # 将字符位置转换为 token 位置
            # 找到 step_start_pos 和 step_end_pos 对应的 token 位置
            step_text_before = generated_text[:step_start_pos]
            step_text_until_end = generated_text[:step_end_pos]
            
            tokens_before = tokenizer.encode(step_text_before, add_special_tokens=False)
            tokens_until_end = tokenizer.encode(step_text_until_end, add_special_tokens=False)
            
            token_start = len(tokens_before)
            token_end = len(tokens_until_end)
            
            # 确保 token 位置在有效范围内（使用生成的 logprobs 的长度）
            token_start = min(token_start, len(all_confs) - 1)
            token_end = min(token_end, len(all_confs))
            
            if token_start >= token_end:
                logger.warning(f"Invalid token range for step {step_idx}: [{token_start}, {token_end}]")
                step_confidences.append({
                    'step_text': step,
                    'step_index': step_idx,
                    'confidence': 0.0,
                    'token_start': token_start,
                    'token_end': token_end
                })
                continue
            
            # 计算该 token 区间的平均置信度：每个 token 的置信度相加再除以长度
            step_confs = all_confs[token_start:token_end]
            avg_confidence = np.mean(step_confs) if step_confs else 0.0
            
            step_confidences.append({
                'step_text': step,
                'step_index': step_idx,
                'confidence': avg_confidence,
                'token_start': token_start,
                'token_end': token_end
            })
            
            logger.debug(f"Step {step_idx}: tokens [{token_start}, {token_end}), confidence: {avg_confidence:.3f}")
            
            current_pos = step_end_pos
        # print(author)

        return generated_text, step_confidences

    def mutation(self, problem, solution, gt_ans):
        mut_pmt = read_prompt(self.args.mutation_prompt)
        mut_pmt = mut_pmt.format_map(
            {"problem": problem, "answer": gt_ans, "solution": solution})
        mut_inputs = format_input(mut_pmt, self.tokenizer)

        response = self.client.completions.create(
            model=self.args.model_path,  # 与启动时指定的模型名称一致
            prompt=mut_inputs,
            max_tokens=self.args.max_mutation_len,
            temperature=self.args.gen_mutation_temp,
            top_p=0.95,
            n=1,
            stop=stop_words,
        )
        mutation = response.choices[0].text
        # print(mutation)
        return mutation

    def pipeline(self, problem, gt_ans):

        # 初始种群
        best_fitness = 0
        best_solution = None
        solutions = self.init_population(problem, gt_ans)
        for iter in range(1, self.args.iter_num+1):
            solution_1, solution_2 = self.select(solutions, gt_ans)
            new_solution = self.crossover(problem, solution_1, solution_2, gt_ans)
            mutation = self.mutation(problem, new_solution, gt_ans)

            solutions.extend([new_solution, mutation])
            fitness_vals = np.array(self.calculate_fitness(solutions, gt_ans))

            # 找到当前代的最优解
            max_fitness_idx = np.argmax(fitness_vals)
            if fitness_vals[max_fitness_idx] > best_fitness:
                best_fitness = fitness_vals[max_fitness_idx]
                best_solution = solutions[max_fitness_idx]
            logger.info(f"进化第{iter}次, 最大fitness为{best_fitness:.3f}")

        logger.debug(f"进化完成, 最大fitness为{best_fitness:.3f}")
        # logger.info(best_solution)
        return best_solution, solutions


class BatchEvolOpt(object):
    def __init__(self, args, vllm, tokenizer):
        """进化问题problem的solution
            vllm: 部署的vllm
            pop_size: 初始种群大小，也就是初次生成的个数；
            tokenizer：分词器用于进行fitness计算
        """
        self.args = args
        self.vllm = vllm
        self.tokenizer = tokenizer
        self.stop_words = stop_words

    def init_population(self, problem, gt_ans):
        pass

    def select(self, solutions, gt_ans):
        pass

    def calculate_fitness(self, solutions, gt_ans):
        pass

    def crossover(self, problem, solution_1, solution_2, gt_ans):
        pass

    def mutation(self, problem, solution, gt_ans):
        pass

    def pipeline(self, problem, gt_ans):
        pass