
import numpy as np
import re
from loguru import logger
from rouge_score import rouge_scorer
from vllm import SamplingParams
import math

from base_evol import EvolOpt, BatchEvolOpt
from python_tool import gen_code_post, floatify_ans
from utils import clean_solution, remove_dup_solutions, read_prompt, format_input, softmax, format_code_input
from fitness import cosine_scaled_reward, accuracy_reward, format_reward, cosine_lang_reward

import random
def compute_step_entropies(logprobs):
    """
    计算 LLM 生成的每个 token 的条件信息熵 (H_j)。
    假设 logprobs 包含每个时间步 Top-K 个 token 的对数概率（ln p）。
    """
    entropies = []
    ln2 = math.log(2)  # ln(2)，用于换底公式

    for token_logprobs in logprobs:
        H_j = 0.0
        
        # 提取该时间步所有 Top-K token 的 logprobs (ln p)
        # 注意：这里需要遍历所有返回的 logprobs，不仅仅是 rank=1 的那个
        for logprob_obj in token_logprobs.values():
            ln_p = logprob_obj.logprob 
            
            # 1. 转换为概率 p: p = exp(ln p)
            p = math.exp(ln_p) 
            
            # 2. 如果 p > 0，计算熵项 -p * log2(p)
            if p > 0:
                # 换底公式: log2(p) = ln(p) / ln(2)
                log2_p = ln_p / ln2
                
                # 香农熵公式：H = - Sum(p * log2(p))
                # 注意：如果只使用 Top-K，这只是一个近似的熵值（下限估计）
                H_j += p * log2_p
        
        # H_j 此时是 Sum(p * log2(p)) 的负值
        entropies.append(-H_j)
            
    return entropies

            
        
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]


rwd_funcs = {
    "len_rwd": cosine_scaled_reward,
    "acc_rwd": accuracy_reward,
    "format_rwd": format_reward,
    "lang_rwd": cosine_lang_reward,
}


class BatchEvolOptV1(BatchEvolOpt):
    def __init__(self, args, vllm, tokenizer):
        """进化问题problem的solution
            vllm: 部署的vllm的model
            pop_size: 初始种群大小，也就是初次生成的个数；
            tokenizer：分词器用于进行fitness计算
        """
        super().__init__(args, vllm, tokenizer)

    def init_population(self, problems, gt_anses):
        """

        Args:
            problems: batch problems
            gt_anses: batch gt_anses

        Returns:

        """
        init_gen_pmt = read_prompt(self.args.gen_responses_prompt)
        gen_inputs = []
        for problem in problems:
            gen_pmt = init_gen_pmt.format_map({"problem": problem, "occup": "<your answer>"})
            gen_inputs.append(format_input(gen_pmt, self.tokenizer))
        ## 问题batch化 将其他的没有解决的在放到一个batch里面 同步提高temp
        cnt = 0
        gen_responses_temp = self.args.gen_responses_temp
        all_problems_solutions = {
            i: {"input": gen_inputs[i], "solutions": [], "problem": problems[i], "answer": gt_anses[i]}
            for i in range(len(gen_inputs))}
        all_lens_flag = [len(value["solutions"]) < self.args.pop_size for _, value in all_problems_solutions.items()]
        while any(all_lens_flag):
            idx = [ix for ix, flag in enumerate(all_lens_flag) if flag]
            prompts = [gen_inputs[i] for i in idx]

            if cnt > 5:
                gen_responses_temp = gen_responses_temp + 0.1
                logger.debug(f"多样性回复难度较大，提高温度{gen_responses_temp - 0.1:.1f}->{gen_responses_temp:.1f}")
                # logger.debug(f"多样性回复难度较大，提高温度{gen_responses_temp - 0.1:.1f}->{gen_responses_temp:.1f}")

            # 相同的temp的或者cnt的batch到一起生成 记录ix
            # 优先多batch的，注意顺序
            outputs = self.vllm.generate(
                prompts,
                SamplingParams(
                    temperature=gen_responses_temp,
                    top_p=0.95,

                    max_tokens=self.args.max_response_len,
                    n=self.args.pop_size,
                    stop=stop_words,
                    stop_token_ids=[151645, 151643]
                ),
            )
            outputs = sorted(
                outputs, key=lambda x: int(x.request_id)
            )  # sort outputs by request_id

            def parse_output(outputs):
                all_responses = []
                for output in outputs:
                    responses = []
                    for res in output.outputs:
                        responses.append(res.text)
                    all_responses.append(responses)
                return all_responses
            responses = parse_output(outputs)

            if cnt < 10:
                for i, solutions in enumerate(responses):
                    solutions = remove_dup_solutions(solutions)
                    ids = idx[i]
                    all_problems_solutions[ids]["solutions"].extend(solutions)
                cnt += 1
            else:
                logger.debug(f"难度很大，丢弃去重操作")
                break

            all_lens_flag = [len(value["solutions"]) < self.args.pop_size for _, value in
                             all_problems_solutions.items()]
        return_solutions = [
            (value["problem"], value["solutions"], value["answer"]) for _, value in all_problems_solutions.items()]
        return return_solutions

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

    def select(self, solutions_list, gt_answers):

        def get_select(solutions, gt_ans):
            fitness_vals = self.calculate_fitness(solutions, gt_ans)
            prob = softmax(fitness_vals) # 计算每个个体被选择的概率
            # solutions = np.array(solutions)
            # selected_idx = np.random.choice(range(len(solutions)), size=2, p=prob)
            # 随机选择
            selected_idx = np.random.choice(len(solutions), size=2, replace=False)
            # return list(solutions[selected_idx])
            return [solutions[i] for i in selected_idx]

        return_selections = []
        for i, solutions in enumerate(solutions_list):
            selected_solutions = get_select(solutions, gt_answers[i])
            return_selections.append(selected_solutions)
        return return_selections
    
    def split_solution_into_steps(self, solution: str) -> list:
        """
        将 solution 按步骤切割，支持如下格式的 Step 标记：
        - Step 1
        - step1
        - ### Step 1
        - **Step 1**
        - - Step 1
        - Step 1:
        - Step 1.
        """
        step_pattern = r'(?i)(?:^|\n)\s*(?:[#*\-\d\.]+\s*)?(step\s*\d+[.:]?\s*)(.*?)(?=\n\s*(?:[#*\-\d\.]+\s*)?step\s*\d+[.:]?|$)'
        
        matches = re.findall(step_pattern, solution, re.DOTALL)

        if matches:
            steps = []
            for header, content in matches:
                combined = (header + content).strip()
                if combined:
                    steps.append(combined)
        else:
            # 无 step 关键词则按行分割
            steps = [s.strip() for s in solution.split("\n") if s.strip()]
        if len(steps) > 10 or len(steps) == 1:
            return None

        return steps if steps else [solution]


    def crossover(self, problems, select_solutions, gt_answers):
        crossover_data = []
        critic_pmts_list = []
        critic_inputs = []
        # 加载三个不同的 Critic 模板
        # pmt_correct_fusion = read_prompt(self.args.critic_correct_fusion_pmt)
        # pmt_correct_guidance = read_prompt(self.args.critic_correct_guidance_pmt)
        # pmt_wrong_exploration = read_prompt(self.args.critic_wrong_exploration_pmt)
        
        # 加载no_gt的critique
        pmt_judge_critique = read_prompt(self.args.judge_critique_pmt)

        for i, solutions in enumerate(select_solutions):
            problem = problems[i]
            gt_answer = gt_answers[i]
            solution_1, solution_2 = solutions
            critic_pmts = pmt_judge_critique.format_map({
                "problem": problem, "solution_1": solution_1, "solution_2": solution_2
            })
            # 1. 适应度判断 (保持不变)
            # print(accuracy_reward(gt_answer, [solution_1]))
            is_correct_1 = accuracy_reward(gt_answer, [solution_1])[0] >= 0.5
            is_correct_2 = accuracy_reward(gt_answer, [solution_2])[0] >= 0.5

            # 5. 场景分类与动态 Prompt 构造
        #     if is_correct_1 and is_correct_2:
        #         scene = "TwoCorrect"
        #         # 选择两正解模板
        #         critic_pmts = pmt_correct_fusion.format_map({
        #             "problem": problem, "solution_1": solution_1, "solution_2": solution_2, "correct_answer": gt_answer
        #         })
            
        #     elif is_correct_1 != is_correct_2:
        #         scene = "OneCorrectOneWrong"
        #         S_C = solution_1 if is_correct_1 else solution_2
        #         S_W = solution_2 if is_correct_1 else solution_1
        #         # 选择一正一负模板
        #         critic_pmts = pmt_correct_guidance.format_map({
        #             "problem": problem, "correct_solution": S_C, "wrong_solution": S_W, "correct_answer": gt_answer
        #         })
                
        #     else: # Both are wrong
        #         scene = "TwoWrong"
        #         # 选择两负解模板
        #         critic_pmts = pmt_wrong_exploration.format_map({
        #             "problem": problem, "solution_1": solution_1, "solution_2": solution_2, "correct_answer": gt_answer
        #         })
            
        #     critic_pmts_list.append(critic_pmts)
        #     crossover_data.append({'problem': problem, 'gt_answer': gt_answer}) # 存储所需数据

        # critic_inputs = [format_input(pmt, self.tokenizer) for pmt in critic_pmts_list]
                

            # critic_pmts = critic_pmt.format_map(
            #     {"problem": problems[i], "answer": gt_answers[i], "solution_1": solution_1,
            #      "solution_2": solution_2})
            critic_input = format_input(critic_pmts, self.tokenizer)
            critic_inputs.append(critic_input)
      
        outputs = self.vllm.generate(
            critic_inputs,
            SamplingParams(
                temperature=self.args.gen_critic_temp,
                top_p=0.95,
                max_tokens=self.args.max_critic_len,
                n=1,
                stop=stop_words,
                stop_token_ids=[151645, 151643]
            ),
        )
        outputs = sorted(
            outputs, key=lambda x: int(x.request_id)
        )  # sort outputs by request_id
        critics = [output.outputs[0].text for output in outputs]

        # 根据feedback增进
        author_inputs = []
        author_pmt = read_prompt(self.args.author_prompt)
        for i, solutions in enumerate(select_solutions):
            solution_1, solution_2 = solutions
            author_pmts = author_pmt.format_map(
                {"problem": problems[i], "solution_1": solution_1,
                 "solution_2": solution_2, "critic_feedback": critics[i]})
            author_input = format_input(author_pmts, self.tokenizer)
            author_inputs.append(author_input)

        outputs = self.vllm.generate(
            author_inputs,
            SamplingParams(
                temperature=self.args.gen_author_temp,
                top_p=0.95,
                max_tokens=self.args.max_author_len,
                n=1,
                logprobs=5,
                stop=stop_words,
                stop_token_ids=[151645, 151643]
            ),
        )
        outputs = sorted(
            outputs, key=lambda x: int(x.request_id)
        )  # sort outputs by request_id
        author = [output.outputs[0].text for output in outputs]
                
        # 计算每个 output 的步骤置信度
        wrong_step_indexs = []
        confs = []
        
        for output in outputs:
            all_step_confidences = []  # 存储每个 output 的步骤置信度列表
            output_obj = output.outputs[0]
            text = output_obj.text
            # print(text)
            
            # 将 output 分成 step
            steps = self._split_solution_into_steps(text)
            # print(steps)

            if len(steps) <= 10 and len(steps) > 1:
                logger.debug(f"Crossover output split into {len(steps)} steps")
                
                # 计算每个 step 的置信度
                if output_obj.logprobs:
                    step_confidences = self._calculate_step_confidences_from_logprobs(
                        text, steps, output_obj.logprobs, self.tokenizer
                    )
                    all_step_confidences.append(step_confidences)
                    best_step = max(step_confidences, key=lambda x: x["confidence"])
                    wrong_index = best_step["step_index"]
                    wrong_step_indexs.append(wrong_index)
                    conf = best_step['confidence']
                    confs.append(conf)
                    
                    # # 记录每个步骤的置信度
                    # for step_conf in step_confidences:
                    #     logger.debug(f"Crossover Step {step_conf['step_index']}: tokens [{step_conf['token_start']}, {step_conf['token_end']}), confidence: {step_conf['confidence']:.3f}")
                else:
                    logger.warning("No logprobs available for crossover output")
                    all_step_confidences.append([])
                    wrong_step_indexs.append([])
                    confs.append([])
            else:
                wrong_step_indexs.append([])
                confs.append([])
        return author, wrong_step_indexs, confs
    
    def _split_solution_into_steps(self, solution: str) -> list:
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
    
    def _calculate_step_confidences_from_logprobs(self, text: str, steps: list, logprobs, tokenizer) -> list:
        """从已有的 logprobs 计算每个步骤的置信度
        
        Args:
            text: 生成的文本
            steps: 步骤列表
            logprobs: 已有的 logprobs
            tokenizer: tokenizer
            
        Returns:
            list of dict: 每个步骤的置信度信息，包含 step_text, step_index, confidence, token_start, token_end
        """
        if not logprobs:
            logger.warning("No logprobs provided")
            return []
        
        # 计算每个 token 的置信度
        all_confs = compute_step_entropies(logprobs)
        
        # 找到每个步骤在 text 中的字符位置，然后转换为 token 位置
        step_confidences = []
        current_pos = 0
        
        for step_idx, step in enumerate(steps):
            # 找到当前步骤在 text 中的起始位置
            step_start_pos = text.find(step, current_pos)
            if step_start_pos == -1:
                logger.warning(f"Step {step_idx} not found in text")
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
            step_text_before = text[:step_start_pos]
            step_text_until_end = text[:step_end_pos]
            
            tokens_before = tokenizer.encode(step_text_before, add_special_tokens=False)
            tokens_until_end = tokenizer.encode(step_text_until_end, add_special_tokens=False)
            
            token_start = len(tokens_before)
            token_end = len(tokens_until_end)
            
            # 确保 token 位置在有效范围内
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
            step_Hs = all_confs[token_start:token_end]
            avg_entropy = np.mean(step_Hs) if step_Hs else 0.0
            
            # 更改: 将字典键 'confidence' 替换为 'entropy'
            step_confidences.append({
                'step_text': step,
                'step_index': step_idx,
                'confidence': avg_entropy, # 更改: 存储平均熵
                'token_start': token_start,
                'token_end': token_end
            })
            
            
            logger.debug(f"Step {step_idx}: tokens [{token_start}, {token_end}), confidence: {avg_entropy:.3f}")
            
            current_pos = step_end_pos
        
        return step_confidences
    

    def mutation(self, problems, solutions, gt_anes, wrong_indexs, confs):
        temps = []
        if confs:
            # confs: 应该是一个列表/数组，其中每个元素是该 solution 的最大熵 H_max
            # 确保 temps 也是一个列表，长度与 mut_inputs (即 problems) 匹配
            # 这里假设 confs 已经和 problems 顺序匹配
            for h_max in confs:
                if h_max:
                    temp = self.args.gen_mutation_temp * (1 + 5 * h_max)
                    temps.append(temp)
                else:
                    temps.append(self.args.gen_author_temp)
            # temps = [self.args.gen_mutation_temp * (1 + 10 * h_max) for h_max in confs]
        else:
            # 如果没有熵信息，使用默认温度
            temps = [self.args.gen_mutation_temp] * len(problems)
        mut_pmt_1 = read_prompt(self.args.mutation_prompt_1)
        mut_pmt_2 = read_prompt(self.args.mutation_prompt_2)
        mut_pmt_3 = read_prompt(self.args.mutation_prompt_3)
        mut_inputs = []
        for i, (solution, wrong_index) in enumerate(zip(solutions, wrong_indexs)):
            # print('$$$$$$$$$$$$')
            # print(f"Current index i: {i}")
            # # print(f"Solution:{solution}")
            # print(f"Problem: {problems[i]}, Type: {type(problems[i])}, Length: {len(problems[i])}")
            # print(f"Answer: {gt_anes[i]}, Type: {type(gt_anes[i])}, Length: {len(gt_anes[i])}")
            # print(f"Solution: {solution}, Type: {type(solution)}, Length: {len(solution)}")
            if wrong_index:
                # mut_pmts = mut_pmt_1.format_map(
                #     {"problem": problems[i], "solution": solution, "wrong_step_index": wrong_index})
                if wrong_index != 0:
                    # mut_pmts = mut_pmt_1.format_map(
                    #     {"problem": problems[i], "wrong_step_index": solution[:wrong_index], "correct_answer": gt_anes[i]})
                    mut_pmts = mut_pmt_1.format_map(
                        {"problem": problems[i], "wrong_step_index": solution[:wrong_index]})
                    mut_input = format_input(mut_pmts, self.tokenizer)
                else:
                    # mut_pmts = mut_pmt_3.format_map(
                    #     {"problem": problems[i], "correct_answer": gt_anes[i]})
                    mut_pmts = mut_pmt_3.format_map(
                        {"problem": problems[i]})
                mut_inputs.append(mut_pmts)
            else:
                # mut_pmts = mut_pmt_2.format_map(
                #     {"problem": problems[i], "solution": solution, "correct_answer": gt_anes[i]})
                mut_pmts = mut_pmt_2.format_map(
                    {"problem": problems[i], "solution": solution})
                mut_input = format_input(mut_pmts, self.tokenizer)
                mut_inputs.append(mut_input)
        sampling_params_list = []
        for i, prompt in enumerate(mut_inputs):
            # 为第 i 个请求使用第 i 个计算出的温度 temps[i]
            params = SamplingParams(
                temperature=float(temps[i]),  # 使用该请求特有的温度
                top_p=0.95,
                max_tokens=self.args.max_mutation_len,
                n=1,
                stop=stop_words,
                stop_token_ids=[151645, 151643]
            )
            sampling_params_list.append(params)
        
        outputs = self.vllm.generate(
            mut_inputs,
            sampling_params=sampling_params_list, # 传入列表
        )        

        # outputs = self.vllm.generate(
        #     mut_inputs,
        #     SamplingParams(
        #         temperature=self.args.gen_mutation_temp,
        #         top_p=0.95,
        #         max_tokens=self.args.max_mutation_len,
        #         n=1,
        #         stop=stop_words,
        #         stop_token_ids=[151645, 151643]
        #     ),
        # )
        outputs = sorted(
            outputs, key=lambda x: int(x.request_id)
        )  # sort outputs by request_id
        mutation = [output.outputs[0].text for output in outputs]
        # print(mutation)
        return mutation

    def pipeline(self, problems, gt_anes):

        # 初始种群, batch problems and gt_anes
        # list 包含了problem和solutions ()
        data_pop = self.init_population(problems, gt_anes)

        def get_best_solution(solutions, gt_ans):
            fitness_vals = np.array(self.calculate_fitness(solutions, gt_ans))
            # 找到当前代的最优解
            # best_fitness = 0.0
            # max_fitness_idx = np.argmax(fitness_vals)
            # if fitness_vals[max_fitness_idx] > best_fitness:
            #     best_fitness = fitness_vals[max_fitness_idx]
            #     best_solution = solutions[max_fitness_idx]
            best_solution = random.choice(solutions)
            best_fitness = 0.0
            return best_solution, best_fitness

        # 自进化
        all_problems = problems
        all_answers = gt_anes
        all_solutions = [dp_pop[1] for dp_pop in data_pop]
        best_solutions, best_fitnesses = [], []
        for iter in range(1, self.args.iter_num+1):
            select_solutions = self.select(all_solutions, all_answers)
            new_solutions = []
            while len(new_solutions) != len(all_problems):
                new_solutions, wrong_indexs, confs = self.crossover(all_problems, select_solutions, all_answers)
                if len(new_solutions) != len(all_problems):
                    logger.debug(f"This round wrong!"
                                 f"problems:{len(all_problems)}, new_solutions: "
                                 f"{len(new_solutions)}, gt_anes: {len(gt_anes)}")
            mutations = self.mutation(all_problems, new_solutions, all_answers, wrong_indexs, confs)

            for i, new_solution in enumerate(new_solutions):
                all_solutions[i].extend([new_solution, mutations[i]])
                # all_solutions[i].extend([new_solution])
                # all_solutions[i].extend([mutations[i]])
                best_solution, best_fitness = get_best_solution(all_solutions[i], gt_anes[i])
                best_solutions.append(best_solution)
                best_fitnesses.append(best_fitness)
            logger.info(f"进化第{iter}次, 最大fitness为{np.mean(best_fitnesses):.3f}")
            best_solutions, best_fitnesses = [], []

        for i, solutions in  enumerate(all_solutions):
            best_solution, best_fitness = get_best_solution(solutions, gt_anes[i])
            best_solutions.append(best_solution)
            best_fitnesses.append(best_fitness)
        logger.debug(f"进化完成, 最大fitness为{np.mean(best_fitnesses):.3f}")
            # logger.info(best_solution)
        return best_solutions, all_solutions


class EvolOptV1_1(EvolOpt):

    def __init__(self, args, client, tokenizer):
        """进化问题problem的solution
            client: 部署的vllm的client
            pop_size: 初始种群大小，也就是初次生成的个数；
            tokenizer：分词器用于进行fitness计算
        """
        self.args = args
        self.client = client
        self.tokenizer = tokenizer

        self.pop_size = self.args.pop_size

        # 1）初始种群的改进：不同模板的few-shot（代码解决或者math解决）；
        #    - 控制few-shot（相当于模板）和温度
        #    - 每个类型生成多样性4个 一共8个
        # 2）选择的改进：基于支配-差异性分数 [多样性和支配性]
        #    - 支配-差异性分数计算：支配mask+相似性计算(两两相似)
        # 3）交叉的改进：抄袭检测  交叉和变异的步骤暂时不改变
        # 4）种群管理：最优种群 + 进化过程种群 （原始种群+进化种群+进化是否成功检测是否保留到最佳的里面）
        #    - 最优个数一共8个，每次更新替换其中2个
        #    - 记录进化过程

    def init_code_population(self, problem, gt_ans, pop_size):
        gen_code_q = f"Problem: {problem}"
        gen_code_i = format_code_input(gen_code_q, self.tokenizer)

        cnt = 0
        solutions = []
        gen_responses_temp = self.args.gen_responses_temp

        while len(solutions) < self.args.pop_size:
            # 生成多样性回复困难则提高温度
            if cnt > 5:
                gen_responses_temp = gen_responses_temp + 0.1
                logger.debug(f"code多样性回复难度较大，提高温度{gen_responses_temp - 0.1:.1f}->{gen_responses_temp:.1f}")

            response = self.client.completions.create(
                model=self.args.model_path,  # 与启动时指定的模型名称一致
                prompt=gen_code_i,
                max_tokens=self.args.max_response_len,
                temperature=gen_responses_temp,
                top_p=0.95,
                # repetition_penalty=1.1,
                n=pop_size,
                stop=stop_words,
            )
            for chose in response.choices:
                text = chose.text
                if "```python" in text:
                    # 执行+处理
                    solution = gen_code_post(text, gt_ans)
                    solutions.append(clean_solution(solution))

            # 如果生成的多样性回复难度太大 那么不需要在去重了
            if cnt < 10:
                solutions = remove_dup_solutions(solutions)
                cnt += 1
            else:
                logger.debug(f"code生成难度很大，丢弃去重操作")
                break

        return solutions

    def init_cot_population(self, problem, pop_size):

        gen_pmt = read_prompt(self.args.gen_responses_prompt)
        gen_pmt = gen_pmt.format_map({"problem": problem, "occup": "<your answer>"})
        gen_inputs = format_input(gen_pmt, self.tokenizer)
        # logger.info(f"Formated input:\n {gen_inputs}")

        cnt = 0
        solutions = []
        gen_responses_temp = self.args.gen_responses_temp

        while len(solutions) < self.args.pop_size:
            # 生成多样性回复困难则提高温度
            if cnt > 2:
                gen_responses_temp = gen_responses_temp + 0.1
                logger.debug(f"多样性回复难度较大，提高温度{gen_responses_temp - 0.1:.1f}->{gen_responses_temp:.1f}")

            response = self.client.completions.create(
                model=self.args.model_path,  # 与启动时指定的模型名称一致
                prompt=gen_inputs,
                max_tokens=self.args.max_response_len,
                temperature=gen_responses_temp,
                top_p=0.95,
                # repetition_penalty=1.1,
                n=pop_size,
                stop=stop_words,
            )
            for chose in response.choices:
                solutions.append(clean_solution(chose.text))

            # 如果生成的多样性回复难度太大 那么不需要在去重了
            if cnt < 5:
                solutions = remove_dup_solutions(solutions)
                cnt += 1
            else:
                logger.debug(f"难度很大，丢弃去重操作")
                break

        return solutions

    def init_population(self, problem, gt_ans):
        # code_solutions = self.init_code_population(problem, gt_ans, self.args.pop_size)
        # print(code_solutions)
        cot_solutions = self.init_cot_population(problem, 2*self.args.pop_size)
        # solutions = code_solutions + cot_solutions
        solutions = cot_solutions
        return solutions

    def compute_dominance_dissimilarity_score(self, details_info, similarity_threshold=0.9, penalty_lambda=1.0):
        n = len(details_info)
        # 提取文本和适应度分数
        texts = [detail_info["solutions"] for detail_info in details_info]
        fitness_scores = np.array([detail_info["rwd"] for detail_info in details_info])

        # 计算两两RougeL相似度分数
        dissimilarity_matrix = np.zeros((n, n))
        similarity_matrix = np.zeros((n, n))  # 额外存储相似性矩阵
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        for i in range(n):
            for j in range(n):
                if i != j:
                    score = scorer.score(texts[i], texts[j])["rougeL"].fmeasure
                    similarity_matrix[i, j] = score  # 存储相似度
                    dissimilarity_matrix[i, j] = min(-score, -0.1)  # 设定最小差异性

        # 计算支配性矩阵（注意方向调整）
        dominance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if fitness_scores[i] < fitness_scores[j]:  # solution_j 支配 solution_i
                    dominance_matrix[j, i] = 1

        # 计算支配-差异性分数矩阵
        dominance_dissimilarity_matrix = dominance_matrix * dissimilarity_matrix

        # 计算每个 solution 的支配-差异性分数（按行求和）
        dds_scores = np.sum(dominance_dissimilarity_matrix, axis=0)

        # 精准相似性惩罚项（仅惩罚索引小的 solutions）
        similarity_penalty = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):  # 只对 `i < j` 进行惩罚
                # if similarity_matrix[i, j] > similarity_threshold and fitness_scores[j] > fitness_threshold:
                if similarity_matrix[i, j] >= similarity_threshold:
                    similarity_penalty[i] += penalty_lambda  # 仅惩罚序列号小的 solutions
        # 计算最终的支配-差异性分数，并保留 3 位小数
        final_dds_scores = [round(score - similarity_penalty[i], 3) for i, score in enumerate(dds_scores)]

        return final_dds_scores

    def calculate_fitness(self, solutions, gt_ans):
        # based on dominance_dissimilarity
        # fitness越大并且越多样性的solution保存下来
        len_rwd = cosine_scaled_reward(gt_ans, solutions, self.tokenizer)
        acc_rwd = accuracy_reward(gt_ans, solutions)
        format_rwd = format_reward(gt_ans, solutions)
        # 丢弃语言一致性 rwd
        # lang_rwd = cosine_lang_reward(gt_ans, solutions, self.tokenizer)
        rwd = []
        details_info = []
        for i in range(len(acc_rwd)):
            rwd.append(float(len_rwd[i] + acc_rwd[i] + format_rwd[i]))
            details_info.append(
                {"solutions": solutions[i], "rwd": rwd[i], "len_rwd": len_rwd[i], "acc_rwd": acc_rwd[i],
                 "format_rwd": format_rwd[i],}
            )
        dds_scores = self.compute_dominance_dissimilarity_score(details_info)
        return dds_scores

    def keep_best_population(self, solutions, fitness_vals, pop_size):
        sorted_solutions = [sol for _, sol in sorted(zip(fitness_vals, solutions))]
        return sorted_solutions[-pop_size:]

    def pipeline(self, problem, gt_ans):

        # 初始种群
        best_fitness = 0
        best_solution = None
        all_solutions = self.init_population(problem, gt_ans)
        all_fitness_vals = self.calculate_fitness(all_solutions, gt_ans)
        best_solutions = self.keep_best_population(all_solutions, all_fitness_vals, 2*self.args.pop_size)
        for iter in range(1, self.args.iter_num+1):
            solution_1, solution_2 = self.select(best_solutions, gt_ans)
            new_solution = self.crossover(problem, solution_1, solution_2, gt_ans)
            mutation = self.mutation(problem, new_solution, gt_ans)

            all_solutions.extend([new_solution, mutation])
            best_solutions.extend([new_solution, mutation])
            fitness_vals = np.array(self.calculate_fitness(best_solutions, gt_ans))
            best_solutions = self.keep_best_population(best_solutions, fitness_vals, 2*self.args.pop_size)

            # 找到当前代的最优解
            max_fitness_idx = np.argmax(fitness_vals)
            if fitness_vals[max_fitness_idx] > best_fitness:
                best_fitness = fitness_vals[max_fitness_idx]
                best_solution = all_solutions[max_fitness_idx]
            logger.info(f"进化第{iter}次, 最大支配-差异性为{best_fitness:.3f}")

        logger.debug(f"进化完成, 最大fitness为{best_fitness:.3f}")
        # logger.info(best_solution)
        return best_solutions, all_solutions
