
import numpy as np
from loguru import logger
from rouge_score import rouge_scorer
from vllm import SamplingParams

from base_evol import EvolOpt, BatchEvolOpt
from python_tool import gen_code_post, floatify_ans
from utils_evol import clean_solution, remove_dup_solutions, read_prompt, format_input, softmax, format_code_input
from fitness import cosine_scaled_reward, accuracy_reward, format_reward, cosine_lang_reward

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]


rwd_funcs = {
    "len_rwd": cosine_scaled_reward,
    "acc_rwd": accuracy_reward,
    "format_rwd": format_reward,
    "lang_rwd": cosine_lang_reward,
}


class BatchEvolOptV1(BatchEvolOpt):
    def __init__(self, args, llm, tokenizer):
        """进化问题problem的solution
            vllm: 部署的vllm的model
            pop_size: 初始种群大小，也就是初次生成的个数；
            tokenizer：分词器用于进行fitness计算
        """
        super().__init__(args, llm, tokenizer)

    def init_population(self, problems):
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
        gen_responses_temp =0.6
        all_problems_solutions = {
            i: {"input": gen_inputs[i], "solutions": [], "problem": problems[i]}
            for i in range(len(gen_inputs))}
        all_lens_flag = [len(value["solutions"]) < 4 for _, value in all_problems_solutions.items()]
        while any(all_lens_flag):
            idx = [ix for ix, flag in enumerate(all_lens_flag) if flag]
            prompts = [gen_inputs[i] for i in idx]

            if cnt > 5:
                gen_responses_temp = gen_responses_temp + 0.1
                logger.debug(f"多样性回复难度较大，提高温度{gen_responses_temp - 0.1:.1f}->{gen_responses_temp:.1f}")
                # logger.debug(f"多样性回复难度较大，提高温度{gen_responses_temp - 0.1:.1f}->{gen_responses_temp:.1f}")

            # 相同的temp的或者cnt的batch到一起生成 记录ix
            # 优先多batch的，注意顺序
            outputs = self.llm.generate(
                prompts,
                SamplingParams(
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=2048,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=[151645, 151643],
                ),
            )
            # outputs = self.vllm.generate(
            #     prompts,
            #     SamplingParams(
            #         temperature=gen_responses_temp,
            #         top_p=0.95,
            #         max_tokens=self.args.max_response_len,
            #         n=self.args.pop_size,
            #         stop=stop_words,
            #         stop_token_ids=[151645, 151643]
            #     ),
            # )
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

            all_lens_flag = [len(value["solutions"]) < 4 for _, value in
                             all_problems_solutions.items()]
        return_solutions = [
            (value["problem"], value["solutions"]) for _, value in all_problems_solutions.items()]
        return return_solutions

    def calculate_fitness(self, solutions):
        len_rwd = cosine_scaled_reward(solutions, self.tokenizer)

        # acc_rwd = accuracy_reward(gt_ans, solutions)

        format_rwd = format_reward(solutions)

        lang_rwd = cosine_lang_reward(solutions, self.tokenizer)

        rwd = []
        details_info = []
        for i in range(len(len_rwd)):
            rwd.append(float(len_rwd[i] + format_rwd[i] + lang_rwd[i]))
            details_info.append(
                {"solutions": solutions[i], "rwd": rwd[i], "len_rwd": len_rwd[i],
                 "format_rwd": format_rwd[i], "lang_rwd": lang_rwd[i]}
            )
        # sel_p = softmax(rwd)
        # return sel_p, details_info
        return rwd

    def select(self, solutions_list):

        def get_select(solutions):
            fitness_vals = self.calculate_fitness(solutions)
            prob = softmax(fitness_vals) # 计算每个个体被选择的概率
            solutions = np.array(solutions)
            selected_idx = np.random.choice(range(len(solutions)), size=2, p=prob)
            return list(solutions[selected_idx])

        return_selections = []
        for i, solutions in enumerate(solutions_list):
            selected_solutions = get_select(solutions)
            return_selections.append(selected_solutions)
        return return_selections

    def crossover(self, problems, select_solutions):
        critic_pmt = read_prompt(self.args.critic_prompt)
        critic_inputs = []
        for i, solutions in enumerate(select_solutions):
            solution_1, solution_2 = solutions
            critic_pmts = critic_pmt.format_map(
                {"problem": problems[i], "solution_1": solution_1,
                 "solution_2": solution_2})
            critic_input = format_input(critic_pmts, self.tokenizer)
            critic_inputs.append(critic_input)
        outputs = self.llm.generate(
            critic_inputs,
            SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048,
                n=1,
                stop=stop_words,
                stop_token_ids=[151645, 151643],
            ),
        )
        # outputs = self.vllm.generate(
        #     critic_inputs,
        #     SamplingParams(
        #         temperature=self.args.gen_critic_temp,
        #         top_p=0.95,
        #         max_tokens=self.args.max_critic_len,
        #         n=1,
        #         stop=stop_words,
        #         stop_token_ids=[151645, 151643]
        #     ),
        # )
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

        outputs = self.llm.generate(
            author_inputs,
            SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048,
                n=1,
                stop=stop_words,
                stop_token_ids=[151645, 151643],
            ),
        )        
        # outputs = self.vllm.generate(
        #     author_inputs,
        #     SamplingParams(
        #         temperature=self.args.gen_author_temp,
        #         top_p=0.95,
        #         max_tokens=self.args.max_author_len,
        #         n=1,
        #         stop=stop_words,
        #         stop_token_ids=[151645, 151643]
        #     ),
        # )
        outputs = sorted(
            outputs, key=lambda x: int(x.request_id)
        )  # sort outputs by request_id
        author = [output.outputs[0].text for output in outputs]

        return author

    def mutation(self, problems, solutions):
        mut_pmt = read_prompt(self.args.mutation_prompt)
        mut_inputs = []
        for i, solution in enumerate(solutions):
            # print('$$$$$$$$$$$$')
            # print(f"Current index i: {i}")
            # # print(f"Solution:{solution}")
            # print(f"Problem: {problems[i]}, Type: {type(problems[i])}, Length: {len(problems[i])}")
            # print(f"Answer: {gt_anes[i]}, Type: {type(gt_anes[i])}, Length: {len(gt_anes[i])}")
            # print(f"Solution: {solution}, Type: {type(solution)}, Length: {len(solution)}")
            mut_pmts = mut_pmt.format_map(
                {"problem": problems[i], "solution": solution})
            mut_input = format_input(mut_pmts, self.tokenizer)
            mut_inputs.append(mut_input)
        
        outputs = self.llm.generate(
            mut_inputs,
            SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048,
                n=1,
                stop=stop_words,
                stop_token_ids=[151645, 151643],
            ),
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

    def pipeline(self, problems):

        # 初始种群, batch problems and gt_anes
        # list 包含了problem和solutions ()
        data_pop = self.init_population(problems)

        def get_best_solution(solutions):
            fitness_vals = np.array(self.calculate_fitness(solutions))
            # 找到当前代的最优解
            best_fitness = 0.0
            max_fitness_idx = np.argmax(fitness_vals)
            if fitness_vals[max_fitness_idx] > best_fitness:
                best_fitness = fitness_vals[max_fitness_idx]
                best_solution = solutions[max_fitness_idx]
            return best_solution, best_fitness

        # 自进化
        all_problems = problems
        all_solutions = [dp_pop[1] for dp_pop in data_pop]
        best_solutions, best_fitnesses = [], []
        for iter in range(1, 4):
            select_solutions = self.select(all_solutions)
            new_solutions = []
            while len(new_solutions) != len(all_problems):
                new_solutions = self.crossover(all_problems, select_solutions)
                if len(new_solutions) != len(all_problems):
                    logger.debug(f"This round wrong!"
                                 f"problems:{len(all_problems)}, new_solutions: "
                                 f"{len(new_solutions)}")
            mutations = self.mutation(all_problems, new_solutions)

            for i, new_solution in enumerate(new_solutions):
                all_solutions[i].extend([new_solution, mutations[i]])
                best_solution, best_fitness = get_best_solution(all_solutions[i])
                best_solutions.append(best_solution)
                best_fitnesses.append(best_fitness)
            logger.info(f"进化第{iter}次, 最大fitness为{np.mean(best_fitnesses):.3f}")
            best_solutions, best_fitnesses = [], []

        for i, solutions in  enumerate(all_solutions):
            best_solution, best_fitness = get_best_solution(solutions)
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
