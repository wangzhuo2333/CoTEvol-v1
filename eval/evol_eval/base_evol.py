import numpy as np
from loguru import logger
from rouge_score import rouge_scorer

# from python_tool import gen_code_post, floatify_ans
from utils_evol import clean_solution, remove_dup_solutions, read_prompt, format_input, softmax, format_code_input
from fitness import cosine_scaled_reward, accuracy_reward, format_reward, cosine_lang_reward
from vllm import LLM, SamplingParams

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]


class EvolOpt(object):

    def __init__(self, args, llm, tokenizer):
        """进化问题problem的solution
            client: 部署的vllm的client
            pop_size: 初始种群大小，也就是初次生成的个数；
            tokenizer：分词器用于进行fitness计算
        """
        self.args = args
        self.tokenizer = tokenizer
        self.llm = llm
        self.stoe_words = stop_words

    def init_population(self, problem):
        gen_pmt = read_prompt(self.args.gen_responses_prompt)
        gen_pmt = gen_pmt.format_map({"problem": problem, "occup": "<your answer>"})
        gen_inputs = format_input(gen_pmt, self.tokenizer)
        # logger.info(f"Formated input:\n {gen_inputs}")

        cnt = 0
        solutions = []
        # gen_responses_temp = self.args.gen_responses_temp
        gen_responses_temp = 0.6

        # while len(solutions) < self.args.pop_size:
        while len(solutions) < 4:
            # 生成多样性回复困难则提高温度
            if cnt > 5:
                gen_responses_temp = gen_responses_temp + 0.1
                logger.debug(f"多样性回复难度较大，提高温度{gen_responses_temp - 0.1:.1f}->{gen_responses_temp:.1f}")
            responses = self.llm.generate(
                gen_inputs,
                SamplingParams(
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=2048,
                    n=4,
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in self.args.model_name_or_path.lower()
                        else None
                    ),
                ),
            )
            # print(len(response))
            # outputs = sorted(
            #     response, key=lambda x: int(x.request_id)
            # )  # sort outputs by request_id
            # outputs = [output.outputs[0].text for output in outputs]
            # response = self.client.completions.create(
            #     model=self.args.model_name_or_path,  # 与启动时指定的模型名称一致
            #     prompt=gen_inputs,
            #     # max_tokens=self.args.max_response_len,
            #     max_tokens=2048,
            #     temperature=gen_responses_temp,
            #     top_p=0.95,
            #     # repetition_penalty=1.1,
            #     n=4,
            #     stop=stop_words,
            # )
            for response in responses:
                solutions.append(clean_solution(response.outputs[0].text))

            # 如果生成的多样性回复难度太大 那么不需要在去重了
            if cnt < 10:
                solutions = remove_dup_solutions(solutions)
                cnt += 1
            else:
                logger.debug(f"难度很大，丢弃去重操作")
                break

        return solutions

    def calculate_fitness(self, solutions):
        len_rwd = cosine_scaled_reward(solution=solutions, tokenizer=self.tokenizer)

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

    def select(self, solutions):
        fitness_vals = self.calculate_fitness(solutions)
        prob = softmax(fitness_vals) # 计算每个个体被选择的概率
        solutions = np.array(solutions)
        selected_idx = np.random.choice(range(len(solutions)), size=2, p=prob)
        return list(solutions[selected_idx])

    def crossover(self, problem, solution_1, solution_2):
        critic_pmt = read_prompt(self.args.critic_prompt)
        critic_pmt = critic_pmt.format_map(
            {"problem": problem, "solution_1": solution_1,
             "solution_2": solution_2})
        critic_inputs = format_input(critic_pmt, self.tokenizer)
        responses = self.llm.generate(
            critic_inputs,
            SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048,
                n=1,
                stop=stop_words,
                stop_token_ids=(
                    [151645, 151643]
                    if "qwen2" in self.args.model_name_or_path.lower()
                    else None
                ),
            ),
        )
        # response = self.client.completions.create(
        #     model=self.args.model_name_or_path,  # 与启动时指定的模型名称一致
        #     prompt=critic_inputs,
        #     max_tokens=2048,
        #     temperature=0.6,
        #     top_p=0.95,
        #     n=1,
        #     stop=stop_words,
        # )
        for response in responses:
            critic = response.outputs[0].text
        # print(critic)

        author_pmt = read_prompt(self.args.author_prompt)
        author_pmt = author_pmt.format_map(
            {"problem": problem, "solution_1": solution_1,
             "solution_2": solution_2, "critic_feedback": critic})
        author_inputs = format_input(author_pmt, self.tokenizer)
        responses = self.llm.generate(
            author_inputs,
            SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048,
                n=1,
                stop=stop_words,
                stop_token_ids=(
                    [151645, 151643]
                    if "qwen2" in self.args.model_name_or_path.lower()
                    else None
                ),
            ),
        )
        # response = self.client.completions.create(
        #     model=self.args.model_name_or_path,  # 与启动时指定的模型名称一致
        #     prompt=author_inputs,
        #     max_tokens=2048,
        #     temperature=0.6,
        #     top_p=0.95,
        #     n=1,
        #     stop=stop_words,
        # )
        for response in responses:
            author = response.outputs[0].text
        # print(author)

        return author

    def mutation(self, problem, solution):
        mut_pmt = read_prompt(self.args.mutation_prompt)
        mut_pmt = mut_pmt.format_map(
            {"problem": problem, "solution": solution})
        mut_inputs = format_input(mut_pmt, self.tokenizer)
        responses = self.llm.generate(
            mut_inputs,
            SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048,
                n=1,
                stop=stop_words,
                stop_token_ids=(
                    [151645, 151643]
                    if "qwen2" in self.args.model_name_or_path.lower()
                    else None
                ),
            ),
        )
        # response = self.client.completions.create(
        #     model=self.args.model_name_or_path,  # 与启动时指定的模型名称一致
        #     prompt=mut_inputs,
        #     max_tokens=2048,
        #     temperature=0.6,
        #     top_p=0.95,
        #     n=1,
        #     stop=stop_words,
        # )
        for response in responses:
            mutation = response.outputs[0].text
        # print(mutation)
        return mutation

    def pipeline(self, problem):

        # 初始种群
        best_fitness = 0
        best_solution = None
        solutions = self.init_population(problem)
        for iter in range(1, 4):
            solution_1, solution_2 = self.select(solutions)
            new_solution = self.crossover(problem, solution_1, solution_2)
            mutation = self.mutation(problem, new_solution)

            solutions.extend([new_solution, mutation])
            fitness_vals = np.array(self.calculate_fitness(solutions))

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
    def __init__(self, args, llm, tokenizer):
        """进化问题problem的solution
            vllm: 部署的vllm
            pop_size: 初始种群大小，也就是初次生成的个数；
            tokenizer：分词器用于进行fitness计算
        """
        self.args = args
        self.llm = llm
        self.tokenizer = tokenizer
        self.stop_words = stop_words

    def init_population(self, problem):
        pass

    def select(self, solutions):
        pass

    def calculate_fitness(self, solutions):
        pass

    def crossover(self, problem, solution_1, solution_2):
        pass

    def mutation(self, problem, solution):
        pass

    def pipeline(self, problem):
        pass