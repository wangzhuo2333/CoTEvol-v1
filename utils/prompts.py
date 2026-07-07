GSM_COT_8_SHOT = "Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the " \
                 "cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many " \
                 "kids came to the cookout in 2006? \n Let's think step by step In 2005, 60/2=30 kids came to the " \
                 "cookout. \n In 2006, 30/3*2=20 kids came to the cookout. \n The answer is 20  \nQuestion: Zilla " \
                 "spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the " \
                 "rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings " \
                 "account in a month? \n Let's think step by step Since $133 is equal to 7% of her earnings, " \
                 "then 1% is equal to $133/7 = $19. \n The total monthly earning of Zilla is represented by 100%, " \
                 "so $19 x 100 = $1900 is her monthly earnings. \n So, $1900/2 = $950 is spent on her other monthly " \
                 "expenses. \n The total amount spent on the rent and other monthly expenses is $133 + $950 = $1083. " \
                 "\n Hence, she saves $1900 - $1083 = $817 per month. \n The answer is 817  \nQuestion: If Buzz " \
                 "bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the " \
                 "ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the " \
                 "waiter ate? \n Let's think step by step The total ratio representing the slices of pizza that Buzz " \
                 "bought is 5+8=13 \n If he shared the slices of pizza with the waiter, the waiter received a " \
                 "fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices \n Twenty less " \
                 "the number of slices of pizza that the waiter ate is 48-20 = 28 \n The answer is 28  \nQuestion: " \
                 "Jame gets a raise to $20 per hour and works 40 hours a week. His old job was $16 an hour for 25 " \
                 "hours per week. How much more money does he make per year in his new job than the old job if he " \
                 "works 52 weeks a year? \n Let's think step by step He makes 20*40=$800 per week \n He used to make " \
                 "16*25=$400 per week \n So his raise was 800400=$400 per week \n So he makes 400*52=$20,800 per year " \
                 "more \n The answer is 20800  \nQuestion: Mr. Gardner bakes 20 cookies, 25 cupcakes, and 35 brownies " \
                 "for his second-grade class of 20 students. If he wants to give each student an equal amount of " \
                 "sweet treats, how many sweet treats will each student receive? \n Let's think step by step Mr. " \
                 "Gardner bakes a total of 20 + 25 + 35 = 80 sweet treats \n Each student will receive 80 / 20 = 4 " \
                 "sweet treats \n The answer is 4  \nQuestion: A used car lot has 24 cars and motorcycles (in total) " \
                 "for sale. A third of the vehicles are motorcycles, and a quarter of the cars have a spare tire " \
                 "included. How many tires are on the used car lot’s vehicles in all? \n Let's think step by step The " \
                 "used car lot has 24 / 3 = 8 motorcycles with 2 tires each. \n The lot has 24 - 8 = 16 cars for sale " \
                 "\n There are 16 / 4 = 4 cars with a spare tire with 5 tires each. \n The lot has 16 - 4 = 12 cars " \
                 "with 4 tires each. \n Thus, the used car lot’s vehicles have 8 * 2 + 4 * 5 + 12 * 4 = 16 + 20 + 48 " \
                 "= 84 tires in all. \n The answer is 84  \nQuestion: Norma takes her clothes to the laundry. She " \
                 "leaves 9 T-shirts and twice as many sweaters as T-shirts in the washer. When she returns she finds " \
                 "3 sweaters and triple the number of T-shirts. How many items are missing? \n Let's think step by " \
                 "step Norma left 9 T-shirts And twice as many sweaters, she took 9 * 2= 18 sweaters \n Adding the " \
                 "T-shirts and sweaters, Norma left 9 + 18 = 27 clothes \n When she came back, she found 3 sweaters " \
                 "And triple the number of T-shirts, she found 3 * 3 = 9 T-shirts \n Adding the T-shirts and " \
                 "sweaters, Norma found 3 + 9 = 12 clothes \n Subtracting the clothes she left from the clothes she " \
                 "found, 27 - 12 = 15 clothes are missing \n The answer is 15  \nQuestion: Adam has an orchard. Every " \
                 "day for 30 days he picks 4 apples from his orchard. After a month, Adam has collected all the " \
                 "remaining apples, which were 230. How many apples in total has Adam collected from his orchard? \n " \
                 "Let's think step by step During 30 days Adam picked 4 * 30 = 120 apples. \n So in total with all " \
                 "the remaining apples, he picked 120 + 230 = 350 apples from his orchard. \n The answer is 350  " \
                 "\nQuestion: {question} \n Let's think step by step "

# math_message = "You are a math teacher. Given a math problem, please use formal " \
#                "mathematical expressions to provide the reasoning process step by step. The final answer should be " \
#                "formatted as $\\boxed{xxx}$. "

math_message = (
    "You are an expert in math. "
    "Below is a math question. Write a response that appropriately answers the question. "
    "The final answer should be formatted as $\\boxed{<your answer>}$. Let's think step by step."
)

math_thought_message = (
    "You are an expert in math. "
    "Below is a math question. Write a response that appropriately answers the question. "
    "Let's think step by step."
)