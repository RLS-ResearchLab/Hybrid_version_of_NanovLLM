"""8-shot CoT prompt construction for GSM8K -- pure Python, no GPU/model
dependency (tokenization happens elsewhere, via the real tokenizer).

EXEMPLARS below are the STANDARD Cobbe et al. (2021) 8-shot chain-of-thought
set -- fetched VERBATIM from lm-evaluation-harness's own task definition
(the same file `lm-eval --tasks gsm8k_cot` loads at run time), not
transcribed by hand:

    https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml
    (fetched 2026-08-03; fewshot_config.samples, sampler: first_n)

Prompt assembly mirrors that same file's `doc_to_text: "Q: {{question}}\n\nA:"`
plus lm-evaluation-harness's ConfigurableTask defaults for the two
delimiters that YAML doesn't override:
  - target_delimiter = " "   (between "A:" and the answer text)
  - fewshot_delimiter = "\n\n"  (between successive shots, and between the
    last shot and the actual test question)
So each shot renders as "Q: {question}\n\nA: {answer}", and shots (plus the
final, answer-less question) are joined with "\n\n".
"""

EXEMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in "
        "the grove today. After they are done, there will be 21 trees. How many "
        "trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some "
        "more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how "
        "many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The "
        "answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how "
        "many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total "
        "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer "
        "is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has "
        "12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to "
        "Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. How many computers are "
        "now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The "
        "answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. How many golf balls did he have at the end of "
        "wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he "
        "had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The "
        "answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money "
        "does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 "
        "dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    },
]

assert len(EXEMPLARS) == 8, f"expected 8 canonical exemplars, got {len(EXEMPLARS)}"


def build_prompt(question: str, exemplars=EXEMPLARS) -> str:
    blocks = [f"Q: {ex['question']}\n\nA: {ex['answer']}" for ex in exemplars]
    blocks.append(f"Q: {question}\n\nA:")
    return "\n\n".join(blocks)
