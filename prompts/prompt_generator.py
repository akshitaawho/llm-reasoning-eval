import json


def _build_base_prompt(sample):
    prompt = f"""You are an expert medical reasoning assistant.

Answer the following multiple-choice question.

Question:
{sample["question"]}

Options:
"""

    for label, option in sample["options"].items():
        prompt += f"{label}. {option}\n"

    return prompt


def _answer_format():
    return """

Return ONLY the final answer on the last line.

The last line must be EXACTLY one of:

FINAL ANSWER: A
FINAL ANSWER: B
FINAL ANSWER: C
FINAL ANSWER: D

Do not add any text after the final answer.
"""


def generate_cot_prompt(sample):
    prompt = _build_base_prompt(sample)

    prompt += """

Think step by step before answering.
"""

    prompt += _answer_format()

    return prompt


def generate_few_shot_prompt(sample):
    prompt = """You are an expert medical reasoning assistant.

Here are two solved examples.

Example 1

Question:
Vitamin C deficiency causes which disease?

Options:
A. Rickets
B. Pellagra
C. Scurvy
D. Beriberi

Reasoning:
Vitamin C deficiency leads to defective collagen synthesis, resulting in scurvy.

FINAL ANSWER: C


Example 2

Question:
Insulin is secreted by which organ?

Options:
A. Liver
B. Pancreas
C. Kidney
D. Spleen

Reasoning:
Insulin is produced by the beta cells of the pancreas.

FINAL ANSWER: B


Now answer the next question.

"""

    prompt += _build_base_prompt(sample)

    prompt += """

Think step by step before answering.
"""

    prompt += _answer_format()

    return prompt


def generate_role_prompt(sample):
    prompt = """You are an experienced physician and medical educator.

Carefully analyze the medical concepts before selecting the correct answer.

"""

    prompt += _build_base_prompt(sample)

    prompt += """

Think step by step before answering.
"""

    prompt += _answer_format()

    return prompt


def generate_contrastive_prompt(sample):
    prompt = _build_base_prompt(sample)

    prompt += """

Think step by step.

Before selecting the final answer, briefly explain why the incorrect options are less appropriate.

"""

    prompt += _answer_format()

    return prompt


def generate_step_back_prompt(sample):
    prompt = _build_base_prompt(sample)

    prompt += """

First identify the underlying medical concept needed to solve the question.

Then use that concept to determine the correct answer.

Think step by step.

"""

    prompt += _answer_format()

    return prompt


def generate_self_consistency_prompt(sample):
    return generate_cot_prompt(sample)


if __name__ == "__main__":

    with open("data/unified/medmcqa.jsonl", encoding="utf-8") as f:
        sample = json.loads(next(f))

    print(generate_few_shot_prompt(sample))