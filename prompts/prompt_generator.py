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

Do not write "<OPTION>".
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

Here are two examples.

Example 1

Question:
Which vitamin deficiency causes scurvy?

Options:
A. Vitamin A
B. Vitamin C
C. Vitamin D
D. Vitamin K

Reasoning:
Scurvy results from Vitamin C deficiency.

FINAL ANSWER: B


Example 2

Question:
Which organ produces insulin?

Options:
A. Liver
B. Kidney
C. Pancreas
D. Spleen

Reasoning:
Insulin is produced by the pancreas.

FINAL ANSWER: C

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

Before selecting the final answer, briefly consider why the other options are incorrect.

"""

    prompt += _answer_format()

    return prompt


def generate_step_back_prompt(sample):
    prompt = _build_base_prompt(sample)

    prompt += """

First identify the underlying medical concept required to solve the question.

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

    print(generate_cot_prompt(sample))