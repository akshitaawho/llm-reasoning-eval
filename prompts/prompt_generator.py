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


def generate_cot_prompt(sample):
    prompt = _build_base_prompt(sample)

    prompt += """

Think step by step before answering.

Return ONLY the final answer on the last line.

The last line must be EXACTLY one of:

FINAL ANSWER: A
FINAL ANSWER: B
FINAL ANSWER: C
FINAL ANSWER: D

Do not write "<OPTION>".
Do not add any text after the final answer.
"""

    return prompt


def generate_self_consistency_prompt(sample):
    prompt = _build_base_prompt(sample)

    prompt += """

Think step by step before answering.

Return ONLY the final answer on the last line.

The last line must be EXACTLY one of:

FINAL ANSWER: A
FINAL ANSWER: B
FINAL ANSWER: C
FINAL ANSWER: D

Do not write "<OPTION>".
Do not add any text after the final answer.
"""

    return prompt


if __name__ == "__main__":

    with open("data/unified/medmcqa.jsonl", encoding="utf-8") as f:
        sample = json.loads(next(f))

    print(generate_cot_prompt(sample))