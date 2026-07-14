import json

def generate_prompt(sample):
    prompt = f"""You are an expert medical reasoning assistant.

Answer the following multiple-choice question.

Question:
{sample["question"]}

Options:
"""

    for label, option in sample["options"].items():
        prompt += f"{label}. {option}\n"

    prompt += """

Think step by step before answering.

Return your final answer exactly in the format:

FINAL ANSWER: <OPTION>
"""

    return prompt


if __name__ == "__main__":

    with open("data/unified/medmcqa.jsonl", encoding="utf-8") as f:
        sample = json.loads(next(f))

    print(generate_prompt(sample))