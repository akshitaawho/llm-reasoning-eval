import json
import argparse
from collections import Counter

from experiments.runner import ModelRunner
from parser.answer_parser import extract_answer
from prompts.prompt_generator import (
    generate_cot_prompt,
    generate_self_consistency_prompt,
    generate_few_shot_prompt,
    generate_role_prompt,
    generate_contrastive_prompt,
    generate_step_back_prompt,
)

parser = argparse.ArgumentParser()

parser.add_argument("--model-name", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--prompt", default="cot")
parser.add_argument("--samples", type=int, default=10)

args = parser.parse_args()

DATASET_PATH = "data/unified/medmcqa.jsonl"

PROMPT_GENERATORS = {
    "cot": generate_cot_prompt,
    "fewshot": generate_few_shot_prompt,
    "role": generate_role_prompt,
    "contrastive": generate_contrastive_prompt,
    "stepback": generate_step_back_prompt,
    "self_consistency": generate_self_consistency_prompt,
}

if args.prompt not in PROMPT_GENERATORS:
    raise ValueError(f"Unknown prompt type: {args.prompt}")

runner = ModelRunner(args.model_path)

results = []

with open(DATASET_PATH, "r", encoding="utf-8") as f:

    for i, line in enumerate(f):

        if i == args.samples:
            break

        sample = json.loads(line)

        prompt = PROMPT_GENERATORS[args.prompt](sample)

        if args.prompt != "self_consistency":

            response = runner.generate(prompt)

            parsed = extract_answer(response)

        else:

            responses = []
            answers = []

            for _ in range(5):

                response = runner.generate(
                    prompt,
                    do_sample=True,
                    temperature=0.7,
                )

                responses.append(response)

                parsed = extract_answer(response)

                if parsed["success"]:
                    answers.append(parsed["answer"])

            if answers:

                majority = Counter(answers).most_common(1)[0][0]

                parsed = {
                    "answer": majority,
                    "success": True,
                }

            else:

                parsed = {
                    "answer": None,
                    "success": False,
                }

            response = "\n\n====================\n\n".join(responses)

        result = {
            "id": sample["id"],
            "model": args.model_name,
            "prompt_type": args.prompt,
            "ground_truth": sample["correct_answer"],
            "predicted": parsed["answer"],
            "success": parsed["success"],
            "correct": parsed["answer"] == sample["correct_answer"],
            "raw_response": response,
        }

        results.append(result)

        print(
            f"{i+1}/{args.samples} | "
            f"GT={result['ground_truth']} | "
            f"Pred={result['predicted']} | "
            f"Correct={result['correct']}"
        )

print()

accuracy = sum(r["correct"] for r in results) / len(results)

print(f"Accuracy: {accuracy:.2%}")

with open(
    f"data/responses/{args.model_name}_{args.prompt}.jsonl",
    "w",
    encoding="utf-8",
) as f:

    for row in results:
        f.write(json.dumps(row) + "\n")

print("Saved results.")