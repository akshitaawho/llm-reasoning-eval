import json
import argparse
from collections import Counter

from experiments.runner import ModelRunner
from parser.answer_parser import extract_answer
from prompts.prompt_generator import (
    generate_cot_prompt,
    generate_self_consistency_prompt,
)

parser = argparse.ArgumentParser()

parser.add_argument("--model-name", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--prompt", default="cot")
parser.add_argument("--samples", type=int, default=10)

args = parser.parse_args()

DATASET_PATH = "data/unified/medmcqa.jsonl"

runner = ModelRunner(args.model_path)

results = []

with open(DATASET_PATH, "r", encoding="utf-8") as f:

    for i, line in enumerate(f):

        if i == args.samples:
            break

        sample = json.loads(line)

        if args.prompt == "cot":
            prompt = generate_cot_prompt(sample)
        elif args.prompt == "self_consistency":
            prompt = generate_self_consistency_prompt(sample)
        else:
            raise ValueError(f"Unknown prompt type: {args.prompt}")

        if args.prompt == "cot":

            response = runner.generate(prompt)

            parsed = extract_answer(response)

            responses = [response]
            parsed_answers = [parsed["answer"]]

        else:

            responses = []
            parsed_answers = []

            for _ in range(5):

                response = runner.generate(
                    prompt,
                    do_sample=True,
                    temperature=0.7,
                )

                responses.append(response)

                parsed = extract_answer(response)

                parsed_answers.append(parsed["answer"])

            valid_answers = [
                a for a in parsed_answers
                if a is not None
            ]

            if valid_answers:

                counts = Counter(valid_answers)
                most_common = counts.most_common()

                if (
                    len(most_common) > 1
                    and most_common[0][1] == most_common[1][1]
                ):
                    parsed = {
                        "answer": None,
                        "success": False,
                    }
                else:
                    parsed = {
                        "answer": most_common[0][0],
                        "success": True,
                    }

            else:

                parsed = {
                    "answer": None,
                    "success": False,
                }

        result = {
            "id": sample["id"],
            "model": args.model_name,
            "prompt_type": args.prompt,
            "ground_truth": sample["correct_answer"],
            "predicted": parsed["answer"],
            "success": parsed["success"],
            "correct": parsed["answer"] == sample["correct_answer"],
            "responses": responses,
            "parsed_answers": parsed_answers,
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

output_path = f"data/responses/{args.model_name}_{args.prompt}.jsonl"

with open(output_path, "w", encoding="utf-8") as f:

    for row in results:
        f.write(json.dumps(row) + "\n")

print(f"Saved results to {output_path}")