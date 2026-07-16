import json
import argparse

from experiments.runner import ModelRunner
from parser.answer_parser import extract_answer
from prompts.prompt_generator import generate_prompt

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

        prompt = generate_prompt(sample)

        response = runner.generate(prompt)

        parsed = extract_answer(response)

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
            f"{i+1}/10 | "
            f"GT={result['ground_truth']} | "
            f"Pred={result['predicted']} | "
            f"Correct={result['correct']}"
        )

print()

accuracy = sum(r["correct"] for r in results) / len(results)

print(f"Accuracy: {accuracy:.2%}")

with open(f"data/responses/{args.model_name}_{args.prompt}.jsonl", "w", encoding="utf-8") as f:

    for row in results:
        f.write(json.dumps(row) + "\n")

print("Saved results.")