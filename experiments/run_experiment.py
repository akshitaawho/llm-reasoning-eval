import json

from experiments.runner import ModelRunner
from parser.answer_parser import extract_answer
from prompts.prompt_generator import generate_prompt

MODEL_PATH = "/media/nas_mount/research3/llm-models/phi4-mini-instruct"
DATASET_PATH = "data/unified/medmcqa.jsonl"

runner = ModelRunner(MODEL_PATH)

results = []

with open(DATASET_PATH, "r", encoding="utf-8") as f:

    for i, line in enumerate(f):

        if i == 10:
            break

        sample = json.loads(line)

        prompt = generate_prompt(sample)

        response = runner.generate(prompt)

        parsed = extract_answer(response)

        result = {
            "id": sample["id"],
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

with open(
    "data/responses/phi4_cot.jsonl",
    "w",
    encoding="utf-8",
) as f:

    for row in results:
        f.write(json.dumps(row) + "\n")

print("Saved results.")