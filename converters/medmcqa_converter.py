import json
from pathlib import Path

from datasets import load_from_disk

# Load dataset
dataset = load_from_disk("data/raw/medmcqa")

# Create output directory
OUTPUT_DIR = Path("data/unified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Convert integer labels to option letters
answer_map = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}

output_file = OUTPUT_DIR / "medmcqa.jsonl"

total_samples = 0
missing_answers = 0

with open(output_file, "w", encoding="utf-8") as f:

    for split in ["train", "validation", "test"]:

        print(f"Processing {split}...")

        for sample in dataset[split]:

            # Test set has cop = -1 (hidden labels)
            correct_answer = answer_map.get(sample["cop"])

            if correct_answer is None:
                missing_answers += 1

            unified = {
                "id": sample["id"],
                "dataset": "MedMCQA",
                "question": sample["question"],
                "options": {
                    "A": sample["opa"],
                    "B": sample["opb"],
                    "C": sample["opc"],
                    "D": sample["opd"]
                },
                "correct_answer": correct_answer,
                "metadata": {
                    "split": split,
                    "subject_name": sample["subject_name"],
                    "topic_name": sample["topic_name"],
                    "choice_type": sample["choice_type"],
                    "explanation": sample["exp"]
                }
            }

            f.write(json.dumps(unified, ensure_ascii=False) + "\n")
            total_samples += 1

print("\nConversion complete!")
print(f"Total samples: {total_samples}")
print(f"Samples with missing answers: {missing_answers}")
print(f"Saved to: {output_file}")