import json

VALID_ANSWERS = {"A", "B", "C", "D", "E"}

input_file = "data/unified/medmcqa.jsonl"

total = 0
valid = 0
invalid = 0

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        total += 1

        sample = json.loads(line)

        try:
            assert sample["id"]
            assert sample["dataset"]
            assert sample["question"]
            assert isinstance(sample["options"], dict)
            assert len(sample["options"]) >= 2
            assert "metadata" in sample

            answer = sample["correct_answer"]

            if answer is not None:
                assert answer in sample["options"]
                assert answer in VALID_ANSWERS

            valid += 1

        except AssertionError:
            invalid += 1

print("=" * 40)
print(f"Total Samples : {total}")
print(f"Valid Samples : {valid}")
print(f"Invalid Samples : {invalid}")
print("=" * 40)