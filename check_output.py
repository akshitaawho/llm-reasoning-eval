import json

with open("data/unified/medmcqa.jsonl", "r", encoding="utf-8") as f:
    sample = json.loads(next(f))

print(sample)