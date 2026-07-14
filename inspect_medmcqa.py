from datasets import load_from_disk

dataset = load_from_disk("data/raw/medmcqa")

print(dataset)
print()

print("Train sample:")
print(dataset["train"][0])