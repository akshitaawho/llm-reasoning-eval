from datasets import load_dataset

dataset = load_dataset("openlifescienceai/medmcqa")

print(dataset)

dataset.save_to_disk("data/raw/medmcqa")

print("Downloaded successfully!")