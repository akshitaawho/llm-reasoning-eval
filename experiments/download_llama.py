from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_PATH = "/media/nas_mount/research3/llm-models/llama-3-8b-instruct"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="cpu",
)

print(f"Saving to {SAVE_PATH}...")
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print("Done!")