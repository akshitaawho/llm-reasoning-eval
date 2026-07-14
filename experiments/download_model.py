from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/Phi-4-mini-instruct"
SAVE_PATH = "/media/nas_mount/research3/llm-models/phi4-mini-instruct"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="cpu"
)

print(f"Saving to {SAVE_PATH}...")
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print("Done!")