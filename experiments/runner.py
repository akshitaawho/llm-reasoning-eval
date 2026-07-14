from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class ModelRunner:
    def __init__(self, model_path: str):
        print(f"Loading model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.model.eval()
        print("Model loaded.\n")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()


if __name__ == "__main__":

    MODEL_PATH = "/media/nas_mount/research3/llm-models/phi4-mini-instruct"

    runner = ModelRunner(MODEL_PATH)

    prompt = "What is the capital of France?\nFINAL ANSWER:"

    response = runner.generate(prompt)

    print(response)