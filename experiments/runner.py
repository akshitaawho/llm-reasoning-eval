from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class ModelRunner:
    def __init__(self, model_path: str):
        print(f"Loading model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )

        self.model.eval()
        print("Model loaded.\n")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens=1024,
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

    import json
    from prompts.prompt_generator import generate_prompt

    MODEL_PATH = "/media/nas_mount/research3/llm-models/phi4-mini-instruct"

    runner = ModelRunner(MODEL_PATH)

    with open("data/unified/medmcqa.jsonl", "r", encoding="utf-8") as f:
        sample = json.loads(next(f))

    prompt = generate_prompt(sample)

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prompt)

    print("\n" + "=" * 80)
    print("MODEL RESPONSE")
    print("=" * 80)

    response = runner.generate(prompt)

    print(response)