from transformers import AutoTokenizer, AutoModelForCausalLM
from parser.answer_parser import extract_answer
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
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.9

        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()


if __name__ == "__main__":

    import json
    from prompts.prompt_generator import generate_cot_prompt

    MODEL_PATH = "/media/nas_mount/research3/llm-models/phi4-mini-instruct"

    runner = ModelRunner(MODEL_PATH)

    with open("data/unified/medmcqa.jsonl", "r", encoding="utf-8") as f:
        sample = json.loads(next(f))

    prompt = generate_cot_prompt(sample)

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prompt)

    print("\n" + "=" * 80)
    print("MODEL RESPONSE")
    print("=" * 80)

    response = runner.generate(prompt)

    print(response)

    parsed = extract_answer(response)

    print("\n" + "=" * 80)
    print("PARSED ANSWER")
    print("=" * 80)
    print(parsed)