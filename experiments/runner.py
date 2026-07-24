from vllm import LLM, SamplingParams


class ModelRunner:
    def __init__(self, model_path: str):
        print(f"Loading model from {model_path}...")

        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
        )

        print("Model loaded.\n")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=0.9 if do_sample else 1.0,
        )

        outputs = self.llm.generate(
            [prompt],
            sampling_params,
        )

        return outputs[0].outputs[0].text.strip()


if __name__ == "__main__":

    MODEL_PATH = "/media/nas_mount/research3/dheeraj/checkpoints/Ministral-8B-Instruct-2410-awq-w4a16-asym-g128"

    runner = ModelRunner(MODEL_PATH)

    response = runner.generate(
        "What is 2 + 2? Answer in one sentence."
    )

    print("=" * 80)
    print("MODEL RESPONSE")
    print("=" * 80)
    print(response)