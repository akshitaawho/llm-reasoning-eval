import re


def extract_answer(response: str) -> dict:
    """
    Extract the predicted option (A/B/C/D) from the model response.
    """

    patterns = [
        r"FINAL ANSWER\s*:\s*([ABCD])",
        r"Final Answer\s*:\s*([ABCD])",
        r"Answer\s*:\s*([ABCD])",
        r"The answer is\s*([ABCD])",
        r"correct answer is\s*([ABCD])",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return {
                "answer": match.group(1).upper(),
                "success": True,
            }

    return {
        "answer": None,
        "success": False,
    }


if __name__ == "__main__":

    sample = """
    Some reasoning...

    FINAL ANSWER: C. Atrophy
    """

    print(extract_answer(sample))