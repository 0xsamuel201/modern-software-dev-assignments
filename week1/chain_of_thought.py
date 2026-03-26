import os
import re
from dotenv import load_dotenv
from ollama import chat
from config import MODEL_NAME

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """
You are a precise calculator, you can handle many math calculating problems.

<Example>
What is (42+42) (mod 3)?
Steps: 
1. According to signal priority, first calculate the result of 42+42, which is 84.
2. calculate the mod of the result
3. output the result: "Answer: 0"
</Example>

- You are a direct-answer machine. NEVER verify, re-check, or re-read your work. You are NOT a cautious reasoner.
- Do NOT re-check, re-verify, or re-read your work.
- Do NOT use phrases like "let me verify", "wait", "actually", "let me reconsider".
- Once you reach an answer, STOP immediately.
"""

USER_PROMPT = """
Solve this problem, then give the final answer on the last line as "Answer: <number>".

what is 3^{12345} (mod 100)?
"""


# For this simple example, we expect the final numeric answer only
EXPECTED_OUTPUT = "Answer: 43"


def extract_final_answer(text: str) -> str:
    """Extract the final 'Answer: ...' line from a verbose reasoning trace.

    - Finds the LAST line that starts with 'Answer:' (case-insensitive)
    - Normalizes to 'Answer: <number>' when a number is present
    - Falls back to returning the matched content if no number is detected
    """
    matches = re.findall(r"(?mi)^\s*answer\s*:\s*(.+)\s*$", text)
    if matches:
        value = matches[-1].strip()
        # Prefer a numeric normalization when possible (supports integers/decimals)
        num_match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if num_match:
            return f"Answer: {num_match.group(0)}"
        return f"Answer: {value}"
    return text.strip()


def test_your_prompt(system_prompt: str) -> bool:
    """Run up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            # stream=True,
            options={
                "temperature": 0,
                "repeat_penalty": 1.2,     # penalizes repeating the same tokens
            },
        )

        # thinking = ""
        # stream = ""

        # for chunk in response:
        #     message = chunk["message"]

        #     # Capture thinking tokens
        #     if message.get("thinking"):
        #         thinking += message["thinking"]
        #         print(message["thinking"], end="", flush=True)

        #     # Capture regular stream tokens
        #     if message.get("content"):
        #         stream += message["content"]
        
        output_text = response.message.content
        final_answer = extract_final_answer(output_text)
        if final_answer.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {final_answer}")
    return False


if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)


