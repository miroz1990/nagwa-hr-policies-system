import os
import json
import requests
import re


API_KEY = os.getenv("OPENROUTER_API_KEY")

EVALUATOR_MODEL = "meta-llama/llama-3-8b-instruct"


def evaluate_with_llm(question, expected_answer, generated_answer, contexts):
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set")

    context_text = "\n\n".join([
        f"""
Document: {c.get("document", "Unknown")}
Section: {c.get("section", "Unknown")}
Text: {c.get("text", "")}
"""
        for c in contexts
    ])

    prompt = f"""
You are evaluating an internal HR RAG assistant.

Evaluate the generated answer against the expected answer and the retrieved context.

Question:
{question}

Expected answer:
{expected_answer}

Generated answer:
{generated_answer}

Retrieved context:
{context_text}

Scoring rules:
- correctness: score from 1 to 5, where 5 means fully correct.
- grounding: score from 1 to 5, where 5 means fully supported by the retrieved context.
- completeness: score from 1 to 5, where 5 means all required details are included.
- If the generated answer is shorter but semantically correct, give high scores.
- If the generated answer misses important conditions, thresholds, or exceptions, lower completeness.
- If the answer says "Not mentioned in policy" while the context contains the answer, give correctness 1.
- If the answer adds unsupported information, give grounding 1.

Return ONLY valid JSON:
{{
  "correctness": 5,
  "grounding": 5,
  "completeness": 5,
  "passed": true,
  "explanation": "short explanation"
}}
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG Evaluator"
        },
        json={
            "model": EVALUATOR_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        },
        timeout=60
    )

    data = response.json()

    if "choices" not in data:
        raise Exception(f"Evaluator LLM Error: {data}")

    content = data["choices"][0]["message"]["content"].strip()

    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)

        if not match:
            raise ValueError(f"Evaluator did not return valid JSON:\n{content}")

        return json.loads(match.group())
    except json.JSONDecodeError:
        raise ValueError(f"Evaluator did not return valid JSON:\n{content}")