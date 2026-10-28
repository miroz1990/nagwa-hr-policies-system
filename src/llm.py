import os
import re
import requests

API_KEY = os.getenv("OPENROUTER_API_KEY")


def clean_answer(answer):
    answer = answer.strip()

    # Remove source-style lines only
    patterns = [
        r"(?im)^source:.*$",
        r"(?im)^document:.*$",
        r"(?im)^section:.*$",
        r"(?im)^context\s*\d+.*$",
    ]

    for p in patterns:
        answer = re.sub(p, "", answer)

    # Remove phrases, not whole answer lines
    answer = re.sub(r"(?i)\baccording to context\s*\d+,?\s*", "", answer)
    answer = re.sub(r"(?i)\baccording to the policy,?\s*", "", answer)
    answer = re.sub(r"(?i)\baccording to the provided policy,?\s*", "", answer)
    answer = re.sub(r"(?i)\bbased on the provided context,?\s*", "", answer)

    answer = re.sub(
    r"(?i)^based on the provided policy context,?\s*here is the answer:?\s*",
    "",
    answer,
    )

    answer = re.sub(
    r"(?i)^here is the answer:?\s*",
    "",
    answer,
    )

    answer = re.sub(
    r"(?im)^note:.*$",
    "",
    answer,
    )

    answer = answer.strip()

    if not answer:
        return "I found relevant policy information, but could not generate a clear answer. Please rephrase your question."

    return answer


def generate_answer(query, contexts):
    if not API_KEY:
        raise ValueError("❌ OPENROUTER_API_KEY is not set")

    context_text = "\n\n".join([
    f"""
Document: {c.get("document", "Unknown")}
Section: {c.get("section", "Unknown")}
Policy text:
{c.get("text", "")}
"""
    for c in contexts
])

    prompt = f"""
You are an internal HR policy assistant.

Your job is to answer employee questions using ONLY the policy information provided below.

IMPORTANT:
All provided policies belong to Nagwa. Treat phrases like "in Nagwa", "at Nagwa", "inside the company", or "company policy" as referring to the same internal Nagwa policy set. Do not require the word "Nagwa" to appear in the policy text.

RULES:
1. Use ONLY the provided policy context.
2. Do NOT guess.
3. Do NOT use outside knowledge.
4. If the answer cannot be found in the policy context, respond exactly:
Not mentioned in policy.
5. If the answer exists anywhere in the retrieved context, answer it.
6. If the answer requires combining multiple policy statements, combine them.
7. If multiple conditions, limits, exceptions, or requirements apply, include ALL of them.
8. Do not omit relevant details.
9. Be concise but complete.
10. If the policy gives a reason, restriction, location, condition, exception, or consequence, include it in the answer.
11. Do NOT generate summaries, conclusions, recommendations, notes, warnings, introductions, or closing remarks unless they are explicitly stated in the policy text.
12. For follow-up questions such as "Why?", "Why not?", or "Explain", use the previous question and previous answer included in the question text, and explain using the same policy context.
13. If the answer contains information from multiple sections, group the information under clear headings and preserve the structure used in the policy.

When a policy contains "Appropriate" and "Inappropriate"
items, present them as separate lists.

YES/NO QUESTIONS:
Only start with "Yes." or "No." when the question is clearly a yes/no question.

For yes/no questions:
- Start with "Yes." or "No."
- Then include the policy reason or condition if available.
- Do NOT answer with only "Yes." or only "No." if the context contains explanatory details.

Examples:
Question: Can I smoke at Nagwa?
Answer: No. Smoking is not allowed inside the building, corridors, or sidewalks, and it is not allowed during working hours.

Question: Are part-timers entitled to sick leave?
Answer: No. Part-timers are not entitled to sick, annual, or any other type of leave.

Question: Are part-timers entitled to medical insurance?
Answer: No. Part-timers are not provided with medical or health insurance.

NON-YES/NO QUESTIONS:
For questions starting with What, When, Where, Why, Who, Which, How, How many, or Explain:
- Do not start with "Yes." or "No."
- Answer directly.
- If it is a "Why?" follow-up, explain the reason using the previous context.

ANSWER STYLE:
Never mention:
- Context 1
- Context 2
- Context 3
- Source 1
- Source 2
- document names
- section names
- citations
- retrieved chunks
- provided context
- source ids

Never start the answer with:
- Here is the answer
- Based on the provided policy context
- Based on the policy text
- Note:
- According to the provided policy

Begin immediately with the answer itself.

Never write:
- "According to Context 1"
- "According to the policy"
- "Based on the provided context"

Return only the final answer.

Do not add phrases such as:
- Please note that...
- Keep in mind that...
- Employees are expected to...
- The policy states that...
- The dress code includes...
- In summary...
- Overall...

unless those exact ideas appear in the policy text.

If multiple rules apply, use bullet points.

If the answer requires information from multiple contexts,
combine the information naturally without mentioning where it came from.

Every sentence in the answer must be directly supported by the retrieved policy text.
If a sentence cannot be traced to the policy text, do not include it.



Context:
{context_text}

Question:
{query}
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "HR Policy Assistant",
        },
        json={
            "model": "meta-llama/llama-3.1-8b-instruct",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        },
        timeout=60
    )

    data = response.json()

    if "choices" not in data:
        raise Exception(f"LLM Error: {data}")

    answer = data["choices"][0]["message"]["content"]
    return clean_answer(answer)