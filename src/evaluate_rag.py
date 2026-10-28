import os
import json
import csv
from main import ask
from llm_evaluator import evaluate_with_llm


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_FILE = os.path.join(BASE_DIR, "data", "evaluation_questions.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "rag_evaluation_results.csv")


def normalize(text):
    return text.lower().strip()


def keyword_score(answer, expected_answer):
    answer_words = set(normalize(answer).replace(".", "").replace(",", "").split())
    expected_words = set(normalize(expected_answer).replace(".", "").replace(",", "").split())

    if not expected_words:
        return 0.0

    overlap = answer_words.intersection(expected_words)
    return round(len(overlap) / len(expected_words), 2)


def source_match_score(contexts, expected_document, expected_section):
    expected_document = normalize(expected_document)
    expected_section = normalize(expected_section)

    for ctx in contexts:
        doc = normalize(ctx.get("document", ""))
        section = normalize(ctx.get("section", ""))

        if expected_document in doc and expected_section in section:
            return 1

    return 0


def evaluate():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []

    for i, item in enumerate(test_cases, start=1):
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_document = item["expected_document"]
        expected_section = item["expected_section"]

        print(f"\n[{i}/{len(test_cases)}] Question: {question}")

        try:
            answer, contexts = ask(question)

            k_score = keyword_score(answer, expected_answer)
            s_score = source_match_score(contexts, expected_document, expected_section)

            llm_eval = evaluate_with_llm(
                question=question,
                expected_answer=expected_answer,
                generated_answer=answer,
                contexts=contexts
            )

            correctness = int(llm_eval.get("correctness", 0))
            grounding = int(llm_eval.get("grounding", 0))
            completeness = int(llm_eval.get("completeness", 0))

            passed = (
                correctness >= 4
                and grounding >= 4
                and completeness >= 4
                and s_score == 1
            )

            results.append({
                "id": i,
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": answer,
                "expected_document": expected_document,
                "expected_section": expected_section,

                "keyword_score": k_score,
                "source_match": s_score,

                "correctness": correctness,
                "grounding": grounding,
                "completeness": completeness,
                "llm_passed": llm_eval.get("passed", False),
                "passed": passed,
                "llm_explanation": llm_eval.get("explanation", ""),

                "top_document": contexts[0].get("document", "") if contexts else "",
                "top_section": contexts[0].get("section", "") if contexts else "",
                "top_context": contexts[0].get("text", "") if contexts else ""
            })

            print(f"Keyword Score: {k_score}")
            print(f"Source Match: {s_score}")
            print(f"Correctness: {correctness}")
            print(f"Grounding: {grounding}")
            print(f"Completeness: {completeness}")
            print(f"Passed: {passed}")

        except Exception as e:
            results.append({
                "id": i,
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": f"ERROR: {str(e)}",
                "expected_document": expected_document,
                "expected_section": expected_section,

                "keyword_score": 0,
                "source_match": 0,

                "correctness": 0,
                "grounding": 0,
                "completeness": 0,
                "llm_passed": False,
                "passed": False,
                "llm_explanation": f"ERROR: {str(e)}",

                "top_document": "",
                "top_section": "",
                "top_context": ""
            })

            print(f"ERROR: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])

    avg_keyword = round(sum(r["keyword_score"] for r in results) / total, 2)
    avg_source = round(sum(r["source_match"] for r in results) / total, 2)
    avg_correctness = round(sum(r["correctness"] for r in results) / total, 2)
    avg_grounding = round(sum(r["grounding"] for r in results) / total, 2)
    avg_completeness = round(sum(r["completeness"] for r in results) / total, 2)

    print("\n==============================")
    print("RAG Evaluation Summary")
    print("==============================")
    print(f"Total questions: {total}")
    print(f"Passed: {passed_count}/{total}")
    print(f"Pass rate: {round((passed_count / total) * 100, 2)}%")
    print(f"Average keyword score: {avg_keyword}")
    print(f"Average source match: {avg_source}")
    print(f"Average correctness: {avg_correctness}")
    print(f"Average grounding: {avg_grounding}")
    print(f"Average completeness: {avg_completeness}")
    print(f"Saved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()