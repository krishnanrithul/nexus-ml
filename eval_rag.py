"""
eval_rag.py — Run after python main.py

Measures:
1. Intent routing accuracy    — does the classifier pick the right record type?
2. Retrieval precision        — do retrieved chunks match the expected type?
3. Fallback rate              — how often does typed search return nothing?
4. Latency                    — how fast is retrieval?

Usage:
    python eval_rag.py
"""

import time
from query_engine import ask, _classify_intent, _retrieve

# ─────────────────────────────────────────────
# TEST SET
# Questions where we know the right record type
# ─────────────────────────────────────────────

TEST_CASES = [
    # (question, expected_intent, expected_record_type)

    # Narrative questions
    ("Which model performed better?",              "narrative",   "narrative"),
    ("What features were most important?",         "narrative",   "narrative"),
    ("How did the model perform overall?",         "narrative",   "narrative"),
    ("What was the R² score?",                     "narrative",   "narrative"),
    ("Give me a summary of results",               "narrative",   "narrative"),

    # Segment questions
    ("Which neighborhood had the worst error?",    "segment",     "segment_summary"),
    ("What was the RMSE for Downtown?",            "segment",     "segment_summary"),
    ("Which area had the best predictions?",       "segment",     "segment_summary"),
    ("Show me segment breakdown",                  "segment",     "segment_summary"),
    ("Which region was hardest to predict?",       "segment",     "segment_summary"),

    # Prediction questions
    ("Show me rows with high prediction error",    "prediction",  "prediction"),
    ("Which individual predictions were worst?",   "prediction",  "prediction"),
    ("Show me properties the model missed badly",  "prediction",  "prediction"),
    ("Find predictions above 15% error",           "prediction",  "prediction"),
    ("What were the worst individual cases?",      "prediction",  "prediction"),
]


def evaluate():
    print("=" * 60)
    print("NexusML RAG Evaluation")
    print("=" * 60)

    intent_correct   = 0
    retrieval_correct = 0
    fallback_count   = 0
    latencies        = []

    results_log = []

    for question, expected_intent, expected_rtype in TEST_CASES:
        # 1. Intent classification
        detected_intent = _classify_intent(question)
        intent_hit = detected_intent == expected_intent

        # 2. Retrieval + latency
        t0 = time.time()
        chunks = _retrieve(question, detected_intent, top_k=5)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        # 3. Retrieval precision — what % of returned chunks are the right type?
        if chunks:
            correct_chunks = sum(1 for c in chunks if c.get("record_type") == expected_rtype)
            precision = correct_chunks / len(chunks)
        else:
            precision = 0.0
            fallback_count += 1

        if intent_hit:
            intent_correct += 1
        if precision >= 0.6:  # 60% of chunks are correct type = pass
            retrieval_correct += 1

        results_log.append({
            "question":        question,
            "expected_intent": expected_intent,
            "detected_intent": detected_intent,
            "intent_hit":      intent_hit,
            "precision":       precision,
            "chunks_returned": len(chunks),
            "latency_ms":      latency_ms,
        })

        status = "✅" if (intent_hit and precision >= 0.6) else "❌"
        print(f"{status} [{detected_intent:10}] {question[:50]:<50}  precision={precision:.0%}  {latency_ms:.0f}ms")

    # ─────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────
    n = len(TEST_CASES)
    intent_acc    = intent_correct / n
    retrieval_acc = retrieval_correct / n
    avg_latency   = sum(latencies) / len(latencies)
    p95_latency   = sorted(latencies)[int(0.95 * len(latencies))]

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Intent routing accuracy : {intent_acc:.0%}  ({intent_correct}/{n})")
    print(f"Retrieval precision     : {retrieval_acc:.0%}  ({retrieval_correct}/{n})")
    print(f"Fallback rate           : {fallback_count}/{n} queries returned nothing")
    print(f"Avg latency             : {avg_latency:.0f}ms")
    print(f"P95 latency             : {p95_latency:.0f}ms")

    # ─────────────────────────────────────────────
    # FAILURES — what broke and why
    # ─────────────────────────────────────────────
    failures = [r for r in results_log if not r["intent_hit"] or r["precision"] < 0.6]
    if failures:
        print()
        print("FAILURES")
        print("-" * 60)
        for f in failures:
            print(f"  Q: {f['question']}")
            print(f"     Expected {f['expected_intent']}, got {f['detected_intent']}")
            print(f"     Retrieval precision: {f['precision']:.0%}")

    # ─────────────────────────────────────────────
    # WHAT TO PUT IN YOUR README
    # ─────────────────────────────────────────────
    print()
    print("README SNIPPET")
    print("-" * 60)
    print(f"Intent routing accuracy : {intent_acc:.0%}")
    print(f"Retrieval precision     : {retrieval_acc:.0%}")
    print(f"Query latency (P95)     : {p95_latency:.0f}ms")


if __name__ == "__main__":
    evaluate()