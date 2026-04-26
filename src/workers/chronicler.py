import re
from datetime import datetime
from src.state import client, FactoryState
from src.tools.vector_ops import index_chunks


def _generate_narrative(model_results: dict) -> str:
    """LLM writes a 5-7 sentence summary of overall model performance."""
    seg = model_results.get("segment_results", {})
    seg_summary = ""
    if seg:
        worst    = max(seg, key=lambda s: seg[s]["rmse"])
        best_seg = min(seg, key=lambda s: seg[s]["rmse"])
        seg_summary = (f"Segment analysis shows '{worst}' had the highest error "
                       f"(RMSE {seg[worst]['rmse']:.2f}, MAPE {seg[worst]['mape']:.1f}%) "
                       f"while '{best_seg}' performed best "
                       f"(RMSE {seg[best_seg]['rmse']:.2f}, MAPE {seg[best_seg]['mape']:.1f}%).")

    top_features = list(model_results.get("feature_importance", {}).keys())[:3]

    prompt = f"""
You are a Senior Data Scientist writing a concise model performance report.

Results:
- Target: {model_results['target_column']}
- Features used: {model_results['feature_names']}
- Train/test split: {model_results['n_train']} / {model_results['n_test']} samples
- Linear Regression → RMSE: {model_results['lr_rmse']:.2f}, R²: {model_results['lr_r2']:.4f}, MAPE: {model_results['lr_mape']:.2f}%
- Random Forest     → RMSE: {model_results['rf_rmse']:.2f}, R²: {model_results['rf_r2']:.4f}, MAPE: {model_results['rf_mape']:.2f}%
- Best model: {model_results['best_model']}
- Top 3 features by importance: {top_features}
- {seg_summary}

Write 5-7 sentences covering: what was predicted, how each model performed,
which model won and why, what the top features reveal, and one practical recommendation.
Professional prose only. No bullet points. No markdown.
"""
    response = client.generate(model='mistral', prompt=prompt)
    return response['response'].strip()


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _build_narrative_chunks(narrative: str, model_results: dict,
                             timestamp: str) -> list[dict]:
    """
    TYPE 1 — narrative sentences.
    Good for: 'how did the model do?', 'which model was better?',
              'what features mattered?'
    """
    sentences = _split_sentences(narrative)
    return [
        {
            "text":           s,
            "record_type":    "narrative",
            "target_column":  model_results["target_column"],
            "best_model":     model_results["best_model"],
            "segment":        "all",
            "error_pct":      -1.0,
            "timestamp":      timestamp,
        }
        for s in sentences
    ]


def _build_segment_chunks(model_results: dict, timestamp: str) -> list[dict]:
    """
    TYPE 2 — one record per segment with its metrics as natural language.
    Good for: 'which segment had worst predictions?',
              'what was the error in Riverside?'
    """
    seg_results = model_results.get("segment_results", {})
    seg_col     = model_results.get("segment_column", "segment")
    chunks      = []

    for seg_name, metrics in seg_results.items():
        text = (
            f"{seg_col.capitalize()} '{seg_name}': "
            f"RMSE {metrics['rmse']:.2f}, "
            f"MAPE {metrics['mape']:.1f}%, "
            f"{metrics['n']} test samples. "
            f"Model: {model_results['best_model']}. "
            f"Target: {model_results['target_column']}."
        )
        chunks.append({
            "text":           text,
            "record_type":    "segment_summary",
            "target_column":  model_results["target_column"],
            "best_model":     model_results["best_model"],
            "segment":        seg_name,
            "error_pct":      float(metrics["mape"]),
            "timestamp":      timestamp,
        })

    return chunks


def _build_prediction_chunks(model_results: dict, timestamp: str) -> list[dict]:
    """
    TYPE 3 — one record per test row.
    Good for: 'show rows with error above 15%',
              'worst individual predictions in Downtown'
    Only stores rows where error > 5% to avoid flooding the index
    with easy predictions — high-error rows are what users ask about.
    """
    row_preds = model_results.get("row_predictions", [])
    chunks    = []

    for row in row_preds:
        # Only index rows worth asking about
        if row["error_pct"] < 5.0:
            continue

        seg_label = f" in segment '{row['segment']}'" if row["segment"] != "unknown" else ""
        text = (
            f"Prediction{seg_label}: "
            f"actual {row['actual']:.2f}, "
            f"predicted {row['predicted']:.2f}, "
            f"error {row['error_pct']:.1f}%. "
            f"Model: {model_results['best_model']}. "
            f"Target: {row['target']}."
        )
        chunks.append({
            "text":           text,
            "record_type":    "prediction",
            "target_column":  row["target"],
            "best_model":     model_results["best_model"],
            "segment":        row["segment"],
            "error_pct":      float(row["error_pct"]),
            "timestamp":      timestamp,
        })

    return chunks


def chronicler_node(state: FactoryState):
    """
    Worker: Generates narrative + indexes THREE record types to LanceDB.

    record_type = 'narrative'        → aggregate model story (6-8 chunks)
    record_type = 'segment_summary'  → one per segment with RMSE/MAPE
    record_type = 'prediction'       → one per high-error test row

    This three-tier structure is what makes the RAG layer answer
    specific questions rather than just paraphrasing the narrative.
    """
    model_results = state["model_results"]
    timestamp     = datetime.utcnow().isoformat()

    print("Chronicler: ✍️  Generating narrative...")

    try:
        # 1. Narrative
        narrative        = _generate_narrative(model_results)
        narrative_chunks = _build_narrative_chunks(narrative, model_results, timestamp)
        print(f"Chronicler: {len(narrative_chunks)} narrative chunks")

        # 2. Segment summaries
        segment_chunks = _build_segment_chunks(model_results, timestamp)
        print(f"Chronicler: {len(segment_chunks)} segment summary chunks")

        # 3. Row-level predictions (high-error only)
        prediction_chunks = _build_prediction_chunks(model_results, timestamp)
        print(f"Chronicler: {len(prediction_chunks)} prediction chunks (error > 5%)")

        # Index all three types — vector_ops handles embedding + storage
        all_chunks = narrative_chunks + segment_chunks + prediction_chunks
        index_chunks(all_chunks, {
            "target_column": model_results["target_column"],
            "best_model":    model_results["best_model"],
            "timestamp":     timestamp,
        })

        # Only narrative sentences go into state for the manager to check
        report_chunks = [c["text"] for c in narrative_chunks]

        return {
            "report_chunks": report_chunks,
            "errors":        [],
            "messages":      [f"Chronicler: Indexed {len(all_chunks)} chunks "
                              f"({len(narrative_chunks)} narrative, "
                              f"{len(segment_chunks)} segments, "
                              f"{len(prediction_chunks)} predictions)."],
        }

    except Exception as e:
        error_msg = f"Chronicler Error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"errors": [error_msg]}