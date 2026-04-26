import os
import lancedb
import numpy as np
import pyarrow as pa
from typing import List, Dict, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer

_LANCEDB_PATH = "db/"
_TABLE_NAME   = "model_reports"
_EMBED_MODEL  = SentenceTransformer("all-MiniLM-L6-v2")
_VECTOR_DIM   = 384


# ─────────────────────────────────────────────────────────────
# SCHEMA
# Three record types share one table, distinguished by record_type.
# Extra fields (segment, error_pct) enable filtered queries.
# ─────────────────────────────────────────────────────────────

_SCHEMA = pa.schema([
    pa.field("vector",        pa.list_(pa.float32(), _VECTOR_DIM)),
    pa.field("text",          pa.utf8()),
    pa.field("record_type",   pa.utf8()),   # 'narrative' | 'segment_summary' | 'prediction'
    pa.field("target_column", pa.utf8()),
    pa.field("best_model",    pa.utf8()),
    pa.field("segment",       pa.utf8()),   # segment name or 'all' / 'unknown'
    pa.field("error_pct",     pa.float32()), # -1.0 for narrative records
    pa.field("timestamp",     pa.utf8()),
])


def _embed(texts: List[str]) -> np.ndarray:
    """
    Single embed function used everywhere — never instantiate _EMBED_MODEL elsewhere.
    Query and index vectors must come from the same model instance.
    """
    return _EMBED_MODEL.encode(texts, convert_to_numpy=True).astype(np.float32)


def _get_or_create_table(db) -> lancedb.table.Table:
    if _TABLE_NAME in db.table_names():
        return db.open_table(_TABLE_NAME)
    return db.create_table(_TABLE_NAME, schema=_SCHEMA)


def _strip_vectors(records: List[Dict]) -> List[Dict]:
    """Remove vector arrays before returning to callers — they're large and useless outside LanceDB."""
    return [{k: v for k, v in r.items() if k != "vector"} for r in records]


# ─────────────────────────────────────────────────────────────
# WRITE
# ─────────────────────────────────────────────────────────────

def index_chunks(chunks: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    """
    Embed and store a list of chunk dicts.

    Each chunk dict must have at minimum: 'text', 'record_type'.
    The full schema fields (segment, error_pct, etc.) should be
    set by the caller (chronicler) — defaults applied here if missing.

    Called by:
        chronicler_node → index_chunks(all_chunks, model_results)
    """
    if not chunks:
        print("[VectorOps] No chunks to index.")
        return

    os.makedirs(_LANCEDB_PATH, exist_ok=True)
    db        = lancedb.connect(_LANCEDB_PATH)
    timestamp = metadata.get("timestamp", datetime.utcnow().isoformat())
    texts     = [c["text"] for c in chunks]
    embeddings = _embed(texts)

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "vector":        embeddings[i].tolist(),
            "text":          chunk["text"],
            "record_type":   chunk.get("record_type",   "narrative"),
            "target_column": chunk.get("target_column", metadata.get("target_column", "unknown")),
            "best_model":    chunk.get("best_model",    metadata.get("best_model",    "unknown")),
            "segment":       chunk.get("segment",       "all"),
            "error_pct":     float(chunk.get("error_pct", -1.0)),
            "timestamp":     chunk.get("timestamp",     timestamp),
        })

    table = _get_or_create_table(db)
    table.add(records)
    print(f"[VectorOps] Indexed {len(records)} chunks into '{_TABLE_NAME}'.")


# ─────────────────────────────────────────────────────────────
# READ
# ─────────────────────────────────────────────────────────────

def query_chunks(question: str,
                 top_k: int = 5,
                 record_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Embed question, run ANN search, return top_k results (vectors stripped).

    Args:
        question:    Natural language question from the user.
        top_k:       Number of results to return.
        record_type: Optional filter — one of:
                       'narrative'       → aggregate model story
                       'segment_summary' → per-segment metrics
                       'prediction'      → individual row predictions
                     None = search all record types.

    Returns list of dicts with: text, record_type, segment, error_pct,
    best_model, target_column, timestamp, _distance.

    Called by:
        query_engine.py → query_chunks(question, top_k=5)
        query_engine.py → query_chunks(question, record_type='segment_summary')
    """
    db = lancedb.connect(_LANCEDB_PATH)

    if _TABLE_NAME not in db.table_names():
        print("[VectorOps] Table not found. Run the pipeline first.")
        return []

    table        = db.open_table(_TABLE_NAME)
    query_vector = _embed([question])[0].tolist()

    search = table.search(query_vector).limit(top_k * 3 if record_type else top_k)
    raw    = search.to_list()

    # Post-filter by record_type if specified
    # (LanceDB OSS doesn't support pre-filter on ANN — filter after retrieval)
    if record_type:
        raw = [r for r in raw if r.get("record_type") == record_type]

    return _strip_vectors(raw[:top_k])


def query_segments(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience wrapper — searches only segment_summary records.
    Use for questions like 'which segment had worst predictions?'
    """
    return query_chunks(question, top_k=top_k, record_type="segment_summary")


def query_predictions(question: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Convenience wrapper — searches only prediction records.
    Use for questions like 'show rows with high error' or
    'worst predictions in Riverside'.
    """
    return query_chunks(question, top_k=top_k, record_type="prediction")


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def get_all_chunks(record_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all stored chunks, optionally filtered by record_type. Vectors stripped."""
    db = lancedb.connect(_LANCEDB_PATH)
    if _TABLE_NAME not in db.table_names():
        return []
    df = db.open_table(_TABLE_NAME).to_pandas().drop(columns=["vector"], errors="ignore")
    if record_type:
        df = df[df["record_type"] == record_type]
    return df.to_dict(orient="records")


def clear_table() -> None:
    """Drop the table. Call at start of each test run for a clean slate."""
    db = lancedb.connect(_LANCEDB_PATH)
    if _TABLE_NAME in db.table_names():
        db.drop_table(_TABLE_NAME)
        print(f"[VectorOps] Cleared '{_TABLE_NAME}'.")
    else:
        print(f"[VectorOps] Table '{_TABLE_NAME}' doesn't exist.")


def table_stats() -> Dict[str, Any]:
    """Summary of what's currently in the table — useful for test verification."""
    db = lancedb.connect(_LANCEDB_PATH)
    if _TABLE_NAME not in db.table_names():
        return {"status": "table not found"}

    df = db.open_table(_TABLE_NAME).to_pandas()
    by_type = df.groupby("record_type").size().to_dict() if "record_type" in df.columns else {}

    return {
        "status":         "ok",
        "total_chunks":   len(df),
        "by_record_type": by_type,        # {"narrative": 7, "segment_summary": 4, "prediction": 38}
        "unique_models":  df["best_model"].unique().tolist()    if "best_model"    in df.columns else [],
        "unique_targets": df["target_column"].unique().tolist() if "target_column" in df.columns else [],
        "segments":       df[df["record_type"] == "segment_summary"]["segment"].tolist()
                          if "record_type" in df.columns else [],
    }