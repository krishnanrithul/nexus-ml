# NexusML

**Agentic ML pipeline with self-correcting orchestration and conversational RAG**

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic-purple?style=flat-square)
![LanceDB](https://img.shields.io/badge/LanceDB-vector--store-green?style=flat-square)
![Mistral](https://img.shields.io/badge/Mistral-local--LLM-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-teal?style=flat-square)

NexusML is a self-correcting agentic ML pipeline that cleans arbitrary tabular datasets, trains and evaluates models at aggregate and segment level, indexes results into a vector store, and lets you query findings conversationally via a RAG layer. Built to close a personal skill gap between classical ML depth and modern LLM systems architecture — every component has a specific reason for existing and a known failure mode.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    ▼                                     │
CSV ──► MANAGER ──► WRANGLER ──► MANAGER ──► MODELER ──► MANAGER ──► CHRONICLER
         (LLM)     (LLM code    (diagnoses   (segment     (routes       (3 record
                    generation   failures,    RMSE +       or retries)    types →
                    + exec())    passes fix)  SHAP)                      LanceDB)
                                                                            │
                                                                            ▼
                                                                        LanceDB
                                                                      (narrative +
                                                                       segment +
                                                                       prediction
                                                                        records)
                                                                            │
                                                              ┌─────────────┘
                                                              ▼
                                                       query_engine
                                                      (intent routing)
                                                              │
                                                              ▼
                                                          FastAPI
                                                              │
                                                              ▼
                                                       Browser UI
                                                    (chat + sources)
```

Every worker feeds back to the manager. The manager reads full pipeline state, decides the next action via an LLM call, and on failure diagnoses the root cause before routing a retry.

---

## Key Architectural Decisions

### 1. LLM-powered manager vs if/elif routing

The manager is a Mistral call, not a switch statement. An if/elif manager can only handle cases anticipated at build time — an LLM manager reasons over novel state combinations without hardcoding every edge case. The tradeoff is non-determinism in routing, mitigated by a `_VALID_ACTIONS` guard that catches hallucinated actions before they reach LangGraph, and a `_MAX_RETRIES` hard cap that prevents infinite loops.

The manager returns a structured `ACTION / REASONING / DIAGNOSIS` response. The diagnosis field is the key addition — it passes a targeted one-sentence fix to the retrying worker rather than just the raw error message.

### 2. Retry-by-reasoning vs retry-by-hope

Most self-correction loops re-prompt with the error and hope the model does better. NexusML's manager diagnoses the failure first: *"unhashable type: Index means an Index object was passed as a dict key in df.replace() — use a for loop over column names instead."* That diagnosis is injected into the wrangler's retry prompt. The worker gets told what the fix is, not just what broke.

### 3. Three LanceDB record types vs a single narrative blob

Storing only narrative sentences means the system can only answer questions the LLM already answered when writing them. Ask "which neighborhood had the worst predictions?" and there's literally no data to retrieve. NexusML stores three types:

- `narrative` — aggregate model story (answers "how did the model do?")
- `segment_summary` — one record per segment with RMSE/MAPE (answers "which segment was worst?")
- `prediction` — one record per high-error test row (answers "show me rows with error above 15%")

This three-tier structure is what makes RAG retrieval useful rather than cosmetic.

### 4. Intent routing before retrieval

`query_engine` classifies the question before searching LanceDB. A segment question searches only `segment_summary` records; a prediction question searches only `prediction` records. Searching all record types for every question dilutes results — the closest narrative sentence is rarely the most relevant answer to a segment-specific question. Falls back to unfiltered search gracefully if the typed search returns nothing.

Pre-filtering is applied at the LanceDB query level via `.where()` — not post-filtered in Python after retrieval. This means only the relevant record type is scanned, not the full table.

### 5. exec() in the wrangler vs hardcoded cleaning logic

The wrangler needs to generalise across arbitrary datasets — it doesn't know column names, types, or cleaning requirements ahead of time. LLM-generated code adapts to whatever schema it sees. The alternative (hardcoded cleaning) works for one dataset and breaks on any other.

The tradeoff: `exec()` runs arbitrary code in the Python process with full environment access. Mitigated locally by a strict prompt with banned patterns, exact code templates, and the manager's diagnosis on retry. In production this would run in a sandboxed container.

---

## RAG Evaluation

`eval_rag.py` benchmarks the query engine on 15 canned questions across all three record types.

| Metric | Result |
|---|---|
| Intent routing accuracy | 80% (12/15) |
| Retrieval precision | 80%+ (after segment fix) |
| Query latency (avg) | 46ms |
| Query latency (P95) | 310ms |
| Fallback rate | 0/15 |

**Known routing edge cases:**
- `"How did the model perform overall?"` — "overall" scores as prediction without explicit narrative keyword matching. Fixed by adding a `_NARRATIVE_KEYWORDS` check before segment/prediction scoring.
- `"What was the RMSE for Downtown?"` — routes to narrative without domain-specific segment terms in `_SEGMENT_KEYWORDS`. Fixed by adding metric terms (`rmse`, `mape`) and neighborhood names to the segment keyword set.

---

## What You Can Ask

After running the pipeline, `query_engine` routes questions to the right record type automatically:

| Question | Intent | Record type searched |
|---|---|---|
| "which model performed better?" | narrative | aggregate model story |
| "what features were most important?" | narrative | aggregate model story |
| "which segment had the worst error?" | segment | segment_summary records |
| "what was the RMSE for Riverside?" | segment | segment_summary records |
| "show me rows with high prediction error" | prediction | row-level prediction records |
| "worst predictions in Downtown?" | prediction | row-level prediction records |

---

## How to Run

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.ai) installed locally.

```bash
# 1. Clone and install
git clone https://github.com/krishnanrithul/nexus-ml
cd nexus-ml
pip install -r requirements.txt

# 2. Pull Mistral locally
ollama pull mistral

# 3. Generate the dataset
python generate_data.py

# 4. Run the pipeline
python main.py

# 5. Evaluate RAG quality
python eval_rag.py

# 6. Start the query interface
uvicorn app.main:app --reload
# Open http://localhost:8000
```

---

## Failure Modes and Mitigations

| Component | Failure | Why it happens | Mitigation |
|---|---|---|---|
| Manager | LLM hallucinates unknown action | Mistral returns `ACTION: retrain` which doesn't exist | `_VALID_ACTIONS` guard defaults to `end` |
| Manager | Infinite retry loop | Worker keeps failing, manager keeps routing back | `_MAX_RETRIES = 2` hard stop before LLM is called |
| Wrangler | `exec()` defines function but never calls it | LLMs generate definitions, not calls | `clean_data(df)` appended to generated code before exec |
| Wrangler | LLM uses `df.join()` after `get_dummies` | Common training data pattern causes column overlap | Explicit BANNED list + exact code templates in prompt |
| Wrangler | Data leakage via temporal columns | `sale_date` OHE produces 365 columns perfectly correlated with price | Removed from dataset generation — no temporal features |
| Vector ops | Query and index vectors in different spaces | Different embedding model instances used | Single `_EMBED_MODEL` instance in `vector_ops.py` — never instantiate elsewhere |
| Vector ops | Schema mismatch on second run | LanceDB infers wrong vector dtype without explicit schema | PyArrow schema with `pa.list_(pa.float32(), 384)` |
| Vector ops | Post-filter wastes retrieval budget | Fetching `top_k * 3` then filtering in Python | Pre-filter via `.where(record_type = '...')` at query time |
| Query engine | LLM answers from training data not context | Mistral ignores retrieval context for generic questions | Prompt explicitly instructs: answer ONLY from context |
| Query engine | Ambiguous queries misroute | "overall" scores as prediction without narrative keywords | `_NARRATIVE_KEYWORDS` checked first before segment/prediction |

---

## What I'd Do Differently at Production Scale

- **Sandbox `exec()`** — containerise the wrangler in Docker with no network access and a restricted filesystem. `exec()` in a local POC is acceptable; in production it's a security boundary.
- **Production LLM API** — replace Ollama/Mistral with an API-served model with rate limiting, retry logic, and fallback. Local Mistral is fast to iterate with; production needs reliability guarantees.
- **Retrieval-optimized embeddings** — replace `all-MiniLM-L6-v2` (trained on sentence similarity) with a retrieval-specific model such as Jina v3 or Voyage AI. Same LanceDB, better vectors.
- **Hybrid search** — add BM25 alongside dense vector search. Exact terms like neighborhood names and metric abbreviations (`RMSE`, `MAPE`) are better matched by keyword search than semantic similarity alone.
- **Structured logging** — replace `print()` statements with structured logs. Right now debugging means reading terminal output; in production you need queryable logs with correlation IDs.
- **Persistent state** — LangGraph state is currently held in memory and lost between runs. Persist state to a database so the pipeline can resume after failure and audit history is maintained.

---

## Roadmap

- [ ] LLM-based intent classifier — replace keyword scoring with a Mistral call for robust query routing across all question types
- [ ] Hybrid search (BM25 + dense) — improve retrieval on exact terms like neighborhood names and metric abbreviations
- [ ] FP&A domain dataset — replace house prices with a synthetic cash collection / invoice aging dataset that maps to real financial forecasting use cases
- [ ] Manager quality gate — route back to modeler if RMSE exceeds a threshold relative to target mean, with LLM-generated feature engineering suggestions

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| Orchestration | LangGraph | Declarative graph with conditional edges and built-in state management |
| LLM | Mistral via Ollama | Runs locally, no API key, fast iteration |
| Vector store | LanceDB | No server, no port, stores as files on disk — right for a local pipeline |
| Embeddings | all-MiniLM-L6-v2 | Small, fast, 384 dimensions — sufficient for semantic similarity on short sentences |
| ML | scikit-learn | LR + RF — deterministic, no reason to exec() generated sklearn code |
| Backend | FastAPI | Thin HTTP wrapper around query_engine.ask() |

---

## Project Structure

```
nexus_ml/
├── src/
│   ├── state.py          # FactoryState — shared memory across all agents
│   ├── manager.py        # LLM orchestrator — reasons over state, diagnoses failures
│   └── workers/
│       ├── wrangler.py   # LLM code generation + exec() for dataset-agnostic cleaning
│       ├── modeler.py    # LR + RF training, segment RMSE, SHAP feature importance
│       └── chronicler.py # Narrative generation + three-tier LanceDB indexing
├── src/tools/
│   ├── vector_ops.py     # LanceDB read/write with schema, pre-filtering, embedding
│   └── executor.py       # Centralised exec() sandbox
├── app/
│   ├── main.py           # FastAPI — /ask, /stats endpoints
│   └── static/
│       └── index.html    # Chat UI
├── main.py               # LangGraph graph definition and entry point
├── query_engine.py       # Intent routing + RAG + cited answers
├── eval_rag.py           # RAG evaluation — intent routing accuracy + retrieval precision
├── generate_data.py      # Synthetic dataset generation with real-world messiness
└── data/raw/             # Drop your CSV here
```