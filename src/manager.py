from src.state import client, FactoryState

# Valid actions the LLM can return
_VALID_ACTIONS = {"wrangler", "wrangler_retry", "modeler", "modeler_retry", "chronicler", "end"}

# Hard cap on retries — prevents infinite self-correction loops
_MAX_RETRIES = 2


def _build_routing_prompt(state: FactoryState) -> str:
    """
    Summarises the current pipeline state for the LLM to reason over.
    Gives the manager everything it needs to make a decision.
    """
    # Summarise what's done so far
    completed = []
    if state.get("cleaned_data_path"):
        completed.append(f"- Data cleaned and saved to: {state['cleaned_data_path']}")
    if state.get("model_results"):
        mr = state["model_results"]
        completed.append(
            f"- Models trained. Best: {mr.get('best_model')}, "
            f"RMSE: {mr.get('rf_rmse', mr.get('lr_rmse', 'unknown')):.2f}, "
            f"Target: {mr.get('target_column')}"
        )
    if state.get("report_chunks"):
        completed.append(f"- Narrative indexed: {len(state['report_chunks'])} chunks in LanceDB")

    completed_str = "\n".join(completed) if completed else "- Nothing completed yet"

    # Summarise errors
    error_str = "None"
    diagnosis_instruction = ""
    if state.get("errors"):
        last_error = state["errors"][-1]
        retry_count = state.get("retry_count", 0)
        error_str = f"{last_error} (retry attempt {retry_count} of {_MAX_RETRIES})"
        diagnosis_instruction = f"""
IMPORTANT: There is an active error. Before deciding the action, diagnose it:
- What specifically caused: "{last_error}"
- What one targeted fix should the worker apply on retry?
Include your diagnosis in the 'reasoning' field.
"""

    return f"""
You are the orchestration manager of NexusML, an agentic ML pipeline.
Your job is to assess the current state and decide what happens next.

PIPELINE STAGES (in order):
1. wrangler   — cleans raw CSV data
2. modeler    — trains LinearRegression and RandomForest, picks best
3. chronicler — generates narrative from results, indexes to LanceDB
4. end        — all stages complete

CURRENT STATE:
Completed steps:
{completed_str}

Last error: {error_str}
Messages: {state.get('messages', [])}
{diagnosis_instruction}
AVAILABLE ACTIONS:
- wrangler        : run data cleaning (first time)
- wrangler_retry  : re-run wrangler with error context (only if wrangler failed)
- modeler         : run model training
- modeler_retry   : re-run modeler with error context (only if modeler failed)
- chronicler      : generate narrative and index to LanceDB
- end             : pipeline is complete or unrecoverable

Respond in this exact format, nothing else:
ACTION: <action>
REASONING: <one sentence explaining why>
DIAGNOSIS: <if there is an error, one sentence on the specific fix needed. Otherwise write 'none'>
"""


def _parse_response(response: str) -> tuple[str, str, str]:
    """
    Parses ACTION, REASONING, and DIAGNOSIS from the LLM response.
    Falls back gracefully if the model doesn't follow the format exactly.
    """
    action = "end"
    reasoning = "Could not parse response."
    diagnosis = "none"

    for line in response.strip().splitlines():
        line = line.strip()
        if line.startswith("ACTION:"):
            action = line.replace("ACTION:", "").strip().lower()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
        elif line.startswith("DIAGNOSIS:"):
            diagnosis = line.replace("DIAGNOSIS:", "").strip()

    # Guard against hallucinated actions
    if action not in _VALID_ACTIONS:
        print(f"Manager: ⚠️ LLM returned unknown action '{action}'. Defaulting to 'end'.")
        action = "end"

    return action, reasoning, diagnosis


def manager_node(state: FactoryState) -> dict:
    """
    LLM-powered orchestrator. Reads full pipeline state, reasons about
    what to do next, and returns the action + a diagnosis for retries.
    """
    print("\n--- [MANAGER] Assessing System State ---")

    # Hard stop — if we've hit max retries don't even ask the LLM
    retry_count = state.get("retry_count", 0)
    if retry_count >= _MAX_RETRIES and state.get("errors"):
        print(f"Manager: 🛑 Max retries ({_MAX_RETRIES}) reached. Terminating.")
        return {"next_step": "end", "retry_count": 0}

    # Ask the LLM to reason over the state
    prompt = _build_routing_prompt(state)
    response = client.generate(model="mistral", prompt=prompt)
    raw = response["response"]

    action, reasoning, diagnosis = _parse_response(raw)

    print(f"Manager: 🧠 Action    → {action}")
    print(f"Manager: 💬 Reasoning → {reasoning}")
    if diagnosis != "none":
        print(f"Manager: 🔧 Diagnosis → {diagnosis}")

    # Track retry count — increments on retry actions, resets otherwise
    new_retry_count = retry_count + 1 if "retry" in action else 0

    return {
        "next_step":     action,
        "retry_count":   new_retry_count,
        "manager_diagnosis": diagnosis,  # workers read this on retry to apply the fix
        "errors":        [],             # clear errors after manager has processed them
        "messages":      [f"Manager: Routed to '{action}'. Reason: {reasoning}"],
    }