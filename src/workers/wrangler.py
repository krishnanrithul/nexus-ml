import pandas as pd
import os
import re
from src.state import client, FactoryState


def _extract_code(raw_response: str) -> str:
    """
    Robustly strips markdown fences and leading/trailing prose from LLM output.
    """
    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = raw_response.strip().splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        if re.match(r"^(import |from |def |class |#|if |for |while |try:|except|return|    |\w+ ?=)", line):
            in_code = True
        if in_code:
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else raw_response.strip()


def _build_prompt(df: pd.DataFrame, processed_path: str, last_error: str = None, diagnosis: str = None) -> str:
    """
    Builds the cleaning prompt for initial attempt or self-correction.
    On retry, injects the manager's diagnosis as a targeted fix instruction.
    """
    summary = df.isna().sum().to_dict()
    dtypes = df.dtypes.astype(str).to_dict()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    error_block = f"""
Your previous attempt failed with this error:
{last_error}

The orchestration manager diagnosed the issue as:
{diagnosis or "No specific diagnosis provided — fix the error above."}

Apply this specific fix. Return the full corrected function.
""" if last_error else ""

    return f"""
You are a Senior Data Engineer writing a Python data cleaning function.
{error_block}
Dataset info:
- Columns and dtypes: {dtypes}
- Missing value counts: {summary}
- Categorical columns (object dtype): {cat_cols}

Write a function called `clean_data(df)` that does exactly these steps in order:

STEP 1 — Fill numerical NaNs. Use EXACTLY this code, no variation:
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

STEP 2 — Fill categorical NaNs. Use EXACTLY this loop, no variation:
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

STEP 3 — One-hot encode categorical columns. Use EXACTLY this line, no variation:
    df = pd.get_dummies(df, columns={cat_cols})

STEP 4 — Save and return. Use EXACTLY these two lines:
    df.to_csv('{processed_path}', index=False)
    return df

BANNED — using any of these will cause a runtime error, do not use them:
- df.replace() for filling NaNs
- df.join() or df.merge() after get_dummies
- sklearn, LabelEncoder, scipy, or any library other than pandas and numpy
- inplace=True on fillna applied to the whole dataframe at once
- Passing an Index object as a dict key anywhere

Return ONLY the function definition. No imports. No explanation. No markdown fences. No example calls.
"""


def wrangler_node(state: FactoryState):
    """
    Worker: Automates data cleaning with self-correction.
    Uses LLM-generated code to generalize across arbitrary datasets.
    """
    raw_path = state["raw_data_path"]
    processed_path = "data/processed/house_prices_cleaned.csv"
    os.makedirs("data/processed", exist_ok=True)

    is_retry = state.get("next_step") == "wrangler_retry"
    df_fresh = pd.read_csv(raw_path)

    if is_retry:
        last_error = state["errors"][-1]
        diagnosis = state.get("manager_diagnosis", "none")
        print(f"Wrangler: ⚠️ Self-correcting. Last error: {last_error}")
        print(f"Wrangler: 🔧 Manager diagnosis: {diagnosis}")
        prompt = _build_prompt(df_fresh, processed_path, last_error=last_error, diagnosis=diagnosis)
    else:
        print("Wrangler: 🔍 Initial attempt - Analyzing raw data...")
        prompt = _build_prompt(df_fresh, processed_path)

    response = client.generate(model='mistral', prompt=prompt)
    code = _extract_code(response['response'])
    full_code = f"{code}\n\nclean_data(df)"

    print(f"Wrangler: Generated code preview:\n{full_code[:400]}...")

    try:
        exec(full_code, {"pd": pd, "df": df_fresh, "processed_path": processed_path})

        return {
            "cleaned_data_path": processed_path,
            "errors": [],
            "messages": [f"Wrangler: {'Self-corrected' if is_retry else 'Initially cleaned'} successfully."],
        }

    except Exception as e:
        error_msg = f"Wrangler Error: {str(e)}\n--- Generated Code ---\n{full_code}"
        print(f"❌ Wrangler failed: {error_msg}")
        return {"errors": [error_msg]}