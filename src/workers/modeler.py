import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.state import client, FactoryState


def _infer_target_column(columns: list) -> str:
    prompt = f"""
You are a Data Scientist. Given these column names from a dataset:
{columns}

Which single column is most likely the TARGET variable (the one to predict)?
Reply with ONLY the exact column name. No explanation. No punctuation.
"""
    response = client.generate(model='mistral', prompt=prompt)
    candidate = response['response'].strip().strip('"').strip("'")
    if candidate in columns:
        return candidate
    print(f"Modeler: ⚠️ LLM returned '{candidate}' — not valid. Falling back to last column.")
    return columns[-1]


def _infer_segment_column(df: pd.DataFrame, feature_cols: list) -> str | None:
    """
    Detect best column for segment analysis.
    Prefers raw categorical; falls back to detecting OHE prefix (e.g. neighborhood_X).
    Returns None if nothing suitable found.
    """
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        return cat_cols[0]

    # Detect OHE columns — wrangler produces neighborhood_Riverside style
    ohe_cols = [c for c in feature_cols if "_" in c and df[c].nunique() <= 2]
    if ohe_cols:
        prefixes = list({c.rsplit("_", 1)[0] for c in ohe_cols})
        return prefixes[0]

    return None


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100)
    return {"rmse": rmse, "r2": r2, "mape": mape}


def _segment_rmse(df_test: pd.DataFrame, y_true: np.ndarray,
                  y_pred: np.ndarray, segment_prefix: str) -> dict:
    """
    Compute RMSE + MAPE per segment value.
    Handles both raw categorical and OHE columns.

    Returns:
        {"Riverside": {"rmse": 14200.0, "mape": 12.4, "n": 45}, ...}
    """
    results = {}
    ohe_cols = [c for c in df_test.columns if c.startswith(f"{segment_prefix}_")]

    if ohe_cols:
        for col in ohe_cols:
            seg_name = col.replace(f"{segment_prefix}_", "")
            mask = df_test[col].astype(bool).values
            if mask.sum() < 3:
                continue
            m = _evaluate(y_true[mask], y_pred[mask])
            results[seg_name] = {"rmse": round(m["rmse"], 2),
                                 "mape": round(m["mape"], 2),
                                 "n":    int(mask.sum())}
    elif segment_prefix in df_test.columns:
        for seg_val in df_test[segment_prefix].unique():
            mask = (df_test[segment_prefix] == seg_val).values
            if mask.sum() < 3:
                continue
            m = _evaluate(y_true[mask], y_pred[mask])
            results[str(seg_val)] = {"rmse": round(m["rmse"], 2),
                                     "mape": round(m["mape"], 2),
                                     "n":    int(mask.sum())}
    return results


def _feature_importance(model, X_train: np.ndarray, feature_cols: list) -> dict:
    """
    SHAP mean absolute values if shap is installed,
    otherwise falls back to RF feature_importances_.
    Returns top 10 features sorted descending.
    """
    try:
        import shap
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_train[:200])
        importance = np.abs(shap_vals).mean(axis=0)
    except ImportError:
        print("Modeler: ⚠️ shap not installed — using feature_importances_ instead.")
        importance = model.feature_importances_

    ranked = sorted(zip(feature_cols, importance.tolist()),
                    key=lambda x: x[1], reverse=True)
    return {k: round(float(v), 4) for k, v in ranked[:10]}


def _build_row_predictions(df_test: pd.DataFrame, y_true: np.ndarray,
                            y_pred: np.ndarray, target_col: str,
                            segment_prefix: str | None) -> list[dict]:
    """
    One dict per test row — stored as individual 'prediction' records in LanceDB.
    This is what enables the RAG layer to answer row-specific questions like
    'which properties had error above 15%?' or 'worst predictions in Riverside?'
    """
    records = []
    for i in range(len(y_true)):
        actual    = float(y_true[i])
        predicted = float(y_pred[i])
        error_pct = abs(actual - predicted) / (abs(actual) if actual != 0 else 1) * 100

        rec = {
            "actual":    round(actual, 2),
            "predicted": round(predicted, 2),
            "error_pct": round(error_pct, 2),
            "target":    target_col,
            "segment":   "unknown",
        }

        if segment_prefix:
            ohe_cols = [c for c in df_test.columns if c.startswith(f"{segment_prefix}_")]
            if ohe_cols:
                for col in ohe_cols:
                    if df_test.iloc[i][col]:
                        rec["segment"] = col.replace(f"{segment_prefix}_", "")
                        break
            elif segment_prefix in df_test.columns:
                rec["segment"] = str(df_test.iloc[i][segment_prefix])

        records.append(rec)

    return records


def modeler_node(state: FactoryState):
    """
    Worker: Trains LR + RF. Produces three levels of output for the RAG layer:
      1. Aggregate metrics  → narrative record in LanceDB
      2. Segment RMSE/MAPE  → segment_summary records in LanceDB
      3. Row predictions    → prediction records in LanceDB

    Without all three, query_engine can only answer generic questions.
    With all three, it can answer 'which segment was worst' and
    'show me rows with error above 15%' — the questions that matter.
    """
    cleaned_path = state["cleaned_data_path"]
    print("Modeler: 📊 Loading cleaned data...")

    try:
        df = pd.read_csv(cleaned_path)
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        columns    = df.columns.tolist()
        target_col = _infer_target_column(columns)
        print(f"Modeler: 🎯 Target: '{target_col}'")

        feature_cols = [c for c in columns if c != target_col]
        X = df[feature_cols].values
        y = df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Keep test rows aligned with predictions
        df_test = df.iloc[len(X_train):].reset_index(drop=True)

        # ── Train ──────────────────────────────────────────────
        print("Modeler: Training LinearRegression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds   = lr.predict(X_test)
        lr_metrics = _evaluate(y_test, lr_preds)
        print(f"  LR  RMSE {lr_metrics['rmse']:.2f}  R² {lr_metrics['r2']:.4f}  MAPE {lr_metrics['mape']:.2f}%")

        print("Modeler: Training RandomForest...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds   = rf.predict(X_test)
        rf_metrics = _evaluate(y_test, rf_preds)
        print(f"  RF  RMSE {rf_metrics['rmse']:.2f}  R² {rf_metrics['r2']:.4f}  MAPE {rf_metrics['mape']:.2f}%")

        best_model = "RandomForest" if rf_metrics["rmse"] < lr_metrics["rmse"] else "LinearRegression"
        best_preds = rf_preds if best_model == "RandomForest" else lr_preds
        print(f"Modeler: ✅ Winner: {best_model}")

        # ── Segment analysis ───────────────────────────────────
        seg_col     = _infer_segment_column(df_test, feature_cols)
        seg_results = {}
        if seg_col:
            seg_results = _segment_rmse(df_test, y_test, best_preds, seg_col)
            if seg_results:
                worst    = max(seg_results, key=lambda s: seg_results[s]["rmse"])
                best_seg = min(seg_results, key=lambda s: seg_results[s]["rmse"])
                print(f"Modeler: 📍 Segment '{seg_col}': "
                      f"worst={worst} ({seg_results[worst]['rmse']:.2f}), "
                      f"best={best_seg} ({seg_results[best_seg]['rmse']:.2f})")

        # ── Feature importance ─────────────────────────────────
        print("Modeler: 🔍 Feature importance...")
        feat_imp = _feature_importance(rf, X_train, feature_cols)
        print(f"  Top 3: {list(feat_imp.keys())[:3]}")

        # ── Row-level predictions ──────────────────────────────
        row_preds = _build_row_predictions(df_test, y_test, best_preds, target_col, seg_col)
        print(f"Modeler: Built {len(row_preds)} row-level prediction records")

        return {
            "model_results": {
                # Aggregate
                "lr_rmse":            lr_metrics["rmse"],
                "lr_r2":              lr_metrics["r2"],
                "lr_mape":            lr_metrics["mape"],
                "rf_rmse":            rf_metrics["rmse"],
                "rf_r2":              rf_metrics["r2"],
                "rf_mape":            rf_metrics["mape"],
                "best_model":         best_model,
                "target_column":      target_col,
                "feature_names":      feature_cols,
                "n_train":            len(X_train),
                "n_test":             len(X_test),
                # Segment level
                "segment_column":     seg_col,
                "segment_results":    seg_results,
                # Feature importance
                "feature_importance": feat_imp,
                # Row level — passed to chronicler for LanceDB indexing
                "row_predictions":    row_preds,
            },
            "errors":   [],
            "messages": [f"Modeler: {best_model} wins. "
                         f"{len(seg_results)} segments. "
                         f"{len(row_preds)} prediction records."],
        }

    except Exception as e:
        error_msg = f"Modeler Error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"errors": [error_msg]}