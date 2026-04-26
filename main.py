from langgraph.graph import StateGraph, END

from src.state import FactoryState
from src.manager import manager_node
from src.workers.wrangler import wrangler_node
from src.workers.modeler import modeler_node
from src.workers.chronicler import chronicler_node
from src.tools.vector_ops import clear_table, table_stats


# ─────────────────────────────────────────────
# ROUTER
# Called after every manager_node execution.
# Reads next_step from state and returns the
# name of the edge LangGraph should follow.
# ─────────────────────────────────────────────

def route(state: FactoryState) -> str:
    return state["next_step"]


# ─────────────────────────────────────────────
# GRAPH DEFINITION
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(FactoryState)

    # Register nodes
    graph.add_node("manager",    manager_node)
    graph.add_node("wrangler",   wrangler_node)
    graph.add_node("modeler",    modeler_node)
    graph.add_node("chronicler", chronicler_node)

    # Entry point — manager always runs first
    graph.set_entry_point("manager")

    # Manager decides what runs next via the router
    graph.add_conditional_edges(
        "manager",
        route,
        {
            "wrangler":       "wrangler",
            "wrangler_retry": "wrangler",
            "modeler":        "modeler",
            "modeler_retry":  "modeler",
            "chronicler":     "chronicler",
            "end":            END,
        }
    )

    # Every worker reports back to the manager after completing
    graph.add_edge("wrangler",   "manager")
    graph.add_edge("modeler",    "manager")
    graph.add_edge("chronicler", "manager")

    return graph.compile()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run(raw_data_path: str, fresh: bool = True):
    """
    Run the full NexusML pipeline.

    Args:
        raw_data_path: Path to the raw CSV file.
        fresh:         If True, clears LanceDB before running so results
                       reflect only this run. Set False to accumulate
                       results across multiple datasets.
    """
    if fresh:
        print("🗑️  Clearing previous LanceDB state...")
        clear_table()

    initial_state: FactoryState = {
        "raw_data_path":     raw_data_path,
        "cleaned_data_path": None,
        "model_results":     {},
        "report_chunks":     [],
        "messages":          [],
        "next_step":         "",
        "errors":            [],
        "retry_count":       0,
    }

    print(f"🚀 Starting NexusML pipeline on: {raw_data_path}\n")

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # ── Summary ──────────────────────────────
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)

    model_results = final_state.get("model_results", {})
    if model_results:
        print(f"✅ Best model    : {model_results.get('best_model')}")
        print(f"   Target       : {model_results.get('target_column')}")
        print(f"   LR  RMSE/R²  : {model_results.get('lr_rmse'):.2f} / {model_results.get('lr_r2'):.4f}")
        print(f"   RF  RMSE/R²  : {model_results.get('rf_rmse'):.2f} / {model_results.get('rf_r2'):.4f}")

    stats = table_stats()
    if stats.get("status") == "ok":
        print(f"✅ LanceDB       : {stats['total_chunks']} chunks indexed")

    if final_state.get("errors"):
        print(f"⚠️  Errors        : {final_state['errors']}")

    print("\n▶️  Run `python query_engine.py` to query your results.")

    return final_state


if __name__ == "__main__":
    run("data/raw/house_prices.csv")