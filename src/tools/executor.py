from typing import Any, Dict


def safe_exec(code: str, globals_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Centralised sandbox for executing LLM-generated code.
    All workers that use exec() must go through here — never call exec() directly.

    Returns:
        {
            "success": bool,
            "error": str | None,    # None on success
            "globals": dict          # the globals_dict after execution (mutated in place)
        }

    Usage in a worker:
        from src.tools.executor import safe_exec

        result = safe_exec(full_code, {"pd": pd, "df": df_fresh})
        if not result["success"]:
            return {"errors": [f"Wrangler Error: {result['error']}"]}
    """
    try:
        exec(code, globals_dict)  # noqa: S102
        return {"success": True, "error": None, "globals": globals_dict}
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"SyntaxError on line {e.lineno}: {e.msg}\n--- Code ---\n{code}",
            "globals": globals_dict,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}\n--- Code ---\n{code}",
            "globals": globals_dict,
        }