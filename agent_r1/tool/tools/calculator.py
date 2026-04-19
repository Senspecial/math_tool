import math
from typing import Any

from ..base import BaseTool
from ..schema import ToolResponse

SAFE_GLOBALS = {"__builtins__": {}}
SAFE_LOCALS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

MAX_EXPRESSION_LENGTH = 1024


@BaseTool.register("calculator")
class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = (
        "A calculator that evaluates mathematical expressions. "
        "Supports arithmetic (+, -, *, /, //, %, **), "
        "math functions (sqrt, abs, round, min, max, sum, log, ceil, floor, sin, cos, tan), "
        "and constants (pi, e). Example: '(15 + 27) * 3' or 'sqrt(144) + max(3, 5)'."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g. '(15 + 27) * 3'",
            },
        },
        "required": ["expression"],
    }

    async def execute(self, args: dict[str, Any], **kwargs) -> tuple[ToolResponse, float | None, dict]:
        expression = args.get("expression", "")
        if not isinstance(expression, str):
            expression = str(expression)

        expression = expression.strip()
        if not expression:
            return ToolResponse(text="Error: empty expression"), None, {}

        if len(expression) > MAX_EXPRESSION_LENGTH:
            return (
                ToolResponse(text=f"Error: expression too long (max {MAX_EXPRESSION_LENGTH} chars)"),
                None,
                {},
            )

        try:
            result = eval(expression, SAFE_GLOBALS, SAFE_LOCALS)  # noqa: S307
            result_str = str(result)
        except ZeroDivisionError:
            result_str = "Error: division by zero"
        except Exception as exc:
            result_str = f"Error: {type(exc).__name__}: {exc}"

        return ToolResponse(text=result_str), None, {"expression": expression, "result": result_str}
