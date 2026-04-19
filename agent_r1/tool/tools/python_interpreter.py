import asyncio
import logging
import os
import subprocess
import tempfile
from typing import Any

from ..base import BaseTool
from ..schema import ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_TIMEOUT = 10
MAX_OUTPUT_LENGTH = 4096
MAX_CODE_LENGTH = 8192


@BaseTool.register("python_interpreter")
class PythonInterpreterTool(BaseTool):
    name: str = "python_interpreter"
    description: str = (
        "Execute Python code and return stdout/stderr. "
        "Use this for complex calculations, data processing, or any logic "
        "that goes beyond simple arithmetic. The code runs in an isolated subprocess "
        "with a timeout limit."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Use print() to output results.",
            },
        },
        "required": ["code"],
    }

    async def execute(self, args: dict[str, Any], **kwargs) -> tuple[ToolResponse, float | None, dict]:
        code = args.get("code", "")
        if not isinstance(code, str):
            code = str(code)

        code = code.strip()
        if not code:
            return ToolResponse(text="Error: empty code"), None, {}

        if len(code) > MAX_CODE_LENGTH:
            return (
                ToolResponse(text=f"Error: code too long (max {MAX_CODE_LENGTH} chars)"),
                None,
                {},
            )

        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, self._run_code, code)

        return ToolResponse(text=output), None, {"code": code, "output": output}

    @staticmethod
    def _run_code(code: str) -> str:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUT,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if stdout and stderr:
                output = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            elif stderr:
                output = f"Error:\n{stderr}"
            else:
                output = stdout if stdout else "(no output)"

        except subprocess.TimeoutExpired:
            output = f"Error: execution timed out after {DEFAULT_TIMEOUT} seconds"
        except Exception as exc:
            output = f"Error: {type(exc).__name__}: {exc}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + f"\n... (truncated, total {len(output)} chars)"

        return output
