# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Math benchmark tools for RL training on MATH / AIME / Minerva datasets.

Three tools are provided:

* ``python_math_executor``  – runs arbitrary Python code (numeric computation,
  sympy symbolic math, etc.) in a sandboxed subprocess and returns stdout.
  Does **not** carry a reward signal – it is a helper for intermediate
  calculation steps.

* ``submit_math_answer``    – the terminal tool.  The agent calls this once
  it is confident in its answer.  The tool compares the submitted answer
  against the ground-truth stored in ``tools_kwargs`` using the same
  ``is_equiv`` logic from verl's ``math_reward`` module (LaTeX-aware string
  normalisation + optional sympy equivalence check).  Returns reward 1.0 on
  correct, 0.0 on incorrect.

Both tools follow the same ``BaseTool`` interface as the existing
``calculator`` tool so they plug straight into ``ToolEnv``.
"""

import asyncio
import re
import subprocess
import sys
import tempfile
from typing import Any

from ..base import BaseTool
from ..schema import ToolResponse

# ---------------------------------------------------------------------------
# Utility: math answer equivalence
# ---------------------------------------------------------------------------

def _strip_string(string: str) -> str:
    """Normalise a math answer string for comparison.

    Mirrors the logic in ``verl/verl/utils/reward_score/math_reward.py`` so
    that the tool's inline reward is consistent with the offline scorer used
    during the reward-loop fallback.
    """
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    # remove units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            string = splits[0]
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    # strip trivial LHS like "x = "
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    # fix sqrt3 -> sqrt{3}
    if "\\sqrt" in string:
        parts = string.split("\\sqrt")
        new_str = parts[0]
        for part in parts[1:]:
            if part and part[0] != "{":
                new_str += "\\sqrt{" + part[0] + "}" + part[1:]
            else:
                new_str += "\\sqrt" + part
        string = new_str
    string = string.replace(" ", "")
    # fix \frac without braces
    substrs = string.split("\\frac")
    new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if not substr or substr[0] == "{":
            new_str += substr
        elif len(substr) >= 2:
            a, b = substr[0], substr[1]
            rest = substr[2:]
            if b != "{":
                new_str += "{" + a + "}{" + b + "}" + rest
            else:
                new_str += "{" + a + "}" + b + rest
        else:
            new_str += substr
    string = new_str
    if string == "0.5":
        string = "\\frac{1}{2}"
    # a/b -> \frac{a}{b}
    parts = string.split("/")
    if len(parts) == 2:
        try:
            a_int, b_int = int(parts[0]), int(parts[1])
            if string == f"{a_int}/{b_int}":
                string = f"\\frac{{{a_int}}}{{{b_int}}}"
        except ValueError:
            pass
    return string


def _is_equiv(str1: str | None, str2: str | None) -> bool:
    """Return True if *str1* and *str2* represent the same math value."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


def _extract_boxed(solution_str: str) -> str | None:
    """Extract the last ``\\boxed{...}`` expression from *solution_str*."""
    # handle \boxed <space> form
    if "\\boxed " in solution_str:
        return solution_str.split("\\boxed ")[-1].split("$")[0]
    idx = solution_str.rfind("\\boxed")
    if idx < 0:
        idx = solution_str.rfind("\\fbox")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    open_count = 0
    while i < len(solution_str):
        if solution_str[i] == "{":
            open_count += 1
        if solution_str[i] == "}":
            open_count -= 1
            if open_count == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    raw = solution_str[idx: right_brace_idx + 1]
    # strip outer \boxed{ ... }
    left = "\\boxed{"
    if raw.startswith(left) and raw.endswith("}"):
        return raw[len(left):-1]
    return raw


def _compute_math_reward(submitted: str, ground_truth: str) -> float:
    """Return 1.0 if *submitted* equals *ground_truth*, else 0.0.

    Matching order:
    1. Try direct equivalence after normalisation.
    2. Try extracting a ``\\boxed`` expression from *submitted* first.
    3. Optionally try sympy symbolic equivalence (best-effort).
    """
    # Direct normalised match
    if _is_equiv(submitted, ground_truth):
        return 1.0

    # Maybe the model wrapped its answer in \boxed{}
    boxed = _extract_boxed(submitted)
    if boxed is not None and _is_equiv(boxed, ground_truth):
        return 1.0

    # Sympy symbolic check (best-effort, skip on error or timeout)
    try:
        import signal  # noqa: PLC0415

        from sympy import simplify, sympify  # noqa: PLC0415

        def _timeout_handler(signum, frame):
            raise TimeoutError("sympy check timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)  # 5-second timeout
        try:
            expr1 = sympify(submitted.replace("\\", "").replace("{", "(").replace("}", ")"))
            expr2 = sympify(ground_truth.replace("\\", "").replace("{", "(").replace("}", ")"))
            if simplify(expr1 - expr2) == 0:
                return 1.0
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except Exception:
        pass

    return 0.0


# ---------------------------------------------------------------------------
# Tool 1: python_math_executor
# ---------------------------------------------------------------------------

MAX_CODE_LENGTH = 8192
MAX_OUTPUT_LENGTH = 4096
EXECUTION_TIMEOUT = 15  # seconds
MAX_CONCURRENT_EXECUTIONS = 32

_exec_semaphore: asyncio.Semaphore | None = None


def _get_exec_semaphore() -> asyncio.Semaphore:
    """Lazily create a per-event-loop semaphore to cap concurrent subprocesses."""
    global _exec_semaphore
    if _exec_semaphore is None:
        _exec_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXECUTIONS)
    return _exec_semaphore


@BaseTool.register("python_math_executor")
class PythonMathExecutorTool(BaseTool):
    """Execute Python code (including sympy) for intermediate math computation.

    This tool does **not** return a reward – it is a helper that lets the
    agent run numeric/symbolic calculations mid-trajectory.  The final answer
    must still be submitted via ``submit_math_answer``.
    """

    name: str = "python_math_executor"
    description: str = (
        "Execute Python code to perform mathematical computations. "
        "Supports standard library modules and sympy for symbolic math. "
        "Use this for intermediate calculations, equation solving, or "
        "numerical verification. Do NOT use this to submit your final answer; "
        "use `submit_math_answer` for that. "
        "Example: 'from sympy import *; x = symbols(\"x\"); print(solve(x**2 - 4, x))'"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python code to execute. The output printed to stdout will be "
                    "returned. Max length: 8192 characters."
                ),
            },
        },
        "required": ["code"],
    }

    async def execute(
        self, args: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float | None, dict]:
        code = args.get("code", "")
        if not isinstance(code, str):
            code = str(code)
        code = code.strip()

        if not code:
            return ToolResponse(text="Error: empty code"), None, {}

        if len(code) > MAX_CODE_LENGTH:
            return (
                ToolResponse(
                    text=f"Error: code too long (max {MAX_CODE_LENGTH} chars)"
                ),
                None,
                {},
            )

        async with _get_exec_semaphore():
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, self._run_code, code)

        # No process reward: follow ToRL's philosophy where only the final
        # answer correctness (0/1) drives learning. This matches the reward
        # design in ToRL paper (Li et al. 2025, arXiv:2503.23383).
        # If you want to add a tool-use bonus later, uncomment the next line:
        # process_reward = 0.05 if not output.startswith(("Error", "Execution error")) else 0.0
        return ToolResponse(text=output), None, {"code": code, "output": output}

    @staticmethod
    def _run_code(code: str) -> str:
        import os  # noqa: PLC0415

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT,
            )
            stdout = result.stdout
            stderr = result.stderr

            if len(stdout) > MAX_OUTPUT_LENGTH:
                stdout = stdout[:MAX_OUTPUT_LENGTH] + "\n[output truncated]"
            if len(stderr) > MAX_OUTPUT_LENGTH:
                stderr = stderr[:MAX_OUTPUT_LENGTH] + "\n[stderr truncated]"

            if result.returncode != 0:
                output = f"Execution error (exit {result.returncode}):\n{stderr}"
                if stdout:
                    output = f"stdout:\n{stdout}\n" + output
            elif stderr:
                output = f"stdout:\n{stdout}\nstderr:\n{stderr}" if stdout else f"stderr:\n{stderr}"
            else:
                output = stdout if stdout else "(no output)"

        except subprocess.TimeoutExpired:
            output = f"Error: execution timed out after {EXECUTION_TIMEOUT} seconds"
        except Exception as exc:
            output = f"Error: {type(exc).__name__}: {exc}"
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        return output


# ---------------------------------------------------------------------------
# Tool 2: submit_math_answer  (terminal tool – carries reward)
# ---------------------------------------------------------------------------


@BaseTool.register("submit_math_answer")
class SubmitMathAnswerTool(BaseTool):
    """Submit the final answer to a math problem and receive a reward signal.

    This is the terminal tool (``terminal = True``).  ``ToolEnv.step()`` will
    set ``done=True`` as soon as this tool is executed, stopping the rollout.
    Call it exactly once when you are confident in your answer.

    The tool compares your answer against the ground truth and returns reward
    1.0 for a correct answer, 0.0 otherwise.  The feedback message intentionally
    does **not** reveal the ground truth so the reward signal stays clean.

    Accepted answer formats
    -----------------------
    * Plain numbers:          ``42``, ``-3``, ``0.5``
    * Fractions:              ``1/2``, ``\\\\frac{1}{2}``
    * Expressions:            ``2\\\\sqrt{3}``, ``\\\\pi/4``
    * Boxed LaTeX:            ``\\\\boxed{42}``
    * Comma-separated list:   ``1, 2, 3``  (for multi-answer problems)
    """

    name: str = "submit_math_answer"
    # terminal=True tells ToolEnv to set done=True after this tool executes
    terminal: bool = True
    description: str = (
        "Submit your final answer for the math problem. "
        "Returns whether your answer is correct (reward=1.0) or not (reward=0.0). "
        "Call this tool only once, after you have verified your answer. "
        "Accepted formats: plain numbers, fractions (1/2 or \\frac{1}{2}), "
        "expressions (2\\sqrt{3}), or boxed LaTeX (\\boxed{42})."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Your final answer. Can be a number, fraction, expression, or "
                    "LaTeX string. Example: '42', '\\\\frac{3}{4}', '2\\\\sqrt{3}'."
                ),
            },
        },
        "required": ["answer"],
    }

    async def execute(
        self, args: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float | None, dict]:
        answer = args.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
        answer = answer.strip()

        tools_kwargs = kwargs.get("tools_kwargs") or {}

        # Support flat and tool-namespaced layouts
        ground_truth: str | None = None
        if "ground_truth" in tools_kwargs:
            ground_truth = tools_kwargs["ground_truth"]
        elif (
            "submit_math_answer" in tools_kwargs
            and "ground_truth" in tools_kwargs["submit_math_answer"]
        ):
            ground_truth = tools_kwargs["submit_math_answer"]["ground_truth"]

        if ground_truth is None:
            raise ValueError(
                "ground_truth is required in tools_kwargs for submit_math_answer"
            )

        reward = _compute_math_reward(answer, str(ground_truth))
        is_correct = reward > 0.0

        # Do NOT reveal the ground truth in the feedback: the agent must not be
        # able to read the correct answer out of the tool response and exploit it
        # in subsequent steps (and the reward signal itself is the only learning
        # signal that should matter).
        feedback = (
            "Your answer has been submitted and is correct."
            if is_correct
            else "Your answer has been submitted but is incorrect."
        )

        extra_info = {
            "submitted_answer": answer,
            "ground_truth": ground_truth,
            "reward": float(reward),
            "is_correct": is_correct,
        }

        return ToolResponse(text=feedback), float(reward), extra_info
