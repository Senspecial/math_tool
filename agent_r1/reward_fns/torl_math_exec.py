"""ToRL-aligned math reward WITH execution-failure penalty.

Differences vs ``torl_math.py``:
    - Correct answer:    +1.0   (unchanged)
    - Wrong answer:      +0.0   (unchanged; equivalent to -1 under GRPO due to group normalization)
    - Code execution failure in this trajectory: additional **-0.5** penalty
      (stacked with answer score), matching ToRL paper section 3.2.

    > "We implemented a rule-based reward function where correct answers receive
    > a reward of 1, incorrect answers receive -1. In addition, the code
    > interpreter naturally provides feedback on code executability. Based on
    > the correlation between successful code execution and problem-solving
    > accuracy, we introduced an execution-based penalty: responses containing
    > non-executable code incur a -0.5 reward reduction."

The ``any_code_failed`` flag is supplied via ``extra_info`` by the agent flow
(see ``agent_r1/agent_flow/agent_env_loop.py`` and
``agent_r1/env/envs/tool.py``). It is True iff **any** python_interpreter call
in this trajectory exited with non-zero status / timed out / raised.

Usage (in training script):
    custom_reward_function.path=agent_r1/reward_fns/torl_math_exec.py
    custom_reward_function.name=compute_score
"""

import os
from typing import Optional

# Debug: set REWARD_FN_DEBUG_N=5 in env to print first 5 reward computations
_DEBUG_N = int(os.environ.get("REWARD_FN_DEBUG_N", "0"))
_debug_count = 0

# Penalty magnitude (ToRL paper: 0.5)
EXEC_FAIL_PENALTY = float(os.environ.get("EXEC_FAIL_PENALTY", "0.5"))


def _last_boxed(s: str) -> Optional[str]:
    """Extract content inside the last \\boxed{...} expression (brace-balanced)."""
    idx = s.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    start = i
    while i < len(s):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i]
        i += 1
    return None


def _normalize(s: str) -> str:
    """Light normalization for math answer comparison."""
    if s is None:
        return ""
    s = s.strip()
    s = " ".join(s.split())
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.rstrip(".,")
    s = s.replace(" ", "")
    return s


def _read_flag(extra_info: Optional[dict], key: str) -> bool:
    """Robustly read a boolean flag from extra_info.

    ``extra_info`` may come from verl with values wrapped as numpy scalars
    (e.g. 0-d object arrays). Normalize to a plain Python bool.
    """
    if not extra_info:
        return False
    val = extra_info.get(key)
    if val is None:
        return False
    # Unwrap numpy 0-d arrays if needed.
    try:
        item = val.item() if hasattr(val, "item") else val
    except Exception:
        item = val
    return bool(item)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Reward = (1.0 if answer correct else 0.0) - 0.5 * exec_failed.

    Returns a dict with {"score": ..., "acc": ..., "pred": ..., "exec_failed": ...}
    so that verl's reward manager can log the breakdown.
    """
    # --- Answer correctness (same as torl_math.py) ---
    pred_raw = _last_boxed(solution_str)
    if pred_raw is None and "\\boxed " in solution_str:
        tail = solution_str.split("\\boxed ")[-1]
        pred_raw = tail.split("$")[0].strip()

    if pred_raw is None:
        is_correct = False
        pred_norm = "[NO_BOXED]"
        gt_norm = _normalize(str(ground_truth))
    else:
        pred_norm = _normalize(pred_raw)
        gt_norm = _normalize(str(ground_truth))
        is_correct = pred_norm == gt_norm
        if not is_correct:
            try:
                if int(pred_norm) == int(gt_norm):
                    is_correct = True
            except (ValueError, TypeError):
                pass

    acc = 1.0 if is_correct else 0.0

    # --- Execution-based penalty (ToRL paper) ---
    # any_code_failed is populated by agent_env_loop; True iff any python_interpreter
    # call in this trajectory failed (non-zero exit / timeout / exception).
    exec_failed = _read_flag(extra_info, "any_code_failed")
    penalty = EXEC_FAIL_PENALTY if exec_failed else 0.0

    score = acc - penalty

    # Optional debug dump
    global _debug_count
    if _debug_count < _DEBUG_N:
        _debug_count += 1
        print(f"\n===== [REWARD_FN_EXEC DEBUG #{_debug_count}] =====")
        print(f"[data_source]: {data_source}")
        print(f"[ground_truth]: {ground_truth!r}")
        print(f"[pred_raw]: {pred_raw!r}")
        print(f"[pred_norm]: {pred_norm!r}  vs  gt_norm={gt_norm!r}")
        print(f"[acc]: {acc}   [exec_failed]: {exec_failed}   [penalty]: {penalty}")
        print(f"[score]: {score}")
        print(f"[extra_info keys]: {list(extra_info.keys()) if extra_info else None}")
        print(f"[solution_str tail 500]: ...{solution_str[-500:]!r}")
        print(f"=====================================================\n", flush=True)

    return {
        "score": score,
        "acc": acc,
        "pred": pred_raw if pred_raw is not None else "[NO_BOXED]",
        "exec_failed": 1.0 if exec_failed else 0.0,
    }


# Quick self-test: python agent_r1/reward_fns/torl_math_exec.py
if __name__ == "__main__":
    cases = [
        # (sol, gt, extra_info, expected_score)
        ("The answer is $\\boxed{42}$.", "42", {}, 1.0),
        ("The answer is $\\boxed{42}$.", "42", {"any_code_failed": True}, 0.5),
        ("Wrong $\\boxed{99}$", "42", {}, 0.0),
        ("Wrong $\\boxed{99}$", "42", {"any_code_failed": True}, -0.5),
        ("No boxed here", "42", {"any_code_failed": True}, -0.5),
        ("computed: \\boxed{042}", "42", {}, 1.0),  # integer fuzzy
        ("Right $\\boxed{42}$", "42", None, 1.0),  # None extra_info
    ]
    for sol, gt, ei, exp in cases:
        r = compute_score("math_dapo", sol, gt, extra_info=ei)
        ok = "OK " if abs(r["score"] - exp) < 1e-9 else "FAIL"
        print(f"{ok}  sol={sol[:40]!r:45s}  gt={gt!r}  ei={ei}  -> {r}  (expected {exp})")
