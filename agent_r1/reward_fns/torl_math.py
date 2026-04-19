"""Custom reward function for ToRL-aligned math training.

Uses `\\boxed{}` extraction + string match (ToRL-style) instead of verl's
default math_dapo which uses "Answer: xxx" regex. Returns 0/1.

Usage (in training script):
    custom_reward_function.path=agent_r1/reward_fns/torl_math.py
    custom_reward_function.name=compute_score
"""

import os
from typing import Optional

# Debug: set REWARD_FN_DEBUG_N=5 in env to print first 5 reward computations
# Useful for inspecting model outputs during val_before_train.
_DEBUG_N = int(os.environ.get("REWARD_FN_DEBUG_N", "0"))
_debug_count = 0


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
    # Collapse whitespace
    s = " ".join(s.split())
    # Normalize common TeX variants
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    # Remove outer parens that wrap whole expression: "(x)" -> "x"  (crude)
    # Strip trailing dot/comma
    s = s.rstrip(".,")
    # Remove spaces again
    s = s.replace(" ", "")
    return s


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """0/1 reward: 1.0 if model's last \\boxed{...} matches ground_truth, else 0.0.

    Returns a dict with {"score": ..., "pred": ..., "acc": ...}
    so that verl's reward manager can log the extracted prediction.
    """
    # Extract the last boxed answer from the model's solution
    pred_raw = _last_boxed(solution_str)

    # Fallback: if model wrote "\boxed " (no brace) form, try simple split
    if pred_raw is None and "\\boxed " in solution_str:
        tail = solution_str.split("\\boxed ")[-1]
        pred_raw = tail.split("$")[0].strip()

    if pred_raw is None:
        return {"score": 0.0, "acc": 0.0, "pred": "[NO_BOXED]"}

    # Normalize and compare
    pred_norm = _normalize(pred_raw)
    gt_norm = _normalize(str(ground_truth))

    is_correct = (pred_norm == gt_norm)

    # Additional fuzzy matching: integer equivalence (e.g. "42" vs "042")
    if not is_correct:
        try:
            if int(pred_norm) == int(gt_norm):
                is_correct = True
        except (ValueError, TypeError):
            pass

    score = 1.0 if is_correct else 0.0

    # Optional debug dump for the first few reward computations
    global _debug_count
    if _debug_count < _DEBUG_N:
        _debug_count += 1
        print(f"\n===== [REWARD_FN DEBUG #{_debug_count}] =====")
        print(f"[data_source]: {data_source}")
        print(f"[ground_truth]: {ground_truth!r}")
        print(f"[pred_raw]: {pred_raw!r}")
        print(f"[pred_norm]: {pred_norm!r}  vs  gt_norm={gt_norm!r}")
        print(f"[score]: {score}")
        print(f"[solution_str tail 500]: ...{solution_str[-500:]!r}")
        print(f"===============================================\n", flush=True)

    return {"score": score, "acc": score, "pred": pred_raw}


# Quick self-test (run with: python agent_r1/reward_fns/torl_math.py)
if __name__ == "__main__":
    cases = [
        ("math_dapo", "The answer is $\\boxed{42}$.", "42", 1.0),
        ("aime24", "Final: \\boxed{204}", "204", 1.0),
        ("aime24", "Oops $\\boxed{205}$", "204", 0.0),
        ("math500", "So $\\boxed{\\frac{1}{2}}$", "\\frac{1}{2}", 1.0),
        ("math500", "Thus $\\boxed{-\\dfrac{2}{3}}$", "-\\frac{2}{3}", 1.0),
        ("math500", "No boxed answer here", "42", 0.0),
        ("aime24", "computed: \\boxed{042}", "42", 1.0),  # integer fuzzy
    ]
    for src, sol, gt, exp in cases:
        r = compute_score(src, sol, gt)
        ok = "OK " if r["score"] == exp else "FAIL"
        print(f"{ok}  src={src}  sol={sol[:40]!r}  gt={gt!r}  -> {r}  (expected {exp})")
