"""Convert ToRL's data format to Agent-R1's format (ToRL-aligned: no submit tool).

ToRL format columns: [data_source, prompt, ability, reward_model, extra_info]
Agent-R1 needs additionally: [agent_name, env_kwargs] and a system prompt.

Design choice (ToRL-aligned):
- Only include `python_math_executor` as a tool (no submit_math_answer).
- Ask the model to put the final answer in \\boxed{}.
- Rely on verl's default_compute_score fallback to extract \\boxed{} and grade.
- Map ToRL's train data_source ("ToRL") to "math_dapo" so fallback works.

Usage:
    python examples/data_preprocess/convert_torl_data.py \
        --input /path/to/torl/train.parquet \
        --output /path/to/out/train.parquet
"""

import argparse
import json
import os

import pandas as pd


SYSTEM_PROMPT = """\
You are an expert mathematician. Solve the problem step by step.

You have access to one tool:

`python_math_executor` – Run Python code (including sympy) for intermediate \
calculations, equation solving, or numerical verification. Use this freely \
whenever computation helps.

When you have the final answer, put it within \\boxed{} in your response. \
For example: "The answer is \\boxed{42}".
"""


def convert_row(row: dict) -> dict:
    """Convert one ToRL row to Agent-R1 format (no submit tool)."""
    rm = dict(row["reward_model"])
    gt = rm.get("ground_truth")
    # Fix ToRL's typo: 'stype' -> 'style'
    if "stype" in rm and "style" not in rm:
        rm["style"] = rm.pop("stype")

    original_prompt = list(row["prompt"])
    user_msg = None
    for m in original_prompt:
        if m["role"] == "user":
            user_msg = m["content"]
            break
    if user_msg is None:
        user_msg = original_prompt[-1]["content"]

    new_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # Keep original data_source for eval-time per-subset analysis (aime24, aime25, math500, ToRL).
    # The reward function mapping is handled by a custom reward_fn loaded at training time,
    # which routes all these sources to math_dapo.compute_score (boxed extraction).
    mapped_src = row.get("data_source", "ToRL")

    return {
        "data_source": mapped_src,
        "agent_name": "agent_env_loop",
        "prompt": new_prompt,
        "ability": "math",
        "reward_model": rm,
        "extra_info": dict(row.get("extra_info", {}) or {}),
        "env_kwargs": json.dumps(
            {
                "env_type": "tool",
                "tools": ["python_math_executor"],
                "tool_format": "hermes",
                "tools_kwargs": {"ground_truth": gt},
            }
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet path (ToRL format)")
    parser.add_argument("--output", required=True, help="Output parquet path (Agent-R1 format)")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Columns: {df.columns.tolist()}")

    converted = df.apply(lambda row: convert_row(row.to_dict()), axis=1, result_type="expand")
    print(f"Converted {len(converted)} rows")
    print(f"New columns: {converted.columns.tolist()}")
    print(f"data_source distribution: {converted['data_source'].value_counts().to_dict()}")

    row0 = converted.iloc[0]
    print("--- Sample converted row ---")
    for c in converted.columns:
        v = str(row0[c])
        print(f"[{c}]: {v[:200]}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    converted.to_parquet(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
