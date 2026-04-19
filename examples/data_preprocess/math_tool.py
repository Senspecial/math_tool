//# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Preprocess math reasoning datasets (MATH, AIME, Minerva, etc.) into the
parquet format consumed by Agent-R1's AgentEnvLoop + ToolEnv pipeline.

Supported datasets
------------------
* ``lighteval/MATH``            – Hendrycks MATH (algebra, geometry, …)
* ``DigitalLearningGmbH/MATH-lighteval``  – alternative MATH mirror
* ``HuggingFaceH4/MATH-500``   – 500-problem evaluation subset
* ``math_dapo``                 – DAPO math benchmark
* ``aime``                      – AIME competition problems (integer answers)
* ``openai/gsm8k``              – GSM8K (falls back to calc_gsm8k_reward)

Tool pipeline
-------------
The agent has access to two tools:
  1. ``python_math_executor``  – run Python / sympy for intermediate steps
  2. ``submit_math_answer``    – submit the final answer (carries reward)

The system prompt explicitly tells the model to:
  * think step-by-step inside <think>…</think> tags,
  * use ``python_math_executor`` freely for computation,
  * call ``submit_math_answer`` exactly once with its final answer.

Usage
-----
  python examples/data_preprocess/math_tool.py \
      --data_source lighteval/MATH \
      --local_save_dir ~/data/math_tool

  # Use a local dataset cache:
  python examples/data_preprocess/math_tool.py \
      --data_source lighteval/MATH \
      --local_dataset_path /path/to/local/math \
      --local_save_dir ~/data/math_tool

  # Upload to HDFS as well:
  python examples/data_preprocess/math_tool.py \
      --data_source lighteval/MATH \
      --local_save_dir ~/data/math_tool \
      --hdfs_dir hdfs://path/to/math_tool
"""

import argparse
import json
import os
import re

import datasets

# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def _extract_boxed_answer(solution_str: str) -> str | None:
    r"""Extract content from the last ``\boxed{...}`` in *solution_str*.

    Returns ``None`` if no boxed expression is found.
    """
    # Handle ``\boxed <space>`` form (rare but present in some datasets)
    if "\\boxed " in solution_str:
        candidate = solution_str.split("\\boxed ")[-1]
        # take up to the first $ or end-of-string
        return candidate.split("$")[0].strip()

    idx = solution_str.rfind("\\boxed")
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
    left = "\\boxed{"
    if raw.startswith(left) and raw.endswith("}"):
        return raw[len(left):-1].strip()
    return None


def _extract_gsm8k_answer(solution_str: str) -> str | None:
    """Extract the numeric answer after ``####`` (GSM8K format)."""
    match = re.search(r"####\s*(-?[\d,\.]+)", solution_str)
    if match:
        return match.group(1).replace(",", "")
    return None


def _extract_aime_answer(solution_str: str) -> str | None:
    """Extract integer answer from AIME solution (integer in [000, 999])."""
    # Try boxed first
    boxed = _extract_boxed_answer(solution_str)
    if boxed is not None:
        return boxed
    # Fall back to last standalone integer
    matches = re.findall(r"\b(\d{1,3})\b", solution_str)
    if matches:
        return matches[-1]
    return None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert mathematician. Solve the problem step by step.

You have access to the following tools:

1. `python_math_executor` – Run Python code (including sympy) for intermediate \
calculations, equation solving, or numerical verification. Use this as many \
times as needed.

2. `submit_math_answer` – Submit your **final** answer. Call this tool exactly \
**once** when you are confident. Accepted formats: plain numbers (42), \
fractions (1/2 or \\frac{1}{2}), expressions (2\\sqrt{3}), or boxed LaTeX \
(\\boxed{42}).

Workflow:
- Think step by step.
- Use `python_math_executor` freely to verify intermediate results.
- When you have the final answer, call `submit_math_answer` with it.
- Do NOT call `submit_math_answer` more than once.
"""


# ---------------------------------------------------------------------------
# Dataset-specific configuration
# ---------------------------------------------------------------------------

# Maps data_source -> (hf_name, split_field, solution_field, answer_extractor)
_DATASET_CONFIGS: dict[str, dict] = {
    "lighteval/MATH": {
        "hf_name": "lighteval/MATH",
        "hf_config": "all",
        "train_split": "train",
        "test_split": "test",
        "question_field": "problem",
        "solution_field": "solution",
        "answer_extractor": _extract_boxed_answer,
        "extra_fields": ["type", "level"],
    },
    "DigitalLearningGmbH/MATH-lighteval": {
        "hf_name": "DigitalLearningGmbH/MATH-lighteval",
        "hf_config": None,
        "train_split": "train",
        "test_split": "test",
        "question_field": "problem",
        "solution_field": "solution",
        "answer_extractor": _extract_boxed_answer,
        "extra_fields": ["type", "level"],
    },
    "HuggingFaceH4/MATH-500": {
        "hf_name": "HuggingFaceH4/MATH-500",
        "hf_config": None,
        "train_split": None,   # no train split; use test only
        "test_split": "test",
        "question_field": "problem",
        "solution_field": "solution",
        "answer_extractor": _extract_boxed_answer,
        "extra_fields": ["subject", "level"],
    },
    "openai/gsm8k": {
        "hf_name": "openai/gsm8k",
        "hf_config": "main",
        "train_split": "train",
        "test_split": "test",
        "question_field": "question",
        "solution_field": "answer",
        "answer_extractor": _extract_gsm8k_answer,
        "extra_fields": [],
    },
    # Large competition-math training set: AIME 1983-2024 (~1000 problems, integer answers 000-999)
    # Fields: Year, Type, Problem, Question, Solution
    "hxlhassam/AIME_Problem_Set_1983-2024": {
        "hf_name": "hxlhassam/AIME_Problem_Set_1983-2024",
        "hf_config": None,
        "train_split": "train",
        "test_split": None,
        "question_field": "Question",
        "solution_field": "Solution",
        "answer_extractor": _extract_aime_answer,
        "extra_fields": ["Year", "Type", "Problem"],
    },
    # DeepScaleR: ~40K mixed competition math (AIME 1984-2023 / AMC / Omni-MATH / Still)
    # — recommended for training; does NOT overlap with AIME 2024/2025
    "agentica-org/DeepScaleR-Preview-Dataset": {
        "hf_name": "agentica-org/DeepScaleR-Preview-Dataset",
        "hf_config": None,
        "train_split": "train",
        "test_split": None,
        "question_field": "problem",
        "solution_field": "answer",
        "answer_extractor": lambda s: s.strip() if s else None,
        "extra_fields": ["solution"],
    },
    # AIME 2024 (30 problems) — evaluation only
    "HuggingFaceH4/aime_2024": {
        "hf_name": "HuggingFaceH4/aime_2024",
        "hf_config": None,
        "train_split": None,
        "test_split": "train",  # only has one split named 'train'
        "question_field": "problem",
        "solution_field": "answer",
        "answer_extractor": lambda s: str(s).strip() if s is not None else None,
        "extra_fields": ["id"],
    },
    # AIME 2025 (30 problems) — evaluation only; definitely out-of-distribution for DeepScaleR
    "yentinglin/aime_2025": {
        "hf_name": "yentinglin/aime_2025",
        "hf_config": None,
        "train_split": None,
        "test_split": "train",
        "question_field": "problem",
        "solution_field": "answer",
        "answer_extractor": lambda s: str(s).strip() if s is not None else None,
        "extra_fields": ["id"],
    },
}

# Datasets where the test set has no labels (use val/test heuristically)
_AIME_LIKE_SOURCES = {"aime", "math_dapo", "math_dapo_reasoning"}


# ---------------------------------------------------------------------------
# Core preprocessing function
# ---------------------------------------------------------------------------

def build_agent_sample(
    example: dict,
    idx: int,
    split: str,
    data_source: str,
    question_field: str,
    solution_field: str,
    answer_extractor,
    extra_fields: list[str],
    tools: list[str] | None = None,
) -> dict | None:
    """Convert a single dataset example into an Agent-R1 training record.

    Returns ``None`` if the answer cannot be extracted (skipped silently).
    """
    if tools is None:
        tools = ["python_math_executor", "submit_math_answer"]

    question = example.get(question_field, "").strip()
    solution = example.get(solution_field, "").strip()

    if not question or not solution:
        return None

    ground_truth = answer_extractor(solution)
    if ground_truth is None:
        # Skip examples where we can't determine the ground truth
        return None

    extra_info: dict = {"split": split, "index": idx, "solution": solution}
    for field in extra_fields:
        if field in example:
            extra_info[field] = example[field]

    return {
        "data_source": data_source,
        "agent_name": "agent_env_loop",
        "prompt": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": extra_info,
        "env_kwargs": json.dumps(
            {
                "env_type": "tool",
                "tools": tools,
                "tool_format": "hermes",
                "tools_kwargs": {"ground_truth": ground_truth},
            }
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess math reasoning datasets for Agent-R1 tool-use RL training."
    )
    parser.add_argument(
        "--data_source",
        default="lighteval/MATH",
        choices=list(_DATASET_CONFIGS.keys()),
        help="HuggingFace dataset identifier (default: lighteval/MATH).",
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Path to a locally cached dataset (skips HuggingFace download).",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/math_tool",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS path to copy the parquet files to (optional).",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["python_math_executor", "submit_math_answer"],
        help=(
            "Tools to include in env_kwargs. "
            "Default: python_math_executor submit_math_answer"
        ),
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="Limit the number of training examples (for debugging).",
    )
    args = parser.parse_args()

    cfg = _DATASET_CONFIGS[args.data_source]

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset: {args.data_source}")
    if args.local_dataset_path is not None:
        raw = datasets.load_dataset(
            args.local_dataset_path,
            cfg.get("hf_config"),
            trust_remote_code=True,
        )
    else:
        load_kwargs: dict = {"trust_remote_code": True}
        if cfg.get("hf_config"):
            load_kwargs["name"] = cfg["hf_config"]
        raw = datasets.load_dataset(cfg["hf_name"], **load_kwargs)

    # ------------------------------------------------------------------
    # Build map function
    # ------------------------------------------------------------------
    def make_map_fn(split: str):
        def process_fn(example: dict, idx: int) -> dict:
            return build_agent_sample(
                example=example,
                idx=idx,
                split=split,
                data_source=args.data_source,
                question_field=cfg["question_field"],
                solution_field=cfg["solution_field"],
                answer_extractor=cfg["answer_extractor"],
                extra_fields=cfg.get("extra_fields", []),
                tools=args.tools,
            )
        return process_fn

    # ------------------------------------------------------------------
    # Process splits
    # ------------------------------------------------------------------
    train_dataset = None
    test_dataset = None

    train_split_name = cfg.get("train_split")
    test_split_name = cfg.get("test_split")

    if train_split_name and train_split_name in raw:
        train_dataset = raw[train_split_name]
        if args.train_size is not None:
            train_dataset = train_dataset.select(range(min(args.train_size, len(train_dataset))))
        train_dataset = train_dataset.map(
            function=make_map_fn("train"),
            with_indices=True,
            desc=f"Processing {train_split_name}",
        )
        # Drop examples where answer extraction failed (process_fn returns None)
        train_dataset = train_dataset.filter(lambda x: x is not None and x.get("data_source") is not None)

    if test_split_name and test_split_name in raw:
        test_dataset = raw[test_split_name]
        test_dataset = test_dataset.map(
            function=make_map_fn("test"),
            with_indices=True,
            desc=f"Processing {test_split_name}",
        )
        test_dataset = test_dataset.filter(lambda x: x is not None and x.get("data_source") is not None)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    if train_dataset is not None:
        train_path = os.path.join(local_save_dir, "train.parquet")
        train_dataset.to_parquet(train_path)
        print(f"Saved {len(train_dataset)} train examples to {train_path}")

    if test_dataset is not None:
        test_path = os.path.join(local_save_dir, "test.parquet")
        test_dataset.to_parquet(test_path)
        print(f"Saved {len(test_dataset)} test examples to {test_path}")

    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs  # noqa: PLC0415
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            print(f"Copied data to HDFS: {args.hdfs_dir}")
        except ImportError:
            print("Warning: verl.utils.hdfs_io not available; skipping HDFS copy.")


if __name__ == "__main__":
    main()
