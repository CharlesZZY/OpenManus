"""Unified dataset loader.

Reads JSON files produced by ``benchmarks/download_datasets.py`` and returns
a list of :class:`BenchmarkSample` instances with the original prompt intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.serving_benchmark.core.schema import (
    SUITE_DATASETS,
    BenchmarkSample,
    Suite,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Maps dataset name → (category sub-dir, file stem pattern, prompt extractor)
# The prompt extractor receives a raw JSON item and returns the prompt string.


def _prompt_gsm8k(item: Dict[str, Any]) -> str:
    return item["question"]


def _ref_gsm8k(item: Dict[str, Any]) -> str:
    return item["answer"]


def _prompt_humaneval(item: Dict[str, Any]) -> str:
    return item["prompt"]


def _ref_humaneval(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "canonical_solution": item["canonical_solution"],
        "test": item["test"],
        "entry_point": item["entry_point"],
        "task_id": item.get("task_id", ""),
    }


def _prompt_mmlu(item: Dict[str, Any]) -> str:
    choices = item.get("choices", [])
    options = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return f"{item['question']}\n{options}"


def _ref_mmlu(item: Dict[str, Any]) -> int:
    return item["answer"]  # 0-3


def _prompt_truthfulqa(item: Dict[str, Any]) -> str:
    return item["question"]


def _ref_truthfulqa(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mc1_targets": item.get("mc1_targets"),
        "mc2_targets": item.get("mc2_targets"),
    }


def _prompt_longbench(item: Dict[str, Any]) -> str:
    context = item.get("context", "")
    question = item.get("question", item.get("input", ""))
    if context:
        return f"{context}\n\n{question}"
    return question


def _ref_longbench(item: Dict[str, Any]) -> str:
    return item.get("answer", item.get("answers", ""))


def _prompt_selective_context(item: Dict[str, Any]) -> str:
    return item.get("text", item.get("content", item.get("document", "")))


def _ref_selective_context(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "summary": item.get("summary", ""),
        "questions": item.get("questions", []),
        "answers": item.get("answers", []),
    }


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "category": "math",
        "file_pattern": "gsm8k_{split}.json",
        "prompt_fn": _prompt_gsm8k,
        "ref_fn": _ref_gsm8k,
        "suite": Suite.R,
    },
    "humaneval": {
        "category": "codegen",
        "file_pattern": "humaneval_{split}.json",
        "prompt_fn": _prompt_humaneval,
        "ref_fn": _ref_humaneval,
        "suite": Suite.R,
    },
    "mmlu": {
        "category": "history",
        "file_pattern": "mmlu_{split}.json",
        "prompt_fn": _prompt_mmlu,
        "ref_fn": _ref_mmlu,
        "suite": Suite.S,
    },
    "truthfulqa": {
        "category": "commonsense",
        "file_pattern": "truthfulqa_mc_{split}.json",
        "prompt_fn": _prompt_truthfulqa,
        "ref_fn": _ref_truthfulqa,
        "suite": Suite.S,
    },
    "longbench_v2": {
        "category": "long_context",
        "file_pattern": "longbench_v2_{split}.json",
        "prompt_fn": _prompt_longbench,
        "ref_fn": _ref_longbench,
        "suite": Suite.L,
    },
    "selective_context": {
        "category": "long_document",
        "file_pattern": "selective_context_arxiv_{split}.json",
        "prompt_fn": _prompt_selective_context,
        "ref_fn": _ref_selective_context,
        "suite": Suite.L,
    },
}


def load_dataset(
    name: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> List[BenchmarkSample]:
    """Load a dataset and return :class:`BenchmarkSample` instances.

    Parameters
    ----------
    name : dataset key (gsm8k, humaneval, mmlu, truthfulqa, longbench_v2, selective_context)
    split : dataset split (test, validation, train)
    max_samples : cap the number of samples (None = all)
    data_dir : override default data directory
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY)}"
        )

    info = DATASET_REGISTRY[name]
    root = data_dir or DATA_DIR
    file_path = root / info["category"] / info["file_pattern"].format(split=split)

    # Try common split aliases
    if not file_path.exists():
        for alt in ("validation", "test", "train"):
            alt_path = root / info["category"] / info["file_pattern"].format(split=alt)
            if alt_path.exists():
                file_path = alt_path
                break

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}. "
            f"Run 'python benchmarks/download_datasets.py' first."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    if max_samples is not None:
        raw_items = raw_items[:max_samples]

    samples: List[BenchmarkSample] = []
    for item in raw_items:
        samples.append(
            BenchmarkSample(
                id=item.get("id", ""),
                dataset=name,
                suite=info["suite"].value,
                raw_prompt=info["prompt_fn"](item),
                reference=info["ref_fn"](item),
                metadata=item,
            )
        )
    return samples


def load_suite(
    suite: Suite,
    split: str = "test",
    max_samples_per_dataset: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> List[BenchmarkSample]:
    """Load all datasets belonging to a suite."""
    datasets = SUITE_DATASETS[suite]
    all_samples: List[BenchmarkSample] = []
    for ds_name in datasets:
        try:
            all_samples.extend(
                load_dataset(ds_name, split, max_samples_per_dataset, data_dir)
            )
        except FileNotFoundError:
            pass  # skip unavailable datasets
    return all_samples
