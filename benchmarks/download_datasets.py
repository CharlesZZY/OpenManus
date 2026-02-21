#!/usr/bin/env python3
"""
下载所有 benchmark 数据集

数据集分类：
1. 数学：GSM8K, SVAMP
2. 历史：MMLU (全科), SocialIQA
3. 常识：TruthfulQA
4. 代码生成：HumanEval
5. 现实长上下文多任务：LongBench v2
6. 长文档数据处理任务：Selective_Context (arxiv, bbc_news, sharegpt)

所有数据集均从 HuggingFace Hub 下载完整版本。
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# 尝试导入 datasets 库
try:
    from datasets import Dataset, DatasetDict, load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

DATA_DIR = Path(__file__).parent / "data"


# ============================================================
# 工具函数
# ============================================================


def ensure_dir(path: Path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)


def save_to_json(data: List[Dict[str, Any]], path: Path):
    """将数据保存为 JSON 文件（完整，不截断）"""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  ✅ 保存 {len(data)} 条样本 -> {path}  ({size_mb:.2f} MB)")


def save_dataset_info(name: str, info: Dict[str, Any], category_dir: Path):
    """保存数据集元信息"""
    ensure_dir(category_dir)
    info_path = category_dir / f"{name}_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def timer(func):
    """计时装饰器"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  ⏱️  耗时: {elapsed:.1f}s")
        return result

    return wrapper


# ============================================================
# 1. 数学数据集
# ============================================================


@timer
def download_gsm8k() -> Dict[str, Any]:
    """
    下载 GSM8K 数学推理数据集（完整版）

    来源: openai/gsm8k
    规模: train ~7.5K, test ~1.3K
    字段: question, answer
    """
    print("\n📥 [1/8] 下载 GSM8K (数学推理)...")
    dataset = load_dataset("gsm8k", "main")

    result = {"name": "gsm8k", "category": "math", "splits": {}}
    category_dir = DATA_DIR / "math"

    for split_name in dataset:
        split_data = dataset[split_name]
        samples = []
        for i, item in enumerate(split_data):
            samples.append(
                {
                    "id": f"gsm8k_{split_name}_{i}",
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )
        save_to_json(samples, category_dir / f"gsm8k_{split_name}.json")
        result["splits"][split_name] = len(samples)

    save_dataset_info("gsm8k", result, category_dir)
    return result


@timer
def download_svamp() -> Dict[str, Any]:
    """
    下载 SVAMP 数学推理数据集（完整版）

    来源: ChilleD/SVAMP
    规模: train 700, test 300
    字段: Body, Question, Equation, Answer, Type
    """
    print("\n📥 [2/8] 下载 SVAMP (数学推理)...")
    dataset = load_dataset("ChilleD/SVAMP")

    result = {"name": "svamp", "category": "math", "splits": {}}
    category_dir = DATA_DIR / "math"

    for split_name in dataset:
        split_data = dataset[split_name]
        samples = []
        for i, item in enumerate(split_data):
            samples.append(
                {
                    "id": f"svamp_{split_name}_{i}",
                    "body": item["Body"],
                    "question": item["Question"],
                    "question_concat": item.get(
                        "question_concat", f"{item['Body']} {item['Question']}"
                    ),
                    "equation": item.get("Equation", ""),
                    "answer": item["Answer"],
                    "type": item.get("Type", ""),
                }
            )
        save_to_json(samples, category_dir / f"svamp_{split_name}.json")
        result["splits"][split_name] = len(samples)

    save_dataset_info("svamp", result, category_dir)
    return result


# ============================================================
# 2. 历史数据集
# ============================================================


@timer
def download_mmlu() -> Dict[str, Any]:
    """
    下载 MMLU 数据集（全部科目，完整版）

    来源: cais/mmlu, config="all"
    规模: test ~14K, validation ~1.5K, dev ~285 (所有科目合计)
    字段: question, choices, answer (0-3 对应 A-D)
    """
    print("\n📥 [3/8] 下载 MMLU (全科多任务语言理解)...")
    dataset = load_dataset("cais/mmlu", "all")

    result = {"name": "mmlu", "category": "history", "splits": {}}
    category_dir = DATA_DIR / "history"

    for split_name in dataset:
        split_data = dataset[split_name]
        samples = []
        for i, item in enumerate(split_data):
            samples.append(
                {
                    "id": f"mmlu_{split_name}_{i}",
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],  # 0-3 对应 A-D
                    "subject": item.get("subject", "unknown"),
                }
            )
        save_to_json(samples, category_dir / f"mmlu_{split_name}.json")
        result["splits"][split_name] = len(samples)

    # 统计各科目分布
    if "test" in dataset:
        subject_counts = {}
        for item in dataset["test"]:
            subj = item.get("subject", "unknown")
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
        result["subject_distribution"] = dict(sorted(subject_counts.items()))

    save_dataset_info("mmlu", result, category_dir)
    return result


@timer
def download_socialiqa() -> Dict[str, Any]:
    """
    下载 SocialIQA 社会常识推理数据集（完整版）

    来源: allenai/social_i_qa
    规模: train ~33K, validation ~1.9K (test 标签不公开)
    字段: context, question, answerA, answerB, answerC, label
    """
    print("\n📥 [4/8] 下载 SocialIQA (社会常识推理)...")
    dataset = load_dataset("allenai/social_i_qa")

    result = {"name": "socialiqa", "category": "history", "splits": {}}
    category_dir = DATA_DIR / "history"

    for split_name in dataset:
        split_data = dataset[split_name]
        samples = []
        for i, item in enumerate(split_data):
            samples.append(
                {
                    "id": f"socialiqa_{split_name}_{i}",
                    "context": item["context"],
                    "question": item["question"],
                    "answerA": item["answerA"],
                    "answerB": item["answerB"],
                    "answerC": item["answerC"],
                    "label": item["label"],
                }
            )
        save_to_json(samples, category_dir / f"socialiqa_{split_name}.json")
        result["splits"][split_name] = len(samples)

    save_dataset_info("socialiqa", result, category_dir)
    return result


# ============================================================
# 3. 常识数据集
# ============================================================


@timer
def download_truthfulqa() -> Dict[str, Any]:
    """
    下载 TruthfulQA 数据集（完整版，含 generation 和 multiple_choice 两种格式）

    来源: truthful_qa
    规模: validation ~817 (该数据集只有 validation split)
    格式: generation — question, best_answer, correct_answers, incorrect_answers
          multiple_choice — question, mc1_targets, mc2_targets
    """
    print("\n📥 [5/8] 下载 TruthfulQA (真实性问答)...")
    category_dir = DATA_DIR / "commonsense"
    result = {"name": "truthfulqa", "category": "commonsense", "configs": {}}

    # --- generation 格式 ---
    print("    下载 generation 格式...")
    ds_gen = load_dataset("truthful_qa", "generation")
    for split_name in ds_gen:
        split_data = ds_gen[split_name]
        samples = []
        for i, item in enumerate(split_data):
            samples.append(
                {
                    "id": f"truthfulqa_gen_{split_name}_{i}",
                    "question": item["question"],
                    "best_answer": item["best_answer"],
                    "correct_answers": item["correct_answers"],
                    "incorrect_answers": item["incorrect_answers"],
                    "category": item.get("category", ""),
                    "source": item.get("source", ""),
                }
            )
        save_to_json(samples, category_dir / f"truthfulqa_generation_{split_name}.json")
        result["configs"].setdefault("generation", {})[split_name] = len(samples)

    # --- multiple_choice 格式 ---
    print("    下载 multiple_choice 格式...")
    ds_mc = load_dataset("truthful_qa", "multiple_choice")
    for split_name in ds_mc:
        split_data = ds_mc[split_name]
        samples = []
        for i, item in enumerate(split_data):
            samples.append(
                {
                    "id": f"truthfulqa_mc_{split_name}_{i}",
                    "question": item["question"],
                    "mc1_targets": item["mc1_targets"],
                    "mc2_targets": item["mc2_targets"],
                }
            )
        save_to_json(samples, category_dir / f"truthfulqa_mc_{split_name}.json")
        result["configs"].setdefault("multiple_choice", {})[split_name] = len(samples)

    save_dataset_info("truthfulqa", result, category_dir)
    return result


# ============================================================
# 4. 代码生成数据集
# ============================================================


@timer
def download_humaneval() -> Dict[str, Any]:
    """
    下载 HumanEval 代码生成数据集（完整版）

    来源: openai_humaneval
    规模: test 164
    字段: task_id, prompt, canonical_solution, test, entry_point
    """
    print("\n📥 [6/8] 下载 HumanEval (代码生成)...")
    dataset = load_dataset("openai_humaneval")

    result = {"name": "humaneval", "category": "codegen", "splits": {}}
    category_dir = DATA_DIR / "codegen"

    for split_name in dataset:
        split_data = dataset[split_name]
        samples = []
        for i, item in enumerate(split_data):
            samples.append(
                {
                    "id": f"humaneval_{split_name}_{i}",
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                }
            )
        save_to_json(samples, category_dir / f"humaneval_{split_name}.json")
        result["splits"][split_name] = len(samples)

    save_dataset_info("humaneval", result, category_dir)
    return result


# ============================================================
# 5. 长上下文多任务数据集
# ============================================================


@timer
def download_longbench_v2() -> Dict[str, Any]:
    """
    下载 LongBench v2 数据集（完整版）

    来源: THUDM/LongBench-v2
    规模: 503 道多选题
    字段: 涵盖 6 大类长文本任务（单文档QA、多文档QA、长ICL、对话历史、代码仓库理解、结构化数据理解）
    上下文长度: 8K ~ 2M 词
    """
    print("\n📥 [7/8] 下载 LongBench v2 (长上下文多任务)...")
    dataset = load_dataset("THUDM/LongBench-v2")

    result = {"name": "longbench_v2", "category": "long_context", "splits": {}}
    category_dir = DATA_DIR / "long_context"

    for split_name in dataset:
        split_data = dataset[split_name]
        samples = []
        columns = split_data.column_names
        for i, item in enumerate(split_data):
            sample = {"id": f"longbench_v2_{split_name}_{i}"}
            for col in columns:
                sample[col] = item[col]
            samples.append(sample)
        save_to_json(samples, category_dir / f"longbench_v2_{split_name}.json")
        result["splits"][split_name] = len(samples)

    save_dataset_info("longbench_v2", result, category_dir)
    return result


# ============================================================
# 6. 长文档数据处理任务 (Selective_Context)
# ============================================================


@timer
def download_selective_context() -> Dict[str, Any]:
    """
    下载 Selective_Context 论文配套的三个评测数据集（完整版）

    论文: "Compressing Context to Enhance Inference Efficiency of Large Language Models" (EMNLP 2023)
    包含 3 个子数据集：
      - liyucheng/arxiv-march-2023   : arXiv 论文
      - liyucheng/bbc_new_2303       : BBC 新闻
      - liyucheng/sharegpt-500       : ShareGPT 对话
    任务: 摘要、问答、上下文重构、对话
    """
    print("\n📥 [8/8] 下载 Selective_Context (长文档数据处理)...")
    category_dir = DATA_DIR / "long_document"
    result = {
        "name": "selective_context",
        "category": "long_document",
        "sub_datasets": {},
    }

    sub_datasets = {
        "arxiv": "liyucheng/arxiv-march-2023",
        "bbc_news": "liyucheng/bbc_new_2303",
        "sharegpt": "liyucheng/sharegpt-500",
    }

    for sub_name, hf_id in sub_datasets.items():
        print(f"    下载子数据集: {hf_id} ...")
        try:
            ds = load_dataset(hf_id)
            sub_result = {}
            for split_name in ds:
                split_data = ds[split_name]
                samples = []
                columns = split_data.column_names
                for i, item in enumerate(split_data):
                    sample = {"id": f"sc_{sub_name}_{split_name}_{i}"}
                    for col in columns:
                        val = item[col]
                        # 处理不可 JSON 序列化的类型
                        if isinstance(
                            val, (list, dict, str, int, float, bool, type(None))
                        ):
                            sample[col] = val
                        else:
                            sample[col] = str(val)
                    samples.append(sample)
                save_to_json(
                    samples,
                    category_dir / f"selective_context_{sub_name}_{split_name}.json",
                )
                sub_result[split_name] = len(samples)
            result["sub_datasets"][sub_name] = sub_result
        except Exception as e:
            print(f"    ⚠️  子数据集 {hf_id} 下载失败: {e}")
            result["sub_datasets"][sub_name] = {"error": str(e)}

    save_dataset_info("selective_context", result, category_dir)
    return result


# ============================================================
# 主流程
# ============================================================


def main():
    """主函数：按类别依次下载所有数据集"""
    print("=" * 70)
    print("📊 Benchmark 数据集下载工具（完整版）")
    print("=" * 70)
    print(f"数据保存目录: {DATA_DIR.resolve()}")
    print()

    if not HAS_DATASETS:
        print("❌ 未检测到 datasets 库。请先运行:")
        print("   pip install datasets")
        sys.exit(1)

    all_results = {}
    errors = {}

    # 定义下载任务
    tasks = [
        # (分类名, 函数, 关键字)
        ("1. 数学 - GSM8K", download_gsm8k, "gsm8k"),
        ("2. 数学 - SVAMP", download_svamp, "svamp"),
        ("3. 历史 - MMLU (全科)", download_mmlu, "mmlu"),
        ("4. 历史 - SocialIQA", download_socialiqa, "socialiqa"),
        ("5. 常识 - TruthfulQA", download_truthfulqa, "truthfulqa"),
        ("6. 代码 - HumanEval", download_humaneval, "humaneval"),
        ("7. 长上下文 - LongBench v2", download_longbench_v2, "longbench_v2"),
        (
            "8. 长文档 - Selective_Context",
            download_selective_context,
            "selective_context",
        ),
    ]

    total_start = time.time()

    for desc, func, key in tasks:
        print(f"\n{'─' * 50}")
        print(f"  {desc}")
        print(f"{'─' * 50}")
        try:
            result = func()
            all_results[key] = result
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            traceback.print_exc()
            errors[key] = str(e)

    total_elapsed = time.time() - total_start

    # 打印汇总
    print("\n")
    print("=" * 70)
    print("📋 下载结果汇总")
    print("=" * 70)

    for desc, _, key in tasks:
        if key in all_results:
            r = all_results[key]
            if "splits" in r:
                total_samples = sum(r["splits"].values())
                splits_str = ", ".join(f"{k}={v}" for k, v in r["splits"].items())
                print(f"  ✅ {desc:40s}  共 {total_samples:>6} 条  ({splits_str})")
            elif "configs" in r:
                for cfg, splits in r["configs"].items():
                    total_samples = sum(splits.values())
                    splits_str = ", ".join(f"{k}={v}" for k, v in splits.items())
                    print(
                        f"  ✅ {desc:40s}  [{cfg}] 共 {total_samples:>6} 条  ({splits_str})"
                    )
            elif "sub_datasets" in r:
                for sub, splits in r["sub_datasets"].items():
                    if isinstance(splits, dict) and "error" not in splits:
                        total_samples = sum(splits.values())
                        print(f"  ✅ {desc:40s}  [{sub}] 共 {total_samples:>6} 条")
                    else:
                        print(f"  ⚠️  {desc:40s}  [{sub}] 下载失败")
        else:
            print(f"  ❌ {desc:40s}  下载失败: {errors.get(key, '未知错误')}")

    success_count = len(all_results)
    total_count = len(tasks)
    print(f"\n总计: {success_count}/{total_count} 个数据集下载成功")
    print(f"总耗时: {total_elapsed:.1f}s")
    print(f"数据目录: {DATA_DIR.resolve()}")

    # 保存总结到文件
    summary = {
        "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "success_count": success_count,
        "total_count": total_count,
        "datasets": all_results,
        "errors": errors,
    }
    summary_path = DATA_DIR / "download_summary.json"
    ensure_dir(DATA_DIR)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"下载摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
