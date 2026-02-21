#!/usr/bin/env python3
"""
数据集清理脚本：
- 每种类型只保留一套数据集
- 统一拆分为 train / test 两部分
- 删除多余文件
- 记录最终结构
"""

import json
import os
import random
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
random.seed(42)  # 可复现


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  ✅ 保存 {len(data)} 条 -> {path.name}  ({size_mb:.2f} MB)")


def delete_file(path: Path):
    if path.exists():
        path.unlink()
        print(f"  🗑️  删除 {path}")


def split_data(data, train_ratio=0.8):
    """随机拆分为 train / test"""
    shuffled = data.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


# ===========================================================
# 1. math — 保留 GSM8K，删除 SVAMP
# ===========================================================
def process_math():
    print("\n" + "=" * 60)
    print("📐 [1/6] math — 保留 GSM8K，删除 SVAMP")
    print("=" * 60)

    cat_dir = DATA_DIR / "math"

    # 删除 SVAMP 文件
    for f in cat_dir.glob("svamp*"):
        delete_file(f)

    # GSM8K 已有 train / test，直接保留
    train_data = load_json(cat_dir / "gsm8k_train.json")
    test_data = load_json(cat_dir / "gsm8k_test.json")

    # 更新 info
    info = {
        "name": "gsm8k",
        "category": "math",
        "status": "retained",
        "splits": {"train": len(train_data), "test": len(test_data)},
        "fields": ["id", "question", "answer"],
        "description": "Grade School Math 8K — 多步数学推理 (train/test 原始 split)",
    }
    save_json(info, cat_dir / "gsm8k_info.json")
    print(f"  train={len(train_data)}, test={len(test_data)}")
    return info


# ===========================================================
# 2. history — 保留 MMLU，删除 SocialIQA
# ===========================================================
def process_history():
    print("\n" + "=" * 60)
    print("📜 [2/6] history — 保留 MMLU，删除 SocialIQA")
    print("=" * 60)

    cat_dir = DATA_DIR / "history"

    # 删除 SocialIQA 文件
    for f in cat_dir.glob("socialiqa*"):
        delete_file(f)

    # MMLU 有 auxiliary_train / dev / validation / test
    # 保留: auxiliary_train → train, test → test
    # 删除: dev, validation

    # 重命名 auxiliary_train → train
    aux_path = cat_dir / "mmlu_auxiliary_train.json"
    train_path = cat_dir / "mmlu_train.json"
    if aux_path.exists():
        aux_data = load_json(aux_path)
        save_json(aux_data, train_path)
        delete_file(aux_path)
        train_count = len(aux_data)
    else:
        train_data = load_json(train_path)
        train_count = len(train_data)

    test_data = load_json(cat_dir / "mmlu_test.json")
    test_count = len(test_data)

    # 删除多余 split
    delete_file(cat_dir / "mmlu_dev.json")
    delete_file(cat_dir / "mmlu_validation.json")

    info = {
        "name": "mmlu",
        "category": "history",
        "status": "retained",
        "splits": {"train": train_count, "test": test_count},
        "fields": ["id", "question", "choices", "answer", "subject"],
        "description": "MMLU 全科 — 57 学科多选知识问答 (auxiliary_train → train, test 保留)",
    }
    save_json(info, cat_dir / "mmlu_info.json")
    print(f"  train={train_count}, test={test_count}")
    return info


# ===========================================================
# 3. commonsense — TruthfulQA (唯一)
#    只有 validation split，拆分为 train/test
#    保留 generation 格式，删除 mc 格式
# ===========================================================
def process_commonsense():
    print("\n" + "=" * 60)
    print("🧠 [3/6] commonsense — TruthfulQA (拆分 validation → train/test)")
    print("=" * 60)

    cat_dir = DATA_DIR / "commonsense"

    # 加载 generation 格式的 validation
    gen_path = cat_dir / "truthfulqa_generation_validation.json"
    data = load_json(gen_path)

    # 拆分
    train_data, test_data = split_data(data, train_ratio=0.8)

    # 重新编号 id
    for i, item in enumerate(train_data):
        item["id"] = f"truthfulqa_train_{i}"
    for i, item in enumerate(test_data):
        item["id"] = f"truthfulqa_test_{i}"

    save_json(train_data, cat_dir / "truthfulqa_train.json")
    save_json(test_data, cat_dir / "truthfulqa_test.json")

    # 删除旧文件
    delete_file(gen_path)
    delete_file(cat_dir / "truthfulqa_mc_validation.json")

    info = {
        "name": "truthfulqa",
        "category": "commonsense",
        "status": "retained",
        "splits": {"train": len(train_data), "test": len(test_data)},
        "fields": [
            "id",
            "question",
            "best_answer",
            "correct_answers",
            "incorrect_answers",
            "category",
            "source",
        ],
        "description": "TruthfulQA generation 格式 — 真实性问答 (validation 80/20 拆分)",
    }
    save_json(info, cat_dir / "truthfulqa_info.json")
    print(f"  train={len(train_data)}, test={len(test_data)}")
    return info


# ===========================================================
# 4. codegen — HumanEval (唯一)
#    只有 test split，拆分为 train/test
# ===========================================================
def process_codegen():
    print("\n" + "=" * 60)
    print("💻 [4/6] codegen — HumanEval (拆分 test → train/test)")
    print("=" * 60)

    cat_dir = DATA_DIR / "codegen"

    data = load_json(cat_dir / "humaneval_test.json")

    train_data, test_data = split_data(data, train_ratio=0.8)

    for i, item in enumerate(train_data):
        item["id"] = f"humaneval_train_{i}"
    for i, item in enumerate(test_data):
        item["id"] = f"humaneval_test_{i}"

    save_json(train_data, cat_dir / "humaneval_train.json")
    save_json(test_data, cat_dir / "humaneval_test.json")

    info = {
        "name": "humaneval",
        "category": "codegen",
        "status": "retained",
        "splits": {"train": len(train_data), "test": len(test_data)},
        "fields": ["id", "task_id", "prompt", "canonical_solution", "test", "entry_point"],
        "description": "HumanEval — 函数级 Python 代码生成 (原 test 80/20 拆分)",
    }
    save_json(info, cat_dir / "humaneval_info.json")
    print(f"  train={len(train_data)}, test={len(test_data)}")
    return info


# ===========================================================
# 5. long_context — LongBench v2 (唯一)
#    只有 train split，拆分为 train/test
# ===========================================================
def process_long_context():
    print("\n" + "=" * 60)
    print("📖 [5/6] long_context — LongBench v2 (拆分 train → train/test)")
    print("=" * 60)

    cat_dir = DATA_DIR / "long_context"

    data = load_json(cat_dir / "longbench_v2_train.json")

    train_data, test_data = split_data(data, train_ratio=0.8)

    for i, item in enumerate(train_data):
        item["id"] = f"longbench_v2_train_{i}"
    for i, item in enumerate(test_data):
        item["id"] = f"longbench_v2_test_{i}"

    save_json(train_data, cat_dir / "longbench_v2_train.json")
    save_json(test_data, cat_dir / "longbench_v2_test.json")

    info = {
        "name": "longbench_v2",
        "category": "long_context",
        "status": "retained",
        "splits": {"train": len(train_data), "test": len(test_data)},
        "fields": "保留原始字段 (context, question, choice, answer 等)",
        "description": "LongBench v2 — 6 类长文本多选 QA, 8K-2M 词 (原 train 80/20 拆分)",
    }
    save_json(info, cat_dir / "longbench_v2_info.json")
    print(f"  train={len(train_data)}, test={len(test_data)}")
    return info


# ===========================================================
# 6. long_document — Selective Context (唯一)
#    三个子集 (arxiv / bbc_news / sharegpt) 各只有 train
#    合并后统一拆分为 train/test
# ===========================================================
def process_long_document():
    print("\n" + "=" * 60)
    print("📄 [6/6] long_document — Selective Context (合并 3 子集 → train/test)")
    print("=" * 60)

    cat_dir = DATA_DIR / "long_document"

    merged = []
    for sub in ["arxiv", "bbc_news", "sharegpt"]:
        p = cat_dir / f"selective_context_{sub}_train.json"
        if p.exists():
            sub_data = load_json(p)
            # 给每条记录添加 sub_dataset 字段
            for item in sub_data:
                item["sub_dataset"] = sub
            merged.extend(sub_data)
            print(f"  加载 {sub}: {len(sub_data)} 条")

    print(f"  合并总数: {len(merged)} 条")

    train_data, test_data = split_data(merged, train_ratio=0.8)

    for i, item in enumerate(train_data):
        item["id"] = f"selective_context_train_{i}"
    for i, item in enumerate(test_data):
        item["id"] = f"selective_context_test_{i}"

    save_json(train_data, cat_dir / "selective_context_train.json")
    save_json(test_data, cat_dir / "selective_context_test.json")

    # 删除旧的子集文件
    for sub in ["arxiv", "bbc_news", "sharegpt"]:
        delete_file(cat_dir / f"selective_context_{sub}_train.json")

    # 统计子集分布
    train_dist = {}
    for item in train_data:
        s = item.get("sub_dataset", "unknown")
        train_dist[s] = train_dist.get(s, 0) + 1
    test_dist = {}
    for item in test_data:
        s = item.get("sub_dataset", "unknown")
        test_dist[s] = test_dist.get(s, 0) + 1

    info = {
        "name": "selective_context",
        "category": "long_document",
        "status": "retained",
        "splits": {"train": len(train_data), "test": len(test_data)},
        "sub_dataset_distribution": {"train": train_dist, "test": test_dist},
        "fields": "保留原始字段 + sub_dataset (arxiv/bbc_news/sharegpt)",
        "description": "Selective Context — 3 类长文档 (arxiv, bbc_news, sharegpt) 合并后 80/20 拆分",
    }
    save_json(info, cat_dir / "selective_context_info.json")
    print(f"  train={len(train_data)}, test={len(test_data)}")
    print(f"  train 分布: {train_dist}")
    print(f"  test  分布: {test_dist}")
    return info


# ===========================================================
# 主流程
# ===========================================================
def main():
    print("=" * 70)
    print("🧹 数据集清理：每类保留一套，统一 train/test 拆分")
    print("=" * 70)

    results = {}
    results["math"] = process_math()
    results["history"] = process_history()
    results["commonsense"] = process_commonsense()
    results["codegen"] = process_codegen()
    results["long_context"] = process_long_context()
    results["long_document"] = process_long_document()

    # 保存总结
    summary = {
        "description": "清理后的数据集结构 — 每类保留 1 套，统一 train/test",
        "total_categories": len(results),
        "datasets": results,
    }
    save_json(summary, DATA_DIR / "dataset_structure.json")

    # 打印最终结构
    print("\n")
    print("=" * 70)
    print("📋 最终数据集结构")
    print("=" * 70)
    print(f"{'类别':<16} {'数据集':<22} {'train':>8} {'test':>8} {'总计':>8}")
    print("-" * 70)
    total_train = 0
    total_test = 0
    for cat, info in results.items():
        tr = info["splits"]["train"]
        te = info["splits"]["test"]
        total_train += tr
        total_test += te
        print(f"{cat:<16} {info['name']:<22} {tr:>8,} {te:>8,} {tr+te:>8,}")
    print("-" * 70)
    print(f"{'合计':<16} {'':<22} {total_train:>8,} {total_test:>8,} {total_train+total_test:>8,}")
    print()

    # 列出目录下最终的文件
    print("📁 最终文件列表:")
    for cat_dir in sorted(DATA_DIR.iterdir()):
        if cat_dir.is_dir():
            print(f"\n  {cat_dir.name}/")
            for f in sorted(cat_dir.iterdir()):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    {f.name:<50} {size_mb:>8.2f} MB")

    print("\n✅ 清理完成！")


if __name__ == "__main__":
    main()
