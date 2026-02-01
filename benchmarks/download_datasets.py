#!/usr/bin/env python3
"""
下载所有benchmark数据集

数据集分类：
1. 数学推理: GSM8K, MathQA, SVAMP
2. 历史/社会: MMLU (world_history), SocialIQA
3. 真实性/常识: TruthfulQA, Natural Questions
4. 代码生成: HumanEval, MBPP
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

# 尝试导入 datasets 库
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("请先安装 datasets 库: pip install datasets")


DATA_DIR = Path(__file__).parent / "data"


def ensure_dir(path: Path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)


def save_samples(data: List[Dict[str, Any]], path: Path, limit: int = 100):
    """保存样本到JSON文件"""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data[:limit], f, ensure_ascii=False, indent=2)
    print(f"  保存 {min(len(data), limit)} 条样本到 {path}")


def download_gsm8k(limit: int = 100):
    """下载 GSM8K 数学推理数据集"""
    print("\n📥 下载 GSM8K...")
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        samples = [
            {
                "id": f"gsm8k_{i}",
                "question": item["question"],
                "answer": item["answer"],
                "category": "math"
            }
            for i, item in enumerate(dataset)
        ]
        save_samples(samples, DATA_DIR / "math" / "gsm8k.json", limit)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def download_mathqa(limit: int = 100):
    """下载 MathQA 数据集"""
    print("\n📥 下载 MathQA...")
    try:
        dataset = load_dataset("math_qa", split="test")
        samples = [
            {
                "id": f"mathqa_{i}",
                "question": item["Problem"],
                "options": item["options"],
                "answer": item["correct"],
                "category": "math"
            }
            for i, item in enumerate(dataset)
        ]
        save_samples(samples, DATA_DIR / "math" / "mathqa.json", limit)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def download_svamp(limit: int = 100):
    """下载 SVAMP 数学推理数据集"""
    print("\n📥 下载 SVAMP...")
    try:
        dataset = load_dataset("ChilleD/SVAMP", split="test")
        samples = [
            {
                "id": f"svamp_{i}",
                "question": item["Body"] + " " + item["Question"],
                "answer": str(item["Answer"]),
                "equation": item.get("Equation", ""),
                "category": "math"
            }
            for i, item in enumerate(dataset)
        ]
        save_samples(samples, DATA_DIR / "math" / "svamp.json", limit)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def download_mmlu_history(limit: int = 100):
    """下载 MMLU 世界历史子集"""
    print("\n📥 下载 MMLU (world_history)...")
    try:
        dataset = load_dataset("cais/mmlu", "world_history", split="test")
        samples = [
            {
                "id": f"mmlu_history_{i}",
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answer"],
                "category": "history"
            }
            for i, item in enumerate(dataset)
        ]
        save_samples(samples, DATA_DIR / "history" / "mmlu_world_history.json", limit)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def download_socialiqa(limit: int = 100):
    """下载 SocialIQA 社会常识推理数据集"""
    print("\n📥 下载 SocialIQA...")
    try:
        dataset = load_dataset("social_i_qa", split="validation")
        samples = [
            {
                "id": f"socialiqa_{i}",
                "context": item["context"],
                "question": item["question"],
                "answerA": item["answerA"],
                "answerB": item["answerB"],
                "answerC": item["answerC"],
                "label": item["label"],
                "category": "social"
            }
            for i, item in enumerate(dataset)
        ]
        save_samples(samples, DATA_DIR / "social" / "socialiqa.json", limit)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def download_truthfulqa(limit: int = 100):
    """下载 TruthfulQA 数据集"""
    print("\n📥 下载 TruthfulQA...")
    try:
        dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
        samples = [
            {
                "id": f"truthfulqa_{i}",
                "question": item["question"],
                "mc1_targets": item["mc1_targets"],
                "mc2_targets": item["mc2_targets"],
                "category": "truthful"
            }
            for i, item in enumerate(dataset)
        ]
        save_samples(samples, DATA_DIR / "truthful" / "truthfulqa.json", limit)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def download_natural_questions(limit: int = 100):
    """下载 Natural Questions 数据集 (简化版)"""
    print("\n📥 下载 Natural Questions...")
    try:
        # 使用简化版本，完整版太大
        dataset = load_dataset("nq_open", split="validation")
        samples = [
            {
                "id": f"nq_{i}",
                "question": item["question"],
                "answer": item["answer"],
                "category": "qa"
            }
            for i, item in enumerate(dataset)
        ]
        save_samples(samples, DATA_DIR / "qa" / "natural_questions.json", limit)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def download_humaneval(limit: int = 100):
    """下载 HumanEval 代码生成数据集"""
    print("\n📥 下载 HumanEval...")
    try:
        dataset = load_dataset("openai_humaneval", split="test")
        samples = [
            {
                "id": f"humaneval_{i}",
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
                "category": "code"
            }
            for i, item in enumerate(dataset)
        ]
        save_samples(samples, DATA_DIR / "code" / "humaneval.json", limit)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def download_mbpp(limit: int = 100):
    """下载 MBPP 代码生成数据集"""
    print("\n📥 下载 MBPP...")
    try:
        dataset = load_dataset("mbpp", split="test")
        samples = [
            {
                "id": f"mbpp_{i}",
                "task_id": item["task_id"],
                "text": item["text"],
                "code": item["code"],
                "test_list": item["test_list"],
                "category": "code"
            }
            for i, item in enumerate(dataset)
        ]
        save_samples(samples, DATA_DIR / "code" / "mbpp.json", limit)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def create_sample_dataset():
    """创建示例数据集（当无法下载时使用）"""
    print("\n📝 创建示例数据集...")
    
    # 数学示例
    math_samples = [
        {
            "id": "sample_math_1",
            "question": "小明有5个苹果，小红给了他3个，请问小明现在有几个苹果？",
            "answer": "8",
            "category": "math"
        },
        {
            "id": "sample_math_2", 
            "question": "一个长方形的长是10厘米，宽是5厘米，求它的面积。",
            "answer": "50平方厘米",
            "category": "math"
        },
        {
            "id": "sample_math_3",
            "question": "计算: 25 × 4 + 36 ÷ 6 = ?",
            "answer": "106",
            "category": "math"
        }
    ]
    save_samples(math_samples, DATA_DIR / "math" / "sample_math.json", 100)
    
    # 历史示例
    history_samples = [
        {
            "id": "sample_history_1",
            "question": "唐朝是由谁建立的？建立于哪一年？",
            "answer": "唐朝由李渊建立于618年",
            "category": "history"
        },
        {
            "id": "sample_history_2",
            "question": "秦始皇统一六国是在哪一年？",
            "answer": "公元前221年",
            "category": "history"
        },
        {
            "id": "sample_history_3",
            "question": "第一次世界大战爆发的导火索是什么事件？",
            "answer": "萨拉热窝事件（奥匈帝国皇储斐迪南大公遇刺）",
            "category": "history"
        }
    ]
    save_samples(history_samples, DATA_DIR / "history" / "sample_history.json", 100)
    
    # 常识问答示例
    qa_samples = [
        {
            "id": "sample_qa_1",
            "question": "水的沸点是多少摄氏度？",
            "answer": "100摄氏度（标准大气压下）",
            "category": "qa"
        },
        {
            "id": "sample_qa_2",
            "question": "地球上最大的海洋是什么？",
            "answer": "太平洋",
            "category": "qa"
        }
    ]
    save_samples(qa_samples, DATA_DIR / "qa" / "sample_qa.json", 100)
    
    # 代码示例
    code_samples = [
        {
            "id": "sample_code_1",
            "prompt": "写一个Python函数，计算两个数的和",
            "test": "assert add(1, 2) == 3",
            "category": "code"
        },
        {
            "id": "sample_code_2",
            "prompt": "写一个Python函数，判断一个数是否为素数",
            "test": "assert is_prime(7) == True",
            "category": "code"
        }
    ]
    save_samples(code_samples, DATA_DIR / "code" / "sample_code.json", 100)
    
    print("✅ 示例数据集创建完成")


def main():
    """主函数"""
    print("=" * 60)
    print("📊 Benchmark 数据集下载工具")
    print("=" * 60)
    
    if not HAS_DATASETS:
        print("\n⚠️ 未安装 datasets 库，正在安装...")
        os.system("pip install datasets")
        print("请重新运行此脚本")
        return
    
    # 每个数据集下载的样本数量
    SAMPLE_LIMIT = 50  # 用于测试，可调整
    
    results = {}
    
    # 数学推理数据集
    print("\n" + "=" * 40)
    print("📐 数学推理数据集")
    print("=" * 40)
    results["gsm8k"] = download_gsm8k(SAMPLE_LIMIT)
    results["mathqa"] = download_mathqa(SAMPLE_LIMIT)
    results["svamp"] = download_svamp(SAMPLE_LIMIT)
    
    # 历史/社会数据集
    print("\n" + "=" * 40)
    print("📚 历史/社会数据集")
    print("=" * 40)
    results["mmlu_history"] = download_mmlu_history(SAMPLE_LIMIT)
    results["socialiqa"] = download_socialiqa(SAMPLE_LIMIT)
    
    # 真实性/常识数据集
    print("\n" + "=" * 40)
    print("💡 真实性/常识数据集")
    print("=" * 40)
    results["truthfulqa"] = download_truthfulqa(SAMPLE_LIMIT)
    results["natural_questions"] = download_natural_questions(SAMPLE_LIMIT)
    
    # 代码生成数据集
    print("\n" + "=" * 40)
    print("💻 代码生成数据集")
    print("=" * 40)
    results["humaneval"] = download_humaneval(SAMPLE_LIMIT)
    results["mbpp"] = download_mbpp(SAMPLE_LIMIT)
    
    # 创建示例数据集（保底）
    create_sample_dataset()
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("📋 下载结果摘要")
    print("=" * 60)
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\n总计: {success_count}/{total_count} 个数据集下载成功")
    print(f"数据保存目录: {DATA_DIR}")


if __name__ == "__main__":
    main()
