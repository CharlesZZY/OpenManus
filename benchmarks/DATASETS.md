# Benchmark 数据集（清理后）

> 每种任务类型仅保留一套数据集，统一拆分为 `train` / `test`。

## 1. 最终结构总览

| # | 类别 | 保留数据集 | train | test | 总计 | 拆分方式 |
|---|------|-----------|------:|-----:|-----:|---------|
| 1 | 数学推理 (math) | **GSM8K** | 7,473 | 1,319 | 8,792 | 原始 split 保留 |
| 2 | 历史/社会 (history) | **MMLU** (全科) | 99,842 | 14,042 | 113,884 | auxiliary_train→train, test 保留 |
| 3 | 常识 (commonsense) | **TruthfulQA** | 653 | 164 | 817 | validation 80/20 随机拆分 |
| 4 | 代码生成 (codegen) | **HumanEval** | 131 | 33 | 164 | 原 test 80/20 随机拆分 |
| 5 | 长上下文 (long_context) | **LongBench v2** | 402 | 101 | 503 | 原 train 80/20 随机拆分 |
| 6 | 长文档处理 (long_document) | **Selective Context** | 2,373 | 594 | 2,967 | 3子集合并后 80/20 随机拆分 |
| | **合计** | | **110,874** | **16,253** | **127,127** | |

随机拆分 seed = 42，可复现。

---

## 2. 被删除的数据集

| 类别 | 被删除 | 原因 |
|------|--------|------|
| math | SVAMP (1,000 条) | 与 GSM8K 同属小学数学推理，GSM8K 规模更大、更常用 |
| history | SocialIQA (35,364 条) | MMLU 覆盖 57 学科更全面，是业界标准 |

---

## 3. 目录结构

```
benchmarks/data/
├── math/
│   ├── gsm8k_train.json          (7,473 条, 4.29 MB)
│   ├── gsm8k_test.json           (1,319 条, 0.77 MB)
│   └── gsm8k_info.json
├── history/
│   ├── mmlu_train.json           (99,842 条, 166.41 MB)
│   ├── mmlu_test.json            (14,042 条, 8.25 MB)
│   └── mmlu_info.json
├── commonsense/
│   ├── truthfulqa_train.json     (653 条, 0.48 MB)
│   ├── truthfulqa_test.json      (164 条, 0.12 MB)
│   └── truthfulqa_info.json
├── codegen/
│   ├── humaneval_train.json      (131 条, 0.17 MB)
│   ├── humaneval_test.json       (33 条, 0.04 MB)
│   └── humaneval_info.json
├── long_context/
│   ├── longbench_v2_train.json   (402 条, 358.29 MB)
│   ├── longbench_v2_test.json    (101 条, 85.65 MB)
│   └── longbench_v2_info.json
├── long_document/
│   ├── selective_context_train.json  (2,373 条, 24.56 MB)
│   ├── selective_context_test.json   (594 条, 5.78 MB)
│   └── selective_context_info.json
└── dataset_structure.json        (全局结构摘要)
```

---

## 4. 各数据集详情

### 4.1 GSM8K (math)

| 属性 | 值 |
|------|-----|
| **全称** | Grade School Math 8K |
| **HuggingFace ID** | `openai/gsm8k` (config: `main`) |
| **论文** | Cobbe et al., "Training Verifiers to Solve Math Word Problems" (2021) |
| **语言** | 英文 |
| **领域** | 小学数学，2-8 步推理 |

**字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识 `gsm8k_train_0` / `gsm8k_test_0` |
| `question` | string | 数学题目 |
| `answer` | string | 含逐步推理的完整答案，最终数值以 `####` 分隔 |

---

### 4.2 MMLU (history)

| 属性 | 值 |
|------|-----|
| **全称** | Massive Multitask Language Understanding |
| **HuggingFace ID** | `cais/mmlu` (config: `all`) |
| **论文** | Hendrycks et al., "Measuring Massive Multitask Language Understanding" (ICLR 2021) |
| **语言** | 英文 |
| **领域** | 57 学科 (STEM, 人文, 社科, 其他) |

**字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识 |
| `question` | string | 题目文本 |
| `choices` | list[string] | 4 个选项 (A/B/C/D) |
| `answer` | int | 正确选项索引 0-3 |
| `subject` | string | 所属学科 |

train 来自原始 `auxiliary_train` (99,842 条)，test 为原始 `test` (14,042 条)。

---

### 4.3 TruthfulQA (commonsense)

| 属性 | 值 |
|------|-----|
| **全称** | TruthfulQA |
| **HuggingFace ID** | `truthful_qa` (generation 格式) |
| **论文** | Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods" (ACL 2022) |
| **语言** | 英文 |
| **领域** | 38 类常见误解 |

**字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识 |
| `question` | string | 问题文本 |
| `best_answer` | string | 最佳正确答案 |
| `correct_answers` | list[string] | 所有可接受的正确答案 |
| `incorrect_answers` | list[string] | 常见错误答案 |
| `category` | string | 误解类别 |
| `source` | string | 来源出处 |

原数据集仅有 validation split (817 条)，按 80/20 随机拆分为 train (653) / test (164)。
mc (multiple_choice) 格式已删除，仅保留 generation 格式。

---

### 4.4 HumanEval (codegen)

| 属性 | 值 |
|------|-----|
| **全称** | HumanEval |
| **HuggingFace ID** | `openai_humaneval` |
| **论文** | Chen et al., "Evaluating Large Language Models Trained on Code" (2021) |
| **语言** | Python |
| **领域** | 函数级代码补全 |

**字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识 |
| `task_id` | string | 任务标识 `HumanEval/0` 等 |
| `prompt` | string | 函数签名 + docstring |
| `canonical_solution` | string | 参考解法 |
| `test` | string | 单元测试代码 |
| `entry_point` | string | 被测函数名 |

原数据集仅有 test split (164 条)，按 80/20 随机拆分为 train (131) / test (33)。

---

### 4.5 LongBench v2 (long_context)

| 属性 | 值 |
|------|-----|
| **全称** | LongBench v2 |
| **HuggingFace ID** | `THUDM/LongBench-v2` |
| **论文** | "LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks" (2024) |
| **语言** | 中英双语 |
| **领域** | 6 类长文本理解 (单文档QA, 多文档QA, 长ICL, 对话历史, 代码仓库, 结构化数据) |
| **上下文** | 8K - 2M 词 |

保留原始字段 (context, question, choice, answer 等)。
原数据集仅有 train split (503 条)，按 80/20 随机拆分为 train (402) / test (101)。

---

### 4.6 Selective Context (long_document)

| 属性 | 值 |
|------|-----|
| **全称** | Selective Context |
| **论文** | Li et al., "Compressing Context to Enhance Inference Efficiency of Large Language Models" (EMNLP 2023) |
| **语言** | 英文 |
| **领域** | 长文档压缩与处理 |

**3 个子集已合并为一个数据集：**

| 子集 | 来源 HuggingFace ID | train 中 | test 中 | 总计 |
|------|---------------------|------:|------:|-----:|
| arxiv | `liyucheng/arxiv-march-2023` | 399 | 101 | 500 |
| bbc_news | `liyucheng/bbc_new_2303` | 1,516 | 376 | 1,892 |
| sharegpt | `liyucheng/sharegpt-500` | 458 | 117 | 575 |
| **合计** | | **2,373** | **594** | **2,967** |

每条记录包含 `sub_dataset` 字段 (arxiv / bbc_news / sharegpt) 标识来源。

---

## 5. Agent Workflow 测试维度

| 维度 | 数据集 | 单条输入规模 | 推理复杂度 | 预期 Agent 步骤数 | 对 workflow 的压力来源 |
|------|--------|------------|-----------|-----------------|---------------------|
| 短文本 + 推理密集 | GSM8K | 短 (50-200 词) | 高 (多步推理) | 3-5 步 | 推理链长度，多步计算 |
| 短文本 + 知识检索 | MMLU | 短 (50-150 词) | 中 (需背景知识) | 1-3 步 | 大规模并发吞吐 |
| 短文本 + 判断 | TruthfulQA | 短 (20-50 词) | 低-中 (事实判断) | 1-2 步 | 快速调度开销 |
| 中文本 + 生成 | HumanEval | 中 (100-500 词) | 高 (代码+测试) | 3-8 步 | 工具调用，代码执行 |
| 超长文本 + 理解 | LongBench v2 | 长 (8K-2M 词) | 高 (长距离依赖) | 5-10+ 步 | Token 消耗，长推理时间 |
| 长文档 + 处理 | Selective Context | 长 (1K-10K+ 词) | 中 (摘要/提取) | 3-6 步 | 文档切分与拼接策略 |
