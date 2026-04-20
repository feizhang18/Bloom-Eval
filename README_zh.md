# Bloom-Eval: 基于布鲁姆分类学的自动综述生成分层评估基准

[![ACL 2026](https://img.shields.io/badge/ACL-2026-blue.svg)](https://2026.aclweb.org/)
[![Status](https://img.shields.io/badge/Status-Camera--ready-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

这是论文 **"Bloom-Eval: A Hierarchical Evaluation Benchmark for Automatic Survey Generation Based on Bloom's Taxonomy"** 的官方仓库。

当前仓库采用精简的 camera-ready 公开形式，只保留描述 benchmark、公开已发布 topic 以及提供评测脚本所需的最小材料。

## 项目简介

Bloom-Eval 是一个面向自动综述生成（ASG）的六层级分层评估基准。该基准以布鲁姆教育目标分类学（Bloom's Taxonomy）为理论基础，从记忆、理解、应用、分析、评价、创造六个维度对 ASG 系统进行细粒度评估。

## 框架概览

![Bloom-Eval 框架总览](bloom-eval_10_05.png)

*图：Bloom-Eval 框架概览，展示了数据收集流程以及六层级评估体系。*

## 核心特性

### 1. 六层级认知评估体系

- **记忆（Memory）**：评估对实体、核心文献和事实陈述的准确再现能力。
- **理解（Comprehension）**：评估对研究景观、引用忠实度和主题聚焦的把握能力。
- **应用（Application）**：评估对学术规范、文档结构和组织范式的执行能力。
- **分析（Analysis）**：评估层级推理、分析粒度和结构清晰度。
- **评价（Evaluation）**：评估对已有工作结论与局限性的批判性判断能力。
- **创造（Creation）**：评估构建新颖概念框架和提出未来研究方向的能力。

### 2. GRADE 评估方法

我们采用 **GRADE**（Generative Rubric Adaptive Differential Evaluation）这一基于量表的对比评估方法。

### 3. 跨学科 Benchmark

- 完整 benchmark 构建流程覆盖 `3,506` 篇人工核验的专家综述论文
- 覆盖 `60` 个学术会议与期刊
- 覆盖 `14` 个科学领域
- 本 camera-ready 仓库当前公开 `20` 个评测 topic

## 仓库结构

```text
.
├── dataset/            # Bloom-Eval 数据集元数据（当前公开 20 个 topic）
├── scripts/            # 评估脚本（包含确定性算法与 GRADE 相关实现）
├── prompts/            # 各认知层级提取与评分所需的提示词模板说明
└── README.md
```

## 当前公开范围

这个 camera-ready 仓库当前包括：

- `dataset/experimental_topics.csv`：20 个已公开评测 topic
- `scripts/`：按评测层级组织的 benchmark 脚本
- `prompts/`：prompt 占位与发布说明

本仓库不包含完整论文源码，也不包含受版权保护的全文论文。

## 引用

如果你使用了这个仓库，请引用 Bloom-Eval 论文。引用信息可见 `CITATION.cff`。
