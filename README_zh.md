<div align="center">
  <h1>Bloom-Eval：一个基于布鲁姆分类法的自动综述生成分层评测基准</h1>
  <p>
    <strong>Fei Zhang<sup>1</sup>, Zhe Zhao<sup>2</sup>, Haibin Wen<sup>1</sup>, Tianshuo Wei<sup>1</sup>,</strong><br>
    <strong>Zaixi Zhang<sup>3</sup>, Chao Yang<sup>4,*</sup>, and Ye Wei<sup>1,*</sup></strong>
  </p>
  <p>
    <sup>1</sup>City University of Hong Kong &nbsp;&nbsp;
    <sup>2</sup>Stanford University<br>
    <sup>3</sup>Princeton University &nbsp;&nbsp;
    <sup>4</sup>Shanghai Jiaotong University
  </p>
  <p><sup>*</sup>通讯作者</p>
</div>

[![ACL 2026](https://img.shields.io/badge/ACL-2026-blue.svg)](https://2026.aclweb.org/)
[![Status](https://img.shields.io/badge/Status-Camera--ready-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

论文 **"Bloom-Eval: A Hierarchical Evaluation Benchmark for Automatic Survey Generation Based on Bloom's Taxonomy"** 的官方代码仓库。

当前仓库为精简的 camera-ready 公开版本，保留了理解基准设计与复现 topic 级别评测所需的核心内容，包括评测脚本、提示词文件以及 20 个已发布主题样例。

[English Version](README.md)

## 项目简介

Bloom-Eval 是一个面向自动综述生成（Automatic Survey Generation, ASG）的六层级评测基准，其设计依据布鲁姆分类法（Bloom's Taxonomy）展开：

- **Level 1，记忆（Memory）**：评估实体、事实、引用等基础信息的准确回忆能力
- **Level 2，理解（Comprehension）**：评估主题聚焦、引用忠实性以及参考文献覆盖与均衡性
- **Level 3，应用（Application）**：评估综述结构组织与学术写作规范的执行情况
- **Level 4，分析（Analysis）**：评估章节结构、一致性与分析性组织能力
- **Level 5，评价（Evaluation）**：评估对已有工作的批判性判断与优缺点归纳能力
- **Level 6，创造（Creation）**：评估新研究问题、研究方向与综合创新能力

其中，涉及语义比较和主观判断的指标采用 **GRADE**（Generative Rubric Adaptive Differential Evaluation）进行基于规则的 LLM 评审。

## 框架图

![Bloom-Eval framework overview](figs/bloom-eval.png)

*Bloom-Eval 的数据构建流程与六层级评测框架示意图。*

## 当前发布内容

这个 camera-ready 版本包含：

- **16 个核心指标**的评测脚本
- LLM 类指标所需的提示词模板
- 位于 `data/experimental_data` 下的 `20` 个已发布评测主题
- 主题元数据文件 `data/experimental_data/experimental_topics.csv`
- 一个用于单主题批量运行全部指标的便捷脚本

## 仓库结构

```text
Bloom-Eval/
├── data/
│   └── experimental_data/
│       ├── experimental_topics.csv
│       ├── cs_01/ ... cs_10/
│       └── gs_01/ ... gs_10/
├── figs/
├── prompts/
│   ├── level1/ ... level6/
├── scripts/
│   ├── common.py
│   ├── prompt_utils.py
│   ├── level1/ ... level6/
│   └── others/
├── requirements.txt
├── run_topic_all_metrics.sh
└── README_zh.md
```

每个已发布主题当前都包含一个 `human/` 目录，通常包括：

- `content.json`
- `outline.json`
- `reference.json`
- 一个任务文件，例如 `123_topic_title.json`

如果你要评估某个模型在同一主题上的输出，需要自行构造一个对应的 `llm/` 目录，其中至少包含：

- `content.json`
- `outline.json`
- `reference.json`

## 已发布主题

当前版本共发布 `20` 个主题：

- `cs_01` 到 `cs_10`：计算机科学
- `gs_01` 到 `gs_10`：综合科学

每个主题的标题、发表 venue、年份和引用量信息可在 `data/experimental_data/experimental_topics.csv` 中查看。

## 包含的指标

`run_topic_all_metrics.sh` 默认会运行以下 16 个核心指标：

### Level 1

- `EFid`
- `FCons`
- `HIRC`

### Level 2

- `CF`
- `OTC`
- `TBal`
- `TFSim`

### Level 3

- `DSI`
- `FAP`
- `FMI`

### Level 4

- `SCS`
- `SCons`
- `STS`

### Level 5

- `CAA`

### Level 6

- `FNov`
- `ROQ`

此外还提供一个非核心指标脚本：

- `scripts/others/run_Readability.py`

## 单独运行某个指标

所有指标脚本位于 `scripts/level*/run_*.py`。例如：

```bash
python scripts/level2/run_OTC.py \
  --outline_file_human data/experimental_data/cs_01/human/outline.json \
  --outline_file_llm /path/to/cs_01/llm/outline.json \
  --output_dir results/otc_cs_01 \
  --model gpt-5-mini
```

再例如一个不依赖 LLM 的脚本：

```bash
python scripts/level3/run_DSI.py \
  --content_file_human /path/to/human_content.md \
  --content_file_llm /path/to/llm_content.md \
  --output_dir results/dsi_example
```

直接运行单个脚本前，建议先查看该脚本的命令行参数，确认它实际要求的输入文件类型与参数名。

## 输出结果

每个指标都会把结果写入指定输出目录，通常包括：

- `result.json`：结构化结果
- `report.txt`：便于人工阅读的摘要报告
- 各类中间 JSON 文件，例如抽取、匹配、评分阶段的产物
- 如果启用了 `--save_raw_response`，则会额外生成 `logs/` 原始响应日志

## 引用

如果你使用了本仓库，请引用 Bloom-Eval 论文。
