# Experiment Console × ThinkCheck

AI 实验控制台与 ThinkCheck 谐振评估引擎的集成项目。在 Godot 中运行批量 AI 对话实验，每次对话完成后自动调用 ThinkCheck API 进行 U/D/A/H 四维评估，并生成完整的 U 值轨迹追踪报告。

## 核心功能

- **AI 实验控制台**：基于 Godot 的图形化实验平台，支持模板管理、流式对话、批量运行
- **自动评估**：每次 AI 回复完成后，自动调用 ThinkCheck API，评估结果显示在状态栏
- **轨迹追踪**：`trace_u_trajectory_v2.py` 支持矛盾驱动采样、分段独立请求、自适应翻转点检测
- **报告生成**：轨迹数据导出为 JSON、CSV 和分析报告

## 文件结构

```
├── project.godot              # Godot 项目入口
├── scripts/                   # GDScript 源码
│   ├── DeepSeekStreamClient.gd # DeepSeek API 流式客户端
│   ├── ExperimentPanel.gd      # 实验面板主控制器
│   └── ExperimentStore.gd      # 实验文件存储
├── thinkcheck-agent-v6/       # ThinkCheck API 服务
│   ├── api.py                  # FastAPI 入口
│   └── thinkcheck_harmony/     # 核心评估引擎 (已升级至 v9)
├── trace_u_trajectory.py      # U 值轨迹追踪脚本 (基础版)
├── trace_u_trajectory_v2.py   # U 值轨迹追踪脚本 (矛盾驱动采样版)
├── test_b_output.txt          # 测试文本样例
├── test_b_trajectory.json     # 轨迹数据样例
├── test_b_u_trajectory.csv    # CSV 轨迹数据
└── test_b_analysis_report.txt # 分析报告样例
```

## 快速开始

### 1. 启动 ThinkCheck API

```bash
cd thinkcheck-agent-v6
pip install -r requirements.txt
python api.py
```

服务启动后访问 `http://localhost:8000/health` 确认正常运行。

### 2. 打开 Experiment Console

用 Godot 引擎打开 `project.godot`，按 `F5` 运行。选择一个实验模板或手动输入 user 消息，点击“发送”，等待回复完成后状态栏将显示 ThinkCheck 评估结果。

## U 值轨迹追踪

### V2 脚本（推荐）

```bash
# 使用默认测试文本
python trace_u_trajectory_v2.py

# 指定自定义文件
python trace_u_trajectory_v2.py your_test_file.txt
```

V2 脚本特性：

- **矛盾驱动采样**：先粗扫全文识别语言特征变化最大的区域，在这些区域内密集采样，其他区域稀疏采样
- **分段独立请求**：每次 API 调用只发送 2000 字符的局部片段，不累积历史文本，彻底解决超时问题
- **指数退避重试**：请求失败时自动重试（30s → 60s → 120s），最多 3 次
- **自适应翻转点检测**：基于 U 值波动最大区域的 A 值峰值动态定位翻转点

### V1 脚本（基础版）

```bash
python trace_u_trajectory.py your_test_file.txt
```

V1 使用全局累积文本进行评估，适用于短文本的快速测试。

### 输出文件

| 文件 | 说明 |
|------|------|
| `test_b_trajectory.json` | 每步完整的 U/D/A/H 轨迹数据 |
| `test_b_u_trajectory.csv` | CSV 格式，可直接导入 Excel 绘图 |
| `test_b_analysis_report.txt` | 分析报告：翻转点位置、趋势斜率、结论 |

## ThinkCheck 升级说明

本仓库集成的 ThinkCheck API 核心代码已升级至 **v9** 版本。主要改进：

- **A 模块重构**：彻底移除预设语义对立词库，改为基于句子嵌入余弦相似度的关系网络检测
- **构成分解暴露**：每条矛盾边的类型、权重、句子位置完全可追溯
- **U/D 增强**：跨术语一致性检测 + 真伪创新区分
- **H 审计**：λ 权重完全可配置，审计日志记录权重来源和修改历史

完整升级记录见 [ThinkCheck Agent v9 仓库](https://github.com/luoxuejian000/-thinkcheck-lib-/tree/thinkcheck-agent-v9)。

## 相关项目

- [ThinkCheck Agent v9](https://github.com/luoxuejian000/-thinkcheck-lib-/tree/thinkcheck-agent-v9) — 谐振评估引擎
- [OCHR（舟济）](https://github.com/luoxuejian000/OCHR) — AI Agent 集群安全治理层
- [Resonance Inference](https://github.com/luoxuejian000/resonance-inference) — LLM 推理实时温度调度
- [CodeHarmony](https://github.com/luoxuejian000/code-harmony) — 代码和谐度审计工具
- [GitNarrative](https://github.com/luoxuejian000/chronos-resonance) — Git 演化自传生成器

## 作者

**李广好** (luoxuejian000)

## 协议

MIT License
```
