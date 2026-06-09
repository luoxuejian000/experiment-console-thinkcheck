#!/usr/bin/env python3
"""
Test B U值轨迹追踪实验 V3 Pure —— 零预设、零判定、纯记录

核心设计哲学：
- 矛盾动力论：A值的每一步微小攀升都要被捕捉，记录累积速率和差分
- 关系本体论：不仅记录U/D/A/H绝对值，更记录它们之间的差分和耦合关系
- 谐振调谐论：D值波动率、A值累积速率、U值变化模式 —— 揭示四维度之间的关系失调
- 实践介入论：所有参数在报告头部清晰标注，不留任何黑箱

背景与进化：
- V1：固定翻转点+累积文本请求 → 只能看到全局U值，超时严重
- V2：矛盾驱动采样+分段独立请求+自适应翻转点检测 → 能看到局部U值变化
- V3：在V2架构之上，将单维U值观测升级为U/D/A/H四维全息记录

本报告仅记录真实数据，不包含任何判定或结论。所有阈值和判定标准均应由观察者基于数据自行定义。
"""

import json
import requests
import numpy as np
import os
import sys
import time
import csv
from datetime import datetime


# ========================================
# 可配置参数（实践介入论：参数可配置、可审计）
# ========================================
THINKCHECK_API = "http://localhost:8000/evaluate"
DEFAULT_INPUT_FILE = "test_b_output.txt"
OUTPUT_CSV = "test_b_pure_multidim.csv"
OUTPUT_JSON = "test_b_pure_multidim.json"
OUTPUT_REPORT = "test_b_pure_report.txt"

# 采样策略参数
SEGMENT_LENGTH = 2000        # 每次API请求的文本片段长度（防止超时）
COARSE_BLOCK_SIZE = 1000     # 粗扫阶段的块大小
FINE_GRAIN_STEP = 100        # 精细采样步长
SPARSE_GRAIN_STEP = 500      # 稀疏采样步长
NUM_CANDIDATE_REGIONS = 5    # 候选区域数量
MAX_TOTAL_STEPS = 80         # 最大采样步数

# 重试机制参数
MAX_RETRIES = 3              # 最大重试次数
INITIAL_TIMEOUT = 30         # 初始超时时间(秒)
BACKOFF_FACTOR = 2           # 超时翻倍因子
RETRY_DELAY = 1              # 重试间隔(秒)

# 语言特征变化检测参数
CHINESE_WEIGHT = 0.5         # 中文比例变化权重
VOCAB_WEIGHT = 0.3           # 词汇重叠率权重
PUNCTUATION_WEIGHT = 0.2     # 标点密度权重

# 翻转点快照参数
FLIP_SNAPSHOT_WINDOW = 10    # 翻转点前后采样步数


# ========================================
# 工具函数
# ========================================

def is_chinese_char(c):
    """判断是否为中文字符"""
    return '\u4e00' <= c <= '\u9fff'


def count_chinese_ratio(text):
    """计算文本中中文字符的比例"""
    if not text:
        return 0.0
    chinese_count = sum(1 for c in text if is_chinese_char(c))
    return chinese_count / len(text)


def extract_vocab(text, n=2):
    """提取文本的n-gram词汇集合"""
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    words = text.lower().split()
    vocab = set()
    for i in range(len(words) - n + 1):
        vocab.add(' '.join(words[i:i+n]))
    return vocab


def jaccard_similarity(set1, set2):
    """计算两个集合的Jaccard相似度"""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def count_punctuation_density(text):
    """计算标点密度"""
    if not text:
        return 0.0
    punctuation = '，。！？；：、,.!?;:;'
    count = sum(1 for c in text if c in punctuation)
    return count / len(text)


def calculate_change_score(block1, block2):
    """计算相邻块之间的语言特征变化度评分"""
    chinese1 = count_chinese_ratio(block1)
    chinese2 = count_chinese_ratio(block2)
    chinese_diff = abs(chinese1 - chinese2)

    vocab1 = extract_vocab(block1)
    vocab2 = extract_vocab(block2)
    vocab_sim = jaccard_similarity(vocab1, vocab2)
    vocab_diff = 1 - vocab_sim

    punc1 = count_punctuation_density(block1)
    punc2 = count_punctuation_density(block2)
    punc_diff = abs(punc1 - punc2)

    score = (CHINESE_WEIGHT * chinese_diff +
             VOCAB_WEIGHT * vocab_diff +
             PUNCTUATION_WEIGHT * punc_diff)

    return score, {
        'chinese_diff': chinese_diff,
        'vocab_diff': vocab_diff,
        'punc_diff': punc_diff
    }


def exponential_backoff_retry(func, *args, **kwargs):
    """指数退避重试装饰器实现"""
    timeout = INITIAL_TIMEOUT
    for attempt in range(MAX_RETRIES):
        try:
            kwargs['timeout'] = timeout
            return func(*args, **kwargs)
        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                timeout *= BACKOFF_FACTOR
    return None


def evaluate_segment(chunk_pos, text, total_length):
    """评估单个文本片段（带重试机制）

    返回: (U, D, A, H) 元组，失败时返回 None
    """
    try:
        response = exponential_backoff_retry(
            requests.post,
            THINKCHECK_API,
            json={"document": text, "domain": "general"}
        )
        if response is None:
            print(f"  字符位置 {chunk_pos}/{total_length}: API请求失败（重试耗尽），使用占位数据")
            return (0.0, 0.0, 0.0, 0.0)

        data = response.json()
        u = data.get('u', 0.0)
        d = data.get('d', 0.0)
        a = data.get('a', 0.0)
        h = data.get('h', 0.0)
        return (u, d, a, h)
    except Exception as e:
        print(f"  字符位置 {chunk_pos}/{total_length}: 评估异常: {e}，使用占位数据")
        return (0.0, 0.0, 0.0, 0.0)


def validate_api_health():
    """验证ThinkCheck API健康状态"""
    health_url = THINKCHECK_API.replace('/evaluate', '/health')
    try:
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200 and response.json().get("status") == "healthy":
            return True
    except Exception:
        pass
    return False


def compute_h_driven_by(u_diff, d_diff, a_diff, h_diff):
    """计算H值的归因分解

    比较U_diff、D_diff、A_diff的绝对值，找出对H值变化贡献最大的维度。
    如果所有差分的绝对值都很小（< 0.001），标注为"均衡"。
    """
    abs_u = abs(u_diff)
    abs_d = abs(d_diff)
    abs_a = abs(a_diff)
    abs_h = abs(h_diff)

    # 如果H值变化本身就很小，标记为均衡
    if abs_h < 0.001:
        return "均衡"

    # 找出绝对值最大的差分维度
    max_abs = max(abs_u, abs_d, abs_a)
    if max_abs < 0.001:
        return "均衡"

    if abs_u == max_abs:
        return "U"
    elif abs_d == max_abs:
        return "D"
    else:
        return "A"


# ========================================
# 核心采样逻辑（复用V2架构）
# ========================================

def coarse_scan(full_text):
    """第一阶段：粗扫识别矛盾集中区域

    将文本按固定大块切分，计算相邻块的语言特征变化度。
    返回按变化度排序的候选区域列表和所有块。
    """
    print("\n=== 第一阶段：粗扫识别矛盾集中区域 ===")

    blocks = []
    change_scores = []

    for i in range(0, len(full_text), COARSE_BLOCK_SIZE):
        block = full_text[i:i + COARSE_BLOCK_SIZE]
        blocks.append(block)

    print(f"  全文长度: {len(full_text)} 字符")
    print(f"  分块数量: {len(blocks)} (每块 {COARSE_BLOCK_SIZE} 字符)")

    for i in range(len(blocks) - 1):
        score, details = calculate_change_score(blocks[i], blocks[i + 1])
        change_scores.append({
            'index': i,
            'position': (i + 1) * COARSE_BLOCK_SIZE,
            'score': score,
            'details': details
        })
        print(f"  块 {i+1} -> {i+2}: 变化度={score:.4f} "
              f"(中文比例差异={details['chinese_diff']:.4f}, "
              f"词汇差异={details['vocab_diff']:.4f}, "
              f"标点差异={details['punc_diff']:.4f})")

    change_scores.sort(key=lambda x: x['score'], reverse=True)
    candidates = change_scores[:NUM_CANDIDATE_REGIONS]

    print(f"\n  识别到的 {NUM_CANDIDATE_REGIONS} 个高变化区域:")
    for i, candidate in enumerate(candidates):
        print(f"  {i+1}. 位置 {candidate['position']}, 变化度 {candidate['score']:.4f}")

    return candidates, blocks


def generate_sample_points(full_text, candidates):
    """生成最终采样点列表

    候选区域周围使用精细采样步长，其他区域使用稀疏采样步长。
    """
    print("\n=== 生成采样点 ===")

    sample_points = set()
    total_length = len(full_text)

    # 在候选区域周围进行精细采样
    for candidate in candidates:
        region_start = max(0, candidate['position'] - COARSE_BLOCK_SIZE)
        region_end = min(total_length, candidate['position'] + COARSE_BLOCK_SIZE)

        for pos in range(region_start, region_end, FINE_GRAIN_STEP):
            if pos < total_length - 100:
                sample_points.add(pos)

    # 在整个文本范围内进行稀疏采样
    for pos in range(0, total_length, SPARSE_GRAIN_STEP):
        if pos < total_length - 100:
            sample_points.add(pos)

    # 添加文本末尾采样点
    if total_length > SEGMENT_LENGTH:
        sample_points.add(total_length - SEGMENT_LENGTH)

    # 过滤无效点并排序
    valid_points = sorted(sample_points)

    # 如果采样点太多，进行均匀采样压缩
    if len(valid_points) > MAX_TOTAL_STEPS:
        step_size = len(valid_points) // MAX_TOTAL_STEPS
        if step_size < 1:
            step_size = 1
        valid_points = valid_points[::step_size]

    print(f"  候选区域精细采样点: {len(sample_points)} 个")
    print(f"  最终采样点数量: {len(valid_points)} 个")

    return valid_points


def collect_multi_dim_data(full_text, sample_points):
    """采集多维数据

    对每个采样点请求API，记录:
    - 绝对值: U, D, A, H
    - 一阶差分: U_diff, D_diff, A_diff
    - 累积指标: cumul_A
    - 波动指标: D_volatility
    - 归因分解: H_driven_by
    """
    print("\n=== 第三阶段：多维数据采集 ===")

    trajectory = []
    total_length = len(full_text)
    prev_u = 0.0
    prev_d = 0.0
    prev_a = 0.0
    prev_h = 0.0
    cumul_a = 0.0
    d_values_so_far = []

    for step_idx, chunk_pos in enumerate(sample_points):
        # 截取固定长度的文本片段
        end_pos = min(chunk_pos + SEGMENT_LENGTH, total_length)
        segment = full_text[chunk_pos:end_pos]

        if len(segment) < 100:
            continue

        # 请求API
        u, d, a, h = evaluate_segment(chunk_pos, segment, total_length)

        # 计算一阶差分
        u_diff = u - prev_u
        d_diff = d - prev_d
        a_diff = a - prev_a
        h_diff = h - prev_h

        # 更新累积值
        cumul_a += a

        # 更新D值波动率
        d_values_so_far.append(d)
        if len(d_values_so_far) > 1:
            d_volatility = float(np.std(d_values_so_far))
        else:
            d_volatility = 0.0

        # 归因分解
        h_driven_by = compute_h_driven_by(u_diff, d_diff, a_diff, h_diff)

        record = {
            'step': step_idx,
            'char_pos': chunk_pos,
            'U': round(u, 4),
            'D': round(d, 4),
            'A': round(a, 4),
            'H': round(h, 4),
            'U_diff': round(u_diff, 4),
            'D_diff': round(d_diff, 4),
            'A_diff': round(a_diff, 4),
            'H_diff': round(h_diff, 4),
            'H_driven_by': h_driven_by,
            'cumul_A': round(cumul_a, 4),
            'D_volatility': round(d_volatility, 6),
        }
        trajectory.append(record)

        # 实时输出状态
        print(f"  步{step_idx:3d} 位置{chunk_pos:5d}: "
              f"U={u:.3f}({u_diff:+.3f}) D={d:.3f}({d_diff:+.3f}) "
              f"A={a:.3f}({a_diff:+.3f}) H={h:.3f} "
              f"H驱动={h_driven_by} cumulA={cumul_a:.3f} Dvol={d_volatility:.4f}")

        # 更新上一步状态
        prev_u = u
        prev_d = d
        prev_a = a
        prev_h = h

    return trajectory


def detect_flip_point(trajectory):
    """自适应翻转点检测（基于矛盾动力论）

    计算U值差分序列，找到波动最大的若干候选点，
    在其中选择A值最高的作为翻转点。
    仅用于确定高分辨率快照的展示范围，不作为判定结论。
    """
    if len(trajectory) < 4:
        return None

    u_values = [r['U'] for r in trajectory]
    u_diffs = np.abs(np.diff(u_values))

    if len(u_diffs) > 3:
        # 选取U值波动最大的3个候选点
        candidate_indices = np.argsort(u_diffs)[-3:]
        # 选择A值最高的那个
        a_values = [trajectory[i]['A'] for i in candidate_indices if i < len(trajectory)]
        if a_values:
            best_idx = int(candidate_indices[np.argmax(a_values)])
            flip_step = trajectory[best_idx]['step']
        else:
            flip_step = len(trajectory) // 2
    else:
        flip_step = len(trajectory) // 2

    return flip_step


# ========================================
# 输出函数
# ========================================

def write_csv(trajectory):
    """写入CSV文件"""
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'step', 'char_pos',
            'U', 'D', 'A', 'H',
            'U_diff', 'D_diff', 'A_diff',
            'H_driven_by', 'cumul_A', 'D_volatility'
        ])
        for r in trajectory:
            writer.writerow([
                r['step'], r['char_pos'],
                r['U'], r['D'], r['A'], r['H'],
                r['U_diff'], r['D_diff'], r['A_diff'],
                r['H_driven_by'], r['cumul_A'], r['D_volatility']
            ])
    print(f"[OK] CSV数据已保存至 {OUTPUT_CSV}")


def write_json(trajectory):
    """写入JSON文件"""
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON数据已保存至 {OUTPUT_JSON}")


def write_report(trajectory, candidates, flip_step, input_file):
    """写入纯数据报告文本"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Test B 多维U/D/A/H轨迹追踪报告 V3 Pure")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"输入文件: {input_file}")
    report_lines.append(f"全文长度: {sum(len(t) for t in [open(input_file, 'r', encoding='utf-8').read()]) if os.path.exists(input_file) else 'N/A'} 字符")
    report_lines.append("")

    # 采样策略说明
    report_lines.append("-" * 80)
    report_lines.append("1. 采样策略参数（实践介入论：无黑箱）")
    report_lines.append("-" * 80)
    report_lines.append(f"  分段长度        : {SEGMENT_LENGTH} 字符（分段独立请求，防止超时）")
    report_lines.append(f"  粗扫块大小       : {COARSE_BLOCK_SIZE} 字符")
    report_lines.append(f"  精细采样步长     : {FINE_GRAIN_STEP} 字符（高变化区域）")
    report_lines.append(f"  稀疏采样步长     : {SPARSE_GRAIN_STEP} 字符（平稳区域）")
    report_lines.append(f"  候选区域数       : {NUM_CANDIDATE_REGIONS}")
    report_lines.append(f"  最大采样步数     : {MAX_TOTAL_STEPS}")
    report_lines.append(f"  指数退避重试     : 初始{INITIAL_TIMEOUT}s, 翻倍{BACKOFF_FACTOR}x, 最多{MAX_RETRIES}次")
    report_lines.append(f"  翻转点快照窗口   : 前后各{FLIP_SNAPSHOT_WINDOW}步")
    report_lines.append("")

    # 候选区域识别结果
    report_lines.append("-" * 80)
    report_lines.append("2. 矛盾集中区域识别结果（粗扫阶段）")
    report_lines.append("-" * 80)
    report_lines.append(f"  识别方法: 相邻块语言特征变化度加权评分 "
                        f"(中文比例×{CHINESE_WEIGHT} + 词汇×{VOCAB_WEIGHT} + 标点×{PUNCTUATION_WEIGHT})")
    report_lines.append("")
    report_lines.append(f"  {'排名':<5} {'位置':<8} {'变化度':<10}")
    report_lines.append(f"  {'-' * 5} {'-' * 8} {'-' * 10}")
    for i, c in enumerate(candidates):
        report_lines.append(f"  {i+1:<5} {c['position']:<8} {c['score']:<10.4f}")
    report_lines.append("")

    # 完整多维数据表
    report_lines.append("-" * 80)
    report_lines.append("3. 完整多维数据表（U/D/A/H 四维全息记录）")
    report_lines.append("-" * 80)
    report_lines.append("")
    header = (f"  {'step':>5} {'pos':>6} "
              f"{'U':>7} {'ΔU':>7} "
              f"{'D':>7} {'ΔD':>7} "
              f"{'A':>7} {'ΔA':>7} "
              f"{'H':>7} "
              f"{'驱动':>4} {'ΣA':>7} {'σD':>7}")
    report_lines.append(header)
    report_lines.append(f"  {'-' * 5} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} "
                        f"{'-' * 7} {'-' * 7} {'-' * 7} {'-' * 4} {'-' * 7} {'-' * 7}")
    for r in trajectory:
        line = (f"  {r['step']:>5} {r['char_pos']:>6} "
                f"{r['U']:>7.3f} {r['U_diff']:+7.3f} "
                f"{r['D']:>7.3f} {r['D_diff']:+7.3f} "
                f"{r['A']:>7.3f} {r['A_diff']:+7.3f} "
                f"{r['H']:>7.3f} "
                f"{r['H_driven_by']:>4} {r['cumul_A']:>7.3f} {r['D_volatility']:>7.4f}")
        report_lines.append(line)
    report_lines.append("")

    # 翻转点附近高分辨率快照
    report_lines.append("-" * 80)
    report_lines.append("4. 翻转点附近高分辨率快照（基于A值峰值定位，仅作展示范围）")
    report_lines.append("-" * 80)
    report_lines.append("")
    if flip_step is not None:
        report_lines.append(f"  翻转点步号: {flip_step}")
        report_lines.append(f"  展示范围: 步 {max(0, flip_step - FLIP_SNAPSHOT_WINDOW)} "
                            f"至 步 {min(len(trajectory) - 1, flip_step + FLIP_SNAPSHOT_WINDOW)}")
        report_lines.append("")
        report_lines.append(header)
        report_lines.append(f"  {'-' * 5} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} "
                            f"{'-' * 7} {'-' * 7} {'-' * 7} {'-' * 4} {'-' * 7} {'-' * 7}")
        start = max(0, flip_step - FLIP_SNAPSHOT_WINDOW)
        end = min(len(trajectory), flip_step + FLIP_SNAPSHOT_WINDOW + 1)
        for r in trajectory[start:end]:
            marker = " <-- 翻转点" if r['step'] == flip_step else ""
            line = (f"  {r['step']:>5} {r['char_pos']:>6} "
                    f"{r['U']:>7.3f} {r['U_diff']:+7.3f} "
                    f"{r['D']:>7.3f} {r['D_diff']:+7.3f} "
                    f"{r['A']:>7.3f} {r['A_diff']:+7.3f} "
                    f"{r['H']:>7.3f} "
                    f"{r['H_driven_by']:>4} {r['cumul_A']:>7.3f} {r['D_volatility']:>7.4f}{marker}")
            report_lines.append(line)
    else:
        report_lines.append("  数据点不足，无法识别翻转点")
    report_lines.append("")

    # H值归因分解汇总
    report_lines.append("-" * 80)
    report_lines.append("5. H值归因分解汇总（关系本体论：维度间耦合关系）")
    report_lines.append("-" * 80)
    report_lines.append("")
    driver_counts = {}
    for r in trajectory:
        key = r['H_driven_by']
        driver_counts[key] = driver_counts.get(key, 0) + 1
    total = len(trajectory)
    report_lines.append(f"  {'驱动维度':<8} {'步数':<8} {'占比':<10}")
    report_lines.append(f"  {'-' * 8} {'-' * 8} {'-' * 10}")
    for key in ['U', 'D', 'A', '均衡']:
        if key in driver_counts:
            count = driver_counts[key]
            pct = count / total * 100
            report_lines.append(f"  {key:<8} {count:<8} {pct:<10.1f}%")
    for key, count in driver_counts.items():
        if key not in ['U', 'D', 'A', '均衡']:
            pct = count / total * 100
            report_lines.append(f"  {key:<8} {count:<8} {pct:<10.1f}%")
    report_lines.append("")

    # 数据范围摘要
    report_lines.append("-" * 80)
    report_lines.append("6. 数据范围摘要")
    report_lines.append("-" * 80)
    report_lines.append("")
    if trajectory:
        u_values = [r['U'] for r in trajectory]
        d_values = [r['D'] for r in trajectory]
        a_values = [r['A'] for r in trajectory]
        h_values = [r['H'] for r in trajectory]
        report_lines.append(f"  U值范围: [{min(u_values):.3f}, {max(u_values):.3f}]")
        report_lines.append(f"  D值范围: [{min(d_values):.3f}, {max(d_values):.3f}]")
        report_lines.append(f"  A值范围: [{min(a_values):.3f}, {max(a_values):.3f}]")
        report_lines.append(f"  H值范围: [{min(h_values):.3f}, {max(h_values):.3f}]")
        report_lines.append(f"  A累积值范围: [{min(r['cumul_A'] for r in trajectory):.3f}, "
                            f"{max(r['cumul_A'] for r in trajectory):.3f}]")
        report_lines.append(f"  D波动率范围: [{min(r['D_volatility'] for r in trajectory):.4f}, "
                            f"{max(r['D_volatility'] for r in trajectory):.4f}]")
    report_lines.append("")

    # 零判定附注
    report_lines.append("-" * 80)
    report_lines.append("7. 附注（零判定声明）")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append("  本报告仅记录真实数据，不包含任何判定或结论。")
    report_lines.append("  所有阈值和判定标准均应由观察者基于数据自行定义。")
    report_lines.append("")
    report_lines.append("  - 翻转点位置仅用于确定展示范围，不作为异常判定")
    report_lines.append("  - H值归因分解仅呈现数据关系，不做因果断言")
    report_lines.append("  - A值累积速率和D值波动率为衍生观测指标，不设阈值")
    report_lines.append("  - 不输出\"渐进衰减\"或\"状态切换\"等二元判定标签")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("报告结束")
    report_lines.append("=" * 80)

    # 输出到控制台
    print("\n" + "\n".join(report_lines))

    # 保存到文件
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"\n[OK] 分析报告已保存至 {OUTPUT_REPORT}")


# ========================================
# 主函数
# ========================================

def main(input_file=None):
    """主执行函数"""
    input_file = input_file or DEFAULT_INPUT_FILE

    print("=" * 80)
    print("Test B 多维U/D/A/H轨迹追踪实验 V3 Pure")
    print("零预设 · 零判定 · 纯记录")
    print("=" * 80)

    # API健康检查
    print("\n[检查] ThinkCheck API健康状态...")
    if not validate_api_health():
        print("[FAIL] ThinkCheck API不可用，请先启动服务")
        print("       启动命令: cd thinkcheck-agent-v6 && python api.py")
        sys.exit(1)
    print("[OK] ThinkCheck API状态正常")

    # 读取输入文件
    print(f"\n[读取] 输入文件: {input_file}")
    if not os.path.exists(input_file):
        print(f"[FAIL] 文件不存在: {input_file}")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        full_text = f.read()

    print(f"[OK] 读取完成，文本长度: {len(full_text)} 字符")

    # 阶段1: 粗扫
    candidates, blocks = coarse_scan(full_text)

    # 阶段2: 生成采样点
    sample_points = generate_sample_points(full_text, candidates)

    # 阶段3: 多维数据采集
    trajectory = collect_multi_dim_data(full_text, sample_points)

    if not trajectory:
        print("\n[FAIL] 未采集到有效数据，程序退出")
        sys.exit(1)

    print(f"\n[OK] 采集完成，共 {len(trajectory)} 个数据点")

    # 翻转点检测（仅用于展示范围）
    flip_step = detect_flip_point(trajectory)

    # 输出文件
    print("\n=== 输出文件 ===")
    write_csv(trajectory)
    write_json(trajectory)
    write_report(trajectory, candidates, flip_step, input_file)

    print("\n" + "=" * 80)
    print("采集与记录完成")
    print("=" * 80)


if __name__ == "__main__":
    input_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(input_file_arg)
