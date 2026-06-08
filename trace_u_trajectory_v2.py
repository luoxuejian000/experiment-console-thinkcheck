#!/usr/bin/env python3
"""
Test B U值轨迹追踪实验 V2.0 - 矛盾驱动自适应采样版

分析方法：基于矛盾动力论的多层次采样策略
- 第一阶段：粗扫识别矛盾集中区域
- 第二阶段：精细采样关键区域
- 翻转点 = U值波动最大区域内A值峰值位置
- 分段独立请求确保API响应稳定

理论依据：
- 矛盾动力论：矛盾(A值)是系统演化的能量源，翻转点周围应分配最多计算资源
- 关系本体论：采样策略依据文本自身特征动态调整
- 实践介入论：所有参数可配置、可审计，输出透明化

工程质量：
- 参数集中管理、完善日志、异常处理、多重输出格式
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
OUTPUT_JSON = "test_b_trajectory.json"
OUTPUT_CSV = "test_b_u_trajectory.csv"
OUTPUT_REPORT = "test_b_analysis_report.txt"

# 采样策略参数
SEGMENT_LENGTH = 2000        # 每次API请求的文本片段长度（防止超时）
COARSE_BLOCK_SIZE = 1000     # 粗扫阶段的块大小
FINE_GRAIN_STEP = 100        # 精细采样步长
SPARSE_GRAIN_STEP = 500      # 稀疏采样步长
NUM_CANDIDATE_REGIONS = 5    # 候选区域数量
MAX_TOTAL_STEPS = 60         # 最大采样步数

# 重试机制参数
MAX_RETRIES = 3              # 最大重试次数
INITIAL_TIMEOUT = 30         # 初始超时时间(秒)
BACKOFF_FACTOR = 2           # 超时翻倍因子
RETRY_DELAY = 1              # 重试间隔(秒)

# 语言特征变化检测参数
CHINESE_WEIGHT = 0.5         # 中文比例变化权重
VOCAB_WEIGHT = 0.3           # 词汇重叠率权重
PUNCTUATION_WEIGHT = 0.2     # 标点密度权重

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
    # 中文比例差异
    chinese1 = count_chinese_ratio(block1)
    chinese2 = count_chinese_ratio(block2)
    chinese_diff = abs(chinese1 - chinese2)
    
    # 词汇重叠率
    vocab1 = extract_vocab(block1)
    vocab2 = extract_vocab(block2)
    vocab_sim = jaccard_similarity(vocab1, vocab2)
    vocab_diff = 1 - vocab_sim
    
    # 标点密度差异
    punc1 = count_punctuation_density(block1)
    punc2 = count_punctuation_density(block2)
    punc_diff = abs(punc1 - punc2)
    
    # 加权综合评分（越高表示变化越大）
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
        except requests.exceptions.RequestException as e:
            print(f"  重试 {attempt+1}/{MAX_RETRIES} - 错误: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                timeout *= BACKOFF_FACTOR
    return None

def evaluate_segment(step, text, segment_pos, total_length):
    """评估单个文本片段（带重试机制）"""
    try:
        response = exponential_backoff_retry(
            requests.post,
            THINKCHECK_API,
            json={"document": text, "domain": "general"}
        )
        if response is None:
            print(f"步骤 {step}: 采样位置 {segment_pos}/{total_length} - 评估失败（重试耗尽）")
            return None
        
        data = response.json()
        u = data.get('u', 0.0)
        d = data.get('d', 0.0)
        a = data.get('a', 0.0)
        h = data.get('h', 0.0)
        
        print(f"步骤 {step}: 采样位置 {segment_pos}/{total_length} - U={u:.3f} D={d:.3f} A={a:.3f} H={h:.3f}")
        return {
            "step": step,
            "position": segment_pos,
            "U": u,
            "D": d,
            "A": a,
            "H": h,
            "text_length": len(text)
        }
    except Exception as e:
        print(f"步骤 {step}: 采样位置 {segment_pos}/{total_length} - 评估失败: {e}")
        return None

def validate_api_health():
    """验证ThinkCheck API健康状态"""
    try:
        response = requests.get(f"{THINKCHECK_API.replace('/evaluate', '/health')}", timeout=10)
        if response.status_code == 200 and response.json().get("status") == "healthy":
            return True
    except Exception as e:
        print(f"API健康检查失败: {e}")
    return False

# ========================================
# 核心采样逻辑
# ========================================

def coarse_scan(full_text):
    """第一阶段：粗扫识别矛盾集中区域"""
    print("\n=== 第一阶段：粗扫识别矛盾集中区域 ===")
    
    blocks = []
    change_scores = []
    
    # 将文本分块
    for i in range(0, len(full_text), COARSE_BLOCK_SIZE):
        block = full_text[i:i+COARSE_BLOCK_SIZE]
        blocks.append(block)
    
    print(f"全文长度: {len(full_text)} 字符")
    print(f"分块数量: {len(blocks)} (每块 {COARSE_BLOCK_SIZE} 字符)")
    
    # 计算相邻块之间的变化度
    for i in range(len(blocks) - 1):
        score, details = calculate_change_score(blocks[i], blocks[i+1])
        change_scores.append({
            'index': i,
            'position': (i + 1) * COARSE_BLOCK_SIZE,
            'score': score,
            'details': details
        })
        print(f"  块 {i+1} -> {i+2}: 变化度={score:.4f}")
    
    # 按变化度排序，选取前N个候选区域
    change_scores.sort(key=lambda x: x['score'], reverse=True)
    candidates = change_scores[:NUM_CANDIDATE_REGIONS]
    
    print(f"\n识别到的 {NUM_CANDIDATE_REGIONS} 个高变化区域:")
    for i, candidate in enumerate(candidates):
        print(f"  {i+1}. 位置 {candidate['position']}, 变化度 {candidate['score']:.4f}")
    
    return candidates, blocks

def generate_sample_points(full_text, candidates):
    """生成最终采样点列表"""
    print("\n=== 生成采样点 ===")
    
    sample_points = set()
    total_length = len(full_text)
    
    # 在候选区域周围进行精细采样
    for candidate in candidates:
        region_start = max(0, candidate['position'] - COARSE_BLOCK_SIZE)
        region_end = min(total_length, candidate['position'] + COARSE_BLOCK_SIZE)
        
        # 精细采样：100字符步长
        for pos in range(region_start, region_end, FINE_GRAIN_STEP):
            sample_points.add(pos)
    
    # 在整个文本范围内进行稀疏采样（补充候选区域外的点）
    for pos in range(0, total_length, SPARSE_GRAIN_STEP):
        sample_points.add(pos)
    
    # 添加文本末尾
    sample_points.add(total_length - SEGMENT_LENGTH)
    sample_points.add(total_length)
    
    # 过滤无效点并排序
    valid_points = sorted([p for p in sample_points if 0 <= p <= total_length - 100])
    
    # 如果采样点太多，进行均匀采样
    if len(valid_points) > MAX_TOTAL_STEPS:
        step = len(valid_points) // MAX_TOTAL_STEPS
        valid_points = valid_points[::step]
    
    # 确保首尾都有采样点
    if 0 not in valid_points:
        valid_points.insert(0, 0)
    if (total_length - SEGMENT_LENGTH) not in valid_points:
        valid_points.append(total_length - SEGMENT_LENGTH)
    
    # 去重并排序
    valid_points = sorted(list(set(valid_points)))
    
    print(f"候选区域精细采样点: {len(sample_points)} 个")
    print(f"最终采样点数量: {len(valid_points)} 个")
    
    return valid_points

def adaptive_flip_point_detection(trajectory):
    """自适应翻转点检测（基于矛盾动力论）"""
    if len(trajectory) < 4:
        return len(trajectory) // 2
    
    u_values = [t["U"] for t in trajectory]
    
    # 计算U值的差分序列，找到变化最大的区间
    u_diffs = np.abs(np.diff(u_values))
    
    if len(u_diffs) > 3:
        # 找到U值波动最大的3个候选点
        candidate_indices = np.argsort(u_diffs)[-3:]
        # 在这些候选点中，选择A值最高的那个作为翻转点
        a_values = [trajectory[i]["A"] for i in candidate_indices if i < len(trajectory)]
        if a_values:
            best_candidate_idx = candidate_indices[np.argmax(a_values)]
            flip_point = trajectory[best_candidate_idx]["step"]
        else:
            flip_point = len(trajectory) // 2
    else:
        flip_point = len(trajectory) // 2
    
    return flip_point

# ========================================
# 主函数
# ========================================

def main(input_file=None):
    """主执行函数"""
    input_file = input_file or DEFAULT_INPUT_FILE
    
    print("=" * 70)
    print("Test B U值轨迹追踪实验 V2.0")
    print("分析方法：矛盾驱动自适应采样")
    print("=" * 70)
    
    # 验证API健康状态
    print("\n验证 ThinkCheck API 健康状态...")
    if not validate_api_health():
        print("[FAIL] ThinkCheck API 不可用，请先启动服务")
        print("   启动命令: cd thinkcheck-agent-v6 && python api.py")
        sys.exit(1)
    print("[OK] ThinkCheck API 状态正常")
    
    # 读取输入文件
    print(f"\n读取输入文件: {input_file}")
    if not os.path.exists(input_file):
        print(f"[FAIL] 文件不存在: {input_file}")
        sys.exit(1)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    print(f"[OK] 读取完成，文本长度: {len(full_text)} 字符")
    
    # 第一阶段：粗扫
    candidates, blocks = coarse_scan(full_text)
    
    # 第二阶段：生成采样点
    sample_points = generate_sample_points(full_text, candidates)
    
    # 第三阶段：执行采样
    print("\n=== 第三阶段：执行采样 ===")
    trajectory = []
    step = 0
    total_length = len(full_text)
    
    for pos in sample_points:
        # 截取固定长度的文本片段
        end_pos = min(pos + SEGMENT_LENGTH, total_length)
        segment = full_text[pos:end_pos]
        
        # 跳过过短的片段
        if len(segment) < 100:
            continue
        
        result = evaluate_segment(step, segment, pos, total_length)
        if result:
            trajectory.append(result)
        step += 1
    
    # 保存JSON结果
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 轨迹数据已保存至 {OUTPUT_JSON}")
    
    # 保存CSV结果
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'position', 'U', 'D', 'A', 'H', 'text_length'])
        for t in trajectory:
            writer.writerow([t['step'], t['position'], t['U'], t['D'], t['A'], t['H'], t['text_length']])
    print(f"[OK] CSV数据已保存至 {OUTPUT_CSV}")
    
    # 生成分析报告
    generate_report(trajectory, candidates)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)

def generate_report(trajectory, candidates):
    """生成分析报告"""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Test B U值轨迹追踪分析报告")
    report_lines.append("=" * 70)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 采样策略说明
    report_lines.append("1. 采样策略说明")
    report_lines.append("-" * 50)
    report_lines.append(f"   分段长度: {SEGMENT_LENGTH} 字符（独立请求，防止超时）")
    report_lines.append(f"   粗扫块大小: {COARSE_BLOCK_SIZE} 字符")
    report_lines.append(f"   精细采样步长: {FINE_GRAIN_STEP} 字符（高变化区域）")
    report_lines.append(f"   稀疏采样步长: {SPARSE_GRAIN_STEP} 字符（平稳区域）")
    report_lines.append(f"   候选区域数: {NUM_CANDIDATE_REGIONS}")
    report_lines.append("")
    
    # 采样统计
    report_lines.append("2. 采样统计")
    report_lines.append("-" * 50)
    report_lines.append(f"   总采样点: {len(trajectory)}")
    report_lines.append(f"   高变化候选区域: {len(candidates)} 个")
    if candidates:
        report_lines.append("   候选区域位置: " + ", ".join(str(c['position']) for c in candidates))
    report_lines.append("")
    
    # 自适应翻转点检测
    report_lines.append("3. 翻转点分析")
    report_lines.append("-" * 50)
    
    if trajectory:
        flip_point = adaptive_flip_point_detection(trajectory)
        
        pre_flip = [t["U"] for t in trajectory if t["step"] < flip_point]
        post_flip = [t["U"] for t in trajectory if t["step"] >= flip_point]
        
        report_lines.append(f"   翻转点位置: 第 {flip_point} 步")
        report_lines.append(f"   检测方法: 自动检测 — A值峰值定位（基于矛盾动力论）")
        
        if pre_flip and post_flip:
            pre_mean = np.mean(pre_flip)
            post_mean = np.mean(post_flip)
            report_lines.append(f"   翻转前U值均值: {pre_mean:.3f}")
            report_lines.append(f"   翻转后U值均值: {post_mean:.3f}")
            report_lines.append(f"   U值变化: {post_mean - pre_mean:+.3f}")
            
            if len(pre_flip) > 1:
                pre_trend = np.polyfit(range(len(pre_flip)), pre_flip, 1)[0]
                report_lines.append(f"   翻转前U值趋势斜率: {pre_trend:+.4f}")
                
                if pre_trend < -0.01:
                    conclusion = "渐进衰减型 —— 系统有预警信号，可在衰减阶段提前介入加固"
                else:
                    conclusion = "状态切换型 —— 系统无预警，需在架构层面设防火墙"
                report_lines.append(f"   结论: {conclusion}")
            else:
                report_lines.append("   数据点不足以计算翻转前趋势")
        else:
            report_lines.append("   数据点不足，无法进行翻转点分析")
    else:
        report_lines.append("   无有效轨迹数据")
    
    report_lines.append("")
    report_lines.append("4. 轨迹数据摘要")
    report_lines.append("-" * 50)
    if trajectory:
        report_lines.append(f"   U值范围: [{min(t['U'] for t in trajectory):.3f}, {max(t['U'] for t in trajectory):.3f}]")
        report_lines.append(f"   A值范围: [{min(t['A'] for t in trajectory):.3f}, {max(t['A'] for t in trajectory):.3f}]")
        report_lines.append(f"   D值范围: [{min(t['D'] for t in trajectory):.3f}, {max(t['D'] for t in trajectory):.3f}]")
        report_lines.append(f"   H值范围: [{min(t['H'] for t in trajectory):.3f}, {max(t['H'] for t in trajectory):.3f}]")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("报告结束")
    report_lines.append("=" * 70)
    
    # 输出到控制台
    print("\n" + "\n".join(report_lines))
    
    # 保存到文件
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"\n[OK] 分析报告已保存至 {OUTPUT_REPORT}")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(input_file)