"""
Test B U值轨迹追踪实验
分析方法：自适应翻转点检测（基于矛盾动力论）
- 翻转点 = U值波动最大区域内A值峰值对应的位置
- 翻转点不再固定为第8步，而是从数据中动态计算
- A值峰值标志着矛盾张力的释放时刻，对应漂移发生的物理学机制
"""

import json
import requests
import numpy as np
import os

THINKCHECK_API = "http://localhost:8000/evaluate"
TEST_B_LOG_PATH = "test_b_output.txt"
SLIDING_WINDOW_SIZE = 200

def simulate_generation_from_log(log_path):
    if not os.path.exists(log_path):
        print(f"日志文件 {log_path} 不存在")
        return
    with open(log_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    total_steps = len(full_text) // SLIDING_WINDOW_SIZE + 1
    for i in range(0, len(full_text), SLIDING_WINDOW_SIZE):
        segment = full_text[:i + SLIDING_WINDOW_SIZE]
        step = i // SLIDING_WINDOW_SIZE
        yield step, segment, total_steps

def evaluate_segment(step, text):
    try:
        response = requests.post(THINKCHECK_API, json={"document": text, "domain": "general"}, timeout=120)
        data = response.json()
        return {"step": step, "U": data.get('u', 0.0), "D": data.get('d', 0.0), "A": data.get('a', 0.0), "H": data.get('h', 0.0)}
    except Exception as e:
        print(f"步骤 {step} 评估失败: {e}")
        return None

trajectory = []
print("开始追踪 Test B U值轨迹...")

for step, text_segment, total_steps in simulate_generation_from_log(TEST_B_LOG_PATH):
    result = evaluate_segment(step, text_segment)
    if result:
        trajectory.append(result)
        print(f"[{step}/{total_steps}] U={result['U']:.3f} D={result['D']:.3f} A={result['A']:.3f} H={result['H']:.3f}")

with open("test_b_trajectory.json", 'w', encoding='utf-8') as f:
    json.dump(trajectory, f, indent=2, ensure_ascii=False)
print("轨迹数据已保存至 test_b_trajectory.json")

if trajectory:
    steps = [t["step"] for t in trajectory]
    u_values = [t["U"] for t in trajectory]
    
    # 矛盾动力论：翻转点 = A值累积张力释放的峰值位置
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
    
    pre_flip = [t["U"] for t in trajectory if t["step"] < flip_point]
    post_flip = [t["U"] for t in trajectory if t["step"] >= flip_point]
    if pre_flip and post_flip:
        pre_mean = np.mean(pre_flip)
        post_mean = np.mean(post_flip)
        print(f"\n=== 分析结果 ===")
        print(f"翻转点位置: 第 {flip_point} 步 (来源: 自动检测 — A值峰值定位)")
        print(f"翻转前U值均值: {pre_mean:.3f}")
        print(f"翻转后U值均值: {post_mean:.3f}")
        print(f"U值变化: {post_mean - pre_mean:+.3f}")
        if len(pre_flip) > 1:
            pre_trend = np.polyfit(range(len(pre_flip)), pre_flip, 1)[0]
            print(f"翻转前U值趋势斜率: {pre_trend:+.4f}")
            if pre_trend < -0.01:
                print("结论：渐进衰减型 —— 系统有预警信号，可在衰减阶段提前介入加固")
            else:
                print("结论：状态切换型 —— 系统无预警，需在架构层面设防火墙")
        else:
            print("数据点不足以计算翻转前趋势")
    else:
        print("\n提示：数据点不足，无法进行翻转点分析")
        print(f"当前数据点数：{len(trajectory)}，建议使用更长的文本进行测试")
