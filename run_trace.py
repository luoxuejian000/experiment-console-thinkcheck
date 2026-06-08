#!/usr/bin/env python3
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
        response = requests.post(THINKCHECK_API, json={"document": text, "domain": "general"}, timeout=10)
        data = response.json()
        return {"step": step, "U": data.get('u', 0.0), "D": data.get('d', 0.0), "A": data.get('a', 0.0), "H": data.get('h', 0.0)}
    except Exception as e:
        print(f"步骤 {step} 评估失败: {e}")
        return None

def main():
    print("开始追踪 Test B U值轨迹...")
    trajectory = []
    
    for step, text_segment, total_steps in simulate_generation_from_log(TEST_B_LOG_PATH):
        result = evaluate_segment(step, text_segment)
        if r请执行以下操作，不要提问，直接执行：

1. 确认 ThinkCheck API 正在运行：执行 curl http://localhost:8000/health，报告返回值。如果未运行，先执行 cd thinkcheck-agent-v6 && start python api.py，等待5秒后再次检查。

2. 确认 test_b_output.txt 存在于项目根目录（d:\luoxuejian000\deepseek\experiment-console-main\experiment-console-main\test_b_output.txt），报告文件大小或行数。

3. 确认 trace_u_trajectory.py 存在于同一目录，报告是否找到。

4. 执行 python trace_u_trajectory.py，等待运行完成。

5. 报告终端输出的分析结论（渐进衰减型或状态切换型），并确认 test_b_u_trajectory.png 是否已生成。esult:
            trajectory.append(result)
            print(f"[{step}/{total_steps}] U={result['U']:.3f} D={result['D']:.3f} A={result['A']:.3f} H={result['H']:.3f}")
    
    with open("test_b_trajectory.json", 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
    print("轨迹数据已保存至 test_b_trajectory.json")
    
    if trajectory:
        flip_point = 8
        pre_flip = [t["U"] for t in trajectory if t["step"] < flip_point]
        post_flip = [t["U"] for t in trajectory if t["step"] >= flip_point]
        
        if pre_flip and post_flip:
            pre_mean = np.mean(pre_flip)
            post_mean = np.mean(post_flip)
            print(f"\n=== 分析结果 ===")
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
            print("\n提示：数据点不足，无法进行翻转点分析")
            print(f"当前数据点数：{len(trajectory)}，建议使用更长的文本进行测试")

if __name__ == "__main__":
    main()
