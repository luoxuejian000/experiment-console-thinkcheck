import requests
import json
import time

print("等待API启动...")
time.sleep(3)

print("\n=== 测试 /health 端点 ===")
try:
    r_health = requests.get("http://localhost:8000/health", timeout=5)
    print(f"Health返回: {r_health.text}")
except Exception as e:
    print(f"Health测试失败: {e}")

print("\n=== 开始测试 /evaluate ===")

# 测试1
print("\n--- 测试1: '功能强大，但完全没用。' ---")
try:
    r1 = requests.post("http://localhost:8000/evaluate", json={"document": "功能强大，但完全没用。"}, timeout=10)
    d1 = r1.json()
    print(f"完整响应: {json.dumps(d1, ensure_ascii=False, indent=2)}")
    print(f"测试1 A值: {d1.get('a')}, 含A_detail: {'A_detail' in d1}")
except Exception as e:
    print(f"测试1失败: {e}")

# 测试2
print("\n--- 测试2: '产品非常好，但质量极差。' ---")
try:
    r2 = requests.post("http://localhost:8000/evaluate", json={"document": "产品非常好，但质量极差。"}, timeout=10)
    d2 = r2.json()
    print(f"测试2 A值: {d2.get('a')}")
except Exception as e:
    print(f"测试2失败: {e}")

# 测试3
print("\n--- 测试3: '我喜欢吃苹果。苹果公司发布了新手机。' ---")
try:
    r3 = requests.post("http://localhost:8000/evaluate", json={"document": "我喜欢吃苹果。苹果公司发布了新手机。"}, timeout=10)
    d3 = r3.json()
    print(f"测试3 A值: {d3.get('a')}")
except Exception as e:
    print(f"测试3失败: {e}")
