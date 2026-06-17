import subprocess
import time
import urllib.request
import os

def check_api():
    try:
        response = urllib.request.urlopen('http://localhost:8000/health', timeout=5)
        return response.read().decode('utf-8')
    except Exception as e:
        return f"API not running: {str(e)}"

def start_api():
    try:
        proc = subprocess.Popen(
            ['python', 'api.py'],
            cwd='d:/luoxuejian000/deepseek/experiment-console-main/experiment-console-main/thinkcheck-agent-v6',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(5)
        return proc
    except Exception as e:
        return None, f"Failed to start API: {str(e)}"

def count_lines(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        return len(lines)
    return 0

def run_trace_script():
    script_path = 'd:/luoxuejian000/deepseek/experiment-console-main/experiment-console-main/trace_u_trajectory_v3_pure.py'
    if os.path.exists(script_path):
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            cwd='d:/luoxuejian000/deepseek/experiment-console-main/experiment-console-main'
        )
        return result.stdout, result.stderr
    return None, "Script not found"

def main():
    print("=== Step 1: Check ThinkCheck API ===")
    api_status = check_api()
    print(f"API Status: {api_status}")
    
    if "not running" in api_status:
        print("Starting API...")
        proc = start_api()
        if proc:
            time.sleep(3)
            api_status = check_api()
            print(f"After restart: {api_status}")
    
    print("\n=== Step 2: Check test_b_output.txt ===")
    lines = count_lines('d:/luoxuejian000/deepseek/experiment-console-main/experiment-console-main/test_b_output.txt')
    print(f"test_b_output.txt has {lines} lines")
    
    print("\n=== Step 3: Run trace_u_trajectory_v3_pure.py ===")
    stdout, stderr = run_trace_script()
    if stdout:
        print("STDOUT:")
        print(stdout)
    if stderr:
        print("STDERR:")
        print(stderr)

if __name__ == "__main__":
    main()