import subprocess
import sys

result = subprocess.run(
    [sys.executable, '-m', 'uvicorn', 'api:app', '--host', '0.0.0.0', '--port', '8000'],
    capture_output=True,
    text=True,
    timeout=10
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print("\nReturn code:", result.returncode)