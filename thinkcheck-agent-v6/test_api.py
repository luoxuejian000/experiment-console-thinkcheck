import traceback
try:
    exec(open('api.py').read())
except Exception as e:
    traceback.print_exc()