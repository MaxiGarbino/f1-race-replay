import py_compile
import sys

try:
    py_compile.compile('src/f1_data.py', doraise=True)
    print('f1_data.py: OK')
except py_compile.PyCompileError as e:
    print(f'f1_data.py: FAIL - {e}')
    sys.exit(1)

try:
    py_compile.compile('src/interfaces/race_replay.py', doraise=True)
    print('race_replay.py: OK')
except py_compile.PyCompileError as e:
    print(f'race_replay.py: FAIL - {e}')
    sys.exit(1)

print('All files OK!')
