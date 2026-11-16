"""Guard existing experiment inputs before invoking a historical runner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tooltune.input_guard import freeze, snapshot, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'verify', 't3', 't4'])
    parser.add_argument('stage', nargs='?')
    args = parser.parse_args()
    stages = {'t3': {'preflight', 'baseline', 'train'},
              't4': {'freeze', 'baseline', 'adapter'}}
    if args.command in stages:
        if args.stage not in stages[args.command]:
            parser.error('invalid or missing stage')
    elif args.stage is not None:
        parser.error('freeze/verify take no stage')
    current = snapshot(ROOT)
    path = ROOT/'artifacts/input-guard-v1.json'
    if args.command == 'freeze':
        freeze(path, current)
    else:
        verify(path, current)
        if args.command in stages:
            subprocess.run([sys.executable, str(ROOT/'scripts'/f'{args.command}.py'), args.stage],
                           cwd=ROOT, check=True)
    print(json.dumps({'status': 'passed', 'command': args.command, 'guard': str(path)}))


if __name__ == '__main__':
    main()
