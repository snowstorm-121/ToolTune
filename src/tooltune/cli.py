import argparse
import json
from pathlib import Path
import runpy

from .validation import validate_records, verify_contract


def main():
    parser=argparse.ArgumentParser(description='ToolTune local experiment commands')
    parser.add_argument('command',choices=['prepare','validate','smoke'])
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    if args.command=='validate':
        print(json.dumps({'hashed_files_verified':verify_contract(root),
                          'selected_records':validate_records(root)},indent=2))
    else:
        runpy.run_path(str(root/'scripts'/f'{args.command}.py'),run_name='__main__')


if __name__=='__main__':main()
