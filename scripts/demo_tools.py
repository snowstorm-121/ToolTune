"""Run the safe local executor on fixed calls; this does not invoke the model."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tooltune.demo_tools import TOOLS, execute
from tooltune.metrics import parse_prediction


def main():
    calls=['<tool_call>{"name":"sum_integers","arguments":{"values":[12,30]}}</tool_call>',
           '<tool_call>{"name":"text_length","arguments":{"text":"你好A"}}</tool_call>']
    for raw in calls:
        parsed=parse_prediction(raw)
        print(json.dumps({'prediction':parsed,'result':execute(parsed)},ensure_ascii=False))
    print('local_tool_names:',', '.join(t['function']['name'] for t in TOOLS))


if __name__=='__main__':main()
