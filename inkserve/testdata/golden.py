"""Render testdata/cases.json with the model's own chat template through transformers, into golden.json.

Run on the Mac with a Python that has transformers (omlx's venv):
    ~/omlx-eval/venv/bin/python golden.py MODEL_DIR cases.json > golden.json
Tool-call arguments arrive as JSON strings (the OpenAI wire form); the template wants objects, so they are parsed
first, keeping key order.
"""
import json
import sys
from collections import OrderedDict

from transformers import AutoTokenizer

model_dir, cases_path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
cases = json.load(open(cases_path), object_pairs_hook=OrderedDict)
out = []
for c in cases:
    msgs = json.loads(json.dumps(c["messages"]), object_pairs_hook=OrderedDict)
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"], object_pairs_hook=OrderedDict)
    text = tok.apply_chat_template(msgs, tools=c.get("tools"), tokenize=False, reasoning_effort=c["effort"])
    out.append({"name": c["name"], "prompt": text})
json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
