"""End-to-end checks against a running inkserve. Standard library only.

    python3 -I e2e.py http://127.0.0.1:8045

Covers plain and streamed replies, tool calls and the prefix cache across an agent-style turn, the thinking budget,
stop strings, a client that disconnects mid-stream, and errors.
"""
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = "inkling-erisinky"
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    fails += 0 if ok else 1


def post(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def stream(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(dict(body, stream=True)).encode(),
                                 {"Content-Type": "application/json"})
    chunks, done = [], False
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            chunks.append(json.loads(line[6:]))
    return chunks, done


SHELL = {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return its output.",
                                          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                                         "required": ["cmd"]}}}

# 1. plain reply
st, r = post({"model": MODEL, "messages": [{"role": "user", "content": "What is 17 times 23? Answer with the number only."}],
              "reasoning_effort": "low", "temperature": 0})
msg = r["choices"][0]["message"]
check("plain: status 200", st == 200, st)
check("plain: correct answer", msg["content"].strip() == "391", repr(msg["content"]))
check("plain: reasoning separate", bool(msg.get("reasoning_content")) and "391" not in msg["content"][:-3], msg.get("reasoning_content", "")[:60])
check("plain: finish stop", r["choices"][0]["finish_reason"] == "stop")
u = r["usage"]
check("plain: usage adds up", u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] and u["completion_tokens_details"]["reasoning_tokens"] > 0, u)

# 2. streamed reply
chunks, done = stream({"model": MODEL, "messages": [{"role": "user", "content": "Name three primary colors, comma-separated."}],
                       "reasoning_effort": "low", "stream_options": {"include_usage": True}})
content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content") or "" for c in chunks if c["choices"])
finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
check("stream: [DONE] received", done)
check("stream: role first", chunks[0]["choices"][0]["delta"].get("role") == "assistant")
check("stream: content and reasoning", "red" in content.lower() and len(reasoning) > 0, repr(content))
check("stream: one finish_reason, stop", finish == ["stop"], finish)
check("stream: usage chunk last", "usage" in chunks[-1] and chunks[-1]["choices"] == [], chunks[-1].get("usage"))

# 3. agent turn: tool call, then the result, with and without the reasoning sent back
convo = [{"role": "system", "content": "You are a coding agent. Use the run_shell tool to inspect the machine."},
         {"role": "user", "content": "How many files are in /etc? Use the tool; don't guess."}]
st, r = post({"model": MODEL, "messages": convo, "tools": [SHELL], "reasoning_effort": "low", "temperature": 0})
m = r["choices"][0]["message"]
calls = m.get("tool_calls") or []
check("tools: finish tool_calls", r["choices"][0]["finish_reason"] == "tool_calls", r["choices"][0]["finish_reason"])
check("tools: one run_shell call with a cmd", len(calls) >= 1 and calls[0]["function"]["name"] == "run_shell"
      and "cmd" in json.loads(calls[0]["function"]["arguments"]), calls)
prompt1 = r["usage"]["prompt_tokens"]
gen1 = r["usage"]["completion_tokens"]
if calls:
    result = {"role": "tool", "tool_call_id": calls[0]["id"], "content": "87"}
    # (a) without reasoning: the history differs from what was generated, so the snapshot covers the old prompt
    hist = convo + [{"role": "assistant", "content": m.get("content") or None, "tool_calls": calls}, result]
    st, r2 = post({"model": MODEL, "messages": hist, "tools": [SHELL], "reasoning_effort": "low", "temperature": 0})
    c2 = r2["usage"]["prompt_tokens_details"]["cached_tokens"]
    check("cache, reasoning dropped: old prompt reused", c2 >= prompt1 - 2, f"cached {c2} of {r2['usage']['prompt_tokens']}, first prompt {prompt1}")
    check("tools: answer uses the result", "87" in (r2["choices"][0]["message"]["content"] or ""), r2["choices"][0]["message"]["content"])
    # (b) the same turn again, with reasoning: the history now matches the generated tokens
    st, r1b = post({"model": MODEL, "messages": convo, "tools": [SHELL], "reasoning_effort": "low", "temperature": 0})
    m1b = r1b["choices"][0]["message"]
    hist = convo + [{"role": "assistant", "content": m1b.get("content") or None, "tool_calls": m1b.get("tool_calls"),
                     "reasoning_content": m1b.get("reasoning_content")},
                    {"role": "tool", "tool_call_id": (m1b.get("tool_calls") or [{"id": ""}])[0]["id"], "content": "87"}]
    st, r3 = post({"model": MODEL, "messages": hist, "tools": [SHELL], "reasoning_effort": "low", "temperature": 0})
    c3 = r3["usage"]["prompt_tokens_details"]["cached_tokens"]
    want = r1b["usage"]["prompt_tokens"] + r1b["usage"]["completion_tokens"]
    check("cache, reasoning sent back: whole context reused", c3 >= want - 1, f"cached {c3} of {r3['usage']['prompt_tokens']}, live {want}")

# 4. thinking budget
st, r = post({"model": MODEL, "messages": [{"role": "user", "content": "Prove that there are infinitely many primes, briefly."}],
              "reasoning_effort": "high", "thinking_budget": 64, "max_tokens": 1500})
u = r["usage"]
check("budget: reasoning capped", u["completion_tokens_details"]["reasoning_tokens"] <= 64, u["completion_tokens_details"])
check("budget: still answers", len(r["choices"][0]["message"]["content"]) > 40, r["choices"][0]["message"]["content"][:80])

# 5. stop string
st, r = post({"model": MODEL, "messages": [{"role": "user", "content": "Count from 1 to 10, one number per line, nothing else."}],
              "reasoning_effort": "none", "temperature": 0, "stop": ["5"]})
c = r["choices"][0]["message"]["content"]
check("stop: cut before the stop string", "5" not in c and "4" in c, repr(c))

# 6. client disconnects mid-stream; the next request still works
u = urllib.parse.urlparse(BASE)
conn = http.client.HTTPConnection(u.hostname, u.port, timeout=60)
conn.request("POST", "/v1/chat/completions", json.dumps({"model": MODEL, "stream": True, "max_tokens": 4000,
             "messages": [{"role": "user", "content": "Write a 2000-word story."}]}), {"Content-Type": "application/json"})
resp = conn.getresponse()
for _ in range(20):
    resp.fp.readline()
conn.close()
time.sleep(1)
t0 = time.time()
st, r = post({"model": MODEL, "messages": [{"role": "user", "content": "Say OK."}], "reasoning_effort": "none", "temperature": 0})
check("disconnect: next request served", st == 200 and "OK" in r["choices"][0]["message"]["content"], f"{time.time() - t0:.1f}s")

# 7. errors
st, r = post({"model": "nope", "messages": [{"role": "user", "content": "x"}]})
check("error: wrong model is 404", st == 404, st)
st, r = post({"model": MODEL, "messages": [{"role": "developer", "content": "x"}]})
check("error: bad role is 400", st == 400 and "role" in r["error"]["message"], r)
st, r = post({"model": MODEL, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})
check("error: image is 400", st == 400, st)

print("FAILURES:", fails)
sys.exit(1 if fails else 0)
