"""Protocol checks for `inkling serve`. Run on the Mac: python3 -I proto_test.py BINARY MODEL.

Checks the prefix cache's three paths (live context, snapshot, cold), that each gives the same greedy tokens as one
whole-prompt prefill, and the thinking budget.
"""
import os
import subprocess
import sys

BIN, MODEL = sys.argv[1], sys.argv[2]
# INKSERVE_TEST_REF=drafter.gguf: run the engine with DSpark (DSPARK.md)
REF = ["--ref", os.environ["INKSERVE_TEST_REF"]] if os.environ.get("INKSERVE_TEST_REF") else []


class Engine:
    def __init__(self, debug=""):
        env = dict(os.environ, INKSERVE_DEBUG=debug)
        self.p = subprocess.Popen([BIN, "serve", "-m", MODEL, "-c", "32768"] + REF, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        self.ready = self.p.stdout.readline().decode().strip()

    def gen(self, prompt, **kw):
        body = prompt.encode()
        fields = " ".join(f"{k}={v}" for k, v in kw.items())
        self.p.stdin.write(f"G {fields} len={len(body)}\n".encode() + body)
        self.p.stdin.flush()
        toks, text, info = [], b"", {}
        while True:
            line = self.p.stdout.readline().decode().strip()
            kind, rest = line[0], line[2:]
            if kind == "P":
                info["prefill"] = dict(x.split("=") for x in rest.split())
            elif kind == "T":
                tid, _, hx = rest.split()
                toks.append(int(tid))
                text += bytes.fromhex(hx)
            elif kind == "E":
                parts = rest.split()
                info["reason"] = parts[0]
                info.update(x.split("=") for x in parts[1:])
                return toks, text.decode("utf-8", "replace"), info
            else:
                raise RuntimeError(line)

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def turn1(effort):
    return ("<|message_system|><|content_text|>Thinking effort level: " + effort + "<|end_message|>"
            "<|message_user|><|content_text|>Write a Go function that reverses a slice of ints in place. "
            "Code only.<|end_message|>")


USER2 = "<|message_user|><|content_text|>Now make it generic over any element type.<|end_message|>"
G = dict(temp=0, max=400)
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    fails += 0 if ok else 1


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


# A: the client sends back exactly what was generated, so the live context is reused.
a = Engine()
print("ready:", a.ready)
t1, x1, i1 = a.gen(turn1("0.2"), **G)
check("cold prompt: nothing cached", i1["prefill"]["cached"] == "0", str(i1["prefill"]))
check("turn 1 ends with the stop token", i1["reason"] == "stop", i1["reason"])
p2 = turn1("0.2") + x1 + "<|content_model_end_sampling|>" + USER2
t2, _, i2 = a.gen(p2, **G)
# the engine keeps the stop token's state too (the step that drew it already ran), so it is part of the live context
live_n = int(i1["prefill"]["prompt"]) + len(t1) + (1 if os.environ.get("INKSERVE_CPU_SAMPLE") is None else 0)
check("live path: whole context reused", int(i2["prefill"]["cached"]) == live_n,
      f"cached {i2['prefill']['cached']}, expected {live_n}")
a.close()

# B: the client drops the reasoning from history, so only the snapshot (end of turn 1's prompt) matches.
b = Engine()
b.gen(turn1("0.2"), **G)
answer = x1.split("<|content_text|>")[-1].split("<|end_message|>")[0]
p3 = (turn1("0.2") + "<|message_model|><|content_text|>" + answer + "<|end_message|><|content_model_end_sampling|>"
      + USER2)
t3, _, i3 = b.gen(p3, **G)
snap_n = int(i1["prefill"]["prompt"]) - 2
check("snapshot path: prompt prefix reused", int(i3["prefill"]["cached"]) == snap_n,
      f"cached {i3['prefill']['cached']}, expected {snap_n}")
t3b, _, i3b = b.gen(p3, **G)
check("repeat: all but two tokens reused", int(i3b["prefill"]["cached"]) == int(i3b["prefill"]["prompt"]) - 2,
      str(i3b["prefill"]))
check("repeat: same greedy tokens", t3b == t3, f"first difference at {first_diff(t3, t3b)}")
_, _, i4 = b.gen("<|message_user|><|content_text|>Say hi.<|end_message|>", temp=0, max=50)
check("unrelated prompt: nothing cached", i4["prefill"]["cached"] == "0", str(i4["prefill"]))
tb, xb, ib = b.gen(turn1("0.9"), temp=1.0, seed=7, max=2000, budget=40)
check("budget: thinking capped at 40", ib["think"] == "40", f"think {ib['think']}")
check("budget: reply has text after thinking", "<|content_text|>" in xb.split("<|content_thinking|>")[-1], xb[-120:])
check("budget: no second thinking block", xb.count("<|content_thinking|>") == 1)
check("budget: reply ends cleanly", ib["reason"] == "stop", ib["reason"])
print("--- budget reply tail:\n" + xb[-200:])
b.close()

# C: reference, each prompt as one whole prefill in a fresh engine.
c = Engine("nosplit")
t3c, _, i3c = c.gen(p3, **G)
check("snapshot path = whole-prompt prefill (logit hash)", i3c["prefill"]["h"] == i3["prefill"]["h"] == i3b["prefill"]["h"],
      f"{i3['prefill']['h']} / repeat {i3b['prefill']['h']} vs {i3c['prefill']['h']}")
check("snapshot path = whole-prompt prefill (greedy tokens)", t3c == t3, f"first difference at {first_diff(t3, t3c)}")
c.close()
c = Engine("nosplit")
t1c, _, i1c = c.gen(turn1("0.2"), **G)
check("cold split prefill = whole-prompt prefill (logit hash)", i1c["prefill"]["h"] == i1["prefill"]["h"],
      f"{i1['prefill']['h']} vs {i1c['prefill']['h']}")
check("cold split prefill = whole-prompt prefill (greedy tokens)", t1c == t1, f"first difference at {first_diff(t1, t1c)}")
t2c, _, _ = c.gen(p2, **G)
d = first_diff(t2, t2c)
print(f"info: live path vs whole-prompt prefill of p2: {'identical' if d is None else f'first difference at token {d} of {len(t2)}'}"
      " (the live path ran turn 1's reply through decode steps, not prefill, so small differences can appear)")
c.close()
print("FAILURES:", fails)
sys.exit(1 if fails else 0)
