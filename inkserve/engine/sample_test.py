"""Checks of `inkling serve`'s GPU sampler against its CPU sampler. Run on the Mac:

    python3 -I sample_test.py BINARY MODEL [DRAWS]

- the same seed gives the same reply; another seed gives another one
- temperature 0 gives the CPU path's greedy reply
- the first token's distribution over DRAWS seeds (default 400) matches the CPU sampler's, within sampling noise
- the thinking budget ends thinking at the same count

The CPU sampler runs with INKSERVE_CPU_SAMPLE=1. Standard library only.
"""
import collections
import math
import os
import subprocess
import sys

BIN, MODEL = sys.argv[1], sys.argv[2]
# INKSERVE_TEST_REF=drafter.gguf: run the engine with DSpark (DSPARK.md)
REF = ["--ref", os.environ["INKSERVE_TEST_REF"]] if os.environ.get("INKSERVE_TEST_REF") else []
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 400


class Engine:
    def __init__(self, cpu):
        env = dict(os.environ)
        if cpu:
            env["INKSERVE_CPU_SAMPLE"] = "1"
        self.p = subprocess.Popen([BIN, "serve", "-m", MODEL, "-c", "8192"] + REF, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        self.p.stdout.readline()

    def gen(self, prompt, **kw):
        body = prompt.encode()
        fields = " ".join(f"{k}={v}" for k, v in kw.items())
        self.p.stdin.write(f"G {fields} len={len(body)}\n".encode() + body)
        self.p.stdin.flush()
        toks, info = [], {}
        while True:
            line = self.p.stdout.readline().decode().strip()
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("E "):
                f = line.split()
                info = dict(x.split("=") for x in f[2:])
                info["reason"] = f[1]
                return toks, info

    def close(self):
        self.p.stdin.close()
        self.p.wait()


fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    fails += 0 if ok else 1


CODE = "<|message_user|><|content_text|>Write a Go function that reverses a slice of ints.<|end_message|><|message_model|>"
OPEN = "<|message_user|><|content_text|>Name one fruit. Just the name.<|end_message|><|message_model|><|content_text|>"

gpu, cpu = Engine(False), Engine(True)

a, _ = gpu.gen(CODE, temp=1.0, seed=11, max=200)
b, _ = gpu.gen(CODE, temp=1.0, seed=11, max=200)
c, _ = gpu.gen(CODE, temp=1.0, seed=12, max=200)
check("same seed, same reply", a == b, f"{len(a)} tokens")
check("another seed, another reply", a != c)

def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


g0, _ = gpu.gen(CODE, temp=0, max=300)
c0, _ = cpu.gen(CODE, temp=0, max=300)
d = first_diff(g0, c0)
if REF:
    # DSpark's verify pass computes logits in a batch, a little differently from a decode step, so greedy text can
    # part at a near-tie after a few hundred tokens
    check("temperature 0: DSpark = CPU greedy for 100 tokens", d is None or d >= 100, f"first difference at {d}")
else:
    check("temperature 0: GPU = CPU greedy", d is None, f"first difference at {d}")

# At a tiny temperature a draw is the argmax unless two logits are almost tied. Noise that can reach +infinity (a bug
# once: u rounded to 1.0) picks a random token on ~1% of steps, so 300 steps would almost surely differ from greedy.
for seed in (1, 2, 3):
    t, _ = gpu.gen(CODE, temp=0.01, seed=seed, max=300)
    d = first_diff(t, g0)
    if REF:
        # DSpark picks the draft depth per cycle, so this run and the greedy one verify in different batch sizes and
        # can part at a near-tie; the noise bug would part them within ~100 steps (all 3 seeds past 200: ~0.2%)
        check(f"temperature 0.01, seed {seed}: same as greedy for 200 tokens", d is None or d >= 200, f"first difference at {d}")
    else:
        check(f"temperature 0.01, seed {seed}: same as greedy", d is None, f"first difference at {d}")

# first-token distribution: total variation distance between the two samplers' histograms. Temperature 2 flattens the
# distribution, so many tokens take part and the temperature scaling is checked too.
for temp in (1.0, 2.0):
    hg, hc = collections.Counter(), collections.Counter()
    for s in range(DRAWS):
        hg[gpu.gen(OPEN, temp=temp, seed=1000 + s, max=1)[0][0]] += 1
        hc[cpu.gen(OPEN, temp=temp, seed=5000 + s, max=1)[0][0]] += 1
    keys = set(hg) | set(hc)
    tvd = 0.5 * sum(abs(hg[k] - hc[k]) for k in keys) / DRAWS
    # two independent samples of one distribution: E[TVD] is at most about sqrt(k / (pi * N)) for k categories in use
    noise = math.sqrt(len(keys) / (math.pi * DRAWS))
    top = sorted(keys, key=lambda k: -(hg[k] + hc[k]))[:6]
    check(f"first-token distribution matches, temperature {temp}", tvd < 1.5 * noise + 0.02,
          f"{len(keys)} tokens seen, TVD {tvd:.3f}, noise ~{noise:.3f}; top GPU/CPU: " + ", ".join(f"{k}:{hg[k]}/{hc[k]}" for k in top))

_, ig = gpu.gen(CODE, temp=1.0, seed=7, max=2000, budget=40)
_, ic = cpu.gen(CODE, temp=1.0, seed=7, max=2000, budget=40)
check("budget: GPU path stops thinking at 40", ig["think"] == "40", str(ig))
check("budget: CPU path stops thinking at 40", ic["think"] == "40", str(ic))
check("budget: both end cleanly", ig["reason"] == ic["reason"] == "stop", f"{ig['reason']} / {ic['reason']}")

gpu.close()
cpu.close()
print("FAILURES:", fails)
sys.exit(1 if fails else 0)
