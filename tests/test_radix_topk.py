"""radix_topk against torch.topk on XPU: exact values, unique visible indices, -1 padding.

Also times both under XPU graph replay at decode shape (4 rows as wide as a
256K context).

    python3 tests/test_radix_topk.py
"""
import sys
import time

import torch

from qwen38_qsa.ops.radix_topk import radix_topk

dev = "xpu"
torch.manual_seed(0)
K = 512
failed = 0


def reference(logits, visible, k, recent, bonus=None, per=1):
    x = logits.clone()
    col = torch.arange(x.shape[1], device=dev)
    if bonus is not None:
        x.view(-1, per, x.shape[1]).add_(bonus[:, None, :])
    if recent:
        x += (col[None, :] >= visible[:, None] - recent) * 1.0e30
    x.masked_fill_(col[None, :] >= visible[:, None], float("-inf"))
    values, _ = torch.topk(x, min(k, x.shape[1]), dim=1)
    return x, values


def check(name, logits, visible, k=K, recent=0, bonus=None, per=1):
    global failed
    out = torch.full((logits.shape[0], k), 7, dtype=torch.int32, device=dev)
    before = logits.clone()
    radix_topk(logits, visible, k, out, recent=recent, bonus=bonus, rows_per_request=per)
    masked, want = reference(logits, visible, k, recent, bonus, per)
    problems = []
    if not torch.equal(before, logits):
        problems.append("logits modified")
    for r in range(logits.shape[0]):
        n = min(k, int(visible[r]))
        row = out[r]
        picked = row[:n]
        if (row[n:] != -1).any():
            problems.append(f"row {r}: no -1 padding")
        if (picked < 0).any() or (picked >= int(visible[r])).any():
            problems.append(f"row {r}: index out of range")
            continue
        if picked.unique().numel() != n:
            problems.append(f"row {r}: duplicate indices")
        got = masked[r, picked.long()].sort(descending=True).values
        if not torch.equal(got, want[r, :n]):
            problems.append(f"row {r}: values differ")
    ok = not problems
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f": {problems[:3]}"))


cols = 316 * 208
widths = (0, 1, 100, 511, 512, 513, 1024, 4096, 4097, 30001, cols)
for vis in widths:
    logits = torch.relu(torch.randn(4, cols, device=dev)) * 3
    check(f"relu scores, {vis} visible", logits, torch.full((4,), vis, dtype=torch.int32, device=dev))
visible = torch.tensor([1023, 1024, 1025, 1026], dtype=torch.int32, device=dev)
check("per-row visible (MTP rows)", torch.rand(4, cols, device=dev), visible)
check("all zero (every key ties)", torch.zeros(4, cols, device=dev), visible * 40)
check("eight levels (heavy ties)", torch.randint(0, 8, (4, cols), device=dev).float(), visible * 40)
x = 10 + 0.05 * torch.randn(4, cols, device=dev)
x[:, torch.randperm(cols, device=dev)[:192]] += 1e6
check("192 far above the rest", x, torch.full((4,), cols, dtype=torch.int32, device=dev))
check("signed, with -0.0", torch.randn(4, cols, device=dev) * torch.tensor([1.0, -0.0], device=dev).repeat(cols // 2),
      torch.full((4,), cols, dtype=torch.int32, device=dev))
check("recent 8 blocks", torch.rand(4, cols, device=dev), visible * 30, recent=8)
check("recent, short rows", torch.rand(4, cols, device=dev), torch.tensor([3, 9, 600, 0], dtype=torch.int32, device=dev), recent=8)
pinned = torch.zeros(2, cols, device=dev)
pinned[0, 100:140] = 1.0e4
pinned[1, 5000:5020] = 1.0e4
check("tool-call bonus, 2 requests x 4 rows", torch.rand(8, cols, device=dev),
      torch.tensor([900, 901, 902, 903, 7000, 7001, 7002, 7003], dtype=torch.int32, device=dev),
      bonus=pinned, per=4)
check("bonus and recent", torch.rand(8, cols, device=dev),
      torch.tensor([900, 901, 902, 903, 7000, 7001, 7002, 7003], dtype=torch.int32, device=dev),
      recent=8, bonus=pinned, per=4)
check("k 64", torch.rand(16, 9000, device=dev), torch.randint(0, 9000, (16,), dtype=torch.int32, device=dev), k=64)
strided = torch.rand(8, cols + 100, device=dev)[:, :cols]
check("strided rows, 300 rows", torch.rand(300, 20000, device=dev), torch.randint(0, 20000, (300,), dtype=torch.int32, device=dev))
check("strided rows", strided, torch.full((8,), cols, dtype=torch.int32, device=dev))


def timed(fn, n=200):
    fn()
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    for _ in range(10):
        g.replay()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


out = torch.empty(4, K, dtype=torch.int32, device=dev)
logits = torch.relu(torch.randn(4, cols, device=dev))
warm = torch.randn(4096, 4096, device=dev)
for _ in range(200):                      # an idle GPU's clock ramps up first
    warm @ warm
for ctx in (262000, 65536, 4096):
    vis = torch.full((4,), ctx // 4, dtype=torch.int32, device=dev)
    ours = timed(lambda: radix_topk(logits, vis, K, out))
    theirs = timed(lambda: torch.topk(logits, K, dim=1, sorted=False))
    print(f"     {ctx // 1024}K context, graph replay: radix {ours:.3f} ms, torch.topk {theirs:.3f} ms")

print("all passed" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
