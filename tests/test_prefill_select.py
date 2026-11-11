"""Prefill block selection (scorer + top-k) on XPU against torch, and its time.

    docker run --rm --device /dev/dri -v "$PWD:/w" -w /w --entrypoint python3 \
        qwen3.8-qsa-vllm-xpu:v0.31.0 tests/test_prefill_select.py [contexts]
"""
import sys
import time

import torch

from qwen38_qsa.ops.qsa_indexer import qsa_select_paged_prefill

torch.manual_seed(0)
dev = "xpu"
R, TOPK, PAGE, H, D = 4, 2048, 832, 4, 128
CPAGE = PAGE // R
ROWS = 4096           # one --max-num-batched-tokens chunk
failed = 0
for ctx in [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "16384,131072").split(",")]:
    pages = (ctx + PAGE - 1) // PAGE + 1
    bt = torch.randperm(pages + 4, device=dev)[:pages].to(torch.int32).view(1, -1)
    cache = torch.randn(pages + 4, CPAGE, 1, D, device=dev).to(torch.bfloat16)
    q = torch.randn(ROWS, H, D, device=dev).to(torch.bfloat16)
    pos = torch.arange(ctx - ROWS, ctx, device=dev)
    visible = ((pos + 1) // R).to(torch.int32)
    qsl = torch.tensor([0, ROWS], device=dev, dtype=torch.int32)
    out = torch.empty(ROWS, TOPK // R, dtype=torch.int32, device=dev)
    f = lambda: qsa_select_paged_prefill(q, cache, bt, qsl, visible, TOPK, R, ROWS, out, ctx)
    f()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(5):
        f()
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t) / 5 * 1e3
    b = torch.arange(ctx // R, device=dev)
    keys = cache[bt[0, b // CPAGE].long(), b % CPAGE, 0].float()
    checked = range(0, ROWS, 257)
    bad = 0
    for r in checked:
        v = int(visible[r])
        s = torch.relu(q[r].float() @ keys[:v].T).sum(0)
        k = min(v, TOPK // R)
        thr = s.topk(k).values[-1]
        got = out[r, :k].long()
        if (got < 0).any() or (s[got] < thr - 1e-3 * thr.abs()).any() or got.unique().numel() != got.numel():
            bad += 1
    failed += bad
    print(f"ctx {ctx}: {ROWS} rows in {ms:.2f} ms, {len(checked) - bad}/{len(checked)} checked rows correct")
sys.exit(1 if failed else 0)
