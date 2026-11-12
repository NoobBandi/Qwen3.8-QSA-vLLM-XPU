"""Block top-k on XPU against torch.topk, on the score shapes the indexer produces.

vllm-xpu-kernels' top_k_per_row_decode returns duplicate indices -- dropping
genuine candidates -- when a row has many equal scores or a group far above
the rest. The indexer's own _topk must be exact on all of them.

    python3 tests/test_topk.py
"""
import sys
import time

import torch

from qwen38_qsa.ops import qsa_indexer as QI
from vllm import _custom_ops as ops

torch.manual_seed(0)
dev = "xpu"
K = 512                      # 2048-token budget in 4-token blocks


def cases(cols):
    narrow = 10 + 0.05 * torch.randn(4, cols, device=dev)
    yield "normal", torch.randn(4, cols, device=dev)
    yield "narrow", narrow.clone()
    yield "narrow, bf16 ties", narrow.bfloat16().float()
    bonus = narrow.clone()
    bonus[:, -64:] += 1e30
    yield "narrow + 64 far above", bonus


def errors(logits, visible, idx):
    """Rows whose selection is not a top-k multiset of the visible scores."""
    bad = 0
    for r in range(logits.shape[0]):
        v = int(visible[r]); k = min(K, v)
        got = idx[r, :k].long()
        if (got < 0).any() or (got >= v).any() or got.unique().numel() != k or (idx[r, k:] != -1).any():
            bad += 1
            continue
        ref = torch.topk(logits[r, :v], k).values.sort().values
        bad += not torch.equal(logits[r, got].sort().values, ref)
    return bad


failed = 0
for cols in (8192, 34000):
    for name, logits in cases(cols):
        visible = torch.tensor([cols, cols - 1000, 300, cols // 2], dtype=torch.int32, device=dev)
        out = torch.full((4, K), -7, dtype=torch.int32, device=dev)
        ops.top_k_per_row_decode(logits.clone(), 1, visible, out, 4, cols, 1, K)
        stock = errors(logits, visible, out)
        out.fill_(-7)
        QI._topk(logits.clone(), visible, K * 4, 4, out, None)
        ours = errors(logits, visible, out)
        failed += ours
        print(f"{cols:6d} cols, {name:22s}: stock kernel {stock}/4 rows wrong, qwen38_qsa {ours}/4")

# Prefill-sized batches take the fast kernel plus a repair of duplicate rows.
for name, base in (("narrow + 64 far above", 10 + 0.05 * torch.randn(256, 34000, device=dev)),
                   ("narrow, bf16 ties", (10 + 0.05 * torch.randn(256, 34000, device=dev)).bfloat16().float())):
    if "far" in name:
        base[:, -64:] += 1e30
    visible = torch.full((256,), 34000, dtype=torch.int32, device=dev)
    out = torch.full((256, K), -7, dtype=torch.int32, device=dev)
    QI._topk(base.clone(), visible, K * 4, 4, out, None)
    wrong = errors(base, visible, out)
    failed += wrong
    print(f" 34000 cols, 256 rows, {name:22s}: qwen38_qsa {wrong}/256 rows wrong")

for rows in (16, 1024):
    logits = 10 + 0.05 * torch.randn(rows, 34000, device=dev)
    visible = torch.full((rows,), 34000, dtype=torch.int32, device=dev)
    out = torch.empty((rows, K), dtype=torch.int32, device=dev)
    for fn, name in ((lambda: ops.top_k_per_row_decode(logits, 1, visible, out, rows, 34000, 1, K), "stock"),
                     (lambda: QI._topk(logits, visible, K * 4, 4, out, None), "qwen38_qsa")):
        fn(); torch.xpu.synchronize(); t = time.perf_counter()
        for _ in range(10): fn()
        torch.xpu.synchronize()
        print(f"{rows:5d} rows x 34000: {name:10s} {(time.perf_counter() - t) / 10 * 1e3:.3f} ms")
sys.exit(1 if failed else 0)
