"""The MTP draft's FP8 head (qwen38_qsa.draft_head) against the BF16 head, on XPU.

Two tensor-parallel ranks are simulated on one card: each builds its head
from its half of a random lm_head, rows are exchanged as the all-reduce
would, the cards' logits are concatenated as the all-gather would, and the result is checked against the full BF16 product:
kept tokens close to it, every other token -inf, and the same top token as
the BF16 head restricted to the kept tokens.

    python3 tests/test_draft_head.py
"""
import os
import sys
import time
import types

import torch

os.environ["QWEN38_QSA_DRAFT_VOCAB"] = "full"
import vllm._xpu_ops  # noqa: F401,E402  (registers fp8_gemm_w8a16)
import vllm.distributed as dist  # noqa: E402

from qwen38_qsa import draft_head  # noqa: E402

dev = "xpu"
torch.manual_seed(0)
V, H, WORLD = 248320, 5120, 2
failed = 0


def check(name, ok, detail=""):
    global failed
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f" {detail}" if detail else ""))


weight = (torch.randn(V, H, device=dev) * 0.02).to(torch.bfloat16)
hidden = torch.randn(4, H, device=dev).to(torch.bfloat16)
reference = (hidden.float() @ weight.float().t())
processor = types.SimpleNamespace(org_vocab_size=V)


def shard(rank):
    return weight[rank * V // WORLD:(rank + 1) * V // WORLD]


def fetch(_weight, ids, _start, per_rank):
    # What the all-reduce returns: the sum of every rank's owned rows.
    return sum(draft_head.DraftHead.owned_rows(shard(r), ids, r * per_rank, per_rank) for r in range(WORLD))


draft_head.DraftHead._fetch = staticmethod(fetch)


def build(kept, mode):
    draft_head.MODE = mode
    heads = []
    for rank in range(WORLD):
        lm_head = types.SimpleNamespace(weight=shard(rank), shard_indices=types.SimpleNamespace(
            org_vocab_start_index=rank * V // WORLD))
        dist.get_tensor_model_parallel_rank = lambda r=rank: r
        dist.get_tensor_model_parallel_world_size = lambda: WORLD
        heads.append(draft_head.DraftHead(lm_head, processor, kept))
    return heads


def run(heads):
    gathered = torch.cat([h.local(hidden) for h in heads], dim=-1)   # what all_gather returns
    return heads[0].assemble(gathered)


for mode in ("fp8", "bf16"):
    for label, kept in (("all tokens", torch.arange(V)),
                        ("147,457 kept, most from the first shard", torch.cat([torch.arange(0, 120000),
                         torch.randperm(V - 120000)[:27457] + 120000]).sort().values)):
        heads = build(kept, mode)
        logits = run(heads)
        mask = torch.zeros(V, dtype=torch.bool, device=dev)
        mask[kept.to(dev)] = True
        ref = reference.masked_fill(~mask, float("-inf"))
        err = ((logits.float()[:, mask] - ref[:, mask]).norm() / ref[:, mask].norm()).item()
        check(f"{mode}, {label}: kept logits", err < (0.05 if mode == "fp8" else 0.01), f"(rel err {err:.4f})")
        check(f"{mode}, {label}: others -inf", bool(torch.isneginf(logits[:, ~mask]).all()))
        check(f"{mode}, {label}: top token", torch.equal(logits.float().argmax(1), ref.argmax(1)))

check("rows split evenly", heads[0].rows == heads[1].rows == 73729, f"({heads[0].rows} per card)")
heads = build(torch.arange(147456), "fp8")
for fn, name in ((lambda: hidden @ weight[: V // 2].t(), "bf16 head, one card"),
                 (lambda: heads[0].local(hidden), "fp8 head over 147,456 tokens, one card")):
    fn(); torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(50):
        fn()
    torch.xpu.synchronize()
    print(f"     {name}: {(time.perf_counter() - t) / 50 * 1e3:.3f} ms")
print("all passed" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
