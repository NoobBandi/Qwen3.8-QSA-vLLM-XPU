"""Decode-path kernels on XPU against a torch reference: block selection
(scorer + top-k + expand) and the attn2 sparse attention, plus their time.

The reference follows the release's modeling_qsa.py: a block's score is the
sum over indexer heads of relu(q . k); the query attends the top budget/4
blocks plus its own (partial) block.

    python3 tests/test_kernels.py [seq_lens]
"""
import math
import os
import sys
import time

import torch
import triton

from qwen38_qsa.ops.qsa_attn2 import qsa_sparse_paged_attention_attn2
from qwen38_qsa.ops.qsa_indexer import expand_qsa_block_indices, qsa_select_paged_decode

torch.manual_seed(0)
dev = "xpu"
R = 4                 # tokens per block
TOPK = 2048           # token budget
PAGE = 832            # KV page, tokens
CPAGE = PAGE // R     # compressed rows per page
H_IDX, D_IDX = 4, 128 # indexer heads, dims
NH, NKV, D = 12, 2, 256   # attention heads per card at TP=2
DQL = int(os.environ.get("DQL", "4"))   # query rows per request (MTP3 = 4)

seq_lens = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "9000,1500,2049,30001").split(",")]
max_pages = max(math.ceil(s / PAGE) for s in seq_lens) + 2
num_pages = max_pages * len(seq_lens) + 3
block_table = torch.randperm(num_pages)[: len(seq_lens) * max_pages].view(len(seq_lens), max_pages).to(torch.int32).to(dev)

ccache = torch.randn(num_pages, CPAGE, 1, D_IDX, device=dev).to(torch.bfloat16)
row_req = [r for r in range(len(seq_lens)) for _ in range(DQL)]
row_pos = [s - DQL + i for s in seq_lens for i in range(DQL)]
nrows = len(row_req)
q_idx = torch.randn(nrows, H_IDX, D_IDX, device=dev).to(torch.bfloat16)
token_to_req = torch.tensor(row_req, dtype=torch.int32, device=dev)
positions = torch.tensor(row_pos, dtype=torch.int64, device=dev)
visible = torch.minimum((positions + 1) // R, torch.tensor(seq_lens, device=dev)[token_to_req.long()] // R).to(torch.int32)

# ---------- selection ----------
block_indices = torch.empty(nrows, TOPK // R, dtype=torch.int32, device=dev)
sel = torch.empty(nrows, TOPK + R + 1, dtype=torch.int32, device=dev)   # tokens..., count


def select():
    qsa_select_paged_decode(q_idx, ccache, block_table, visible, TOPK, R, DQL, block_indices)
    expand_qsa_block_indices(block_indices, positions, visible, R, TOPK, sel)


select()
torch.xpu.synchronize()
bad = 0
for row in range(nrows):
    r, p = row_req[row], row_pos[row]
    nb = int(visible[row])
    b = torch.arange(nb, device=dev)
    keys = ccache[block_table[r, b // CPAGE].long(), b % CPAGE, 0].float()
    scores = torch.relu(q_idx[row].float() @ keys.T).sum(0)
    kb = min(nb, TOPK // R)
    thr = scores.topk(kb).values[-1].item() if kb else 0.0
    own = p // R
    want = {t for blk in scores.topk(kb).indices.tolist() for t in range(blk * R, blk * R + R)} if kb else set()
    want |= set(range(own * R, p + 1))
    n = int(sel[row, -1])
    got = sel[row, :n].tolist()
    ok = len(got) == len(set(got)) == len(want) and all(0 <= t <= p for t in got)
    # ties may pick different blocks: each chosen block must score >= the k-th
    ok = ok and all(scores[blk] >= thr - 1e-2 * abs(thr) for blk in {t // R for t in got} - {own})
    ok = ok and set(range(own * R, p + 1)) <= set(got)
    if not ok:
        bad += 1
        print(f"  row {row} (seq {seq_lens[r]}, pos {p}): {n} tokens, expected {len(want)}")
print(f"selection: {nrows - bad}/{nrows} rows correct")

# ---------- sparse attention (attn2) ----------
kv_cache = torch.randn(num_pages, NKV, PAGE, 2 * D, device=dev).to(torch.bfloat16)
key_cache, value_cache = kv_cache.transpose(1, 2).split(D, dim=-1)
q = torch.randn(nrows, NH, D, device=dev).to(torch.bfloat16)
gate = torch.randn(nrows, NH, D, device=dev).to(torch.bfloat16)
o = torch.empty_like(q)


def attend():
    qsa_sparse_paged_attention_attn2(q, key_cache, value_cache, sel, block_table, token_to_req, False, o,
                                     gate, D ** -0.5, 1.0, triton.next_power_of_2(NH // NKV))


attend()
torch.xpu.synchronize()
worst = 0.0
for row in range(nrows):
    r = row_req[row]
    toks = sel[row, : int(sel[row, -1])].long()
    page = block_table[r, toks // PAGE].long()
    k = key_cache[page, toks % PAGE].float()
    v = value_cache[page, toks % PAGE].float()
    att = torch.einsum("ghd,ngd->ghn", q[row].float().view(NKV, NH // NKV, D), k) * D ** -0.5
    ref = torch.einsum("ghn,ngd->ghd", att.softmax(-1), v).reshape(NH, D)
    ref = ref.to(torch.bfloat16).float() * torch.sigmoid(gate[row].float())
    worst = max(worst, ((o[row].float() - ref).abs().max() / ref.abs().max()).item())
print(f"attention: worst relative error {worst:.2e}")


def bench(name, fn, n=20):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    print(f"  {name}: {(time.perf_counter() - t) / n * 1e3:.3f} ms")


print(f"one layer, {nrows} rows:")
bench("select (score + top-k + expand)", select)
bench("attention", attend)
sys.exit(1 if bad or worst > 2e-2 else 0)
