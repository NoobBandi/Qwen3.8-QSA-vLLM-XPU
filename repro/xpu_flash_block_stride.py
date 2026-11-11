"""vLLM's XPU flash attention ignores the KV cache's block stride.

A paged cache whose blocks are followed by other data (here a gap after each
block's K/V, as QSA packs its side caches beside them) is a valid strided
view, and Triton's unified_attention reads it correctly. flash_attn_varlen_func
on XPU indexes blocks as if they were contiguous and returns wrong output.

Expected on vllm/vllm-openai-xpu:v0.31.0 (vllm-xpu-kernels 0.1.15 or 0.1.17.1):
gap 0 is correct for both; with a gap, flash has ~50% relative error, Triton ~3e-3.

    python3 repro/xpu_flash_block_stride.py
"""
import torch
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

torch.manual_seed(0)
dev = "xpu"
P, KVH, PAGE, D, NH = 40, 2, 832, 256, 12
seq = 5000
for gap in (0, 4096):
    for qlen in (1, 4, 300):
        raw = torch.randn(P, KVH * PAGE * 2 * D + gap, device=dev).to(torch.bfloat16)
        kv = raw[:, : KVH * PAGE * 2 * D].view(P, KVH, PAGE, 2 * D)   # block stride includes the gap
        kc, vc = kv.transpose(1, 2).split(D, dim=-1)                 # [P, PAGE, KVH, D] views
        bt = torch.randperm(P, device=dev)[: (seq + PAGE - 1) // PAGE].to(torch.int32).view(1, -1)
        q = torch.randn(qlen, NH, D, device=dev).to(torch.bfloat16)
        cu = torch.tensor([0, qlen], device=dev, dtype=torch.int32)
        used = torch.tensor([seq], device=dev, dtype=torch.int32)
        out_f = torch.empty_like(q)
        flash_attn_varlen_func(q=q, k=kc, v=vc, out=out_f, cu_seqlens_q=cu, max_seqlen_q=qlen, seqused_k=used,
                               max_seqlen_k=seq, softmax_scale=D ** -0.5, causal=True, block_table=bt)
        out_t = torch.empty_like(q)
        unified_attention(q=q, k=kc, v=vc, out=out_t, cu_seqlens_q=cu, max_seqlen_q=qlen, seqused_k=used,
                          max_seqlen_k=seq, softmax_scale=D ** -0.5, causal=True, window_size=(-1, -1),
                          block_table=bt, softcap=0, q_descale=None, k_descale=None, v_descale=None)
        tok = torch.arange(seq, device=dev)
        pg = bt[0, tok // PAGE].long()
        k = kc[pg, tok % PAGE].float()
        v = vc[pg, tok % PAGE].float()
        s = torch.einsum("qghd,ngd->qghn", q.float().view(qlen, KVH, NH // KVH, D), k) * D ** -0.5
        qpos = seq - qlen + torch.arange(qlen, device=dev)
        s = s.masked_fill(tok[None, None, None, :] > qpos[:, None, None, None], float("-inf"))
        ref = torch.einsum("qghn,ngd->qghd", s.softmax(-1), v).reshape(qlen, NH, D)
        err = lambda o: ((o.float() - ref).abs().max() / ref.abs().max()).item()
        print(f"gap {gap:5d} qlen {qlen:3d}: flash {err(out_f):.2e}  triton {err(out_t):.2e}")
