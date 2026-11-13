"""vLLM's XPU flash attention follows a KV block stride only in multiples of 64 rows.

A paged cache whose blocks are followed by other data (QSA packs its index keys
beside each block's K/V) is a valid strided view, and Triton's unified_attention
reads it correctly. flash_attn_varlen_func on XPU (vllm-xpu-kernels 0.1.17.1)
reads it correctly only when the block stride is a multiple of 64 token rows;
any other stride is read wrongly, with no error (strides of 14,600, 14,608 and
14,624 rows fail; 14,592, 14,656 and 14,720 work). A stride that is not a whole
number of token rows raises "page stride must be divisible by K sequence stride".

The workaround the plugin uses: align the block stride to 64 token rows and
show flash the cache as 64-token blocks laid end to end, gaps included, with
the block table translated (block b -> b * stride/64 + 0..12).

Expected on vllm/vllm-openai-xpu:v0.31.0 with vllm-xpu-kernels 0.1.17.1:
17 or 18 K/V sizes apart: ~3e-3 (correct); QSA's 14,586 rows: ~0.6; aligned
to 64 rows: ~3e-3, directly and through the 64-token view.

    python3 repro/xpu_flash_block_stride.py
"""
import torch
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

torch.manual_seed(0)
dev = "xpu"
HQ, HK, D, BS, SUB = 12, 2, 256, 832, 64
ROW = HK * 2 * D                      # elements per token row (K and V of every head)
PAGE = BS * ROW
NB, SEQ, QLEN = 8, 5000, 4


def attend(k, v, bt, q, flash):
    out = torch.empty_like(q)
    common = dict(q=q, k=k, v=v, out=out, cu_seqlens_q=torch.tensor([0, QLEN], device=dev, dtype=torch.int32),
                  max_seqlen_q=QLEN, seqused_k=torch.tensor([SEQ], device=dev, dtype=torch.int32),
                  max_seqlen_k=SEQ, softmax_scale=D ** -0.5, causal=True, block_table=bt)
    if flash:
        flash_attn_varlen_func(**common)
    else:
        unified_attention(**common, window_size=(-1, -1), softcap=0, q_descale=None, k_descale=None, v_descale=None)
    return out


for label, stride in (("17 K/V sizes", 17 * PAGE), ("18 K/V sizes", 18 * PAGE),
                      ("17.53 K/V sizes (QSA)", 14936064), ("QSA, aligned to 64 rows", 14942208)):
    raw = torch.randn(NB * stride, device=dev).to(torch.bfloat16)
    kv = torch.as_strided(raw, (NB, BS, HK, 2 * D), (stride, ROW, 2 * D, 1))
    kc, vc = kv.split(D, dim=-1)
    bt = torch.randperm(NB, device=dev)[: (SEQ + BS - 1) // BS].to(torch.int32).view(1, -1)
    q = torch.randn(QLEN, HQ, D, device=dev).to(torch.bfloat16)
    ref = attend(kc, vc, bt, q, flash=False)
    err = lambda o: ((o.float() - ref.float()).norm() / ref.float().norm()).item()
    line = f"{label:24s} flash {err(attend(kc, vc, bt, q, flash=True)):.2e}"
    if stride % (SUB * ROW) == 0:
        per, pieces = stride // (SUB * ROW), BS // SUB
        shape, strides = ((NB - 1) * per + pieces, SUB, HK, D), (SUB * ROW, ROW, 2 * D, 1)
        vbt = (bt[:, :, None] * per + torch.arange(pieces, device=dev, dtype=torch.int32)).view(1, -1)
        line += f"   64-token view {err(attend(kc.as_strided(shape, strides), vc.as_strided(shape, strides), vbt, q, flash=True)):.2e}"
    print(line)
