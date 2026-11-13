# SPDX-License-Identifier: Apache-2.0
"""Exact top-k over each row's visible columns by radix select, for XPU decode.

``torch.topk`` over a decode row costs the same whatever the context: under a
captured graph the logits are as wide as the longest context the cache can
hold (65,728 columns at 256K), so 4 rows took about 0.15 ms in every one of the
17 QSA layers of a step, three quarters of QSA's per-step selection cost. This
reads only each row's visible columns and selects in five small kernels:

1. ``init`` clears the row's histograms and counters and fills its output with -1.
2. Three histogram passes narrow the k-th largest key down 11, 11 and 10 bits
   at a time (floats map to order-preserving 32-bit keys), each pass counting
   only the keys that share the bits found so far. Every program re-derives
   the earlier passes' thresholds from their histograms, so no pass waits on
   another program.
3. ``compact`` writes every key above the threshold, then as many keys equal to
   it as fill k: exactly k indices, unique, the same values torch.topk picks
   (ties are broken arbitrarily, as with ``sorted=False``). Rows with fewer than
   k visible columns get all of them, padded with -1.

The order of a row's indices is arbitrary. Before selecting, ``bonus`` (one
row per request, ``rows_per_request`` query rows each) is added to the scores,
then 1e30 to each row's newest ``recent`` visible columns, in the order and
precision the unfused ``logits.add_`` calls used; the logits stay unmodified.

Each program adds its histogram's non-empty bins only (a few dozen of 2048),
which is what makes this fast: adding every bin cost 5x more. Triton on XPU
drops a masked ``atomic_add`` whose mask derives straight from
``tl.histogram``'s result, so the histogram goes through the program's own
scratch row first.
"""

import torch

from vllm.triton_utils import tl, triton

_B1, _B2, _B3 = 2048, 2048, 1024
_WORK = _B1 + _B2 + _B3 + 8          # three histograms, then the two counters
_CHUNK = 4096


@triton.jit
def _keys(logits_ptr, stride_row, row, columns, visible, bonus_ptr, stride_bonus,
          ROWS_PER_REQUEST: tl.constexpr, BONUS: tl.constexpr, RECENT: tl.constexpr, recent):
    valid = columns < visible
    x = tl.load(logits_ptr + row * stride_row + columns, mask=valid, other=0.0)
    if BONUS:
        x += tl.load(bonus_ptr + (row // ROWS_PER_REQUEST) * stride_bonus + columns,
                     mask=valid, other=0.0)
    if RECENT:
        x = tl.where(columns >= visible - recent, x + 1.0e30, x)
    bits = x.to(tl.uint32, bitcast=True)
    key = tl.where((bits >> 31) != 0, ~bits, bits | 0x80000000)
    return key, valid


@triton.jit
def _threshold(hist_ptr, NBINS: tl.constexpr, wanted):
    """Highest bin b whose keys at or above it number >= wanted, and the count above b."""
    bins = tl.arange(0, NBINS)
    h = tl.load(hist_ptr + bins)
    at_or_above = tl.sum(h, 0) - tl.cumsum(h, 0) + h
    t = tl.max(tl.where(at_or_above >= wanted, bins, -1), 0)
    above = tl.sum(tl.where(bins > t, h, 0), 0)
    return t, above


@triton.jit
def _add_histogram(hist_ptr, scratch_ptr, bins, mask, NBINS: tl.constexpr):
    h = tl.histogram(bins, NBINS, mask=mask)
    b = tl.arange(0, NBINS)
    tl.store(scratch_ptr + b, h)
    tl.debug_barrier()
    h = tl.load(scratch_ptr + b)
    tl.atomic_add(hist_ptr + b, h, mask=h > 0)


@triton.jit
def _init_kernel(work_ptr, out_ptr, stride_out, WORK: tl.constexpr, K: tl.constexpr,
                 K_PADDED: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, 8192)
    tl.store(work_ptr + row * WORK + i, tl.zeros([8192], tl.int32), mask=i < WORK)
    j = tl.arange(0, K_PADDED)
    tl.store(out_ptr + row * stride_out + j, tl.full([K_PADDED], -1, tl.int32), mask=j < K)


@triton.jit
def _pass_kernel(logits_ptr, stride_row, visible_ptr, work_ptr, scratch_ptr, out_ptr, stride_out,
                 bonus_ptr, stride_bonus, recent,
                 LEVEL: tl.constexpr, K: tl.constexpr, ROWS_PER_REQUEST: tl.constexpr,
                 BONUS: tl.constexpr, RECENT: tl.constexpr,
                 CHUNK: tl.constexpr, WORK: tl.constexpr,
                 B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr):
    row = tl.program_id(0)
    start = tl.program_id(1) * CHUNK
    visible = tl.load(visible_ptr + row)
    if start >= visible:
        return
    columns = start + tl.arange(0, CHUNK)
    key, valid = _keys(logits_ptr, stride_row, row, columns, visible, bonus_ptr, stride_bonus,
                       ROWS_PER_REQUEST, BONUS, RECENT, recent)
    h1 = work_ptr + row * WORK
    h2 = h1 + B1
    h3 = h2 + B2
    counters = h3 + B3
    scratch = scratch_ptr + (row * tl.num_programs(1) + tl.program_id(1)) * B1
    if LEVEL == 1:
        _add_histogram(h1, scratch, (key >> 21).to(tl.int32), valid, B1)
        return
    t1, above1 = _threshold(h1, B1, K)
    take_all = t1 < 0                              # fewer than K visible columns
    if LEVEL == 2:
        if take_all:
            return
        match = valid & ((key >> 21).to(tl.int32) == t1)
        _add_histogram(h2, scratch, ((key >> 10) & 2047).to(tl.int32), match, B2)
        return
    want2 = K - above1
    t2, above2 = _threshold(h2, B2, want2)
    prefix2 = (t1.to(tl.uint32) << 11) | t2.to(tl.uint32)
    if LEVEL == 3:
        if take_all:
            return
        match = valid & ((key >> 10) == prefix2)
        _add_histogram(h3, scratch, (key & 1023).to(tl.int32), match, B3)
        return
    # LEVEL 4: compact.
    want3 = want2 - above2
    t3, above3 = _threshold(h3, B3, want3)
    threshold = (prefix2 << 10) | t3.to(tl.uint32)
    greater_needed = above1 + above2 + above3
    if take_all:
        greater = valid
        equal = valid & False
    else:
        greater = valid & (key > threshold)
        equal = valid & (key == threshold)
    g = greater.to(tl.int32)
    base = tl.atomic_add(counters, tl.sum(g, 0))
    slot = base + tl.cumsum(g, 0) - 1
    tl.store(out_ptr + row * stride_out + slot, columns, mask=greater)
    e = equal.to(tl.int32)
    base_e = tl.atomic_add(counters + 1, tl.sum(e, 0))
    rank = base_e + tl.cumsum(e, 0) - 1
    tl.store(out_ptr + row * stride_out + greater_needed + rank, columns,
             mask=equal & (rank < K - greater_needed))


def radix_topk(logits: torch.Tensor, visible: torch.Tensor, k: int, out: torch.Tensor,
               recent: int = 0, bonus: torch.Tensor | None = None,
               rows_per_request: int = 1) -> None:
    """Write each row's top-k column indices among its first ``visible[row]`` columns.

    ``logits`` [rows, columns] float32 (row stride free, columns contiguous) is
    not modified. ``out`` [rows, k] int32 gets the indices in arbitrary order,
    -1 where a row has fewer than k visible columns. Graph-capturable.
    """
    rows, columns = logits.shape
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert out.shape == (rows, k) and out.dtype == torch.int32 and out.stride(1) == 1
    assert visible.shape[0] >= rows and k <= 8192
    if bonus is not None:
        assert bonus.dtype == torch.float32 and bonus.stride(1) == 1
        assert bonus.shape[0] * rows_per_request >= rows and bonus.shape[1] >= columns
    if rows == 0:
        return
    grid = (rows, triton.cdiv(columns, _CHUNK))
    work = torch.empty((rows, _WORK), dtype=torch.int32, device=logits.device)
    scratch = torch.empty((rows, grid[1], _B1), dtype=torch.int32, device=logits.device)
    _init_kernel[(rows,)](work, out, out.stride(0), WORK=_WORK, K=k,
                          K_PADDED=triton.next_power_of_2(k), num_warps=4)
    for level in (1, 2, 3, 4):
        _pass_kernel[grid](logits, logits.stride(0), visible, work, scratch, out, out.stride(0),
                           logits if bonus is None else bonus,
                           0 if bonus is None else bonus.stride(0), recent,
                           LEVEL=level, K=k, ROWS_PER_REQUEST=rows_per_request,
                           BONUS=bonus is not None, RECENT=bool(recent), CHUNK=_CHUNK, WORK=_WORK,
                           B1=_B1, B2=_B2, B3=_B3, num_warps=8)
