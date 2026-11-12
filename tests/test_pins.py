"""Tool-call pinning (qwen38_qsa.pins) on CPU: which blocks get the bonus.

    python3 tests/test_pins.py
"""
import sys

import torch

from qwen38_qsa import pins

R, ROWS = 4, 208            # tokens per block, compressed rows per page
PAGE = R * ROWS             # tokens per KV page (832)
TOOL_CALL = 248058

pins.CAP = 6
pins.STEP = torch.zeros(4096, dtype=torch.uint8)
pins.SLOTS = torch.zeros(pins._SLOT_CAPACITY + 1, dtype=torch.uint8)
pins.TOKEN_IDS = torch.tensor([TOOL_CALL])

# One request on pages 7 and 3 (logical page 0 -> physical 7, 1 -> 3).
page_table = torch.tensor([[7, 3]])
columns = 2 * ROWS


def write(tokens, logical_positions):
    """Mark a step's tokens and store them where the main KV slot mapping puts them."""
    pins.mark_tokens(torch.tensor(tokens))
    pos = torch.tensor(logical_positions)
    slots = page_table[0, pos // PAGE] * PAGE + pos % PAGE
    pins.write_slots(slots)


failed = 0


def check(name, got, want):
    global failed
    ok = got == want
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (want {want})"))


def pinned():
    bonus = pins.block_bonus(page_table, ROWS, columns, R)
    return (bonus[0] > 0).nonzero().flatten().tolist()


# A <tool_call> at token 41 (block 10) pins blocks 10, 11, 12.
write([1, TOOL_CALL, 2], [40, 41, 42])
check("one call", pinned(), [10, 11, 12])

# A second call on the second page (token 900 -> block 225): both windows.
write([TOOL_CALL], [900])
check("two calls", pinned(), [10, 11, 12, 225, 226, 227])

# A third call: CAP=6 keeps only the newest six pinned blocks.
write([TOOL_CALL], [1200])
check("cap keeps the newest", pinned(), [225, 226, 227, 300, 301, 302])

# A rejected draft token at 1200 is overwritten by the accepted one: no pin.
write([5], [1200])
check("overwritten token", pinned(), [10, 11, 12, 225, 226, 227])

# Padding (slot -1) lands in the sink, not in a real slot.
pins.mark_tokens(torch.tensor([TOOL_CALL, TOOL_CALL]))
pins.write_slots(torch.tensor([-1, -1]))
check("padding ignored", pinned(), [10, 11, 12, 225, 226, 227])

sys.exit(1 if failed else 0)
