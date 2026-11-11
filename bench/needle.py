"""Three passphrases hidden at 15%, 50% and 85% of a long context; can the model recall all three?"""
import random

from common import args, chars_per_token, stream

a = args(__doc__, "20000,112000,230000")
cpt = chars_per_token(a)
random.seed(7)
failed = 0
for n in a.lengths:
    text = a.text[: int(n * cpt)]
    codes = [f"{random.randint(1000, 9999)}-{random.choice(['blue-whale', 'firebird', 'green-pine', 'ginkgo'])}" for _ in range(3)]
    for i in reversed(range(3)):
        at = int(len(text) * (0.15 + 0.35 * i))
        text = text[:at] + f"\n// SECRET-{i}: the passphrase for vault {i} is {codes[i]}\n" + text[at:]
    q = "List the passphrases for vault 0, vault 1 and vault 2 exactly as written above, one per line."
    usage, ttft, secs, out = stream(a, text + "\n\n" + q, 200)
    hits = sum(c in out for c in codes)
    failed += hits < 3
    print(f"{usage['prompt_tokens']} tokens: {hits}/3 found ({ttft + secs:.0f}s) {out.strip()[:120]!r}", flush=True)
raise SystemExit(1 if failed else 0)
