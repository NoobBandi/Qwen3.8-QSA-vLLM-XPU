"""Cold prefill speed (prompt tokens / time to first token) at several context lengths."""
import json
import random

from common import args, chars_per_token, fit, stream

a = args(__doc__, "16000,64000,128000")
cpt = chars_per_token(a)
for n in a.lengths:
    # A fresh prefix each run, so the prefix cache cannot answer it.
    prompt = f"[run {random.random()}]\n" + fit(a, n, cpt) + "\n\nSummarise the text above in one sentence."
    usage, ttft, _, _ = stream(a, prompt, 32)
    p = usage["prompt_tokens"]
    print(json.dumps({"prompt_tokens": p, "ttft_s": round(ttft, 2), "prefill_tok_s": round(p / ttft, 1)}), flush=True)
