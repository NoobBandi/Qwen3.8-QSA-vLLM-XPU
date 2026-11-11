"""Decode speed and MTP acceptance at fixed context lengths (greedy, 3 runs each)."""
import json
import urllib.request

from common import args, chars_per_token, fit, stream

a = args(__doc__, "2000,64000,230000")


def spec():
    text = urllib.request.urlopen(a.url + "/metrics", timeout=30).read().decode()
    get = lambda key: sum(float(l.split()[-1]) for l in text.splitlines() if l.startswith(key))
    return get("vllm:spec_decode_num_drafts_total"), get("vllm:spec_decode_num_accepted_tokens_total")


cpt = chars_per_token(a)
for n in a.lengths:
    prompt = fit(a, n, cpt) + "\n\nWrite a detailed, structured summary of the text above."
    d0, a0 = spec()
    speeds = []
    for _ in range(3):
        usage, _, secs, out = stream(a, prompt, 400)
        speeds.append((usage["completion_tokens"] - 1) / secs)
    d1, a1 = spec()
    print(json.dumps({"prompt_tokens": usage["prompt_tokens"], "decode_tok_s": [round(x, 1) for x in speeds],
                      "accepted_per_step": round((a1 - a0) / max(d1 - d0, 1), 2)}), flush=True)
