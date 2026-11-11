"""Shared bits of the benchmarks: arguments, the corpus, one streamed chat."""
import argparse
import json
import time
import urllib.request


def args(doc, lengths):
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL")
    ap.add_argument("--model", default="qwen38-27b-qsa")
    ap.add_argument("--corpus", default="corpus.txt", help="long text to fill the context (see README)")
    ap.add_argument("--lengths", default=lengths, help="context lengths in tokens, comma separated")
    a = ap.parse_args()
    a.lengths = [int(x) for x in a.lengths.split(",")]
    a.text = open(a.corpus, encoding="utf-8").read()
    return a


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def chars_per_token(a):
    sample = a.text[:200000]
    return len(sample) / json.load(post(a.url + "/tokenize", {"model": a.model, "prompt": sample}))["count"]


def stream(a, content, max_tokens, **extra):
    """Greedy streamed chat; returns (usage, seconds to first token, decode seconds, text)."""
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, **extra}
    t0 = time.time()
    first = usage = None
    out = ""
    for line in post(a.url + "/v1/chat/completions", body):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        for ch in d.get("choices", []):
            c = ch.get("delta", {}).get("content")
            if c:
                first = first or time.time()
                out += c
    return usage, first - t0, time.time() - first, out
