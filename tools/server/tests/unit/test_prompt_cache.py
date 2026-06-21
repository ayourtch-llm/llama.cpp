import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_ctx = 2048
    server.n_predict = 4
    server.temperature = 0.0
    server.cache_ram = 1024


def complete(prompt):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body["timings"]


def base_tokens():
    res = server.make_request("POST", "/tokenize", data={
        "content": "the quick brown fox jumps over the lazy dog. " * 64,
        "add_special": False,
    })
    assert res.status_code == 200
    g = res.body["tokens"]
    assert len(g) >= 500
    return g


def test_prompt_cache_partial_match_picks_longest_prefix():
    # With a single slot and the RAM prompt cache enabled, displaced prompts are
    # stored in the cache. When a new request partially matches several cached
    # prompts, the one sharing the longest prefix with the request must be picked.
    global server
    # disable slot LCP selection so displaced prompts always go through the RAM cache
    server.slot_prompt_similarity = 0.0
    server.start()

    g = base_tokens()
    div_a = next(t for t in g if t != g[150])
    div_b = next(t for t in g if t != g[400])
    div_x = next(t for t in g if t != g[0])

    req   = g[:500]                       # the final request
    short = g[:150] + [div_a] * 10        # high f_keep, short shared prefix (150)
    long  = g[:400] + [div_b] * 200       # lower f_keep, long shared prefix (400)
    other = [div_x] * 60                  # unrelated, evicts "long" from the slot

    # populate the slot, then displace each prompt into the RAM cache in order:
    # cache ends up holding [short, long, other], slot holds "other"
    assert complete(short)["prompt_n"] == len(short)          # full process
    assert complete(long)["prompt_n"] == len(long) - 150      # reuse 150 from slot
    complete(other)

    # now "short" and "long" both partially match the request; "long" shares the
    # longer prefix (400 vs 150) so it must be the one restored from the cache
    timings = complete(req)
    assert timings["cache_n"] == 400
    assert timings["prompt_n"] == len(req) - 400


def test_prompt_cache_restore_over_slot_lcp_match():
    # With slot LCP selection left at its default, a request may match the slot
    # well enough to be routed to it, yet a cached prompt shares a much longer
    # prefix. The RAM cache must still be consulted and the better entry restored.
    global server
    # leave server.slot_prompt_similarity at the default (0.1)
    server.start()

    g = base_tokens()
    div_l = next(t for t in g if t != g[400])
    div_f = next(t for t in g if t != g[0])
    div_p = next(t for t in g if t != g[80])

    long   = g[:400] + [div_l] * 100      # cached prompt, long shared prefix (400)
    filler = [div_f] * 200                # unrelated, used to displace "long"
    pre    = g[:80]  + [div_p] * 380      # slot prompt: short prefix (80) match to req
    req    = g[:450]

    # cache "long", then leave an unrelated prompt ("pre") in the slot that shares
    # only a short prefix with the request
    complete(long)
    complete(filler)
    assert complete(pre)["prompt_n"] == len(pre)             # full process, no restore

    # the slot is selected by LCP similarity (80/450 > 0.1) but would keep little
    # of its context, so the cache is consulted and "long" (prefix 400) restored
    timings = complete(req)
    assert timings["cache_n"] == 400
    assert timings["prompt_n"] == len(req) - 400
