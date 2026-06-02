"""
concurrency.py
==============
A tiny helper for running a function on many inputs in parallel using
a thread pool. Used to speed up the API-bound stages of the pipeline
without changing any prompts, models, or outputs.

Why threads (not processes): API calls spend almost all their time
waiting on network IO, which Python releases the GIL for, so threads
give real parallelism here. Processes would also work but are heavier
and add pickling pain.

Default 8 workers. Tune up if your API rate limit allows, down if you
start seeing 429 errors.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed


def parallel_map(items, fn, workers=8, label=None):
    items = list(items)
    results = [None] * len(items)
    if not items:
        return results

    def _wrap(idx_item):
        idx, item = idx_item
        try:
            return idx, fn(item)
        except Exception as e:
            print(f"  [warn] item {idx} failed: {type(e).__name__}: {e}")
            return idx, e

    done = 0
    total = len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_wrap, (i, it)) for i, it in enumerate(items)]
        for fut in as_completed(futures):
            idx, val = fut.result()
            results[idx] = val
            done += 1
            if label and (done % max(1, total // 20) == 0 or done == total):
                print(f"  [{label}] {done}/{total} done")
    return results
