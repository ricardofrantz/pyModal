import os
from concurrent.futures import ThreadPoolExecutor


def get_num_workers():
    """Return worker count from ``OMP_NUM_THREADS`` or ``os.cpu_count()``."""
    env = os.environ.get("OMP_NUM_THREADS")
    try:
        val = int(env) if env is not None else None
    except (TypeError, ValueError):
        val = None
    if val is not None and val > 0:
        return val
    cpu = os.cpu_count() or 1
    return cpu


def parallel_map(func, iterable, workers=None):
    """Map function over iterable using threads."""
    workers = workers or get_num_workers()
    if workers <= 1:
        return [func(x) for x in iterable]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(func, x) for x in iterable]
        for f in futures:
            results.append(f.result())
    return results
