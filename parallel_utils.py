import os
from concurrent.futures import ThreadPoolExecutor


def get_num_workers():
    """Return worker count from OMP_NUM_THREADS or cpu count."""
    try:
        env_val = int(os.environ.get("OMP_NUM_THREADS", "1"))
    except ValueError:
        env_val = 1
    if env_val <= 0:
        env_val = 1
    return env_val


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
