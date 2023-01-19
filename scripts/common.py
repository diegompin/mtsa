import time
from functools import partial
from functools import reduce

def elapsed_time(fun, *args):
    start = time.time()
    end = time.time()
    time_elapsed = end - start
    fun_return = fun(*args)      
    return time_elapsed, fun_return

def multiple_runs(runs, fun, *args):
    run_ids = range(runs)
    # fun_partial = partial(elapsed_time, fun, *args)
    multiple_results = map(lambda x: (x, elapsed_time(fun, *args)), run_ids)
    return multiple_results