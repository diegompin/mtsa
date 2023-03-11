import time
from functools import partial
from functools import reduce


def elapsed_time(fun, *args):
    start = time.time()
    fun_return = fun(*args)
    end = time.time()
    time_elapsed = end - start
    result = {
        "time_elapsed": time_elapsed,
        "fun_return": fun_return
    }
    return result


def multiple_runs(runs, fun, *args):
    list_run_ids = range(runs)
    # fun_partial = partial(elapsed_time, fun, *args)
    # multiple_results = map(lambda x: (x, elapsed_time(fun, *args)), run_ids)

    multiple_results = map(
        lambda run_id:
        {
            'run_id': run_id,
            "run_details": elapsed_time(fun, *args)
        },
        list_run_ids)

    return list(multiple_results)

