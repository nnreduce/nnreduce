from termcolor import colored
from line_profiler import LineProfiler


def succ_print(*args):
    return print(*[colored(x, "green") for x in args])


def fail_print(*args):
    return print(*[colored(x, "red") for x in args])


def note_print(*args):
    return print(*[colored(x, "yellow") for x in args])


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


_PROFILER = None


def get_or_make_profiler():
    global _PROFILER
    if _PROFILER is None:
        _PROFILER = LineProfiler()
    return _PROFILER


def get_profiler():
    return _PROFILER


def print_stats():
    if _PROFILER is not None:
        _PROFILER.print_stats()


def profile(func):
    def inner(*args, **kwargs):
        get_or_make_profiler().add_function(func)
        get_or_make_profiler().enable_by_count()
        return func(*args, **kwargs)

    return inner
