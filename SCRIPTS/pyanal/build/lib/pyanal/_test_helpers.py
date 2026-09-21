"""Trivial workers used by the test suite to exercise the
run_subprocess_in_module_mp spawn/Pipe plumbing.

Side-effect-free at import time so the spawned child process can re-import
this module cleanly via importlib.
"""


def _smoke_worker(x, y):
    return x + y


def _raising_worker():
    raise ValueError("smoke-test-raised")
