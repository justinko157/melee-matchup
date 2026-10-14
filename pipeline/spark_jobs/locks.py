"""One-at-a-time guard for Spark jobs that share state on the checkpoint volume."""

import os
import sys
from typing import IO


def exclusive_lock(directory: str, job: str) -> IO:
    """Hold an exclusive lock on directory/.lock, or exit if another job holds it.

    Keep the returned file open for as long as the job runs.
    """
    import fcntl  # Linux only; imported here so modules still import on Windows hosts

    os.makedirs(directory, exist_ok=True)
    lock_file = open(os.path.join(directory, ".lock"), "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        sys.exit(f"another {job} job is running (lock held on {directory}/.lock)")
    return lock_file
