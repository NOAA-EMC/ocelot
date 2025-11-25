import os
import time

import psutil


def timing_resource_decorator(func):
    """
    A decorator that tracks execution time, memory usage, disk usage, and CPU usage
    before and after function execution.
    """

    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())

        # Record system resource usage BEFORE execution
        mem_before = process.memory_info().rss / (1024**3)  # GB
        disk_before = psutil.disk_usage("/").used / (1024**3)
        cpu_before = psutil.cpu_percent(interval=None)  # Snapshot before execution

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Record system resource usage AFTER execution
        mem_after = process.memory_info().rss / (1024**3)
        disk_after = psutil.disk_usage("/").used / (1024**3)
        cpu_after = psutil.cpu_percent(interval=None)  # Snapshot after execution

        execution_time = end_time - start_time

        # Print execution time and resource usage
        print(
            f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute."
        )
        print(f"  Memory Usage: {mem_before:.2f} GB → {mem_after:.2f} GB")
        print(f"  CPU Usage: {cpu_before:.2f}% → {cpu_after:.2f}%")
        print(f"  Disk Usage: {disk_before:.2f} GB → {disk_after:.2f} GB\n")
        return result

    return wrapper
