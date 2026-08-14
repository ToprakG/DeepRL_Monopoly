"""Helper for the managed_pool orphan test: a real module so workers can
unpickle the target function and actually run it."""
import sys, time
sys.path.insert(0, "/Users/alinebidal/Projects/DeepRL_Monopoly")
from competition_agent.proc import managed_pool


def slow(i):
    time.sleep(120)
    return i


if __name__ == "__main__":
    with managed_pool(4) as pool:
        pool.map(slow, range(8))
