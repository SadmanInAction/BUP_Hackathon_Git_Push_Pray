import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls a real LLM provider (skipped when none is reachable)")


def pytest_runtest_setup(item):
    # Free-tier LLM quotas are per-minute; LIVE_TEST_DELAY=<seconds> paces the live tests.
    import os
    import time

    if item.get_closest_marker("live") and os.environ.get("LIVE_TEST_DELAY"):
        time.sleep(float(os.environ["LIVE_TEST_DELAY"]))
