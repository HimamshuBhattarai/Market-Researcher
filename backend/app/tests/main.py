import sys
from pathlib import Path

import pytest


TEST_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(TEST_DIR), "-q"]))
