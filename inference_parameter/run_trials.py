"""Shim — use `python -m rtdetr_eval run-trials` from the repo root."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rtdetr_eval.trials import main  # noqa: E402

if __name__ == "__main__":
    main()
