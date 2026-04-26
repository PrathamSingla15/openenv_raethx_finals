"""Entry point: ``python -m tradebench.verifier ...``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
