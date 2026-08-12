"""Allow ``python -m oracle.eval_h2h`` style entry via package help."""

from .eval_h2h import main

if __name__ == "__main__":
    raise SystemExit(main())
