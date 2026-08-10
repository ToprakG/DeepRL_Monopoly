"""Allow ``python -m asu_plus`` to print module help."""

from .eval_h2h import main

if __name__ == "__main__":
    raise SystemExit(main())
