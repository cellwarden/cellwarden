"""``python -m cellwarden run|resume|status|doctor ...`` (see cellwarden.runner)."""

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
