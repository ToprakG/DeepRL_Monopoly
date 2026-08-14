"""First-monopoly meta on the jail-v1 body (``toprakthegoat-v1``)."""

from pathlib import Path

from .bootstrap import install_engine_namespace

install_engine_namespace(Path(__file__).resolve().parent.parent)

from .agent import GOAT_ID, GoatAgent, heuristic_action

__all__ = ["GOAT_ID", "GoatAgent", "heuristic_action"]
