"""Smoke tests: verify RQ2/RQ3 have the expected CLI contracts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_rq2_tune_help_has_no_winner_artifact_flag():
    """RQ2 is BERT4Rec-only: no --winner-artifact flag should be exposed."""
    from scripts.rq2_tune_alpha import build_parser

    parser = build_parser()
    args = parser.parse_args(["--benchmark-id", "rq2-x"])
    assert not hasattr(args, "winner_artifact")


def test_rq3_tune_help_has_no_winner_artifact_flag():
    """RQ3 is BERT4Rec-only: no --rq2-winner or --winner-artifact flag."""
    from scripts.rq3_tune_metadata import build_parser

    parser = build_parser()
    args = parser.parse_args(["--benchmark-id", "rq3-x"])
    assert not hasattr(args, "winner_artifact")
    assert not hasattr(args, "rq2_winner")
