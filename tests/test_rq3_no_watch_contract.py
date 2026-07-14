import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rq3_tune_metadata import apply_no_watch_config, build_parser


def test_rq3_forces_no_watch_configuration():
    config = apply_no_watch_config(
        {"watch_mode": "loss", "watch_alpha": 0.5, "hidden_dim": 64}
    )

    assert config["watch_mode"] == "none"
    assert config["watch_alpha"] == 0.0
    assert config["hidden_dim"] == 64


def test_rq3_tune_parser_rejects_rq2_winner_argument():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--rq2-winner", "winner.json"])
