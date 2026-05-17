import pytest

from scripts.train import TRAINABLE_MODELS, build_parser


def test_trainable_models_exclude_heuristics():
    assert "itemcf" not in TRAINABLE_MODELS
    assert "popularity" not in TRAINABLE_MODELS


def test_build_parser_rejects_popularity():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["popularity"])
