import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rq4_ablation import _get_variant_config


def test_get_variant_config_builds_watch_and_metadata_ablation_matrix():
    rq2 = {"best_variant": "wlwe", "best_alpha": 1.0, "backbone": "bert4rec"}
    rq3 = {"best_variant": "M3", "backbone": "bert4rec"}

    v0 = _get_variant_config("V0", rq2, rq3)
    assert v0 == {
        "config_name": "bert4rec",
        "watch_mode": "none",
        "watch_alpha": 0.0,
        "use_structured": False,
        "use_text": False,
    }

    v1 = _get_variant_config("V1", rq2, rq3)
    assert v1["config_name"] == "bert4rec"
    assert v1["watch_mode"] == "both"   # from wlwe
    assert v1["watch_alpha"] == 1.0
    assert v1["use_structured"] is False

    v3 = _get_variant_config("V3", rq2, rq3)
    assert v3["config_name"] == "bert4rec_metadata"
    assert v3["watch_mode"] == "both"
    assert v3["watch_alpha"] == 1.0
    assert v3["use_structured"] is True  # from M3
    assert v3["use_text"] is True
