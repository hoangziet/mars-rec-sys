import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq3_report


def test_rq3_report_rejects_unknown_metadata_variant_tag():
    selected = [{
        "variant": "?",
        "seed": 42,
        "val_ndcg_at_10": 0.1,
        "provenance": {
            "backbone": "gsasrec",
            "benchmark_id": "rq3-x",
            "preprocessing_version": "v1",
            "data_source": "/tmp/data",
        },
        "run_id": "rid",
        "run_name": "run",
    }]

    with pytest.raises(RuntimeError, match="Invalid metadata_variant"):
        rq3_report._validate_variant_names(selected)
