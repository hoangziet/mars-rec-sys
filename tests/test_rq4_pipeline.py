import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rq4_init_protocol import main as rq4_init_main


def test_rq4_protocol_manifest_contains_expected_runs(tmp_path, monkeypatch):
    out = tmp_path / "rq4"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rq4_init_protocol.py",
            "--benchmark-id", "rq4-ablation-v1",
            "--best-alpha", "0.5",
            "--best-variant", "M3",
            "--output-dir", str(out),
        ],
    )
    rq4_init_main()
    manifest = json.loads((out / "rq4_protocol_manifest.json").read_text())
    assert manifest["variants"] == ["V0", "V1", "V2", "V3"]
    assert len(manifest["neural_seeds"]) == 10
    assert manifest["expected_runs"] == 40


def test_rq4_protocol_manifest_is_not_overwritten(tmp_path, monkeypatch):
    out = tmp_path / "rq4"
    out.mkdir(parents=True)
    path = out / "rq4_protocol_manifest.json"
    path.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rq4_init_protocol.py",
            "--benchmark-id", "rq4-ablation-v1",
            "--best-alpha", "0.5",
            "--best-variant", "M3",
            "--output-dir", str(out),
        ],
    )
    with pytest.raises(RuntimeError, match="already exists"):
        rq4_init_main()
