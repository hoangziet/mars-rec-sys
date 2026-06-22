import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.builder import _validate_text_embedding_manifest


def _write_valid_setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a minimal valid embeddings.pt + manifest.json + source files."""
    emb_path = tmp_path / "text_embeddings.pt"
    manifest_path = tmp_path / "text_embeddings_manifest.json"
    map_path = tmp_path / "item_id_map.csv"
    meta_path = tmp_path / "item_metadata.csv"

    n_items = 5
    emb = torch.zeros(n_items + 1, 16)
    emb[1:] = torch.randn(n_items, 16)
    torch.save(emb, emb_path)

    map_path.write_text("item_id,item_idx\na,1\nb,2\nc,3\nd,4\ne,5\n")
    meta_path.write_text("item_idx,item_id,title\n1,a,t1\n2,b,t2\n3,c,t3\n4,d,t4\n5,e,t5\n")

    map_sha = hashlib.sha256(map_path.read_bytes()).hexdigest()
    meta_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()

    manifest = {
        "n_items": n_items,
        "embedding_dim": 16,
        "padding_row": 0,
        "item_id_map_sha256": map_sha,
        "metadata_sha256": meta_sha,
        "encoder": "dangvantuan/sentence-camembert-base",
        "encoder_revision": "main",
        "text_template_version": "v1",
    }
    manifest_path.write_text(json.dumps(manifest))
    return emb_path, manifest_path, map_path


def test_validate_text_embedding_manifest_passes(tmp_path):
    emb_path, manifest_path, map_path = _write_valid_setup(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    _validate_text_embedding_manifest(emb_path, manifest, map_path, str(tmp_path / "item_metadata.csv"), n_items=5)


def test_validate_text_embedding_manifest_fails_on_shape_mismatch(tmp_path):
    emb_path, manifest_path, map_path = _write_valid_setup(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["n_items"] = 99
    with pytest.raises(RuntimeError, match="Embeddings shape mismatch"):
        _validate_text_embedding_manifest(emb_path, manifest, map_path, str(tmp_path / "item_metadata.csv"), n_items=5)


def test_validate_text_embedding_manifest_fails_on_checksum_mismatch(tmp_path):
    emb_path, manifest_path, map_path = _write_valid_setup(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["item_id_map_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="item_id_map_sha256 mismatch"):
        _validate_text_embedding_manifest(emb_path, manifest, map_path, str(tmp_path / "item_metadata.csv"), n_items=5)


def test_validate_text_embedding_manifest_fails_on_padding_row_nonzero(tmp_path):
    emb_path, manifest_path, map_path = _write_valid_setup(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    emb = torch.load(emb_path, weights_only=True)
    emb[0] = torch.randn(16)
    torch.save(emb, emb_path)
    with pytest.raises(RuntimeError, match="padding row 0"):
        _validate_text_embedding_manifest(emb_path, manifest, map_path, str(tmp_path / "item_metadata.csv"), n_items=5)


# ---- build_model data_dir param ----

def test_build_model_uses_custom_data_dir(tmp_path):
    """build_model with a custom data_dir should resolve item_id_map from it.

    The test stubs out heavy paths so we just need to confirm the gsasrec
    branch reads item_id_map_path from the data_dir root, not a hard-coded
    default location.
    """
    import sys
    sys.modules.pop("pipeline.builder", None)
    from pipeline import builder

    custom_data_dir = tmp_path / "my_data"
    mappings_dir = custom_data_dir / "mappings"
    features_dir = custom_data_dir / "item_features"
    mappings_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)

    # Provide the files gsasrec needs
    n_items = 5
    item_map = mappings_dir / "item_id_map.csv"
    item_map.write_text("item_id,item_idx\n" + "\n".join(f"id{i},{i}" for i in range(1, n_items + 1)))

    item_meta = features_dir / "item_metadata.csv"
    rows = ["item_idx,language,difficulty,theme,software,job,type,duration"]
    for i in range(1, n_items + 1):
        rows.append(f"{i},en,easy,t,sw,j,x,30")
    item_meta.write_text("\n".join(rows))

    vocab_path = features_dir / "metadata_vocab.json"
    vocab_path.write_text(json.dumps({
        "categorical": {"language": {"en": 3}, "difficulty": {"easy": 3}},
        "multilabel": {"theme": {"t": 3}, "software": {"sw": 3}, "job": {"j": 3}, "type": {"x": 3}},
        "duration_mean": 3.0,
        "duration_std": 1.0,
    }))

    text_emb = features_dir / "text_embeddings.pt"
    emb = torch.zeros(n_items + 1, 16)
    emb[1:] = torch.randn(n_items, 16)
    torch.save(emb, text_emb)

    manifest_path = features_dir / "text_embeddings_manifest.json"
    manifest = {
        "n_items": n_items,
        "embedding_dim": 16,
        "padding_row": 0,
        "item_id_map_sha256": hashlib.sha256(item_map.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(item_meta.read_bytes()).hexdigest(),
        "encoder": "fake",
        "encoder_revision": "main",
        "text_template_version": "v1",
    }
    manifest_path.write_text(json.dumps(manifest))

    # Build with use_text=True so the manifest validation path is exercised
    model = builder.build_model(
        "gsasrec", n_items=n_items, n_users=10,
        model_kwargs={
            "hidden_dim": 16,
            "item_encoder": {
                "use_structured": True,
                "use_text": True,
                "metadata_vocab_path": str(vocab_path),
                "metadata_csv_path": str(item_meta),
                "text_emb_path": str(text_emb),
            },
        },
        max_len=10,
        data_dir=custom_data_dir,
    )
    assert model is not None


def test_manifest_schema_includes_encoder_revision():
    """The manifest schema is documented to include encoder_revision."""
    # Verify the validator tolerates a real SHA (not "main") and uses it
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        emb_path, manifest_path, map_path = _write_valid_setup(tmp_path)
        manifest = json.loads(manifest_path.read_text())
        # Real SHA-like value
        manifest["encoder_revision"] = "a" * 40
        # Should still pass
        _validate_text_embedding_manifest(emb_path, manifest, map_path, str(tmp_path / "item_metadata.csv"), n_items=5)


# ---- _resolve_encoder_revision ----

def test_resolve_encoder_revision_returns_real_sha(monkeypatch):
    """When the HF API is reachable, _resolve_encoder_revision should return
    the real commit SHA — not the literal string 'main'."""
    import sys
    from scripts import rq3_precompute_embeddings

    class _FakeInfo:
        sha = "abcdef0123456789" * 4  # 40-char SHA-like

    class _FakeApi:
        def model_info(self, model_name):
            return _FakeInfo()

    class _FakeHuggingFaceHub:
        HfApi = _FakeApi

    monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHuggingFaceHub)

    result = rq3_precompute_embeddings._resolve_encoder_revision("any/model")
    assert result == "abcdef0123456789" * 4
    assert result != "main"


def test_resolve_encoder_revision_falls_back_to_main(monkeypatch):
    """When HF API fails, _resolve_encoder_revision should fall back to 'main'."""
    import sys
    from scripts import rq3_precompute_embeddings

    class _BrokenApi:
        def model_info(self, model_name):
            raise RuntimeError("network down")

    class _FakeHuggingFaceHub:
        HfApi = _BrokenApi

    monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHuggingFaceHub)

    result = rq3_precompute_embeddings._resolve_encoder_revision("any/model")
    assert result == "main"
