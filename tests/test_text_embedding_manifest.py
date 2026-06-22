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
