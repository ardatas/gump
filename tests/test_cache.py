from __future__ import annotations

from src.cache import list_tile_paths


def test_list_tile_paths_excludes_feature_caches(tmp_path):
    train_root = tmp_path / "train"
    train_root.mkdir()

    real_tile = train_root / "18NWG_6_6.zarr"
    feature_cache = train_root / "18NWG_6_6.features.zarr"
    other_tile = train_root / "19NBD_4_4.zarr"
    real_tile.mkdir()
    feature_cache.mkdir()
    other_tile.mkdir()

    paths = list_tile_paths(tmp_path, "train")

    assert paths == [real_tile, other_tile]
