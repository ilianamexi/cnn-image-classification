"""Runs the pipeline on a tiny synthetic image folder (no download, no pretrained weights)."""
import argparse
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402


@pytest.fixture()
def tiny_caltech(tmp_path, monkeypatch):
    import tensorflow as tf

    root = tmp_path / "101_ObjectCategories"
    rng = np.random.default_rng(0)
    for c in ["BACKGROUND_Google", "accordion", "airplanes", "anchor"]:
        (root / c).mkdir(parents=True)
        for i in range(36):
            img = (rng.random((40, 40, 3)) * 255).astype("uint8")
            tf.io.write_file(str(root / c / f"image_{i:04d}.jpg"), tf.io.encode_jpeg(img))
    monkeypatch.setattr(train, "DATA_DIR", root)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    return root


def test_background_folder_is_excluded(tiny_caltech):
    paths, labels, classes = train.list_images(None)
    assert classes == ["accordion", "airplanes", "anchor"]
    assert len(paths) == 3 * 36


def test_per_class_split_uses_30_training_images(tiny_caltech):
    paths, labels, _ = train.list_images(None)
    tr, va, te = train.split(paths, labels, "per-class", seed=0)
    assert len(tr) + len(va) == 90 and len(te) == 18
    assert not (set(tr) & set(te)) and not (set(va) & set(te))


def test_alexnet_and_vgg_paths_run(tiny_caltech, monkeypatch):
    import tensorflow as tf

    base = tf.keras.applications.VGG16
    monkeypatch.setattr(tf.keras.applications, "VGG16", lambda **kw: base(**{**kw, "weights": None}))
    for model in ["alexnet", "vgg16-frozen"]:
        args = argparse.Namespace(model=model, split="per-class", classes=None, epochs=1,
                                  finetune_epochs=1, batch=8, seed=0)
        res = train.run(args)
        assert res["n_classes"] == 3 and 0 <= res["accuracy"] <= 1
