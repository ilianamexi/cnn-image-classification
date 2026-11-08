"""Caltech-101 image classification: AlexNet trained from scratch vs. VGG16 transfer learning.

Models
- alexnet      : AlexNet-style CNN trained from random initialisation.
- vgg16-frozen : VGG16 pretrained on ImageNet used as a fixed feature extractor; only a small
                 classifier head is trained (features are computed once and cached).
- vgg16-finetune : as above, then the last convolutional block is unfrozen and trained with a
                 low learning rate (true fine-tuning).

Splits
- ratio     : stratified 70% train / 30% test.
- per-class : 30 training images per class, the rest for testing (the usual Caltech-101 protocol).
In both cases 15% of the training images are held out as a validation set for early stopping;
the test set is used once, at the end.

The background clutter folder (BACKGROUND_Google) is not an object class and is excluded.
Metrics: overall accuracy and mean per-class accuracy (the standard Caltech-101 measure,
because class sizes range from 31 to 800 images).

Usage:
    python train.py --model vgg16-frozen --split per-class
    python train.py --model alexnet --split ratio --epochs 30      # needs a GPU in practice
    python train.py --model vgg16-finetune --classes 50
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "101_ObjectCategories"
CALTECH_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
EXCLUDED = {"BACKGROUND_Google"}
IMG = 224


# ----------------------------------------------------------------------------- data
def download_caltech101() -> None:
    target = ROOT / "data"
    target.mkdir(exist_ok=True)
    zpath = target / "caltech-101.zip"
    if not zpath.exists():
        print(f"downloading {CALTECH_URL}")
        urllib.request.urlretrieve(CALTECH_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(target)
    inner = next(target.rglob("101_ObjectCategories.tar.gz"))
    with tarfile.open(inner) as t:
        t.extractall(target)


def list_images(n_classes: int | None):
    if not DATA_DIR.exists():
        download_caltech101()
    classes = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in EXCLUDED)
    if n_classes:
        classes = classes[:n_classes]
    paths, labels = [], []
    for i, c in enumerate(classes):
        files = sorted(p for p in (DATA_DIR / c).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        paths += [str(p) for p in files]
        labels += [i] * len(files)
    return np.array(paths), np.array(labels), classes


def split(paths, labels, mode: str, seed: int):
    rng = np.random.default_rng(seed)
    if mode == "ratio":
        tr, te = train_test_split(np.arange(len(labels)), test_size=0.3, stratify=labels, random_state=seed)
    else:  # per-class: 30 training images per class
        tr, te = [], []
        for c in np.unique(labels):
            idx = rng.permutation(np.where(labels == c)[0])
            tr += list(idx[:30])
            te += list(idx[30:])
        tr, te = np.array(tr), np.array(te)
    tr, va = train_test_split(tr, test_size=0.15, stratify=labels[tr], random_state=seed)
    return tr, va, te


def dataset(paths, labels, preprocess, batch: int, shuffle: bool, seed: int):
    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG, IMG))  # float32 in [0, 255], RGB
        return preprocess(img), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------- models
def alexnet(n_classes: int) -> tf.keras.Model:
    L = tf.keras.layers
    return tf.keras.Sequential([
        tf.keras.Input((IMG, IMG, 3)),
        L.Conv2D(96, 11, strides=4, activation="relu"), L.MaxPooling2D(3, 2), L.BatchNormalization(),
        L.Conv2D(256, 5, padding="same", activation="relu"), L.MaxPooling2D(3, 2), L.BatchNormalization(),
        L.Conv2D(384, 3, padding="same", activation="relu"),
        L.Conv2D(384, 3, padding="same", activation="relu"),
        L.Conv2D(256, 3, padding="same", activation="relu"), L.MaxPooling2D(3, 2),
        L.Flatten(), L.Dense(4096, activation="relu"), L.Dropout(0.5),
        L.Dense(4096, activation="relu"), L.Dropout(0.5),
        L.Dense(n_classes, activation="softmax"),
    ])


def vgg_head(n_inputs: int, n_classes: int) -> tf.keras.Model:
    L = tf.keras.layers
    return tf.keras.Sequential([tf.keras.Input((n_inputs,)), L.Dense(256, activation="relu"), L.Dropout(0.5),
                                L.Dense(n_classes, activation="softmax")])


def callbacks():
    return [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)]


def plot_confusion(cm: np.ndarray, result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of the true class")
    ax.set(xlabel="Predicted class index", ylabel="True class index",
           title=f"{result['model']} · {result['split']} split · {result['n_classes']} classes\n"
                 f"accuracy {100 * result['accuracy']:.1f}% · mean per-class {100 * result['mean_per_class_accuracy']:.1f}%")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args) -> dict:
    tf.keras.utils.set_random_seed(args.seed)
    paths, labels, classes = list_images(args.classes)
    tr, va, te = split(paths, labels, args.split, args.seed)
    n = len(classes)
    print(f"{n} classes, {len(tr)} train / {len(va)} val / {len(te)} test images")
    compile_kw = dict(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    if args.model == "alexnet":
        pre = lambda x: x / 255.0
        mk = lambda idx, sh: dataset(paths[idx], labels[idx], pre, args.batch, sh, args.seed)
        model = alexnet(n)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), **compile_kw)
        model.fit(mk(tr, True), validation_data=mk(va, False), epochs=args.epochs, callbacks=callbacks(), verbose=2)
        probs = model.predict(mk(te, False), verbose=0)
    else:
        pre = tf.keras.applications.vgg16.preprocess_input  # RGB -> BGR, ImageNet mean subtraction
        base = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(IMG, IMG, 3),
                                           pooling="avg")
        base.trainable = False

        def features(idx):
            return base.predict(dataset(paths[idx], labels[idx], pre, args.batch, False, args.seed), verbose=0)

        f_tr, f_va, f_te = features(tr), features(va), features(te)
        head = vgg_head(f_tr.shape[1], n)
        head.compile(optimizer=tf.keras.optimizers.Adam(1e-3), **compile_kw)
        head.fit(f_tr, labels[tr], validation_data=(f_va, labels[va]), epochs=args.epochs, batch_size=args.batch,
                 callbacks=callbacks(), verbose=2)
        if args.model == "vgg16-frozen":
            probs = head.predict(f_te, verbose=0)
        else:  # fine-tune the last convolutional block together with the trained head
            for layer in base.layers:
                layer.trainable = layer.name.startswith("block5")
            model = tf.keras.Sequential([base, head])
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), **compile_kw)
            mk = lambda idx, sh: dataset(paths[idx], labels[idx], pre, args.batch, sh, args.seed)
            model.fit(mk(tr, True), validation_data=mk(va, False), epochs=args.finetune_epochs,
                      callbacks=callbacks(), verbose=2)
            probs = model.predict(mk(te, False), verbose=0)

    y_true, y_pred = labels[te], probs.argmax(axis=1)
    result = {
        "model": args.model, "split": args.split, "n_classes": n, "seed": args.seed,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mean_per_class_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    name = f"{args.model}_{args.split}_{n}classes"
    (out / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n))
    np.save(out / f"{name}_confusion.npy", cm)
    plot_confusion(cm, result, out / f"{name}_confusion.png")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["alexnet", "vgg16-frozen", "vgg16-finetune"], default="vgg16-frozen")
    p.add_argument("--split", choices=["ratio", "per-class"], default="per-class")
    p.add_argument("--classes", type=int, default=None, help="use only the first N classes (default: all 101)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--finetune-epochs", dest="finetune_epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    run(p.parse_args())


if __name__ == "__main__":
    main()
