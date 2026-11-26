"""EuroSAT RGB loading from parquet shards (blanchon/EuroSAT_RGB mirror, test+validation splits)."""
import io
import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

def load_frame():
    dfs = [pd.read_parquet(p) for p in ["data/eurosat_test.parquet", "data/eurosat_val.parquet"]]
    df = pd.concat(dfs, ignore_index=True)
    # parquet columns: image (dict with 'bytes'), label (int)
    df["png"] = df["image"].apply(lambda d: d["bytes"] if isinstance(d, dict) else d)
    return df[["png", "label"]]

def decode(png_bytes):
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")

def arrays(df):
    X = np.stack([np.asarray(decode(b), dtype=np.uint8) for b in df["png"]])
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y

def stratified_split(y, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    test_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = int(round(len(idx) * test_frac))
        test_idx.extend(idx[:n])
    test_idx = np.array(sorted(test_idx))
    train_idx = np.array(sorted(set(range(len(y))) - set(test_idx)))
    return train_idx, test_idx

import os
CACHE = "data/arrays_cache.npz"
def load_arrays_cached():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["X"], z["y"]
    df = load_frame()
    X, y = arrays(df)
    np.savez(CACHE, X=X, y=y)
    return X, y
