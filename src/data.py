"""Loaders for the prepared FMA features and spectrogram index (run scripts/prepare_data.py first)."""
import numpy as np
import pandas as pd

from src.config import PROCESSED, RAW_SPEC, SPLITS


def load_fma():
    # One row per track (index = track_id): 518 feature columns + "genre" + "split".
    return pd.read_pickle(PROCESSED / "fma_small.pkl")


def fma_feature_cols(df):
    return [c for c in df.columns if c not in ("genre", "split")]


def fma_xy(df, split):
    part = df[df["split"] == split]
    return part[fma_feature_cols(df)], part["genre"]


def load_spec_index():
    # Labels come from the local Kaggle files; only id -> split is committed to the repo.
    index = pd.read_csv(PROCESSED / "spec_index.csv", dtype={"id": str})
    splits = pd.read_csv(SPLITS / "spec_splits.csv", dtype={"id": str})
    return index.merge(splits, on="id", validate="one_to_one")


def load_spectrograms(paths):
    # Each array is 128 x 431 (~220 KB as float32); load in batches, not all 17k at once on a laptop.
    return np.stack([np.load(RAW_SPEC / p) for p in paths])
