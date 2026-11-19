"""Shared paths, random seed, dataset URLs and genre lists."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_FMA = DATA / "raw" / "fma"
# On Kaggle, set SPEC_DIR to the attached competition folder instead of downloading.
RAW_SPEC = Path(os.environ.get("SPEC_DIR", DATA / "raw" / "spec"))
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

SEED = 42

FMA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
KAGGLE_COMPETITION = "music-genre-classification-from-mel-spectrograms-dl4cv"

FMA_GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
              "Instrumental", "International", "Pop", "Rock"]
SPEC_GENRES = ["Classical", "Electronic", "Folk", "Hip-Hop", "Jazz", "Rock"]
