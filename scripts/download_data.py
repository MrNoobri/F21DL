"""Downloads the FMA metadata and the Kaggle spectrograms into data/raw/ (usage: download_data.py [fma|spec])."""
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FMA_URL, KAGGLE_COMPETITION, RAW_FMA, RAW_SPEC  # noqa: E402

# Only these three CSVs are needed from the ~342 MB zip.
FMA_FILES = ["fma_metadata/tracks.csv", "fma_metadata/features.csv", "fma_metadata/genres.csv"]


def _progress(blocks, block_size, total):
    print(f"\r  {blocks * block_size / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)


def download_fma():
    RAW_FMA.mkdir(parents=True, exist_ok=True)
    if all((RAW_FMA / Path(f).name).exists() for f in FMA_FILES):
        print("FMA: already present, skipping.")
        return
    zip_path = RAW_FMA / "fma_metadata.zip"
    if not zip_path.exists():
        print(f"FMA: downloading {FMA_URL}")
        tmp = zip_path.with_suffix(".part")  # renamed only when complete, so an interrupted download is retried
        urllib.request.urlretrieve(FMA_URL, tmp, reporthook=_progress)
        tmp.rename(zip_path)
        print()
    with zipfile.ZipFile(zip_path) as z:
        for f in FMA_FILES:
            with z.open(f) as src, open(RAW_FMA / Path(f).name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"FMA: extracted {len(FMA_FILES)} CSVs to {RAW_FMA}")


def download_spec():
    RAW_SPEC.mkdir(parents=True, exist_ok=True)
    if any(RAW_SPEC.rglob("*.npy")):
        print("Spectrograms: already present, skipping.")
        return
    if shutil.which("kaggle") is None:
        sys.exit("Kaggle CLI not found. Activate the project environment (pip install -r requirements.txt).")
    cmd = ["kaggle", "competitions", "download", KAGGLE_COMPETITION, "-p", str(RAW_SPEC)]
    if subprocess.run(cmd).returncode != 0:
        sys.exit("Kaggle download failed. Check: (1) your API token is saved in ~/.kaggle/access_token "
                 "(Windows: C:\\Users\\<you>\\.kaggle\\access_token); (2) you accepted the rules at "
                 f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules")
    for z in RAW_SPEC.glob("*.zip"):
        print(f"Spectrograms: extracting {z.name}")
        with zipfile.ZipFile(z) as f:
            f.extractall(RAW_SPEC)
    print(f"Spectrograms: {sum(1 for _ in RAW_SPEC.rglob('*.npy'))} .npy files in {RAW_SPEC}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("fma", "all"):
        download_fma()
    if which in ("spec", "all"):
        download_spec()
