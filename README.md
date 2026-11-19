# Music genre classification from audio features and mel-spectrograms

F20DL Data Mining and Machine Learning, 2026–27. UG Group 12, Heriot-Watt University Dubai.

Abdullah Ishraq, Ahmed Khan, Aryadathan Karyadath, Nayela Ehtesham, Luke Nagooroo

## Project

We classify music genre from two representations of tracks in the Free Music Archive (FMA).

- **Pipeline A** uses 518 precomputed audio features per track to predict 8 genres. Models: Gaussian Naive Bayes, logistic regression, decision tree, k-nearest neighbours, random forest and an MLP. We also cluster the tracks with k-means.
- **Pipeline B** uses 128 × 431 mel-spectrograms of 10-second clips to predict 6 genres. Models: logistic regression on per-band summary statistics, and a CNN.

The two datasets cover different genres and tracks. Each pipeline is scored on its own held-out test set, and accuracies are not compared across pipelines.

Research questions:

1. How well do classical models separate the 8 FMA genres, and which genres get confused?
2. Does an MLP beat the best classical model on the same features?
3. Does a CNN on spectrograms beat a linear model on summary statistics of the same spectrograms?

## Data

| | FMA Small features | Mel-spectrograms |
|---|---|---|
| Source | Defferrard et al., *FMA: A Dataset for Music Analysis*, ISMIR 2017. [github.com/mdeff/fma](https://github.com/mdeff/fma), file `fma_metadata.zip` | University of Tartu Kaggle competition, [Music Genre Classification from Mel Spectrograms](https://www.kaggle.com/competitions/music-genre-classification-from-mel-spectrograms-dl4cv/data) |
| Licence | CC BY 4.0 | CC BY-NC 4.0. The competition rules allow non-commercial research and education use and forbid redistribution |
| Size | 8,000 tracks (30 s clips), 1,000 per genre | 17,382 labelled 10 s clips, 2,897 per genre. The 1,446 unlabelled test clips are not used |
| Inputs | 74 audio descriptors (MFCC, chroma, spectral contrast, tonnetz, centroid, bandwidth, roll-off, RMS energy, zero-crossing rate) × 7 statistics | One 128 × 431 float32 mel-spectrogram per clip, in decibels from −80 to 0 |
| Split | Official FMA split, 800/100/100 tracks per genre, which keeps each artist in one split | Stratified 70/15/15 with seed 42 (12,167 / 2,607 / 2,608 clips), stored in `data/splits/spec_splits.csv` |
| Data quality | No missing or infinite values. 17 duplicate feature rows in 13 groups, none crossing splits; one group carries two different genres | No missing values in a 300-clip sample. Clip IDs are sequential and don't identify the source song, so clips from one song may sit in different splits |

Models only see audio features. Track IDs and artist or album metadata are excluded because they leak information about the label.

The spectrograms are not stored here, since the competition rules forbid redistribution. `scripts/download_data.py` fetches them from Kaggle.

<!-- Phase 8: add two example rows from FMA and two example spectrograms. -->

## Setup

Python 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The spectrogram download needs a Kaggle account:

1. Accept the rules on the [competition page](https://www.kaggle.com/competitions/music-genre-classification-from-mel-spectrograms-dl4cv/rules).
2. Sign in from the terminal with the environment active. This opens a browser window and saves your credentials, so there is no token to copy:

   ```powershell
   kaggle auth login
   kaggle competitions list    # prints a table if you are signed in
   ```

If the browser sign-in doesn't work, generate a token under Kaggle → Settings → API Tokens and save it as `~/.kaggle/access_token` (never commit it):

```powershell
mkdir $HOME\.kaggle -Force
Set-Content -NoNewline $HOME\.kaggle\access_token "KGAT_your_token"
```

The CNN notebook (`06_spec_cnn`) runs on a Kaggle GPU notebook with the competition data attached. Everything else runs on a laptop CPU.

## Running the pipeline

```bash
python scripts/download_data.py    # FMA metadata (~342 MB) and Kaggle spectrograms into data/raw/
python scripts/prepare_data.py     # checks the data, builds data/processed/, writes the splits
python -m src.evaluate             # self-test for the metrics code
```

Add `fma` or `spec` to either script to process one dataset.

`prepare_data.py` does the following:

- **FMA:** keeps the 8,000 tracks of the small subset, with their 518 features, genre and official split.
- **Spectrograms:** takes each clip's genre from its folder (`spectrograms_unnormalized/data/<Genre>/`) and creates the stratified split. Clips in `test/` have no labels and are skipped.
- **Checks:** for missing, infinite, duplicate and constant values, plus array shapes and value ranges. Duplicate FMA rows are also checked for crossing splits, which would leak test tracks into training.
- **Reports:** results go to `results/data_check_fma.md` and `results/data_check_spec.md`.

Running it again gives the same splits. Then run the notebooks in order, starting with `notebooks/00_quickstart.ipynb`.

| Date | Pipeline change | Reason |
|---|---|---|
| 18 Nov | First version | |
| 19 Nov | Spectrogram labels read from genre folders instead of a CSV | The only CSV in the download is Kaggle's submission template, which has placeholder genres |
| 19 Nov | Duplicate check reports split crossing and genre conflicts | 17 duplicate FMA rows found; none cross splits |

## Repository layout

```
src/            config.py (paths, seed), data.py (loaders), evaluate.py (metrics, plots, result logging)
scripts/        download_data.py, prepare_data.py
notebooks/      one notebook per phase
data/splits/    frozen splits (committed)
data/raw/       downloaded data (not committed)
data/processed/ prepared tables (not committed)
results/        one JSON file per experiment, plus the data checks
figures/        figures used in the report
```

## Results

<!-- Phase 8: up to 100 words per requirement, covering inputs, outputs, notebook permalink, results table and one figure. -->

### R2: Data analysis

### R3: Baseline models

### R4: Neural networks
