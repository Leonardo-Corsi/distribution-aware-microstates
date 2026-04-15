# Distribution-aware microstates and global field power analysis for EEG abnormality characterization

This repository implements an end-to-end EEG pipeline on TUH/TUAB for:
- preprocessing EEG recordings and extracting clean segments,
- computing distribution-aware microstate and GFP-derived features (MSD, MTMI, PSD),
- running classification and producing manuscript-ready tables/figures.

The code follows the methods workflow described in the manuscript draft (feature construction and modeling pipeline), titled "Distribution-aware microstates and global field power analysis for EEG abnormality characterization".

## Pipeline overview

1. `s0_loading.py`
- Verifies local TUH/TUAB prerequisites.
- Parses TUH headers.
- Builds TUAB-matched header/index artifacts and S0 summary tables.

2. `s1-preprocess.py` + `s1_preprocess_utils.py`
- Loads TUAB records from S0.
- Applies preprocessing, segmentation, and optional ICA.
- Stores stage outputs as EDF+JSON record bundles:
  - `preprocessed_segmented/<record_stem>/metadata.json` + `*-segNN.edf`
  - `preprocessed_segmented_ICA/<record_stem>/metadata.json` + `*-icaNN.edf`
- On failure, writes stage-local error artifact:
  - `<stage_dir>/<record_stem>/<record_stem>-error.txt`

3. `s2-feature_extraction.py` + `s2_microstates_utils.py`
- Reads S1 EDF+JSON artifacts (segmented and ICA variants).
- Runs microstate backfitting + MTMI + PSD + MSD extraction.
- Writes per-record parquet shards, then merged stage parquet tables:
  - `df_MTMI.parquet`, `df_PSD.parquet`, `df_MSD.parquet`
- Writes audit CSVs:
  - `s2_records_index.csv`, `s2_counts_summary.csv`

4. `s2-feature_stats.py`
- Reads S2 parquet outputs and records index.
- Produces validation summaries for microstate, MTMI, and PSD meta-features.

5. `s2-feature_gabstract.py`
- Generates one graphical abstract figure (SVG/PNG) from a selected S1 record/segment.

6. `s3-feature_classify.py`
- Reads S2 parquet feature tables only.
- Builds intermediate S3 parquet tables, runs model selection/optimization (Optuna), and writes final performance tables/figures.
- Supports study reuse by default, with explicit recomputation flag.

## Configuration

All runtime settings are in `configs.json` and loaded via `_config_loader.py`.

- Relative paths are resolved from the appropriate parent keys:
  - `tuh_parent`, `output_parent`
  - `s0.s0_parent`, `s1.s1_parent`, `s2.s2_parent`, `s3.s3_parent`
- Stage-local output files are resolved under each stage parent.
- Core toggles include:
  - `n_jobs` (parallel workers)
  - `s1.apply_ica`
  - `s2.extraction.recompute`
  - `s3.recompute_intermediate_dfs`
  - `s3.force_recompute_studies`

## Environment setup

```bash
conda create -n microstates python=3.12 -y
conda activate microstates
pip install .
```

Parquet support requires `pyarrow` or `fastparquet` (`pyarrow` is pinned in `pyproject.toml`).

## How to run

Run each stage from the repository root:

```bash
python s0_loading.py --config configs.json
```

```bash
python s1-preprocess.py --config configs.json --start 0 --n 100
```

```bash
python s2-feature_extraction.py --config configs.json
python s2-feature_stats.py --config configs.json
python s2-feature_gabstract.py --config configs.json --seed 42
```

```bash
python s3-feature_classify.py --config configs.json
```

Force Optuna recomputation in S3:

```bash
python s3-feature_classify.py --config configs.json --recompute-studies
```

## Output structure (high level)

```text
<output_parent>/
  s0/...
  s1/<variant>/
    preprocessed_segmented/<record_stem>/{metadata.json,*-segNN.edf or *-error.txt}
    preprocessed_segmented_ICA/<record_stem>/{metadata.json,*-icaNN.edf or *-error.txt}
    preprocess_counts_summary.csv
  s2/<variant>/
    features_without_ica/
      <record_stem>/{status.json,mtmi.parquet,psd.parquet,msd.parquet}
      df_MTMI.parquet, df_PSD.parquet, df_MSD.parquet
      s2_records_index.csv, s2_counts_summary.csv
    features_with_ica/
      ...
    gabstract/...
    feature_validation/...
  s3/<variant>/<with_ica|without_ica>/
    studies/
    intermediate-features/
    tables/
    paperitems/
```

## Notes

- S1 incremental execution is supported via `--start/--end/--n`.
- S2 supports recompute/non-recompute behavior from config.
- S3 reads S2 parquet outputs and produces classification results with optional skip of hyperparameter optimization.

## Citation and license

- Citation metadata: `CITATION.cff`
- License: `LICENSE`
