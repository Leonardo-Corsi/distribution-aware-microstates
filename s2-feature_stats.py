import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, t

from _config_loader import load_configs


def parse_args():
    parser = argparse.ArgumentParser(description="S2 feature validation stage.")
    parser.add_argument("--config", default="configs.json", help="Path to config file.")
    return parser.parse_args()


def _resolve_path(base_dir, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty path string, got: {value!r}")
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(base_dir, expanded))


def _group_from_condition(value):
    text = str(value).strip().lower()
    if "abnormal" in text:
        return "abn"
    if "normal" in text:
        return "nrm"
    return "unk"


def _microstate_label(map_id):
    return chr(ord("A") + int(map_id))


def _load_runtime(cfg):
    s2_cfg = cfg.get("s2", {})
    extraction_cfg = s2_cfg.get("extraction", {})
    validation_cfg = s2_cfg.get("validation", {})
    if "s2_parent" not in s2_cfg:
        raise KeyError("Missing required cfg['s2']['s2_parent'].")

    s2_parent = _resolve_path(cfg["output_parent"], s2_cfg["s2_parent"])
    out_with_ica = _resolve_path(
        s2_parent,
        str(extraction_cfg.get("output_subdir_with_ica", "features_with_ica")),
    )
    out_without_ica = _resolve_path(
        s2_parent,
        str(extraction_cfg.get("output_subdir_without_ica", "features_without_ica")),
    )
    validation_root = _resolve_path(
        s2_parent,
        str(validation_cfg.get("output_subdir", "feature_validation")),
    )
    os.makedirs(validation_root, exist_ok=True)

    n_ms = int(extraction_cfg.get("microstates", {}).get("n_ms", 4))

    return {
        "validation_root": validation_root,
        "n_ms": n_ms,
        "stages": [
            {"label": "without_ica", "feature_dir": out_without_ica},
            {"label": "with_ica", "feature_dir": out_with_ica},
        ],
    }


def _read_stage_inputs(feature_dir):
    path_msd = os.path.join(feature_dir, "df_MSD.parquet")
    path_mtmi = os.path.join(feature_dir, "df_MTMI.parquet")
    path_psd = os.path.join(feature_dir, "df_PSD.parquet")
    path_index = os.path.join(feature_dir, "s2_records_index.csv")
    if not os.path.isfile(path_msd):
        raise FileNotFoundError(f"Missing MSD parquet: {path_msd}")
    if not os.path.isfile(path_mtmi):
        raise FileNotFoundError(f"Missing MTMI parquet: {path_mtmi}")
    if not os.path.isfile(path_psd):
        raise FileNotFoundError(f"Missing PSD parquet: {path_psd}")
    if not os.path.isfile(path_index):
        raise FileNotFoundError(f"Missing records index csv: {path_index}")
    return (
        pd.read_parquet(path_msd),
        pd.read_parquet(path_mtmi),
        pd.read_parquet(path_psd),
        pd.read_csv(path_index),
    )


def _safe_mode(values):
    if values.size == 0:
        return np.nan, np.nan
    vals = np.asarray(values)
    uniq, counts = np.unique(vals, return_counts=True)
    idx = int(np.argmax(counts))
    return float(uniq[idx]), float(counts[idx] / counts.sum())


def _summarize_series(values):
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = arr.size
    if n == 0:
        return {
            "mean_std": "NA",
            "median": np.nan,
            "iqr": np.nan,
            "ci95": "NA",
            "n": 0,
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = float(q3 - q1)
    if n > 1:
        sem = std / math.sqrt(n)
        ci_half = float(t.ppf(0.975, df=n - 1) * sem)
        ci95 = f"{mean:.4f} +/- {ci_half:.4f}"
    else:
        ci95 = "NA"

    return {
        "mean_std": f"{mean:.4f} +/- {std:.4f}",
        "median": round(median, 4),
        "iqr": round(iqr, 4),
        "ci95": ci95,
        "n": int(n),
    }


def _build_summary_table(df_subject, feature_rows):
    rows = []
    for row_label, col_name in feature_rows:
        row = {"feature": row_label}
        for grp in ["nrm", "abn"]:
            stats = _summarize_series(df_subject.loc[df_subject["group2"] == grp, col_name])
            row[f"mean_std ({grp})"] = stats["mean_std"]
            row[f"median ({grp})"] = stats["median"]
            row[f"iqr ({grp})"] = stats["iqr"]
            row[f"ci95 ({grp})"] = stats["ci95"]
            row[f"n ({grp})"] = stats["n"]
        rows.append(row)
    return pd.DataFrame(rows)


def _train_group_map(df_index):
    if df_index.empty or "record_stem" not in df_index.columns or "condition" not in df_index.columns:
        return {}
    df_train = df_index[df_index["condition"].astype(str).str.contains("train", case=False, na=False)].copy()
    mapping = {}
    for _, row in df_train.iterrows():
        rec_id = str(row.get("record_stem", "")).strip()
        if not rec_id:
            continue
        mapping[rec_id] = _group_from_condition(row.get("condition", ""))
    return mapping


def _microstate_subject_features(df_msd, map_id, train_group_map):
    df_map = df_msd[df_msd["map"] == map_id].copy()
    if df_map.empty:
        return pd.DataFrame(columns=["ID", "Group", "group2", "dur_avg", "mode", "prel_mode"])

    rows = []
    for (group, rec_id), grp in df_map.groupby(["Group", "ID"]):
        group2 = train_group_map.get(str(rec_id))
        if group2 is None:
            continue
        durations = pd.to_numeric(grp["Duration"], errors="coerce").dropna().to_numpy(dtype=float)
        mode_value, prel_mode = _safe_mode(durations)
        rows.append(
            {
                "ID": rec_id,
                "Group": group,
                "group2": group2,
                "dur_avg": float(np.mean(durations)) if durations.size else np.nan,
                "mode": mode_value,
                "prel_mode": prel_mode,
            }
        )
    return pd.DataFrame(rows)


def _mtmi_subject_features(df_mtmi, train_group_map):
    rows = []
    for (group, rec_id), grp in df_mtmi.groupby(["Group", "ID"]):
        group2 = train_group_map.get(str(rec_id))
        if group2 is None:
            continue
        values = pd.to_numeric(grp["MTMI Time [ms]"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            skw = np.nan
            krt = np.nan
        elif values.size < 3:
            skw = np.nan
            krt = np.nan
        else:
            skw = float(skew(values, bias=False, nan_policy="omit"))
            krt = float(kurtosis(values, fisher=True, bias=False, nan_policy="omit"))
        rows.append(
            {
                "ID": rec_id,
                "Group": group,
                "group2": group2,
                "skewness": skw,
                "kurtosis": krt,
            }
        )
    return pd.DataFrame(rows)


def _spectral_centroid(freqs, power):
    freqs = np.asarray(freqs, dtype=float)
    power = np.asarray(power, dtype=float)
    denom = float(np.sum(power))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(freqs * power) / denom)


def _psd_subject_features(df_psd, train_group_map):
    rows = []
    for (group, rec_id, ch), grp in df_psd.groupby(["Group", "ID", "channels"]):
        group2 = train_group_map.get(str(rec_id))
        if group2 is None:
            continue
        freq = pd.to_numeric(grp["f"], errors="coerce").to_numpy(dtype=float)
        psd = pd.to_numeric(grp["PSD"], errors="coerce").to_numpy(dtype=float)
        order = np.argsort(freq)
        freq = freq[order]
        psd = psd[order]
        alpha_mask = (freq >= 8.0) & (freq <= 12.0)
        if np.any(alpha_mask):
            alpha_freq = freq[alpha_mask]
            alpha_psd = psd[alpha_mask] * 1e12
            alpha_sum = float(np.trapezoid(alpha_psd, alpha_freq))
        else:
            alpha_sum = np.nan
        rows.append(
            {
                "ID": rec_id,
                "Group": group,
                "group2": group2,
                "channels": ch,
                "spectral_centroid": _spectral_centroid(freq, psd),
                "alpha_8_12_sum": alpha_sum,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["ID", "Group", "group2", "spectral_centroid", "alpha_8_12_sum"])

    df_channels = pd.DataFrame(rows)
    df_subject = (
        df_channels.groupby(["Group", "ID", "group2"], as_index=False)[["spectral_centroid", "alpha_8_12_sum"]]
        .mean()
    )
    return df_subject


def _write_table(df, path_csv, title):
    df.to_csv(path_csv, index=False)
    print(f"\n{title}")
    print(df.to_string(index=False))
    print(f"[saved] {path_csv}")


def run_stage(stage_cfg, runtime):
    feature_dir = stage_cfg["feature_dir"]
    output_dir = os.path.join(runtime["validation_root"], stage_cfg["label"])
    os.makedirs(output_dir, exist_ok=True)

    df_msd, df_mtmi, df_psd, df_index = _read_stage_inputs(feature_dir)
    train_group_map = _train_group_map(df_index)

    if "map" in df_msd.columns:
        df_msd = df_msd[df_msd["map"] != -1].copy()

    print(f"\n[S2-VAL] stage={stage_cfg['label']}")
    print(f"[S2-VAL] input_dir={feature_dir}")
    print(f"[S2-VAL] output_dir={output_dir}")

    micro_rows = []
    for map_id in range(int(runtime["n_ms"])):
        map_label = _microstate_label(map_id)
        df_subject = _microstate_subject_features(df_msd, map_id, train_group_map)
        df_table = _build_summary_table(
            df_subject,
            [
                (f"{map_label} dur.avg.", "dur_avg"),
                (f"{map_label} mode", "mode"),
                (f"{map_label} Prel(mode)", "prel_mode"),
            ],
        )
        df_table.insert(0, "microstate", map_label)
        micro_rows.append(df_table)

    df_microstates = pd.concat(micro_rows, ignore_index=True) if micro_rows else pd.DataFrame()
    _write_table(
        df_microstates,
        os.path.join(output_dir, "microstate_summary.csv"),
        title=f"[S2-VAL] {stage_cfg['label']} - microstates",
    )

    df_mtmi_subject = _mtmi_subject_features(df_mtmi, train_group_map)
    df_mtmi_table = _build_summary_table(
        df_mtmi_subject,
        [
            ("MTMI skewness", "skewness"),
            ("MTMI kurtosis", "kurtosis"),
        ],
    )
    _write_table(
        df_mtmi_table,
        os.path.join(output_dir, "mtmi_summary.csv"),
        title=f"[S2-VAL] {stage_cfg['label']} - MTMI",
    )

    df_psd_subject = _psd_subject_features(df_psd, train_group_map)
    df_psd_table = _build_summary_table(
        df_psd_subject,
        [
            ("PSD spectral centroid", "spectral_centroid"),
            ("PSD sum 8-12Hz", "alpha_8_12_sum"),
        ],
    )
    _write_table(
        df_psd_table,
        os.path.join(output_dir, "psd_summary.csv"),
        title=f"[S2-VAL] {stage_cfg['label']} - PSD",
    )


def main():
    args = parse_args()
    cfg = load_configs(config_path=args.config)
    runtime = _load_runtime(cfg)

    print(f"[S2-VAL] validation_root={runtime['validation_root']}")
    for stage_cfg in runtime["stages"]:
        run_stage(stage_cfg, runtime)


if __name__ == "__main__":
    main()
