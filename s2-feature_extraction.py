import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import shutil
import warnings
from itertools import groupby

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.filter import filter_data
from scipy.signal import find_peaks
from tqdm import tqdm

import s2_microstates_utils as ms
from _config_loader import load_configs

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 12})
_WORKER_MODEL_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(description="S2 feature extraction stage.")
    parser.add_argument("--config", default="configs.json", help="Path to config file.")
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Enable interactive plots (default: disabled/headless).",
    )
    return parser.parse_args()


def _assert_parquet_engine():
    try:
        import pyarrow  # noqa: F401
        return
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401
        return
    except Exception:
        pass
    raise RuntimeError(
        "No parquet engine available. Install 'pyarrow' (recommended) or 'fastparquet' in your environment."
    )


def _resolve_path(base_dir, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty path string, got: {value!r}")
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(base_dir, expanded))


def _path_parts(path):
    return [part for part in str(path).replace("\\", "/").split("/") if part]


def get_condition(description):
    if not isinstance(description, dict):
        return "unknown"
    if "attribuzione_stratificazio" in description:
        value = description["attribuzione_stratificazio"]
        return "unknown" if pd.isna(value) else str(value)
    rec = description.get("record", {})
    rel = rec.get("relative_path")
    if not rel:
        return "unknown"
    parts = _path_parts(rel)
    is_normal = "normal" in parts
    is_train = "train" in parts
    if is_normal and is_train:
        return "normal train"
    if is_normal and not is_train:
        return "normal eval"
    if (not is_normal) and is_train:
        return "abnormal train"
    return "abnormal eval"


def _load_runtime(cfg, cli_show_plots=False):
    s2_cfg = cfg.get("s2", {})
    ext_cfg = s2_cfg.get("extraction", {})
    if "s2_parent" not in s2_cfg:
        raise KeyError("Missing required cfg['s2']['s2_parent'].")
    if not isinstance(ext_cfg, dict):
        raise KeyError("Expected cfg['s2']['extraction'] to be an object.")

    defaults = {
        "output_subdir_with_ica": "features_with_ica",
        "output_subdir_without_ica": "features_without_ica",
        "recompute": True,
        "show_plots": False,
        "psd": {
            "fmin": 0.5,
            "fmax": 32.0,
            "n_fft_sec": 2.0,
            "n_overlap_sec": 1.0,
        },
        "microstates": {
            "n_ms": 4,
            "metamaps_json": "assets/microstates/metamaps_export.json",
            "gfp_h_freq": 32.0,
            "min_peak_distance_ms": 30.0,
        },
    }

    psd_cfg = ext_cfg.get("psd", {})
    if not isinstance(psd_cfg, dict):
        raise KeyError("Expected cfg['s2']['extraction']['psd'] to be an object.")
    micro_cfg = ext_cfg.get("microstates", {})
    if not isinstance(micro_cfg, dict):
        raise KeyError("Expected cfg['s2']['extraction']['microstates'] to be an object.")

    s2_parent = _resolve_path(cfg["output_parent"], s2_cfg["s2_parent"])
    out_with_ica = _resolve_path(
        s2_parent,
        str(ext_cfg.get("output_subdir_with_ica", defaults["output_subdir_with_ica"])),
    )
    out_without_ica = _resolve_path(
        s2_parent,
        str(ext_cfg.get("output_subdir_without_ica", defaults["output_subdir_without_ica"])),
    )
    os.makedirs(out_with_ica, exist_ok=True)
    os.makedirs(out_without_ica, exist_ok=True)

    metamaps_json = _resolve_path(
        cfg["meta"]["project_root"],
        str(micro_cfg.get("metamaps_json", defaults["microstates"]["metamaps_json"])),
    )
    if not os.path.exists(metamaps_json):
        raise FileNotFoundError(f"Metamaps json not found: {metamaps_json}")

    pycrostates_params = micro_cfg.get("pycrostates_params")
    if pycrostates_params is None:
        raise KeyError(
            "Missing required cfg['s2']['extraction']['microstates']['pycrostates_params']."
        )

    show_plots = bool(ext_cfg.get("show_plots", defaults["show_plots"])) or bool(cli_show_plots)
    recompute = ext_cfg.get("recompute")
    if recompute is None:
        recompute = ext_cfg.get("refresh", defaults["recompute"])

    runtime = {
        "s2_parent": s2_parent,
        "n_jobs": int(cfg.get("n_jobs", 1)),
        "recompute": bool(recompute),
        "show_plots": show_plots,
        "montage": cfg["s1"]["montage"],
        "psd": {
            "fmin": float(psd_cfg.get("fmin", defaults["psd"]["fmin"])),
            "fmax": float(psd_cfg.get("fmax", defaults["psd"]["fmax"])),
            "n_fft_sec": float(psd_cfg.get("n_fft_sec", defaults["psd"]["n_fft_sec"])),
            "n_overlap_sec": float(psd_cfg.get("n_overlap_sec", defaults["psd"]["n_overlap_sec"])),
        },
        "microstates": {
            "n_ms": int(micro_cfg.get("n_ms", defaults["microstates"]["n_ms"])),
            "metamaps_json": metamaps_json,
            "gfp_h_freq": float(micro_cfg.get("gfp_h_freq", defaults["microstates"]["gfp_h_freq"])),
            "min_peak_distance_ms": float(
                micro_cfg.get(
                    "min_peak_distance_ms",
                    defaults["microstates"]["min_peak_distance_ms"],
                )
            ),
            "pycrostates_params": pycrostates_params,
        },
        "stages": [
            {
                "label": "without_ica",
                "expected_field": "EEG_seg",
                "stage_dir": os.path.join(cfg["s1"]["s1_parent"], "preprocessed_segmented"),
                "output_dir": out_without_ica,
            },
            {
                "label": "with_ica",
                "expected_field": "EEG_ICA",
                "stage_dir": os.path.join(cfg["s1"]["s1_parent"], "preprocessed_segmented_ICA"),
                "output_dir": out_with_ica,
            },
        ],
    }
    return runtime


def _safe_read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_stage_records(stage_dir, expected_field):
    if not os.path.isdir(stage_dir):
        return [], []

    records = []
    index_rows = []
    for rec_stem in sorted(os.listdir(stage_dir)):
        record_dir = os.path.join(stage_dir, rec_stem)
        if not os.path.isdir(record_dir):
            continue
        metadata_path = os.path.join(record_dir, "metadata.json")
        row = {
            "record_stem": rec_stem,
            "record_dir": record_dir,
            "metadata_path": metadata_path,
            "filename": "",
            "condition": "unknown",
            "status": "excluded",
            "reason": "",
            "n_edf_listed": 0,
            "n_edf_existing": 0,
            "segments_processed": 0,
            "segments_failed": 0,
            "runtime_status": "",
        }

        if not os.path.isfile(metadata_path):
            has_error_txt = any(name.endswith("-error.txt") for name in os.listdir(record_dir))
            row["reason"] = "error_artifact_only" if has_error_txt else "missing_metadata"
            index_rows.append(row)
            continue

        try:
            meta = _safe_read_json(metadata_path)
        except Exception:
            row["reason"] = "invalid_metadata_json"
            index_rows.append(row)
            continue

        row["filename"] = str(meta.get("filename", ""))
        row["condition"] = get_condition(meta.get("description", {}))
        if meta.get("expected_field") != expected_field:
            row["reason"] = f"unexpected_expected_field:{meta.get('expected_field')}"
            index_rows.append(row)
            continue
        if meta.get("errors_log"):
            row["reason"] = "metadata_errors_log"
            index_rows.append(row)
            continue

        edf_files = meta.get("edf_files")
        if not isinstance(edf_files, list) or len(edf_files) == 0:
            row["reason"] = "missing_edf_list"
            index_rows.append(row)
            continue

        edf_paths = [os.path.join(record_dir, rel) for rel in edf_files]
        existing = [p for p in edf_paths if os.path.isfile(p)]
        row["n_edf_listed"] = len(edf_files)
        row["n_edf_existing"] = len(existing)
        if len(existing) == 0:
            row["reason"] = "missing_all_edf_files"
            index_rows.append(row)
            continue

        row["status"] = "included" if len(existing) == len(edf_paths) else "included_partial"
        row["reason"] = "ok" if row["status"] == "included" else "missing_some_edf_files"
        records.append({"record_stem": rec_stem, "metadata": meta, "edf_paths": existing})
        index_rows.append(row)

    return records, index_rows


def _load_raw_with_montage(edf_path, montage_name):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.set_montage(
            mne.channels.make_standard_montage(montage_name),
            match_case=False,
            on_missing="ignore",
            verbose=False,
        )
    return raw


def process_record(record, runtime, meta_modkmeans):
    ms_cfg = {"pycrostates_params": runtime["microstates"]["pycrostates_params"]}
    condition = get_condition(record["metadata"].get("description", {}))
    payloads = []
    failures = []
    for edf_path in record["edf_paths"]:
        try:
            raw = _load_raw_with_montage(edf_path, runtime["montage"])
            ms_seg = ms.microstates_extraction(raw, meta_modkmeans, ms_cfg)
            n_fft = max(1, int(round(raw.info["sfreq"] * runtime["psd"]["n_fft_sec"])))
            n_overlap = max(0, int(round(raw.info["sfreq"] * runtime["psd"]["n_overlap_sec"])))
            if n_overlap >= n_fft:
                n_overlap = max(0, n_fft - 1)
            psd, psd_freqs = mne.time_frequency.psd_array_welch(
                raw.get_data("eeg"),
                raw.info["sfreq"],
                fmin=runtime["psd"]["fmin"],
                fmax=runtime["psd"]["fmax"],
                n_fft=n_fft,
                n_overlap=n_overlap,
                average="mean",
                verbose="ERROR",
            )
            payloads.append(
                {
                    "ms": ms_seg,
                    "psd": psd,
                    "psd_freqs": psd_freqs,
                    "channels": list(raw.info["ch_names"]),
                    "sfreq": float(raw.info["sfreq"]),
                }
            )
        except Exception as e:
            failures.append(str(e))

    if not payloads:
        return {"status": "failed", "condition": condition, "segments_processed": 0, "segments_failed": len(failures)}

    filt_gfps = [
        filter_data(
            p["ms"]["gfp"][np.newaxis, :],
            p["sfreq"],
            l_freq=None,
            h_freq=runtime["microstates"]["gfp_h_freq"],
            verbose="ERROR",
        ).squeeze()
        for p in payloads
    ]
    overall_gfp_std = float(np.std(np.concatenate(filt_gfps)))
    min_peak_distance_ms = runtime["microstates"]["min_peak_distance_ms"]

    mtmi_frames = []
    psd_frames = []
    msd_frames = []
    for p, gfp_flt in zip(payloads, filt_gfps):
        min_distance = max(1, int(round((min_peak_distance_ms / 1000.0) * p["sfreq"])))
        gfp_minima, _ = find_peaks(-gfp_flt, distance=min_distance, prominence=overall_gfp_std)
        mtmi = np.diff(gfp_minima) * (1000.0 / p["sfreq"]) if len(gfp_minima) >= 2 else np.array([])
        mtmi_frames.append(pd.DataFrame({"Group": condition, "ID": record["record_stem"], "MTMI Time [ms]": mtmi}))

        df_spectrum = pd.DataFrame(p["psd"], columns=p["psd_freqs"], index=p["channels"]).reset_index(names="channels")
        df_spectrum = df_spectrum.melt(id_vars=["channels"], var_name="f", value_name="PSD")
        df_spectrum["Group"] = condition
        df_spectrum["ID"] = record["record_stem"]
        psd_frames.append(df_spectrum)

        df_durations = pd.DataFrame(
            [(map_id, len(list(grp))) for map_id, grp in groupby(p["ms"]["sequence"])],
            columns=["map", "Duration"],
        )
        df_durations["Group"] = condition
        df_durations["ID"] = record["record_stem"]
        msd_frames.append(df_durations)

    return {
        "status": "ok",
        "condition": condition,
        "segments_processed": len(payloads),
        "segments_failed": len(failures),
        "mtmi_frames": mtmi_frames,
        "psd_frames": psd_frames,
        "msd_frames": msd_frames,
    }


def _get_worker_model(model_payload):
    key = (
        int(model_payload["n_ms"]),
        model_payload["metamaps_json"],
    )
    cached = _WORKER_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model = ms.PycroModKMeans(
        model_payload["maps"],
        model_payload["gev"],
        model_payload["lbl"],
        plot=False,
    )
    _WORKER_MODEL_CACHE[key] = model
    return model


def _process_record_worker(record, runtime, model_payload):
    model = _get_worker_model(model_payload)
    out = process_record(record, runtime, model)
    out["record_stem"] = record["record_stem"]
    return out


def _concat_or_empty(frames, columns):
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _write_counts_summary(path, stage_cfg, records_index, df_mtmi, df_psd, df_msd):
    n_included_complete = int((records_index["status"] == "included").sum())
    n_included_partial = int((records_index["status"] == "included_partial").sum())
    n_excluded = int((records_index["status"] == "excluded").sum())
    rows = [
        {"metric": "stage_label", "value": stage_cfg["label"]},
        {"metric": "stage_dir", "value": stage_cfg["stage_dir"]},
        {"metric": "output_dir", "value": stage_cfg["output_dir"]},
        {"metric": "records_total", "value": int(len(records_index))},
        {"metric": "records_included", "value": int(n_included_complete + n_included_partial)},
        {"metric": "records_included_complete", "value": n_included_complete},
        {"metric": "records_included_partial", "value": n_included_partial},
        {"metric": "records_excluded", "value": n_excluded},
        {"metric": "records_processed_ok", "value": int((records_index["runtime_status"] == "ok").sum())},
        {"metric": "records_processed_failed", "value": int((records_index["runtime_status"] == "failed").sum())},
        {"metric": "segments_processed", "value": int(records_index["segments_processed"].sum())},
        {"metric": "segments_failed", "value": int(records_index["segments_failed"].sum())},
        {"metric": "rows_MTMI", "value": int(len(df_mtmi))},
        {"metric": "rows_PSD", "value": int(len(df_psd))},
        {"metric": "rows_MSD", "value": int(len(df_msd))},
    ]
    reason_counts = records_index["reason"].fillna("unknown").value_counts()
    rows.extend({"metric": f"records_reason_{reason}", "value": int(n)} for reason, n in reason_counts.items())
    cond_counts = records_index.loc[records_index["runtime_status"] == "ok", "condition"].fillna("unknown").value_counts()
    rows.extend({"metric": f"records_ok_{cond}", "value": int(n)} for cond, n in cond_counts.items())
    pd.DataFrame(rows).to_csv(path, index=False)


def _empty_index_df():
    return pd.DataFrame(
        columns=[
            "record_stem",
            "record_dir",
            "metadata_path",
            "filename",
            "condition",
            "status",
            "reason",
            "n_edf_listed",
            "n_edf_existing",
            "segments_processed",
            "segments_failed",
            "runtime_status",
        ]
    )


def _record_shard_dir(stage_output_dir, record_stem):
    return os.path.join(stage_output_dir, record_stem)


def _write_json_atomic(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _write_parquet_atomic(df, path):
    tmp = f"{path}.tmp.parquet"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _reset_record_shard_dir(stage_output_dir, record_stem):
    shard_dir = _record_shard_dir(stage_output_dir, record_stem)
    if os.path.isdir(shard_dir):
        shutil.rmtree(shard_dir, ignore_errors=True)
    os.makedirs(shard_dir, exist_ok=True)
    return shard_dir


def _clear_stage_record_shards(stage_output_dir):
    if not os.path.isdir(stage_output_dir):
        return
    for name in os.listdir(stage_output_dir):
        path = os.path.join(stage_output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def _write_record_shard(stage_output_dir, row, mtmi_df=None, psd_df=None, msd_df=None):
    record_stem = row["record_stem"]
    shard_dir = _reset_record_shard_dir(stage_output_dir, record_stem)
    status_path = os.path.join(shard_dir, "status.json")
    _write_json_atomic(status_path, row)

    if mtmi_df is not None:
        _write_parquet_atomic(mtmi_df, os.path.join(shard_dir, "mtmi.parquet"))
    if psd_df is not None:
        _write_parquet_atomic(psd_df, os.path.join(shard_dir, "psd.parquet"))
    if msd_df is not None:
        _write_parquet_atomic(msd_df, os.path.join(shard_dir, "msd.parquet"))


def _read_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _can_reuse_completed_shard(stage_output_dir, record_stem):
    rec_dir = _record_shard_dir(stage_output_dir, record_stem)
    status_path = os.path.join(rec_dir, "status.json")
    mtmi_path = os.path.join(rec_dir, "mtmi.parquet")
    psd_path = os.path.join(rec_dir, "psd.parquet")
    msd_path = os.path.join(rec_dir, "msd.parquet")
    if not (os.path.isfile(status_path) and os.path.isfile(mtmi_path) and os.path.isfile(psd_path) and os.path.isfile(msd_path)):
        return False, None
    status = _read_json_safe(status_path)
    if not isinstance(status, dict):
        return False, None
    if status.get("runtime_status") != "ok":
        return False, status
    return True, status


def _aggregate_stage_from_shards(stage_output_dir, allowed_stems=None):
    status_rows = []
    mtmi_frames = []
    psd_frames = []
    msd_frames = []

    if not os.path.isdir(stage_output_dir):
        return _empty_index_df(), _concat_or_empty([], ["Group", "ID", "MTMI Time [ms]"]), _concat_or_empty(
            [], ["channels", "f", "PSD", "Group", "ID"]
        ), _concat_or_empty([], ["map", "Duration", "Group", "ID"])

    allowed = set(allowed_stems) if allowed_stems is not None else None
    for rec_stem in sorted(os.listdir(stage_output_dir)):
        if allowed is not None and rec_stem not in allowed:
            continue
        rec_dir = os.path.join(stage_output_dir, rec_stem)
        if not os.path.isdir(rec_dir):
            continue
        status_path = os.path.join(rec_dir, "status.json")
        if not os.path.isfile(status_path):
            continue
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status_rows.append(json.load(f))
        except Exception:
            continue

        mtmi_path = os.path.join(rec_dir, "mtmi.parquet")
        psd_path = os.path.join(rec_dir, "psd.parquet")
        msd_path = os.path.join(rec_dir, "msd.parquet")
        if os.path.isfile(mtmi_path):
            mtmi_frames.append(pd.read_parquet(mtmi_path))
        if os.path.isfile(psd_path):
            psd_frames.append(pd.read_parquet(psd_path))
        if os.path.isfile(msd_path):
            msd_frames.append(pd.read_parquet(msd_path))

    records_index = (
        pd.DataFrame(status_rows).sort_values(["status", "record_stem"]).reset_index(drop=True)
        if status_rows
        else _empty_index_df()
    )
    df_mtmi = _concat_or_empty(mtmi_frames, ["Group", "ID", "MTMI Time [ms]"])
    df_psd = _concat_or_empty(psd_frames, ["channels", "f", "PSD", "Group", "ID"])
    df_msd = _concat_or_empty(msd_frames, ["map", "Duration", "Group", "ID"])
    return records_index, df_mtmi, df_psd, df_msd


def run_stage(stage_cfg, runtime, meta_modkmeans):
    stage_dir = stage_cfg["stage_dir"]
    output_dir = stage_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    if runtime["recompute"]:
        _clear_stage_record_shards(output_dir)

    if not os.path.isdir(stage_dir):
        print(f"[S2][WARN] Missing stage directory for '{stage_cfg['label']}': {stage_dir}")
        records_index = _empty_index_df()
        df_mtmi = _concat_or_empty([], ["Group", "ID", "MTMI Time [ms]"])
        df_psd = _concat_or_empty([], ["channels", "f", "PSD", "Group", "ID"])
        df_msd = _concat_or_empty([], ["map", "Duration", "Group", "ID"])
    else:
        records, index_rows = discover_stage_records(stage_dir, stage_cfg["expected_field"])
        record_by_stem = {r["record_stem"]: r for r in records}
        idx_map = {row["record_stem"]: row for row in index_rows}
        print(
            f"[S2] stage={stage_cfg['label']} records_to_process={len(records)} "
            f"(n_jobs={runtime['n_jobs']}, recompute={runtime['recompute']})"
        )

        cluster_names = getattr(meta_modkmeans, "_cluster_names", None)
        if cluster_names is None:
            cluster_names = getattr(meta_modkmeans, "cluster_names", None)
        if cluster_names is None:
            cluster_names = [str(i) for i in range(int(runtime["microstates"]["n_ms"]))]

        model_payload = {
            "maps": meta_modkmeans.cluster_centers_,
            "gev": getattr(meta_modkmeans, "_GEV_"),
            "lbl": list(cluster_names),
            "n_ms": int(runtime["microstates"]["n_ms"]),
            "metamaps_json": runtime["microstates"]["metamaps_json"],
        }
        dispatch_records = []
        for row in sorted(index_rows, key=lambda r: r["record_stem"]):
            rec_stem = row["record_stem"]
            if row["status"] in {"included", "included_partial"} and rec_stem in record_by_stem:
                if not runtime["recompute"]:
                    reuse_ok, prev_status = _can_reuse_completed_shard(output_dir, rec_stem)
                    if reuse_ok and prev_status is not None:
                        row["runtime_status"] = prev_status.get("runtime_status", "ok")
                        row["condition"] = prev_status.get("condition", row["condition"])
                        row["segments_processed"] = int(prev_status.get("segments_processed", row["segments_processed"]))
                        row["segments_failed"] = int(prev_status.get("segments_failed", row["segments_failed"]))
                        row["reason"] = prev_status.get("reason", row["reason"])
                        continue
                dispatch_records.append(record_by_stem[rec_stem])

        if dispatch_records:
            future_map = {}
            with ProcessPoolExecutor(max_workers=max(1, int(runtime["n_jobs"]))) as ex:
                for record in dispatch_records:
                    future = ex.submit(_process_record_worker, record, runtime, model_payload)
                    future_map[future] = record["record_stem"]

                for future in tqdm(
                    as_completed(future_map),
                    total=len(future_map),
                    desc=f"S2 processing ({stage_cfg['label']})",
                ):
                    rec_stem = future_map[future]
                    row = idx_map[rec_stem]
                    try:
                        out = future.result()
                    except Exception as e:
                        row["runtime_status"] = "failed"
                        row["reason"] = f"worker_exception:{e}"
                        row["segments_processed"] = 0
                        row["segments_failed"] = int(row.get("n_edf_existing", 0) or 0)
                        _write_record_shard(output_dir, row, mtmi_df=None, psd_df=None, msd_df=None)
                        continue

                    row["runtime_status"] = "ok" if out["status"] == "ok" else "failed"
                    row["condition"] = out["condition"]
                    row["segments_processed"] = int(out["segments_processed"])
                    row["segments_failed"] = int(out["segments_failed"])
                    if out["status"] == "ok":
                        mtmi_df = _concat_or_empty(out["mtmi_frames"], ["Group", "ID", "MTMI Time [ms]"])
                        psd_df = _concat_or_empty(out["psd_frames"], ["channels", "f", "PSD", "Group", "ID"])
                        msd_df = _concat_or_empty(out["msd_frames"], ["map", "Duration", "Group", "ID"])
                        _write_record_shard(output_dir, row, mtmi_df=mtmi_df, psd_df=psd_df, msd_df=msd_df)
                    else:
                        row["reason"] = "no_valid_segments"
                        _write_record_shard(output_dir, row, mtmi_df=None, psd_df=None, msd_df=None)

        for row in sorted(index_rows, key=lambda r: r["record_stem"]):
            if row["runtime_status"] == "":
                # Excluded/undispatched records still get an explicit status marker.
                _write_record_shard(output_dir, row, mtmi_df=None, psd_df=None, msd_df=None)

        records_index, df_mtmi, df_psd, df_msd = _aggregate_stage_from_shards(
            output_dir,
            allowed_stems=[r["record_stem"] for r in index_rows],
        )

    path_mtmi = os.path.join(output_dir, "df_MTMI.parquet")
    path_psd = os.path.join(output_dir, "df_PSD.parquet")
    path_msd = os.path.join(output_dir, "df_MSD.parquet")
    path_counts = os.path.join(output_dir, "s2_counts_summary.csv")
    path_index = os.path.join(output_dir, "s2_records_index.csv")

    df_mtmi.to_parquet(path_mtmi, index=False)
    df_psd.to_parquet(path_psd, index=False)
    df_msd.to_parquet(path_msd, index=False)
    records_index.to_csv(path_index, index=False)
    _write_counts_summary(path_counts, stage_cfg, records_index, df_mtmi, df_psd, df_msd)

    print(f"[S2] stage={stage_cfg['label']} MTMI parquet: {path_mtmi} (rows={len(df_mtmi)})")
    print(f"[S2] stage={stage_cfg['label']} PSD parquet: {path_psd} (rows={len(df_psd)})")
    print(f"[S2] stage={stage_cfg['label']} MSD parquet: {path_msd} (rows={len(df_msd)})")
    print(f"[S2] stage={stage_cfg['label']} records index csv: {path_index}")
    print(f"[S2] stage={stage_cfg['label']} counts summary csv: {path_counts}")

    return {"df_mtmi": df_mtmi, "df_psd": df_psd, "df_msd": df_msd}


def _quick_plots(df_mtmi, df_psd, df_msd):
    import seaborn as sns

    colors = {
        "normal train": "#0072B2",
        "normal eval": "#56B4E9",
        "abnormal train": "#D55E00",
        "abnormal eval": "#F0E442",
        "unknown": "#666666",
    }
    if not df_psd.empty:
        plt.figure()
        df_avg = df_psd.groupby(["Group", "ID", "f"])["PSD"].mean().reset_index()
        sns.lineplot(
            df_avg,
            x="f",
            y="PSD",
            hue="Group",
            estimator="median",
            errorbar=("pi", 50),
            alpha=0.8,
            err_kws={"alpha": 0.3},
            palette=colors,
        )
        plt.gca().set_yscale("log")
        plt.gca().set_ylabel("PSD [dB uV^2/Hz]")
        plt.show(block=False)

    if not df_mtmi.empty:
        tau = 1000 / 128
        max_time_ms = 600
        bins = np.arange(0, max_time_ms + tau, tau) - tau / 2
        edf_rows = []
        for (group, rec_id), grp in df_mtmi.groupby(["Group", "ID"]):
            hist, _ = np.histogram(grp["MTMI Time [ms]"], bins=bins, density=True)
            edf_rows.append(pd.DataFrame({"Group": group, "ID": rec_id, "k": np.arange(len(hist)), "density": hist}))
        if edf_rows:
            plt.figure()
            sns.lineplot(
                pd.concat(edf_rows, ignore_index=True),
                x="k",
                y="density",
                hue="Group",
                estimator="median",
                errorbar=("pi", 50),
                alpha=0.8,
                err_kws={"alpha": 0.3},
                palette=colors,
            )
            plt.show(block=False)

    if not df_msd.empty:
        tau = 1000 / 128
        max_time_ms = 600
        bins = np.arange(0, max_time_ms + tau, tau) - tau / 2
        edf_ms = []
        for (group, rec_id, map_id), grp in df_msd.query("map != -1").groupby(["Group", "ID", "map"]):
            hist, _ = np.histogram(grp["Duration"], bins=bins, density=True)
            edf_ms.append(
                pd.DataFrame({"Group": group, "ID": rec_id, "map": map_id, "k": np.arange(len(hist)), "density": hist})
            )
        if edf_ms:
            df_edf_ms = pd.concat(edf_ms, ignore_index=True)
            unique_maps = sorted(df_edf_ms["map"].unique())
            fig, axes = plt.subplots(len(unique_maps), 1, figsize=(10, 5 * len(unique_maps)), sharex=True)
            axes = [axes] if len(unique_maps) == 1 else axes
            for idx, map_id in enumerate(unique_maps):
                sns.lineplot(
                    data=df_edf_ms[df_edf_ms["map"] == map_id],
                    x="k",
                    y="density",
                    hue="Group",
                    estimator="median",
                    errorbar=("pi", 50),
                    alpha=0.8,
                    err_kws={"alpha": 0.3},
                    ax=axes[idx],
                    palette=colors,
                )
                axes[idx].set_title(f"Map {map_id}")
                axes[idx].set_ylabel("Density")
                axes[idx].grid(True)
            plt.xlabel("k")
            plt.tight_layout()
            plt.show(block=False)


def main():
    args = parse_args()
    cfg = load_configs(config_path=args.config)
    _assert_parquet_engine()
    runtime = _load_runtime(cfg, cli_show_plots=args.show_plots)

    print(f"[S2] s2_parent={runtime['s2_parent']}")
    print(f"[S2] features_without_ica={runtime['stages'][0]['output_dir']}")
    print(f"[S2] features_with_ica={runtime['stages'][1]['output_dir']}")

    meta_maps, meta_gev, meta_lbl = ms.LoadMetamaps(
        filename=runtime["microstates"]["metamaps_json"],
        n_ms=runtime["microstates"]["n_ms"],
        plot=runtime["show_plots"],
    )
    meta_modkmeans = ms.PycroModKMeans(meta_maps, meta_gev, meta_lbl, plot=runtime["show_plots"])

    stage_outputs = []
    for stage_cfg in runtime["stages"]:
        stage_outputs.append(run_stage(stage_cfg, runtime, meta_modkmeans))

    if runtime["show_plots"]:
        for out in stage_outputs:
            _quick_plots(out["df_mtmi"], out["df_psd"], out["df_msd"])


if __name__ == "__main__":
    main()
