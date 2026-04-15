import argparse
import glob
import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed
from _config_loader import load_configs
from tqdm import tqdm

from s0_loading_utils import counts_from_filenames
from s1_preprocess_utils import (
    _build_runtime,
    load_parsed_header,
    process_tuh_entry,
    read_stage_metadata,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="S1 preprocessing orchestration (TUH/TUAB).")
    parser.add_argument("--config", default="configs.json", help="Path to config file.")
    parser.add_argument("--start", type=int, default=0, help="Start index in parsed TUAB header.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive end index in parsed TUAB header.")
    parser.add_argument("--n", type=int, default=None, help="Number of entries to process from start.")
    parser.add_argument(
        "--plot-segmentation",
        action="store_true",
        help="Show segmentation plots while processing.",
    )
    return parser


def _compute_window(total, start=0, end=None, n=None):
    if total < 0:
        raise ValueError("total must be >= 0")

    start_idx = max(int(start or 0), 0)
    if end is not None and n is not None:
        raise ValueError("Use either end or n, not both.")

    if end is not None:
        end_idx = min(int(end), total)
    elif n is not None:
        end_idx = min(start_idx + int(n), total)
    else:
        end_idx = total

    if end_idx < start_idx:
        end_idx = start_idx
    return start_idx, end_idx


def _extract_record_path(instance):
    desc = instance.get("description", {}) if isinstance(instance, dict) else {}
    rec = desc.get("record", {}) if isinstance(desc, dict) else {}
    rel = rec.get("relative_path")
    if rel:
        return str(rel).replace("\\", "/")
    filename = instance.get("filename") if isinstance(instance, dict) else None
    if filename:
        return str(filename).replace("\\", "/")
    return None


def _collect_stage_paths(stage_dir):
    metadata_paths = sorted(glob.glob(os.path.join(stage_dir, "*", "metadata.json")))
    rel_paths = []
    for p in metadata_paths:
        try:
            record_stem = Path(os.path.dirname(p)).name
            meta = read_stage_metadata(stage_dir, record_stem)
            rel = None
            desc = meta.get("description", {}) if isinstance(meta, dict) else {}
            rec = desc.get("record", {}) if isinstance(desc, dict) else {}
            if rec.get("relative_path"):
                rel = str(rec["relative_path"]).replace("\\", "/")
            elif meta.get("filename"):
                rel = str(meta["filename"]).replace("\\", "/")
            if rel:
                rel_paths.append(rel)
        except Exception as e:
            print(f"[S1][WARN] Could not load stage metadata for counts: {p} ({e})")
            continue
    return rel_paths


def _write_counts_summary_csv(runtime):
    stages = [
        ("preprocessed_segmented", runtime["segmented_dir"]),
        ("preprocessed_segmented_ICA", runtime["ica_dir"]),
    ]
    out_path = os.path.join(runtime["s1_output_root"], "preprocess_counts_summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    blocks = []
    for stage_name, stage_dir in stages:
        rel_paths = _collect_stage_paths(stage_dir)
        summary, _ = counts_from_filenames(rel_paths)
        blocks.append((stage_name, summary))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for idx, (stage_name, summary) in enumerate(blocks):
            f.write(f"Stage,{stage_name}\n")
            f.write(summary.to_csv())
            if idx < len(blocks) - 1:
                f.write("\n")
    return out_path


def _categorize_failure(failure):
    msg = str(failure.get("message") or "")
    if "Montage is not set" in msg:
        return "ica_missing_montage"
    if "One PCA component captures most of the explained variance" in msg:
        return "ica_pca_single_component"
    if "seg_no_segments_after_recompute" in msg:
        return "seg_no_segments_after_recompute"
    if "Too many noisy channels in the data to reliably perform RANSAC" in msg:
        return "prep_ransac_insufficient_channels"
    if "No appropriate channels found for the given picks" in msg:
        return "prep_no_channels"
    return "other"


def _safe_error_name(file_path, fallback="unknown"):
    raw = os.path.basename(file_path or fallback)
    stem, _ = os.path.splitext(raw)
    stem = stem or fallback
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def _error_stage_dir(runtime, failure):
    step = str(failure.get("step") or "")
    if step.startswith("ica:"):
        return runtime["ica_dir"]
    if step.startswith("segmented:") or step.startswith("raw:"):
        return runtime["segmented_dir"]
    return runtime["segmented_dir"]


def _write_error_artifact(runtime, failure):
    base = _safe_error_name(failure.get("file_path"), fallback=failure.get("record_stem") or "unknown")
    stage_dir = _error_stage_dir(runtime, failure)
    record_dir = os.path.join(stage_dir, base)
    os.makedirs(record_dir, exist_ok=True)

    # Defensive: keep error artifact as the only stage artifact for failed records.
    for stale_name in os.listdir(record_dir):
        stale_path = os.path.join(record_dir, stale_name)
        if os.path.isfile(stale_path) and (stale_name.lower().endswith(".edf") or stale_name == "metadata.json"):
            try:
                os.remove(stale_path)
            except OSError:
                pass

    path = os.path.join(record_dir, f"{base}-error.txt")

    lines = [
        f"timestamp: {datetime.utcnow().isoformat()}Z",
        f"record_id: {failure.get('record_id', 'unknown')}",
        f"record_stem: {failure.get('record_stem', 'unknown')}",
        f"file_path: {failure.get('file_path', 'unknown')}",
        f"step: {failure.get('step', 'unknown')}",
        f"failure_category: {failure.get('failure_category', 'other')}",
        f"seg_recompute_attempts: {failure.get('seg_recompute_attempts', 0)}",
        f"ica_retry_attempted: {failure.get('ica_retry_attempted', False)}",
        f"ica_retry_succeeded: {failure.get('ica_retry_succeeded', False)}",
        f"apply_ica: {runtime.get('apply_ica')}",
        f"referencing_strategy: {runtime.get('referencing_strategy')}",
        f"segmented_dir: {runtime.get('segmented_dir')}",
        f"ica_dir: {runtime.get('ica_dir')}",
        "",
        "error_message:",
        str(failure.get("message") or ""),
        "",
        "traceback:",
        str(failure.get("traceback") or ""),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    args = build_arg_parser().parse_args()
    cfg = load_configs(config_path=args.config)
    runtime = _build_runtime(cfg)
    parsed_header = load_parsed_header(runtime["parsed_header_path"])

    start_idx, end_idx = _compute_window(len(parsed_header), start=args.start, end=args.end, n=args.n)
    subset = parsed_header[start_idx:end_idx]
    n_jobs = int(cfg["n_jobs"])
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_tuh_entry)(
            entry=entry,
            runtime=runtime,
            plot_segmentation=args.plot_segmentation,
        )
        for entry in tqdm(subset, desc=f"S1 preprocessing dispatch (n_jobs={n_jobs})")
    )

    failures = []
    error_artifacts = []
    for result in results:
        if result["status"] != 0:
            failure = {
                "file_path": result.get("file_path"),
                "record_stem": result.get("record_stem"),
                "record_id": result.get("record_id"),
                "step": result.get("step"),
                "message": result.get("message"),
                "traceback": result.get("traceback"),
                "seg_recompute_attempts": result.get("seg_recompute_attempts", 0),
                "ica_retry_attempted": result.get("ica_retry_attempted", False),
                "ica_retry_succeeded": result.get("ica_retry_succeeded", False),
            }
            failure["failure_category"] = _categorize_failure(failure)
            failures.append(failure)
            err_artifact = _write_error_artifact(runtime, failure)
            error_artifacts.append(err_artifact)
            print(f"[S1][ERROR] {failure['file_path']}")
            print(f"  category: {failure['failure_category']}")
            if failure["message"]:
                print(f"  message: {failure['message']}")
            if failure["step"]:
                print(f"  step: {failure['step']}")
            print(f"  seg_recompute_attempts: {failure['seg_recompute_attempts']}")
            print(f"  ica_retry_attempted: {failure['ica_retry_attempted']}")
            print(f"  error_artifact: {err_artifact}")
            if failure["traceback"]:
                print(f"  traceback:\n{failure['traceback']}")

    summary = {
        "config_path": cfg["meta"]["config_path"],
        "parsed_header_path": runtime["parsed_header_path"],
        "processed_window": {"start": start_idx, "end": end_idx, "n": len(subset)},
        "referencing_strategy": runtime["referencing_strategy"],
        "apply_ica": runtime["apply_ica"],
        "n_jobs": n_jobs,
        "s1_output_root": runtime["s1_output_root"],
        "output_dirs": {
            "segmented": runtime["segmented_dir"],
            "ica": runtime["ica_dir"],
        },
        "ok_count": sum(1 for item in results if item["status"] == 0),
        "err_count": sum(1 for item in results if item["status"] == 1),
        "errors": failures,
        "error_artifacts": error_artifacts,
    }

    if runtime["persist_results"]:
        with open(runtime["results_pickle_path"], "wb") as f:
            pickle.dump(results, f)
        summary["results_pickle"] = runtime["results_pickle_path"]

    counts_summary_csv = _write_counts_summary_csv(runtime)
    summary["counts_summary_csv"] = counts_summary_csv
    summary["summary_json"] = runtime["summary_json_path"]

    with open(runtime["summary_json_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("S1 preprocessing completed")
    print(f"  Window: {summary['processed_window']}")
    print(f"  apply_ica: {summary['apply_ica']}")
    print(f"  Outputs: {summary['output_dirs']}")
    print(f"  OK: {summary['ok_count']}  ERR: {summary['err_count']}")
    print(f"  Summary JSON: {summary['summary_json']}")
    print(f"  Counts summary CSV: {summary['counts_summary_csv']}")
    if "results_pickle" in summary:
        print(f"  Results pickle: {summary['results_pickle']}")


if __name__ == "__main__":
    main()
