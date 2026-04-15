import json
import glob
import os
import warnings

import pandas as pd

from _config_loader import load_configs
from _header_parsing import parse_headers_to_pickle


LOW_EDF_WARNING_THRESHOLD = 2993  # from original publication
EXPECTED_TUAB_SUBSET_SUFFIXES = [
    "edf/train/normal/01_tcp_ar",
    "edf/train/abnormal/01_tcp_ar",
    "edf/eval/normal/01_tcp_ar",
    "edf/eval/abnormal/01_tcp_ar",
]


def _path_parts(path):
    return os.path.normpath(path).split(os.sep)


def snapshot_legacy_outputs(legacy_dir):
    if not os.path.isdir(legacy_dir):
        return {"exists": False, "file_count": 0, "total_bytes": 0, "max_mtime": 0}

    file_count = 0
    total_bytes = 0
    max_mtime = 0.0
    for root, _, files in os.walk(legacy_dir):
        for name in files:
            file_count += 1
            path = os.path.join(root, name)
            stat = os.stat(path)
            total_bytes += stat.st_size
            max_mtime = max(max_mtime, stat.st_mtime)
    return {
        "exists": True,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "max_mtime": max_mtime,
    }


def write_manifest(manifest_path, payload):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _discover_single_file(tuh_parent, filename, description, path_contains=None):
    pattern = os.path.join(tuh_parent, "**", filename)
    needle = (path_contains or "").lower()
    matches = sorted(
        os.path.normpath(path)
        for path in glob.glob(pattern, recursive=True)
        if os.path.isfile(path)
        and (not needle or needle in path.replace("\\", "/").lower())
    )
    if not matches:
        suffix = f" containing '{path_contains}'" if path_contains else ""
        raise FileNotFoundError(
            f"{description} not found under TUH parent '{tuh_parent}'. "
            f"Expected file '{filename}'{suffix}."
        )
    return matches[0]


def _count_files_recursive(root_dir):
    pattern = os.path.join(root_dir, "**", "*")
    return len([path for path in glob.glob(pattern, recursive=True) if os.path.isfile(path)])


def _count_edf_recursive(root_dir):
    pattern = os.path.join(root_dir, "**", "*.edf")
    return len([path for path in glob.glob(pattern, recursive=True) if os.path.isfile(path)])


def _discover_tuab_root_dir(tuh_parent):
    pattern = os.path.join(tuh_parent, "**", "tuh_eeg_abnormal")
    candidates = sorted(
        os.path.normpath(path)
        for path in glob.glob(pattern, recursive=True)
        if os.path.isdir(path)
    )
    if not candidates:
        raise FileNotFoundError(
            f"Directory 'tuh_eeg_abnormal' not found under TUH parent '{tuh_parent}'."
        )

    ranked = sorted(
        [(candidate, _count_edf_recursive(candidate)) for candidate in candidates],
        key=lambda item: (item[1], item[0]),
    )

    chosen_path, chosen_edf_count = ranked[-1]
    if chosen_edf_count <= 0:
        detail = ", ".join(f"{path} (edf={count})" for path, count in ranked)
        raise FileNotFoundError(
            "TUAB root directory exists but contains no EDF files. "
            f"Candidates: {detail}"
        )
    return chosen_path, chosen_edf_count, ranked


def _discover_expected_subset_dirs(tuab_root_abs):
    expected_patterns = [
        os.path.normpath(os.path.join(tuab_root_abs, "**", suffix.replace("/", os.sep)))
        for suffix in EXPECTED_TUAB_SUBSET_SUFFIXES
    ]
    matches_per_suffix = {
        suffix: sorted(
            os.path.normpath(path)
            for path in glob.glob(pattern, recursive=True)
            if os.path.isdir(path)
        )
        for suffix, pattern in zip(EXPECTED_TUAB_SUBSET_SUFFIXES, expected_patterns)
    }
    available = [
        max(matches, key=lambda path: (_count_edf_recursive(path), path))
        for matches in matches_per_suffix.values()
        if matches
    ]
    missing = [suffix for suffix, matches in matches_per_suffix.items() if not matches]
    return expected_patterns, available, missing


def verify_s0_local_prerequisites(tuh_parent, low_edf_warning_threshold=LOW_EDF_WARNING_THRESHOLD):
    tuh_parent_abs = os.path.normpath(tuh_parent)
    if not os.path.isdir(tuh_parent_abs):
        raise FileNotFoundError(f"TUH parent directory does not exist: {tuh_parent_abs}")

    readme_path = _discover_single_file(
        tuh_parent=tuh_parent_abs,
        filename="AAREADME.txt",
        description="TUH README",
        path_contains="tuh_eeg",
    )
    headers_path = _discover_single_file(
        tuh_parent=tuh_parent_abs,
        filename="headers.tar.gz",
        description="TUH headers archive",
        path_contains="tuh_eeg",
    )
    tuab_root_abs, tuab_root_edf_count, tuab_candidates = _discover_tuab_root_dir(tuh_parent_abs)

    if tuab_root_edf_count < low_edf_warning_threshold:
        warnings.warn(
            f"TUAB root contains only {tuab_root_edf_count} EDF files (< {low_edf_warning_threshold}). "
            "Dataset may be incomplete."
        )

    subset_dirs_checked, subset_dirs_available, subset_dirs_missing = _discover_expected_subset_dirs(
        tuab_root_abs
    )
    if not subset_dirs_available:
        raise FileNotFoundError(
            "No expected TUAB subset directories found under TUAB root. "
            f"Checked: {', '.join(subset_dirs_checked)}"
        )
    if subset_dirs_missing:
        warnings.warn(
            "Some expected TUAB subset directories are missing: "
            + ", ".join(subset_dirs_missing)
        )

    subset_available_edf_count = sum(_count_edf_recursive(path) for path in subset_dirs_available)
    if subset_available_edf_count <= 0:
        raise FileNotFoundError(
            "No EDF files found in available TUAB subset directories: "
            + ", ".join(subset_dirs_available)
        )

    return {
        "tuh_parent": tuh_parent_abs,
        "readme_path": readme_path,
        "headers_path": headers_path,
        "tuab_root": tuab_root_abs,
        "tuab_root_file_count": _count_files_recursive(tuab_root_abs),
        "tuab_root_edf_count": tuab_root_edf_count,
        "subset_dirs": subset_dirs_available,
        "subset_missing_count": len(subset_dirs_missing),
        "subset_edf_count": subset_available_edf_count,
    }


def collect_edf_paths(directories):
    return sorted(
        os.path.normpath(path)
        for directory in directories
        for path in glob.glob(os.path.join(directory, "**", "*.edf"), recursive=True)
        if os.path.isfile(path)
    )


def run_local_dataset_checks(config_path="configs.json", reparse_headers=None):
    cfg = load_configs(config_path=config_path)
    s0_cfg = cfg["s0"]
    if reparse_headers is None:
        reparse_headers = s0_cfg["reparse_headers"]

    prerequisites = verify_s0_local_prerequisites(tuh_parent=cfg["tuh_parent"])
    parsed_header, parsed_header_path, header_report = parse_headers_to_pickle(
        header_path=prerequisites["headers_path"],
        output_path=s0_cfg["parsed_header_path"],
        reparse=reparse_headers,
    )
    tuab_dirs = prerequisites["subset_dirs"]
    fpaths = collect_edf_paths(tuab_dirs)

    return {
        "config_path": cfg["meta"]["config_path"],
        "tuh_parent": prerequisites["tuh_parent"],
        "readme_path": prerequisites["readme_path"],
        "headers_path": prerequisites["headers_path"],
        "tuab_root": prerequisites["tuab_root"],
        "tuab_root_file_count": prerequisites["tuab_root_file_count"],
        "tuab_root_edf_count": prerequisites["tuab_root_edf_count"],
        "subset_dir_count": len(prerequisites["subset_dirs"]),
        "subset_missing_count": prerequisites["subset_missing_count"],
        "parsed_header_path": parsed_header_path,
        "n_parsed_header_entries": len(parsed_header),
        "n_tuab_edf_files": len(fpaths),
        "header_path_verified": header_report["absolute_path"],
    }


def counts_from_filenames(
    all_fnames,
    parse_set=lambda x: "train" if "train" in _path_parts(x) else "eval",
    parse_condition=lambda x: "normal" if "normal" in _path_parts(x) else "abnormal",
    transpose=True,
):
    df_filenames = pd.DataFrame({"fpath": all_fnames})
    df_filenames["fname"] = df_filenames["fpath"].apply(os.path.basename)
    df_filenames["set"] = df_filenames["fpath"].apply(parse_set)
    df_filenames["condition"] = df_filenames["fpath"].apply(parse_condition)
    df_filenames["basename"] = df_filenames["fname"].apply(os.path.basename)
    df_filenames["subject"] = df_filenames["basename"].apply(lambda x: x.split("_")[0])

    index = pd.MultiIndex.from_product(
        [["Normal", "Abnormal", "Shared"], ["Subjects", "Recordings"]],
        names=["Condition", "Counts"],
    )
    summary = pd.DataFrame(index=index, columns=["Train", "Evaluation", "Total"])

    summary.loc[("Normal", "Subjects"), "Train"] = df_filenames.query('set == "train" and condition == "normal"')["subject"].nunique()
    summary.loc[("Abnormal", "Subjects"), "Train"] = df_filenames.query('set == "train" and condition == "abnormal"')["subject"].nunique()
    summary.loc[("Normal", "Subjects"), "Evaluation"] = df_filenames.query('set == "eval" and condition == "normal"')["subject"].nunique()
    summary.loc[("Abnormal", "Subjects"), "Evaluation"] = df_filenames.query('set == "eval" and condition == "abnormal"')["subject"].nunique()
    summary.loc[("Normal", "Subjects"), "Total"] = df_filenames.query('condition == "normal"')["subject"].nunique()
    summary.loc[("Abnormal", "Subjects"), "Total"] = df_filenames.query('condition == "abnormal"')["subject"].nunique()

    summary.loc[("Normal", "Recordings"), "Train"] = df_filenames.query('set == "train" and condition == "normal"')["fpath"].count()
    summary.loc[("Abnormal", "Recordings"), "Train"] = df_filenames.query('set == "train" and condition == "abnormal"')["fpath"].count()
    summary.loc[("Normal", "Recordings"), "Evaluation"] = df_filenames.query('set == "eval" and condition == "normal"')["fpath"].count()
    summary.loc[("Abnormal", "Recordings"), "Evaluation"] = df_filenames.query('set == "eval" and condition == "abnormal"')["fpath"].count()
    summary.loc[("Normal", "Recordings"), "Total"] = df_filenames.query('condition == "normal"')["fpath"].count()
    summary.loc[("Abnormal", "Recordings"), "Total"] = df_filenames.query('condition == "abnormal"')["fpath"].count()

    summary.loc[("Shared", "Subjects"), "Train"] = len(
        set(df_filenames.query('set == "train" and condition == "normal"')["subject"]).intersection(
            set(df_filenames.query('set == "train" and condition == "abnormal"')["subject"])
        )
    )
    summary.loc[("Shared", "Subjects"), "Evaluation"] = len(
        set(df_filenames.query('set == "eval" and condition == "normal"')["subject"]).intersection(
            set(df_filenames.query('set == "eval" and condition == "abnormal"')["subject"])
        )
    )
    summary.loc[("Shared", "Subjects"), "Total"] = len(
        set(df_filenames.query('condition == "normal"')["subject"]).intersection(
            set(df_filenames.query('condition == "abnormal"')["subject"])
        )
    )
    summary.loc[("Shared", "Recordings"), "Train"] = len(
        set(df_filenames.query('set == "train" and condition == "normal"')["basename"]).intersection(
            set(df_filenames.query('set == "train" and condition == "abnormal"')["basename"])
        )
    )
    summary.loc[("Shared", "Recordings"), "Evaluation"] = len(
        set(df_filenames.query('set == "eval" and condition == "normal"')["basename"]).intersection(
            set(df_filenames.query('set == "eval" and condition == "abnormal"')["basename"])
        )
    )
    summary.loc[("Shared", "Recordings"), "Total"] = len(
        set(df_filenames.query('condition == "normal"')["basename"]).intersection(
            set(df_filenames.query('condition == "abnormal"')["basename"])
        )
    )

    if transpose:
        summary = summary.T

    return summary, df_filenames


def build_tuab_header_from_tuh_header(
    tuh_parent,
    fpaths,
    parsed_header_path,
    output_path,
):
    import pickle

    paths_relative_to_tuh = [os.path.relpath(fpath, tuh_parent).replace("\\", "/") for fpath in fpaths]
    parsed_header_path = os.path.abspath(os.path.normpath(parsed_header_path))
    with open(parsed_header_path, "rb") as f:
        tuh_header = pickle.load(f)

    tuh_files = [os.path.basename(entry["record"]["relative_path"]) for entry in tuh_header]
    entries_tuab = []
    rpaths_tuab = []
    rpaths_not_found = []

    for rpath in paths_relative_to_tuh:
        fname_tuab = os.path.basename(rpath)
        try:
            entry = tuh_header[tuh_files.index(fname_tuab)]
            entries_tuab.append(entry)
            rpaths_tuab.append(rpath)
        except ValueError:
            rpaths_not_found.append(rpath)

    for entry, rpath in zip(entries_tuab, rpaths_tuab):
        entry["record"]["relative_path"] = rpath

    output_path = os.path.abspath(os.path.normpath(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(entries_tuab, f)

    return entries_tuab, rpaths_not_found, output_path
