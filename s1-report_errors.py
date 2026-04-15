import argparse
import os
import re

import pandas as pd

from _config_loader import load_configs


def parse_args():
    parser = argparse.ArgumentParser(description="S1 error artifacts summary report.")
    parser.add_argument("--config", default="configs.json", help="Path to config file.")
    return parser.parse_args()


def _subject_from_record_stem(record_stem):
    stem = str(record_stem or "").strip()
    m = re.match(r"^(.*)_s\d+_t\d+$", stem)
    if m:
        return m.group(1)
    return stem


def _condition_cell(condition):
    text = str(condition or "").strip().lower()
    tokens = re.findall(r"[a-z]+", text)
    tok = set(tokens)
    if "abnormal" in tok and "train" in tok:
        return "abnormal_train"
    if "abnormal" in tok and ("eval" in tok or "evaluation" in tok):
        return "abnormal_eval"
    if "normal" in tok and "train" in tok:
        return "normal_train"
    if "normal" in tok and ("eval" in tok or "evaluation" in tok):
        return "normal_eval"
    return "unknown"


def _normalize_error_message(msg):
    text = str(msg or "").strip()
    low = text.lower()
    if "seg_no_segments_after_recompute" in low:
        return "seg_no_segments_after_recompute"
    if "too many noisy channels in the data to reliably perform ransac" in low:
        return "prep_ransac_insufficient_channels"
    if "no appropriate channels found for the given picks" in low:
        return "prep_no_channels"
    norm = re.sub(r"\d+(\.\d+)?", "<num>", text)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm if norm else "other"


def _infer_condition_from_file_path(file_path):
    path = str(file_path or "").replace("\\", "/").lower()
    is_train = "/train/" in path
    is_eval = ("/eval/" in path) or ("/evaluation/" in path)
    is_normal = "/normal/" in path
    is_abnormal = "/abnormal/" in path
    if is_train and is_normal:
        return "normal train"
    if is_train and is_abnormal:
        return "abnormal train"
    if is_eval and is_normal:
        return "normal eval"
    if is_eval and is_abnormal:
        return "abnormal eval"
    return "unknown"


def _parse_error_artifact(error_path):
    try:
        txt = open(error_path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return {
            "failure_category": "",
            "error_message": "",
            "error_signature": "unreadable_error_artifact",
            "file_path": "",
        }

    failure_category = ""
    m_cat = re.search(r"^failure_category:\s*(.+)$", txt, flags=re.MULTILINE)
    if m_cat:
        failure_category = m_cat.group(1).strip()

    file_path = ""
    m_fp = re.search(r"^file_path:\s*(.+)$", txt, flags=re.MULTILINE)
    if m_fp:
        file_path = m_fp.group(1).strip()

    error_message = ""
    m_msg = re.search(r"error_message:\s*\n(.*?)\n\s*\ntraceback:", txt, flags=re.DOTALL | re.IGNORECASE)
    if m_msg:
        error_message = m_msg.group(1).strip()
    elif "error_message:" in txt:
        tail = txt.split("error_message:", 1)[1].strip().splitlines()
        if tail:
            error_message = tail[0].strip()

    signature = failure_category.strip() if failure_category else _normalize_error_message(error_message)
    if not signature:
        signature = "other"
    return {
        "failure_category": failure_category,
        "error_message": error_message,
        "error_signature": signature,
        "file_path": file_path,
    }


def _collect_stage_errors(stage_dir):
    details = []
    if not os.path.isdir(stage_dir):
        return pd.DataFrame(details)

    for record_stem in sorted(os.listdir(stage_dir)):
        record_dir = os.path.join(stage_dir, record_stem)
        if not os.path.isdir(record_dir):
            continue
        candidates = [n for n in os.listdir(record_dir) if n.endswith("-error.txt")]
        if not candidates:
            continue
        error_path = os.path.join(record_dir, sorted(candidates)[0])
        parsed = _parse_error_artifact(error_path)
        condition = _infer_condition_from_file_path(parsed.get("file_path", ""))
        details.append(
            {
                "record_stem": record_stem,
                "subject_id": _subject_from_record_stem(record_stem),
                "condition": condition,
                "condition_cell": _condition_cell(condition),
                "error_signature": parsed["error_signature"],
                "failure_category": parsed["failure_category"],
                "error_message": parsed["error_message"],
                "error_path": error_path,
            }
        )
    return pd.DataFrame(details)


def _build_summary(df_details):
    if df_details.empty:
        return pd.DataFrame(
            columns=[
                "error_signature",
                "normal_train_recordings",
                "normal_train_subjects",
                "abnormal_train_recordings",
                "abnormal_train_subjects",
                "normal_eval_recordings",
                "normal_eval_subjects",
                "abnormal_eval_recordings",
                "abnormal_eval_subjects",
                "unknown_recordings",
                "unknown_subjects",
                "total_recordings",
                "total_subjects",
            ]
        )

    ordered_cells = ["normal_train", "abnormal_train", "normal_eval", "abnormal_eval", "unknown"]
    rows = []
    for sig, grp in df_details.groupby("error_signature", dropna=False):
        row = {"error_signature": sig}
        for cell in ordered_cells:
            g = grp[grp["condition_cell"] == cell]
            row[f"{cell}_recordings"] = int(len(g))
            row[f"{cell}_subjects"] = int(g["subject_id"].nunique())
        row["total_recordings"] = int(len(grp))
        row["total_subjects"] = int(grp["subject_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by=["total_recordings", "error_signature"], ascending=[False, True])


def _write_stage_report(stage_label, stage_dir, out_dir):
    try:
        os.makedirs(out_dir, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot create report directory '{out_dir}'. "
            "Close any file explorer/editor locks or choose a writable output location."
        ) from e
    df_details = _collect_stage_errors(stage_dir)
    df_summary = _build_summary(df_details)

    details_path = os.path.join(out_dir, "error_messages_details.csv")
    summary_path = os.path.join(out_dir, "error_messages_summary.csv")
    df_details.to_csv(details_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print(f"\n[S1-ERR] stage={stage_label}")
    print(f"[S1-ERR] stage_dir={stage_dir}")
    print(f"[S1-ERR] details={details_path} (rows={len(df_details)})")
    print(f"[S1-ERR] summary={summary_path} (rows={len(df_summary)})")
    if not df_summary.empty:
        print(df_summary.to_string(index=False))


def main():
    args = parse_args()
    cfg = load_configs(config_path=args.config)
    s1_root = cfg["s1"]["s1_parent"]

    segmented_dir = os.path.join(s1_root, "preprocessed_segmented")
    ica_dir = os.path.join(s1_root, "preprocessed_segmented_ICA")
    report_root = os.path.join(s1_root, "error_reports")

    _write_stage_report("segmented", segmented_dir, os.path.join(report_root, "segmented"))
    _write_stage_report("ica", ica_dir, os.path.join(report_root, "ica"))


if __name__ == "__main__":
    main()
