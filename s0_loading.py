import argparse
import json
import os
from datetime import datetime

from _config_loader import load_configs
from _header_parsing import parse_headers_to_pickle
from s0_loading_utils import (
    build_tuab_header_from_tuh_header,
    collect_edf_paths,
    counts_from_filenames,
    snapshot_legacy_outputs,
    verify_s0_local_prerequisites,
    write_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(description="S0 loader stage.")
    parser.add_argument("--config", default="configs.json", help="Path to configs.json")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_configs(config_path=args.config)
    s0_cfg = cfg["s0"]

    legacy_dir = os.path.join(cfg["meta"]["project_root"], "legacy_outputs")
    legacy_before = snapshot_legacy_outputs(legacy_dir)

    print("Step 0/3: Local TUH prerequisite validation")
    prerequisites = verify_s0_local_prerequisites(
        tuh_parent=cfg["tuh_parent"],
    )
    print("  Local prerequisites passed")

    print("Step 1/3: Required TUH objects verification")
    print("  README verified:", prerequisites["readme_path"])
    print("  Headers archive verified:", prerequisites["headers_path"])
    print("  TUAB root verified:", prerequisites["tuab_root"])
    print("  TUAB root files currently present:", prerequisites["tuab_root_file_count"])
    print("  TUAB root EDF files currently present:", prerequisites["tuab_root_edf_count"])
    print("  TUAB subset dirs available:", len(prerequisites["subset_dirs"]))
    print("  TUAB subset dirs missing:", prerequisites["subset_missing_count"])
    print("  TUAB subset EDF files available:", prerequisites["subset_edf_count"])

    print("Step 2/3: Header parsing verification")
    parsed_header, parsed_header_path, header_report = parse_headers_to_pickle(
        header_path=prerequisites["headers_path"],
        output_path=s0_cfg["parsed_header_path"],
        reparse=s0_cfg["reparse_headers"],
    )
    print(f"  Header entries: {len(parsed_header)}")
    print(f"  Parsed header file: {parsed_header_path}")

    print("Step 3/3: Dataset readiness and counts")
    tuab_dirs = prerequisites["subset_dirs"]
    fpaths = collect_edf_paths(tuab_dirs)
    summary, df_filenames = counts_from_filenames(fpaths)

    entries_tuab, rpaths_not_found, parsed_header_tuab_path = build_tuab_header_from_tuh_header(
        tuh_parent=cfg["tuh_parent"],
        fpaths=fpaths,
        parsed_header_path=s0_cfg["parsed_header_path"],
        output_path=s0_cfg["parsed_header_tuab_path"],
    )

    outputs_dir = os.path.dirname(s0_cfg["manifest_path"])
    summary_csv = os.path.join(outputs_dir, "tuab_counts_summary.csv")
    files_csv = os.path.join(outputs_dir, "tuab_files_index.csv")
    os.makedirs(outputs_dir, exist_ok=True)
    summary.to_csv(summary_csv)
    df_filenames.to_csv(files_csv, index=False)

    legacy_after = snapshot_legacy_outputs(legacy_dir)
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config_path": cfg["meta"]["config_path"],
        "tuh_parent": cfg["tuh_parent"],
        "parsed_header_path": parsed_header_path,
        "parsed_header_tuab_path": parsed_header_tuab_path,
        "summary_csv": summary_csv,
        "files_csv": files_csv,
        "n_parsed_header_entries": len(parsed_header),
        "n_tuab_entries": len(entries_tuab),
        "n_tuab_edf_files": len(fpaths),
        "n_tuab_paths_not_found_in_header": len(rpaths_not_found),
        "header_path_verified": header_report["absolute_path"],
        "tuab_root": prerequisites["tuab_root"],
        "tuab_subset_dir_count": len(prerequisites["subset_dirs"]),
        "legacy_immutability_check_enabled": True,
        "legacy_before": legacy_before,
        "legacy_after": legacy_after,
        "legacy_immutable": legacy_before == legacy_after,
    }
    write_manifest(s0_cfg["manifest_path"], payload)

    print("  EDF files found:", len(fpaths))
    print("  Missing in header:", len(rpaths_not_found))
    print("\nTUAB dataset summary table:")
    print(summary)
    print("\nS0 readiness report:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
