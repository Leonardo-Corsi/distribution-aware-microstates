import json
import os
from copy import deepcopy


REQUIRED_TOP_KEYS = {
    "tuh_parent",  # any level parent of rsync destination
    "output_parent",  # wherever you want the outputs to go
    "s0",  # settings of step s0 (loading)
    "s1",  # settings of step s1 (preprocessing)
    "s2",  # settings of step s2 (extraction and graphical abstract)
    "s3",  # settings of step s3 (classification and final figures)
}

REQUIRED_S0_KEYS = {"s0_parent", "reparse_headers", "parsed_header_path", "parsed_header_tuab_path", "manifest_path"}
REQUIRED_S1_PATH_KEYS = {"s1_parent", "results_path", "summary_path"}
REQUIRED_S2_KEYS = {"s2_parent"}
REQUIRED_S3_KEYS = {"s3_parent"}


def _resolve_path(base_dir, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a non-empty string path, got: {value!r}")
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(base_dir, expanded))


def _require_section_keys(section_name, section_obj, required_keys):
    missing = sorted(required_keys - set(section_obj.keys()))
    if missing:
        raise KeyError(f"Missing required config keys in '{section_name}': {missing}")


def load_configs(config_path="configs.json"):
    config_abspath = os.path.abspath(config_path)
    base_dir = os.path.dirname(config_abspath)

    with open(config_abspath, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    missing = sorted(REQUIRED_TOP_KEYS - set(cfg.keys()))
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    _require_section_keys("s0", cfg["s0"], REQUIRED_S0_KEYS)
    _require_section_keys("s1", cfg["s1"], REQUIRED_S1_PATH_KEYS)
    _require_section_keys("s2", cfg["s2"], REQUIRED_S2_KEYS)
    _require_section_keys("s3", cfg["s3"], REQUIRED_S3_KEYS)

    out = deepcopy(cfg)
    out["meta"] = {
        "config_path": config_abspath,
        "config_dir": base_dir,
        "project_root": base_dir,
    }

    out["tuh_parent"] = _resolve_path(base_dir, out["tuh_parent"])
    out["output_parent"] = _resolve_path(base_dir, out["output_parent"])
    out["s0"]["s0_parent"] = _resolve_path(out["output_parent"], out["s0"]["s0_parent"])
    out["s1"]["s1_parent"] = _resolve_path(out["output_parent"], out["s1"]["s1_parent"])
    out["s2"]["s2_parent"] = _resolve_path(out["output_parent"], out["s2"]["s2_parent"])
    out["s3"]["s3_parent"] = _resolve_path(out["output_parent"], out["s3"]["s3_parent"])
    out["n_jobs"] = int(out.get("n_jobs", 1))

    out["s0"]["parsed_header_path"] = _resolve_path(out["s0"]["s0_parent"], out["s0"]["parsed_header_path"])
    out["s0"]["parsed_header_tuab_path"] = _resolve_path(out["s0"]["s0_parent"], out["s0"]["parsed_header_tuab_path"])
    out["s0"]["manifest_path"] = _resolve_path(out["s0"]["s0_parent"], out["s0"]["manifest_path"])

    out["s1"]["results_path"] = _resolve_path(out["s1"]["s1_parent"], out["s1"]["results_path"])
    out["s1"]["summary_path"] = _resolve_path(out["s1"]["s1_parent"], out["s1"]["summary_path"])

    return out
