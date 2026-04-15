import os
import pickle
import re
import shutil
import traceback
import warnings
from datetime import datetime, timezone
from copy import deepcopy
from itertools import groupby
from pathlib import Path
from traceback import format_exc

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne import Annotations
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep.prep_pipeline import PrepPipeline
from termcolor import colored
from tqdm import tqdm

plt.rcParams.update({"figure.max_open_warning": 0})
S1_STORAGE_SCHEMA_VERSION = 1


def _resolve_path(base_dir, value, make_dir=False):
    if value is None:
        return None
    if os.path.isabs(value):
        resolved = os.path.normpath(value)
    else:
        resolved = os.path.normpath(os.path.join(base_dir, value))
    if make_dir:
        os.makedirs(resolved, exist_ok=True)
    return resolved


def _ensure_abs(path, name):
    if not os.path.isabs(path):
        raise ValueError(f"Expected absolute path for '{name}', got: {path}")
    return os.path.normpath(path)


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _write_json_atomic(path, obj):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        import json

        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def _read_json(path):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _record_stem_from_relpath(relpath):
    return Path(str(relpath)).stem


def _record_paths(stage_dir, record_stem):
    record_dir = os.path.join(stage_dir, record_stem)
    metadata_path = os.path.join(record_dir, "metadata.json")
    return {
        "record_dir": record_dir,
        "metadata_path": metadata_path,
    }


def _reset_stage_record(stage_dir, record_stem):
    paths = _record_paths(stage_dir, record_stem)
    if os.path.isdir(paths["record_dir"]):
        shutil.rmtree(paths["record_dir"], ignore_errors=True)


def _field_suffix(expected_field):
    if expected_field == "EEG_seg":
        return "seg"
    if expected_field == "EEG_ICA":
        return "ica"
    raise ValueError(f"Unsupported expected_field: {expected_field}")


def _validate_metadata_only(stage_dir, record_stem, expected_field, check_edf_headers=False):
    paths = _record_paths(stage_dir, record_stem)
    if not os.path.exists(paths["metadata_path"]):
        return False
    try:
        meta = _read_json(paths["metadata_path"])
    except Exception:
        return False

    if meta.get("errors_log"):
        return False
    if meta.get("schema_version") != S1_STORAGE_SCHEMA_VERSION:
        return False
    if meta.get("expected_field") != expected_field:
        return False

    files = meta.get("edf_files")
    if not isinstance(files, list) or len(files) == 0:
        return False

    for rel_file in files:
        full_path = os.path.join(paths["record_dir"], rel_file)
        if not os.path.exists(full_path):
            return False
        if check_edf_headers:
            try:
                mne.io.read_raw_edf(full_path, preload=False, verbose="ERROR")
            except Exception:
                return False
    return True


def needs_refresh_stage(stage_dir, record_stem, expected_field):
    return not _validate_metadata_only(
        stage_dir=stage_dir,
        record_stem=record_stem,
        expected_field=expected_field,
        check_edf_headers=True,
    )


def read_stage_metadata(stage_dir, record_stem):
    paths = _record_paths(stage_dir, record_stem)
    return _read_json(paths["metadata_path"])


def load_stage_record(stage_dir, record_stem, expected_field, montage_name=None):
    if not _validate_metadata_only(stage_dir, record_stem, expected_field, check_edf_headers=False):
        raise RuntimeError(
            f"Invalid or incomplete stage record: stage_dir={stage_dir}, record_stem={record_stem}, expected_field={expected_field}"
        )

    paths = _record_paths(stage_dir, record_stem)
    meta = _read_json(paths["metadata_path"])
    raws = []
    for rel_file in meta["edf_files"]:
        full_path = os.path.join(paths["record_dir"], rel_file)
        raw = mne.io.read_raw_edf(full_path, preload=True, verbose="ERROR")
        if montage_name:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.set_montage(
                    mne.channels.make_standard_montage(montage_name),
                    match_case=False,
                    on_missing="ignore",
                    verbose=False,
                )
        raws.append(raw)

    instance = {
        "ID": meta.get("ID"),
        "filename": meta.get("filename"),
        "description": meta.get("description"),
        "errors_log": meta.get("errors_log", []),
        expected_field: raws,
    }

    for null_field in meta.get("null_fields", []):
        instance[null_field] = None

    return instance


def save_stage_record(stage_dir, instance, stage):
    if stage not in {"segmented", "ica"}:
        raise ValueError(f"Unsupported stage: {stage}")

    if stage == "segmented":
        expected_field = "EEG_seg"
        suffix = "seg"
        null_fields = ["EEG_pre"]
    else:
        expected_field = "EEG_ICA"
        suffix = "ica"
        null_fields = ["EEG_seg"]

    segments = instance.get(expected_field)
    if not isinstance(segments, list) or len(segments) == 0:
        raise ValueError(f"Cannot persist stage '{stage}': missing or empty '{expected_field}' list.")

    record_stem = _record_stem_from_relpath(instance["filename"])
    paths = _record_paths(stage_dir, record_stem)
    os.makedirs(paths["record_dir"], exist_ok=True)

    edf_files = []
    segment_info = []
    for idx, raw in enumerate(segments, start=1):
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError(f"Segment {idx} of '{expected_field}' is not an MNE Raw object.")
        edf_name = f"{record_stem}-{suffix}{idx:02d}.edf"
        edf_path = os.path.join(paths["record_dir"], edf_name)
        tmp_edf = f"{edf_path}.tmp.edf"
        raw_to_export = raw
        try:
            with warnings.catch_warnings():
                # MNE/EDF exporter pads final block with edge values; this is expected and not actionable.
                warnings.filterwarnings(
                    "ignore",
                    message=r"EDF format requires equal-length data blocks.*",
                    category=RuntimeWarning,
                )
                mne.export.export_raw(tmp_edf, raw_to_export, fmt="edf", overwrite=True)
        except ValueError as e:
            if "EDF only allows dates from 1985 to 2084" not in str(e):
                raise
            # Force a valid EDF date when source meas_date is out of EDF representable range.
            raw_to_export = raw.copy()
            raw_to_export.set_meas_date(datetime(2000, 1, 1, tzinfo=timezone.utc))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"EDF format requires equal-length data blocks.*",
                    category=RuntimeWarning,
                )
                mne.export.export_raw(tmp_edf, raw_to_export, fmt="edf", overwrite=True)
        os.replace(tmp_edf, edf_path)
        edf_files.append(edf_name)
        segment_info.append(
            {
                "index": idx,
                "file": edf_name,
                "n_channels": int(len(raw.ch_names)),
                "sfreq": float(raw.info["sfreq"]),
                "duration_s": float(raw.n_times / raw.info["sfreq"]) if raw.info["sfreq"] else 0.0,
                "n_times": int(raw.n_times),
                "channels": list(raw.ch_names),
            }
        )

    metadata = {
        "schema_version": S1_STORAGE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "expected_field": expected_field,
        "record_stem": record_stem,
        "ID": instance.get("ID"),
        "filename": instance.get("filename"),
        "description": _to_jsonable(instance.get("description")),
        "errors_log": _to_jsonable(instance.get("errors_log", [])),
        "edf_files": edf_files,
        "segment_info": segment_info,
        "null_fields": null_fields,
    }
    _write_json_atomic(paths["metadata_path"], metadata)
    return paths["metadata_path"]


def _tuh_channel_to_standard(ch_name, valid_eeg_channels):
    ch = ch_name.strip()
    up = ch.upper()
    up = re.sub(r"^\s*EEG\s+", "", up)
    up = re.sub(r"\s+", "", up)
    up = re.sub(r"-(REF|LE|AVG)$", "", up)
    up = up.replace("EEG", "")

    if up in {"EKG", "ECG", "ECG1", "EKG1"}:
        return "ECG"

    for target in valid_eeg_channels:
        if up == target.upper():
            return target
    return ch_name


def _normalize_tuh_channels(raw, valid_eeg):
    if not valid_eeg:
        return raw

    rename_map = {}
    used_targets = set(raw.ch_names)
    for ch in raw.ch_names:
        mapped = _tuh_channel_to_standard(ch, valid_eeg)
        if mapped != ch and mapped not in used_targets:
            rename_map[ch] = mapped
            used_targets.add(mapped)
    if rename_map:
        raw.rename_channels(rename_map)
    return raw


def instance_from_tuh_entry(entry, tuh_parent, valid_eeg_channels, montage_name):
    import mne
    import warnings

    rel = entry["record"]["relative_path"].replace("\\", "/").lstrip("/")
    abs_edf = os.path.normpath(os.path.join(tuh_parent, rel))
    instance = {
        "ID": entry["record"]["ID"],
        "filename": rel,
        "description": entry,
        "EEG_raw": None,
        "EEG_clean": None,
        "crop_intervals": None,
        "errors_log": [],
    }

    try:
        if not os.path.exists(abs_edf):
            raise FileNotFoundError(f"EDF file not found: {abs_edf}")

        raw = mne.io.read_raw_edf(abs_edf, preload=True, verbose="ERROR")
        raw = _normalize_tuh_channels(raw, valid_eeg_channels)

        keep_eeg = set(valid_eeg_channels or [])
        keep = [ch for ch in raw.ch_names if ch in keep_eeg or ch == "ECG"]
        if keep:
            raw.pick(keep)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_montage(
                mne.channels.make_standard_montage(montage_name),
                match_case=False,
                on_missing="ignore",
                verbose=False,
            )
        instance["EEG_raw"] = raw
    except Exception as e:
        instance["errors_log"].append((str(e), format_exc()))
    return instance


def _build_runtime(cfg):
    s1_cfg = cfg["s1"]
    tuh_parent = cfg["tuh_parent"]

    parsed_header_path = _ensure_abs(cfg["s0"]["parsed_header_tuab_path"], "s0.parsed_header_tuab_path")
    s1_output_root = _ensure_abs(s1_cfg["s1_parent"], "s1.s1_parent")
    results_pickle_path = _ensure_abs(s1_cfg["results_path"], "s1.results_path")
    summary_json_path = _ensure_abs(s1_cfg["summary_path"], "s1.summary_path")

    channels = s1_cfg["channels"]
    montage = s1_cfg["montage"]
    prep_params = deepcopy(s1_cfg["prep_params"])
    bad_gfp_band = s1_cfg["bad_gfp_band"]
    seed = s1_cfg["seed"]

    os.makedirs(s1_output_root, exist_ok=True)

    runtime = {
        "parsed_header_path": parsed_header_path,
        "tuh_parent": _ensure_abs(tuh_parent, "tuh_parent"),
        "s1_output_root": s1_output_root,
        "segmented_dir": _resolve_path(s1_output_root, "preprocessed_segmented", make_dir=True),
        "ica_dir": _resolve_path(s1_output_root, "preprocessed_segmented_ICA", make_dir=True),
        "results_pickle_path": results_pickle_path,
        "summary_json_path": summary_json_path,
        "persist_results": bool(s1_cfg.get("persist_results", True)),
        "apply_ica": bool(s1_cfg["apply_ica"]),
        "referencing_strategy": s1_cfg["referencing_strategy"],
        "channels": channels,
        "montage": montage,
        "prep_params": prep_params,
        "bad_gfp_band": bad_gfp_band,
        "seed": seed,
        "basic_preprocessing": {
            "l_freq": s1_cfg["l_freq"],
            "h_freq": s1_cfg["h_freq"],
            "l_trans_bandwidth": s1_cfg["l_trans_bandwidth"],
            "h_trans_bandwidth": s1_cfg["h_trans_bandwidth"],
            "notch_freq": s1_cfg["notch_freq"],
            "no_signal_threshold": s1_cfg["no_signal_threshold"],
            "no_signal_threshold_s": s1_cfg["no_signal_threshold_s"],
            "enable_bad_channel_interpolation": bool(s1_cfg["enable_bad_channel_interpolation"]),
            "channels": channels,
            "prep_params": prep_params,
            "montage": montage,
            "prep_seed": seed,
        },
        "segmentation": {
            "env_l_freq": s1_cfg["seg_env_l_freq"],
            "th_env": s1_cfg["seg_th_env"],
            "T_falsenegative": s1_cfg["seg_t_falsenegative"],
            "T_falsepositive": s1_cfg["seg_t_falsepositive"],
            "skirts": s1_cfg["seg_skirts"],
            "reject_if_shorter_than": s1_cfg["seg_reject_if_shorter_than"],
            "enable_bad_interval_detection": bool(s1_cfg["enable_bad_interval_detection"]),
            "bad_gfp_band": bad_gfp_band,
        },
        "ica_params": s1_cfg["ica_params"],
    }

    os.makedirs(os.path.dirname(runtime["results_pickle_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(runtime["summary_json_path"]), exist_ok=True)
    return runtime


def load_parsed_header(parsed_header_path):
    if not os.path.exists(parsed_header_path):
        raise FileNotFoundError(f"Parsed TUAB header not found: {parsed_header_path}")
    with open(parsed_header_path, "rb") as f:
        parsed_header = pickle.load(f)
    if not isinstance(parsed_header, list):
        raise RuntimeError(f"Parsed header is not a list: {parsed_header_path}")
    return parsed_header


def process_tuh_entry(entry, runtime, plot_segmentation=False):
    step = "init"
    seg_recompute_attempts = 0
    ica_retry_attempted = False
    ica_retry_succeeded = False
    segmented_rebuilt = False

    def _has_valid_segments(inst_obj, field):
        segs = inst_obj.get(field) if isinstance(inst_obj, dict) else None
        return isinstance(segs, list) and len(segs) > 0

    def _compute_segmented_from_raw():
        nonlocal step
        step = "raw:load-from-tuh"
        inst_obj = instance_from_tuh_entry(
            entry,
            runtime["tuh_parent"],
            valid_eeg_channels=runtime["channels"],
            montage_name=runtime["montage"],
        )
        if inst_obj.get("errors_log"):
            first_err = inst_obj["errors_log"][0]
            err_msg = first_err[0] if len(first_err) > 0 else "unknown load error"
            err_tb = first_err[1] if len(first_err) > 1 else ""
            raise RuntimeError(f"{inst_obj['ID']} load failed: {err_msg}\n{err_tb}")

        if runtime["referencing_strategy"] != "pyprep":
            raise NotImplementedError(
                "Only referencing_strategy='pyprep' is wired in this refactor commit."
            )
        bp = runtime["basic_preprocessing"]
        step = "segmented:basic-preprocessing"
        inst_obj = basic_preprocessing_tuh(inst_obj, **bp)
        inst_obj["EEG_raw"] = None
        seg = runtime["segmentation"]
        step = "segmented:extract-good-segments"
        inst_obj = extract_good_segments(inst_obj, plot=plot_segmentation, **seg)
        inst_obj["EEG_pre"] = None
        return inst_obj

    try:
        rel_path = entry["record"]["relative_path"].replace("\\", "/").lstrip("/")
        record_stem = _record_stem_from_relpath(rel_path)
        record_id = entry.get("record", {}).get("ID")

        step = "segmented:check-refresh"
        inst = None
        if needs_refresh_stage(runtime["segmented_dir"], record_stem, "EEG_seg"):
            inst = _compute_segmented_from_raw()
            segmented_rebuilt = True
            if not _has_valid_segments(inst, "EEG_seg"):
                seg_recompute_attempts += 1
                step = "segmented:force-recompute-from-raw"
                _reset_stage_record(runtime["segmented_dir"], record_stem)
                inst = _compute_segmented_from_raw()
                segmented_rebuilt = True
            if not _has_valid_segments(inst, "EEG_seg"):
                raise RuntimeError("seg_no_segments_after_recompute")
            step = "segmented:save-new-format"
            _reset_stage_record(runtime["segmented_dir"], record_stem)
            save_stage_record(runtime["segmented_dir"], inst, stage="segmented")
        else:
            step = "segmented:load-new-format"
            inst = load_stage_record(
                runtime["segmented_dir"],
                record_stem,
                "EEG_seg",
                montage_name=runtime["montage"],
            )
            if not _has_valid_segments(inst, "EEG_seg"):
                seg_recompute_attempts += 1
                step = "segmented:force-recompute-from-raw"
                _reset_stage_record(runtime["segmented_dir"], record_stem)
                inst = _compute_segmented_from_raw()
                if not _has_valid_segments(inst, "EEG_seg"):
                    raise RuntimeError("seg_no_segments_after_recompute")
                step = "segmented:save-new-format"
                _reset_stage_record(runtime["segmented_dir"], record_stem)
                save_stage_record(runtime["segmented_dir"], inst, stage="segmented")
                segmented_rebuilt = True

        if runtime["apply_ica"]:
            step = "ica:check-refresh"
            ica_needs_refresh = needs_refresh_stage(runtime["ica_dir"], record_stem, "EEG_ICA")
            if segmented_rebuilt and not ica_needs_refresh:
                step = "ica:force-refresh-after-seg-rebuild"
                _reset_stage_record(runtime["ica_dir"], record_stem)
                ica_needs_refresh = True
            if ica_needs_refresh:
                ica_params = runtime["ica_params"]
                segments = inst.get("EEG_seg")
                if not isinstance(segments, list) or len(segments) == 0:
                    raise RuntimeError(
                        "No EEG segments available for ICA cleaning (EEG_seg is empty or invalid)."
                    )
                all_ica, all_raw = [], []
                ica_errors = []
                for seg_idx, raw in enumerate(tqdm(segments, desc="ICA cleaning", leave=False)):
                    try:
                        step = f"ica:clean-segment:{seg_idx}"
                        raw_cleaned, ica, ica_info = ica_cleaning(
                            raw,
                            ica_params,
                            random_state=runtime["seed"],
                            return_info=True,
                        )
                        ica_retry_attempted = ica_retry_attempted or bool(ica_info.get("retry_attempted"))
                        ica_retry_succeeded = ica_retry_succeeded or bool(ica_info.get("retry_succeeded"))
                        all_ica.append(ica)
                        all_raw.append(raw_cleaned)
                    except Exception as e:
                        ica_errors.append(
                            {
                                "segment_index": seg_idx,
                                "error": str(e),
                                "traceback": format_exc(),
                            }
                        )
                        continue
                if not all_ica:
                    if ica_errors:
                        details = "\n\n".join(
                            [
                                f"[segment {item['segment_index']}] {item['error']}\n{item['traceback']}"
                                for item in ica_errors[:3]
                            ]
                        )
                        raise RuntimeError(
                            "No valid ICA cleaning completed. Segment-level ICA errors "
                            f"(showing up to 3/{len(ica_errors)}):\n{details}"
                        )
                    raise RuntimeError("No valid ICA cleaning completed (no segment-level exception captured).")
                inst.update({"EEG_ICA": all_raw, "ICAobjs": all_ica})
                inst["EEG_seg"] = None
                step = "ica:save-new-format"
                _reset_stage_record(runtime["ica_dir"], record_stem)
                save_stage_record(runtime["ica_dir"], inst, stage="ica")
            output_path = _record_paths(runtime["ica_dir"], record_stem)["metadata_path"]
        else:
            output_path = _record_paths(runtime["segmented_dir"], record_stem)["metadata_path"]

        return {
            "status": 0,
            "file_path": output_path,
            "record_stem": record_stem,
            "record_id": record_id,
            "seg_recompute_attempts": seg_recompute_attempts,
            "ica_retry_attempted": ica_retry_attempted,
            "ica_retry_succeeded": ica_retry_succeeded,
        }

    except Exception as e:
        rec = entry.get("record", {}) if isinstance(entry, dict) else {}
        rec_rel = rec.get("relative_path", "unknown")
        rec_stem = _record_stem_from_relpath(rec_rel) if rec_rel != "unknown" else "unknown"
        # Ensure failed records leave no stale stage artifacts; orchestrator will write stage-local error txt.
        if rec_stem != "unknown":
            _reset_stage_record(runtime["segmented_dir"], rec_stem)
            _reset_stage_record(runtime["ica_dir"], rec_stem)
        return {
            "status": 1,
            "file_path": rec_rel,
            "record_stem": rec_stem,
            "record_id": rec.get("ID"),
            "step": step,
            "message": str(e),
            "traceback": format_exc(),
            "seg_recompute_attempts": seg_recompute_attempts,
            "ica_retry_attempted": ica_retry_attempted,
            "ica_retry_succeeded": ica_retry_succeeded,
        }


#%%
def basic_preprocessing(
    instance,
    crop_samples="auto",
    line_freq="auto",
    verbose=False,
    filter_params_eeg=None,
    prep_params=None,
    montage="standard_1020",
    prep_seed=1,
):
    
    try:
        # name of file and ID
        print(colored(f"{instance['ID']}::{instance['filename']}", 'blue'))
        # get eeg
        rawobj = instance['EEG_raw']
        # parameters
        th_no_gfp = 1e-6
        # functions
        
        ''' Pre-processing : detect line frequency '''
        if line_freq == 'auto':
            line_freq = humming_freq(rawobj)
        else:
            line_freq = float(line_freq) if line_freq is not None else None
        
        ''' Pre-processing : resample '''
        if rawobj.info['sfreq']!=128:
            if verbose: 
                print(f"Resampling {instance['ID']} from {rawobj.info['sfreq']} to 128 Hz")
                rawobj.resample(float(128), n_jobs=1)
        
        ''' Pre-processing : crop '''
        if crop_samples == 'auto' or crop_samples is None:
            # GFP based detection of no-signal intervals
            no_gfp = _detect_nogfp(rawobj,
                                    threshold_uV=th_no_gfp,threshold_s=.030)
            # get intervals of good signal
            (ok_gfp_intervals, _) = _get_true_intervals(
                no_gfp,
                bool_val=False,
                ret_transitions=True,
                sort=True,
            )
            ok_gfp_intervals = [(i_from,i_to) for i_from,i_to in ok_gfp_intervals if i_to-i_from>=10*rawobj.info['sfreq']]
            instance['crop_intervals'] = ok_gfp_intervals 
            if len(ok_gfp_intervals)>1: #todo split rawobj into as many intervals as needed in case of recordings with holes
                warnings.warn(f"Multiple intervals not implemented, taking longest from {instance['ID']}::{instance['filename']}\n"
                            +f"Intervals: {ok_gfp_intervals}")
            crop_min,crop_max = ok_gfp_intervals[0] # biggest interval (first if sort=True)
            
        else:
            crop_min,crop_max = crop_samples
            
        rawobj.crop(tmin=crop_min/rawobj.info['sfreq'],
                    tmax=(crop_max-1)/rawobj.info['sfreq'])
        
        ''' Pre-processing : filter '''
        # 0.5-45 Hz bandpass for eeg
        if filter_params_eeg is None:
            raise ValueError("filter_params_eeg must be provided explicitly.")
        rawobj.filter(**filter_params_eeg,verbose=verbose)
        # notch filter for all non-misc channels
        if line_freq is not None:
            rawobj.notch_filter(line_freq,picks=['eeg','ecg','eog','emg'],verbose=verbose)
        
        ''' Pre-processing : pyprep+interp '''
        rawobj, still_noisy = _apply_prep_and_interp(
            rawobj,
            prep_params=prep_params,
            montage=montage,
            seed=prep_seed,
        )
        rawobj = _force_spline_interp(rawobj, still_noisy)
        
        #TODO implement optional ica and rereference
        # here EEG_clean is only cropped, resampled, filtered and interpolated
        # and finally left with the robust RANSAC reference!
        instance['EEG_clean'] = rawobj
        
    except Exception as e:
        # Log any errors encountered
        print(f"Error loading instance {instance['ID']}:")
        instance['errors_log'].append((e,traceback.format_exc()))
        print(colored(instance['errors_log'][-1][0], 'red'),'\n')
        print(instance['errors_log'][-1][1],'\n')
    
    return instance

def basic_preprocessing_tuh(instance,
                            l_freq=1,
                            h_freq=45,
                            l_trans_bandwidth=.5,
                            h_trans_bandwidth=5,
                            notch_freq=60,
                            no_signal_threshold=5e-6,
                            no_signal_threshold_s=.1,
                            enable_bad_channel_interpolation=True,
                            channels=None,
                            prep_params=None,
                            montage="standard_1020",
                            prep_seed=1,
                            ):
    
    try:
        if not channels:
            raise ValueError("S1 channels are not configured.")
        preprocessed_eeg = instance['EEG_raw'].copy().pick(channels)
        preprocessed_eeg.resample(float(128), n_jobs=1)
        preprocessed_eeg.filter(l_freq,h_freq,
                                method='fir',fir_design='firwin',
                                l_trans_bandwidth=l_trans_bandwidth,
                                h_trans_bandwidth=h_trans_bandwidth)
        preprocessed_eeg.notch_filter([notch_freq],
                                      picks=['eeg',
                                             'ecg',
                                             'eog',
                                             'emg'],
                                      )
        
        no_gfp = _detect_nogfp(preprocessed_eeg,
                                threshold_uV=no_signal_threshold,
                                threshold_s=no_signal_threshold_s)
        # get intervals of good signal
        (ok_gfp_intervals, _) = _get_true_intervals(
            no_gfp,
            bool_val=False,
            ret_transitions=True,
            sort=True,
        )
        ok_gfp_intervals = [(i_from,i_to) for i_from,i_to in ok_gfp_intervals if i_to-i_from>=10*preprocessed_eeg.info['sfreq']]
        crop_min,crop_max = ok_gfp_intervals[0] # biggest interval (first if sort=True)
        preprocessed_eeg.crop(tmin=crop_min/preprocessed_eeg.info['sfreq'],
                        tmax=(crop_max-1)/preprocessed_eeg.info['sfreq'])
        
        if enable_bad_channel_interpolation:
            prep_params_eff = deepcopy(prep_params) if prep_params is not None else {}
            prep_params_eff.setdefault("ref_chs", "eeg")
            prep_params_eff.setdefault("reref_chs", "eeg")
            prep_params_eff.setdefault("line_freqs", [])
            prep_params_eff.setdefault("channel_wise", True)
            preprocessed_eeg, still_noisy = _apply_prep_and_interp(
                preprocessed_eeg,
                prep_params=prep_params_eff,
                montage=montage,
                seed=prep_seed,
            )
            preprocessed_eeg = _force_spline_interp(preprocessed_eeg, still_noisy)
        
        instance['EEG_pre'] = preprocessed_eeg
        
    except Exception as e:
        instance['EEG_pre'] = None
        # Log any errors encountered
        print(f"Error loading instance {instance['ID']}:")
        instance['errors_log']=(e,traceback.format_exc())
        print(colored(instance['errors_log'][0], 'red'),'\n')
        print(instance['errors_log'][1],'\n')
        raise e
        
    return instance

def extract_good_segments(instance,
                        env_l_freq=4,
                        th_env = 100e-6,
                        T_falsenegative = 2.5,
                        T_falsepositive = 0.25,
                        skirts = 0,
                        reject_if_shorter_than=2.5,
                        enable_bad_interval_detection=True,
                        bad_gfp_band=(30, 45),
                        plot=False):
    try:
        preprocessed_eeg = instance['EEG_pre'].copy()
        if not enable_bad_interval_detection:
            instance["EEG_seg"] = [preprocessed_eeg]
            instance["EEG_pre_annot_bad"] = Annotations(
                onset=[], duration=[], description=[], orig_time=preprocessed_eeg.info["meas_date"]
            )
            instance["EEG_pre_annot_good"] = Annotations(
                onset=[], duration=[], description=[], orig_time=preprocessed_eeg.info["meas_date"]
            )
            return instance
        # auto annotations for muscle artifacts and bad envelope (avoid giving ICA trashy data)
        bad_annot,good_annot = auto_bad_muscle(preprocessed_eeg,
                                        l_freq=bad_gfp_band[0],
                                        h_freq=bad_gfp_band[1],
                                        env_l_freq = env_l_freq,
                                        th_env = th_env,
                                        T_falsenegative = T_falsenegative,
                                        T_falsepositive = T_falsepositive,
                                        skirts = skirts,
                                        plot=plot)
        
        eeg_segments = segmentation_by_annotations(preprocessed_eeg,good_annot,
                                                reject_if_shorter_than=reject_if_shorter_than,)
        instance['EEG_seg'] = eeg_segments
        instance['EEG_pre_annot_bad'] = bad_annot
        instance['EEG_pre_annot_good'] = good_annot
        
        if plot:
            preprocessed_eeg.copy().set_annotations(preprocessed_eeg.annotations
                                                + bad_annot
                                                + good_annot
                                                ).plot(duration=10)
    except Exception as e:
        instance['EEG_seg'] = None
        # Log any errors encountered
        print(f"Error loading instance {instance['ID']}:")
        instance['errors_log']=(e,traceback.format_exc())
        print(colored(instance['errors_log'][0], 'red'),'\n')
        print(instance['errors_log'][1],'\n')
        raise e
        
    return instance
    
def _auto_bad_envelope(raw,th=100e-6,
                    T_falsenegative = 2,
                    skirts = 1,
                    check_BAD=True):
    raw_ = raw.copy()
    # detection of absurdly high values in the signal 
    # quite useful to avoid distortions in the z-score of muscle activity
    avg_envelope = raw_.copy().apply_hilbert(envelope=True).get_data().mean(axis=0)
    bad = _detect_overth(avg_envelope, # is the average envelope
                                th=th, # above this threshold?
                                th_samples=T_falsenegative*raw_.info['sfreq'], # (regardless of little fluctuations inside an interval this long)
                                desired=True, # we want those that are above the threshold
                                )
    
    # get intervals of very bad signal
    if skirts:
        (bad_intervals, _) = _get_true_intervals(
            bad,
            bool_val=True,
            ret_transitions=True,
            sort=False,
        )
        # add skirts seconds to each interval to be sure
        skirts_samples = int(skirts*raw_.info['sfreq'])
        bad_intervals = [(max(i_from-skirts_samples,0),
                        min(i_to+skirts_samples,len(raw_.times)-1)) 
                        for i_from,i_to in bad_intervals]
        # merge overlapping intervals
        bad_ = np.zeros_like(avg_envelope)
        for i_from,i_to in bad_intervals:
            bad_[i_from:i_to] = 1
        bad = bad_
    
    bad_intervals,_ = _get_true_intervals(bad,bool_val=True,
                                            ret_transitions=True,
                                            sort=False)

    # additional: find any BAD_* annotations and add them to a bad_any bool as True
    if check_BAD:
        pre_existing_Ann = raw_.annotations
        pre_existing_BAD = pre_existing_Ann[['BAD' in str(ann['description']) for ann in pre_existing_Ann]]
        bad_any = bad.copy()
        for ann in pre_existing_BAD:
            i_from = int(ann['onset']*raw_.info['sfreq'])
            i_to = int((ann['onset']+ann['duration'])*raw_.info['sfreq'])
            bad_any[i_from:i_to] = 1
    
    good_intervals,_ = _get_true_intervals(bad_any,bool_val=False,
                                            ret_transitions=True,
                                            sort=False)
    
    t_vec = raw_.times + raw_.first_time # must be relative to meas_date!!
    
    onset_times = [t_vec[i_from]
                    for i_from, i_to in bad_intervals]
    duration_times = [t_vec[i_to-1] - t_vec[i_from]
                        for i_from, i_to in bad_intervals]
    ann_bad = Annotations(onset=onset_times,
                    duration=duration_times,
                    description='BAD_envelope',
                    orig_time=raw_.info['meas_date'])
    
    if check_BAD and len(pre_existing_BAD):
        ann_bad += pre_existing_BAD
        
    onset_times = [t_vec[i_from] 
                    for i_from, i_to in good_intervals]
    duration_times = [t_vec[i_to-1] - t_vec[i_from]
                        for i_from, i_to in good_intervals]
    ann_good = Annotations(onset=onset_times,
                    duration=duration_times,
                    description='ok',
                    orig_time=raw_.info['meas_date'])
    
    return ann_bad, ann_good

def _muscle_envelope(eeg,filter_freq=None,env_filter=4):
    if filter_freq is None:
        filter_freq = 30, eeg.info['sfreq']//2-5
    
    feeg = eeg.copy().pick('eeg').filter(filter_freq[0],
                                        filter_freq[1],
                                        fir_design="firwin",
                                        pad="reflect_limited",
                                        )
    henv = feeg.apply_hilbert(envelope=True)
    fhenv = henv.copy().filter(0,env_filter,
                            fir_design="firwin",
                            pad="reflect_limited",
                            )
    
    return fhenv.get_data().mean(axis=0)
            
            
            
            
            
            
def auto_bad_muscle(raw, 
                    l_freq=None,
                    h_freq=None,
                    env_l_freq=4,
                    th_env = 100e-6,
                    
                    T_falsepositive=0.1,
                    T_falsenegative = 2,
                    skirts = 0,
                    plot=False):
    
    if h_freq is None:
        h_freq = raw.info['sfreq']//2-1
    if l_freq is None:
        l_freq = 30
    
    raw_ = raw.copy()
    # delete segments of absurdly high envelope
    bad_env, good_env = _auto_bad_envelope(raw_,th=th_env) # already adds the start_time
    
    good_segments = segmentation_by_annotations(raw_,good_env,
                                                reject_if_shorter_than=None,
                                                min_survival_fraction=0.00)
    # each good_segment's .times now is relative to start time of cropped object!
    all_bad, all_good = [], []
    scores_allalong = np.full(len(raw_.times),np.nan)
    tmp_log = mne.set_log_level('ERROR')
    all_menvs = [_muscle_envelope(eeg,
                                  filter_freq=(l_freq,h_freq),
                                  env_filter=env_l_freq) for eeg in good_segments]
    mne.set_log_level(tmp_log)        
    all_muscleenv = np.concatenate(all_menvs)
    th_menv = np.mean(all_muscleenv)+np.std(all_muscleenv)
    for raw_seg,menv_seg, annot_seg in zip(good_segments,all_menvs, good_env):
        
        
        # gfp_rawobj = mne.io.RawArray(raw_seg.get_data().std(axis=0)[np.newaxis,:],
        #                             mne.create_info(ch_names=['GFP_gamma'],
        #                                             ch_types=['eeg'],
        #                                             sfreq=128)
        #                             )
        # _, scores_muscle = annotate_muscle_zscore(raw_seg,
        #                                         ch_type="eeg",
        #                                         threshold=th_zscore,
        #                                         min_length_good=min_length_zscore,
        #                                         filter_freq=(l_freq,h_freq)
        #                                         )    
                         
        # detection of bad intervals
        
        # first remove falsenegatives then remove short bad intervals
        bad = _detect_overth(menv_seg, # is this signal
                            th=th_menv, # above this threshold?
                            th_samples=int(T_falsenegative*raw_seg.info['sfreq']), # (regardless of little fluctuations inside an interval this long)
                            desired=True, # we want those that are above the threshold
                            # NB desired=True means that we want to ensure we have no false negatives and tolerate false positives
                            )
        # remove bad intervals shorter than T_falsepositive
        bad_g = []
        bad_groupby = groupby(bad)
        for i,(val,group) in enumerate(bad_groupby):
            G = list(group)
            if val: # it is bad
                if i == 0 or i == len(bad)-1:
                    # if it is at the edge just keep it
                    bad_g.extend(G)
                else:
                    if len(G) < T_falsepositive*raw_seg.info['sfreq']:
                        # but not long enough
                        bad_g.extend([0]*len(G))
                    else:
                        # and long enough
                        bad_g.extend(G)
            else:
                # good segment, keep it
                bad_g.extend(G)
        bad = np.array(bad_g)
        
        
        # get intervals of bad signal
        if skirts:
            (bad_intervals, _) = _get_true_intervals(
                bad,
                bool_val=True,
                ret_transitions=True,
                sort=False,
            )
            # add skirts seconds to each interval to be sure
            skirts_samples = int(skirts*raw_seg.info['sfreq'])
            bad_intervals = [(max(i_from-skirts_samples,0),
                            min(i_to+skirts_samples,len(raw_seg.times)-1)) 
                            for i_from,i_to in bad_intervals]
            # merge overlapping intervals
            bad_ = np.zeros_like(menv_seg)
            for i_from,i_to in bad_intervals:
                bad_[i_from:i_to] = 1
            bad = bad_

        bad_intervals,_ = _get_true_intervals(bad,bool_val=True,
                                                ret_transitions=True,
                                                sort=False)

        good_intervals,_ = _get_true_intervals(bad,bool_val=False,
                                                ret_transitions=True,
                                                sort=False)
        
        t_vec = raw_.times + annot_seg['onset']
        
        onset_times = [t_vec[i_from]
                       for i_from, i_to in bad_intervals]
        duration_times = [t_vec[i_to-1] - t_vec[i_from]
                          for i_from, i_to in bad_intervals]
        
        ann_bad = Annotations(onset=onset_times,
                      duration=duration_times,
                      description='BAD_muscle',
                      orig_time=raw_.info['meas_date'])
        
        onset_times = [t_vec[i_from]
                       for i_from, i_to in good_intervals]
        duration_times = [t_vec[i_to-1] - t_vec[i_from]
                          for i_from, i_to in good_intervals]
        ann_good = Annotations(onset=onset_times,
                       duration=duration_times,
                       description='ok',
                       orig_time=raw_.info['meas_date'])
        

        all_bad.append(ann_bad)
        all_good.append(ann_good)
        i0 = int(annot_seg['onset']*raw_.info['sfreq']) - raw_.first_samp
        i1 = i0 + len(menv_seg)         
        scores_allalong[i0:i1] = menv_seg
        
    # merging
     
    all_bad = [bad_env, *all_bad] # we consider also the bad_env here
    if len(all_bad):
        bad_annot = all_bad[0]
        for ann in all_bad[1:]:
            bad_annot += ann
    else:
        bad_annot = Annotations(onset=[], 
                                duration=[], 
                                description=[], 
                                orig_time=raw_.info['meas_date'])
    
    if len(all_good):
        good_annot = all_good[0]
        for ann in all_good[1:]:
            good_annot += ann
    else:
        good_annot = Annotations(onset=[], 
                                duration=[], 
                                description=[], 
                                orig_time=raw_.info['meas_date'])
    
    
    if plot:
        print('Rejection based on envelope (blue) and on muscle z-scores')
        for badann in bad_annot:
            if 'env' in badann['description']:
                print(colored(
                      f"{badann['description']}: {badann['onset']:.3f} - {badann['onset']+badann['duration']:.3f}",
                      'blue')
                      )
            else:
                print(f"{badann['description']}: {badann['onset']:.3f} - {badann['onset']+badann['duration']:.3f}"
                      )
        
        
        _, axes = plt.subplots(2, 1, sharex=True)
        eeg_ax = axes[0]
        mnemuscle_ax = axes[1]


        # EEG, 40uV shift for each channel for better visualization
        dat = raw_.get_data(picks='eeg').T
        shift = np.arange(dat.shape[1])[np.newaxis,:]*40e-6
        dat = dat + shift
        eeg_ax.plot(raw_.times+raw_.first_time, 
                    dat, 
                    color='k', lw=0.25,alpha=0.25
                    )
        eeg_ax.set_ylim(shift[0,0]-40e-6,shift[0,-1]+40e-6)
        # Display over EEG
        for badenv in bad_env:
            interval_env = badenv['onset'], badenv['onset'] + badenv['duration']
            eeg_ax.fill_betweenx(np.arange(-1, dat.shape[1] + 1) * 40e-6, 
                                interval_env[0], 
                                interval_env[1], 
                                color='darkred', alpha=0.15)
        for badann in bad_annot:
            interval = badann['onset'], badann['onset']+badann['duration']
                        
            eeg_ax.fill_betweenx(np.arange(-1,dat.shape[1]+1)*40e-6, 
                                interval[0], 
                                interval[1], 
                                color='r', alpha=0.15)

        
        mnemuscle_ax.plot(raw_.times+raw_.first_time, 
                          scores_allalong, 
                          color='k', lw=1.5)
        mnemuscle_ax.hlines(th_menv, raw_.times[0], raw_.times[-1], color='r', linestyle='--',lw=1.5)
        #mnemuscle_ax.set_ylim(-3,9)
        # Display over muscle zscores
        mnemuscle_ax.hlines([th_menv for badann in bad_annot], 
                            [badann['onset'] for badann in bad_annot], 
                            [badann['onset']+badann['duration'] for badann in bad_annot], 
                            color='r', lw=7.5, alpha=.75)
        mnemuscle_ax.plot(raw_.times+raw_.first_time,
                        th_menv*np.ones_like(raw_.times), 
                        color='r', lw=1.5, linestyle='--')


        # adjustments to plots
        #eeg_ax.set_xticks([])
        eeg_ax.set_yticks([])
        #for ax in axes[:-1]: ax.set_xticks([])
        axes[-1].set_xlabel('Time [s]')
        plt.tight_layout(h_pad=0.01)

        
    return bad_annot, good_annot


def segmentation_by_annotations(raw,
                                annotations,
                                reject_if_shorter_than=None,
                                min_survival_fraction=0.10):
    raw_ = raw.copy()
    if annotations is None or len(annotations)==0:
        segments = [raw_]
    elif isinstance(annotations,Annotations):
        segments = raw_.crop_by_annotations(annotations=annotations)
    else:
        raise ValueError('annotations must be None or mne.Annotations,'
                         +f'instead it is {type(annotations)}')
            
    # exclude segments shorter than reject_if_shorter_than
    if reject_if_shorter_than is not None:
            segments = [seg for seg in segments 
                        if seg.times[-1]-seg.times[0]>reject_if_shorter_than]
            
    # check if there are any segments left and if they are long enough
    if len(segments)==0:
        print('No segments left to process')
        return None   
    else:
        # total duration of all segments
        duration = sum([seg.times[-1]-seg.times[0] for seg in segments])
        if duration < min_survival_fraction*raw_.times[-1]:
            warnings.warn(f'Less than 20% of the data is left after rejection!'
                            +f'({duration:.2f}s out of {raw_.times[-1]:.2f}s)')
            return None
        else:
            return segments
        

def shift_annotations_first_samp(annotations,raw):
    '''
    Create a copy of annotations shifted by raw.first_samp/raw.info['sfreq']
    IMPORTANT:
    this is needed when using raw.set_annotations(annotations) because the annotations
    will get shifted by raw.first_samp/raw.info['sfreq'], which is not 0 
    if the raw object has been cropped. 
    IMPORTANT 2:
    this is not needed with mne.io.RawEDF.crop_by_annotations(annotations), since
    the annotations are directly interpreted as a list of dictionaries with 'onset',
    'duration', 'description', 'orig_time' keys compared to the times attribute
    IMPORTANT 3: apparently it is needed instead, but don't know if there's difference with or without orig_time
    '''
    return mne.Annotations(onset=[a['onset'] + raw.first_time
                                    for a in annotations],
                            duration=[a['duration'] 
                                        for a in annotations],
                            description=[a['description']
                                            for a in annotations],
                            orig_time=raw.info['meas_date'])




def ica_cleaning(raw,ica_params,random_state=1,ic_label_th=0, return_info=False):
    warnings.filterwarnings("ignore")
    # this warning is suppressed because when using eeg filtered 1 to 45 Hz it's quite invasive
    info = {"retry_attempted": False, "retry_succeeded": False, "retry_reason": None}

    def _fit_apply(raw_in, params):
        rawica = raw_in.copy()
        rawica.set_eeg_reference('average',projection=False,verbose='ERROR') # otherwise IClabel will complain
        ica = ICA(random_state=random_state, verbose='ERROR', **params)
        ica.fit(rawica,picks='eeg', verbose='ERROR')
        ic_labels = label_components(rawica, ica, method='iclabel') # IClabel
        exclude_with_th = [idx for idx,(lbl,prb) in enumerate(zip(ic_labels['labels'],
                                                                ic_labels['y_pred_proba'])) 
                        if lbl!='brain' and prb>ic_label_th]
        ica_copy = ica.copy()
        ica.apply(rawica, exclude=exclude_with_th, verbose='ERROR')
        return rawica, ica_copy

    try:
        out_raw, out_ica = _fit_apply(raw, deepcopy(ica_params))
    except RuntimeError as e:
        msg = str(e)
        if "One PCA component captures most of the explained variance" not in msg:
            raise
        info["retry_attempted"] = True
        info["retry_reason"] = "ica_pca_single_component"
        ica_params_retry = deepcopy(ica_params)
        ica_params_retry["n_components"] = 0.9999
        out_raw, out_ica = _fit_apply(raw, ica_params_retry)
        info["retry_succeeded"] = True

    if return_info:
        return out_raw, out_ica, info
    return out_raw, out_ica
    
def plot_ica_details(ica,raw, components=False, eog=False):
    ic_labels = label_components(raw, ica, method='iclabel') # IClabel
    
    # plot sources
    ica.plot_sources(raw)#, picks=exclude_idx)
    import matplotlib.patheffects as pe
    plt.annotate(f'ICA ({(ica.method,ica.n_components)}) iters:{ica.n_iter_}\n\n\n'
                    +''.join([f"#{i}::{lbl}::{prb:.2f}\n\n" 
                            for i,(lbl,prb) in enumerate(zip(ic_labels['labels'],
                                                ic_labels['y_pred_proba']))]), 
                    xy=(0.05, 0.55),xycoords='figure fraction', ha='left', va='center', 
                    fontsize=10, color='r',
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    if components:
        # plot properties of non-brain ICs
        for c, (lbl,prb) in enumerate(zip(ic_labels['labels'],ic_labels['y_pred_proba'])):
            if lbl != 'brain':
                ica.plot_properties(raw, picks=[c])
                plt.title(f'#{c}::{lbl}::{prb:.2f}')
    if eog:
        # plot eog evoked from Fp1.Fp2
        eog_evoked = mne.preprocessing.create_eog_epochs(raw,ch_name=['Fp1','Fp2']).average()
        eog_evoked.apply_baseline(baseline=(None,-.2))
        eog_evoked.plot_joint()
#%% Utils

def humming_freq(raw,plot=False):
    # compute welch psd
    psd = raw.compute_psd(fmin=32,fmax=64, remove_dc=False,picks='eeg')
    # get average psd over channels and freqs
    psd_df = psd.to_data_frame()
    freq = psd_df['freq'].to_numpy()
    psd_uV2overHz = psd_df.iloc[:,1:].to_numpy()*1e6**2/freq[:,np.newaxis] # in uV^2/Hz
    avgpsd = 10*np.log10(np.mean(psd_uV2overHz, axis=1))
    psd_dB_uV2overHz = 10*np.log10(psd_uV2overHz) # in uV^2/Hz
    # visual (NB, VISUAL!) aids for peak finding
    stdpsd = np.std(psd_dB_uV2overHz, axis=1)
    avgstdpsd = np.mean(stdpsd)
    
    # plot it in dB
    if plot:
        psd.plot()
        ax = plt.gcf().get_axes()[0]
        # plot mean and overall std as a shaded area with mean line
        ax.fill_between(freq, avgpsd-avgstdpsd, avgpsd+avgstdpsd, color='gray', alpha=0.5)
        ax.plot(freq, avgpsd, color='black', linewidth=1)
    # is there a peak that is at least avgstdpsd above the mean?
    from scipy.signal import find_peaks
    peak_idx, _ = find_peaks(avgpsd, prominence=avgstdpsd)
    freqs_peak = freq[peak_idx]
    if len(freqs_peak)==0:
        print("No humming frequencies found")
        return None
    elif len(freqs_peak)==1:
        # determine if closer to 50 or 60 Hz
        if np.abs(freqs_peak-50)<np.abs(freqs_peak-60):
            print(f"Humming frequency found at {freqs_peak} Hz (closer to 50 Hz)")
            return 50
        else:
            print(f"Humming frequency found at {freqs_peak} Hz (closer to 60 Hz)")
            return 60
    elif len(freqs_peak)>1:
        warnings.warn(f"Possible humming frequencies found at {freqs_peak} Hz")
        # select the one that is closer to 50 or 60 Hz
        close_to50 = np.abs(freqs_peak-50)
        close_to60 = np.abs(freqs_peak-60)
        if np.min(close_to50)<np.min(close_to60):
            print(f"Humming frequency found at {freqs_peak[np.argmin(close_to50)]} Hz (closer to 50 Hz)")
            return 50
        else:
            print(f"Humming frequency found at {freqs_peak[np.argmin(close_to60)]} Hz (closer to 60 Hz)")
            return 60
        
    return None


def _apply_prep_and_interp(rawobj, prep_params, montage, seed):
    tmp = mne.set_log_level('ERROR')
    raw_after_prep = rawobj.copy()
    # Apply PREP pipeline
    prep = PrepPipeline(raw_after_prep, prep_params, montage, random_state=seed)
    prep.fit()
    print(
        "Bad channels/original/after interp: "
        f"{prep.interpolated_channels}/{prep.noisy_channels_original['bad_all']}/{prep.still_noisy_channels}"
    )
    still_noisy = prep.still_noisy_channels
    raw_after_prep = prep.raw
    mne.set_log_level(tmp)
    return raw_after_prep, still_noisy

def _force_spline_interp(rawobj, still_noisy):
    # Reset and Spherical spline interpolation for still noisy channels
    if len(still_noisy) >= 1:
        rawobj.info['bads'] = still_noisy
        rawobj.interpolate_bads(reset_bads=True, mode='spline')
    return rawobj

def _smoothbygroup_or(arr, k=None, sfreq=None, ignore_zero_edges=False):
    if sfreq is not None: # transform k from s to samples
        k = int((k * sfreq))
    
    if k is None:
        return np.array(arr)
    elif k == 0 or k == 1:
        warnings.warn("Threshold in s is too low to be considered. Returning input.")
        return np.array(arr)
    elif k < 0:
        raise ValueError("k must be a positive integer or None.")
    else: 
        arr_ok = []
        Ng = len(list(groupby(arr)))
        for i,(value, group) in enumerate(groupby(arr)):
            group_list = list(group)  # Convert group to a list to check length
            if ignore_zero_edges and i in [0,Ng-1] and value == 0:
                arr_ok.extend([0] * len(group_list)) # ignore short zeros at the edges
            elif value == 0 and len(group_list) < k:
                arr_ok.extend([1] * len(group_list))  # Convert short zeros to ones
            else:
                arr_ok.extend(group_list)
        
        return np.array(arr_ok,dtype=bool)


def _detect_nogfp(rawobj,threshold_uV=1e-6,threshold_s=0.030,desired = False):
    '''
    Detects intervals of no signal in the EEG data based on the global field power (GFP).
    no signal is defined as a flat signal with a standard deviation below or equal to threshold_uV.
    If threshold_s corresponds to at least 2 samples, only flat intervals of at least threshold_s are considered
    '''
    # Compute gfp
    gfp = rawobj.copy().pick('eeg').get_data().std(axis=0)
    # Logical array for flat gfp
    flat_gfp = np.array(gfp <= threshold_uV,dtype=bool)
    
    # flat henv
    # henv = rawobj.copy().pick('eeg').apply_hilbert(envelope=True).get_data().mean(axis=0)
    # flat_gfp = np.array(henv <= threshold_uV,dtype=bool)
    
    if desired == False: # Smooth false negatives
        flat_gfp_sm = ~_smoothbygroup_or(~flat_gfp, k=threshold_s, sfreq=rawobj.info['sfreq'], 
                                            ignore_zero_edges=True)
    elif desired == True: # Smooth false positives
        flat_gfp_sm = _smoothbygroup_or(flat_gfp, k=threshold_s, sfreq=rawobj.info['sfreq'], 
                                            ignore_zero_edges=True)
    else:
        raise ValueError("desired must be either True or False.")
    #print('>>flat_gfp_sm\n',[(val,len(list(g))) for val,g in groupby(flat_gfp_sm)])
    return flat_gfp_sm
   
def _detect_overth(xx,th=None,th_samples=1,desired = False):
    '''
    Detects intervals of xx signal filling holes shorter than th_saples.
    '''
    if th is None: 
        th = np.percentile(xx,50)
    # Detect over th
    over = np.array(xx >= th,dtype=bool)
    # Smooth over
    if desired == False: # Smooth false negatives
        over_sm = ~_smoothbygroup_or(~over, k=th_samples, sfreq=None, 
                                            ignore_zero_edges=True)
    elif desired == True: # Smooth false positives
        over_sm = _smoothbygroup_or(over, k=th_samples, sfreq=None, 
                                            ignore_zero_edges=True)
    else:
        raise ValueError("desired must be either True or False.")
    return over_sm
   
def _get_true_intervals(bool_arr, ret_transitions=False, sort=False, bool_val=True):
    '''
    Given a boolean array, returns a list of tuples with the start and end indexes of the True intervals.
    If ret_transitions is True, also returns the transitions array.
    If sort is True, sorts the intervals by length in descending order.
    '''
    if bool_val is False:
        bool_arr = ~np.array(bool_arr,dtype=bool)
    elif bool_val is True:
        bool_arr = np.array(bool_arr,dtype=bool)
    else:
        raise ValueError("bool_val must be either True or False.")
    
    
    if all(bool_arr): # If all values are True, return the whole array as a single interval
        transitions = np.zeros_like(bool_arr,dtype=float)
        transitions[0] = 1
        transitions[-1] = -1
        return [(0, len(bool_arr))], transitions if ret_transitions else [(0, len(bool_arr))]
    elif all(~bool_arr): # If all values are False, return an empty list
        transitions = np.zeros_like(bool_arr,dtype=float)
        transitions[0] = -1
        transitions[-1] = 1
        return [], transitions if ret_transitions else []
    else: # If there are both True and False values, compute intervals and transitions
        transitions = np.diff(np.pad(bool_arr,mode='edge',pad_width=(0,1)).astype(float))
        
        # Handle edge cases for transitions: if start/end edge has no transition,
        # set it to the opposite of the first or last non-zero value respectively
        if (transitions[0]==0):
            transitions[0] = -transitions[np.nonzero(transitions)[0][0]]
        if (transitions[-1]==0):
            transitions[-1] = -transitions[np.nonzero(transitions)[0][-1]]
        
        # Find the indexes of -1 and +1 events
        to_false = np.where(transitions == -1)[0]
        to_true = np.where(transitions == 1)[0]
        
        # Create intervals of true events: for each to_true, find the closest to_false
        intervals = [
                    (to_true[i], to_false[idx])
                    for i in range(len(to_true))
                    if (idx := np.searchsorted(to_false, to_true[i])) < len(to_false)
                ]
        # Sort intervals by length in descending order if requested
        if sort:
            intervals = sorted(intervals, key=lambda x: x[1] - x[0], reverse=True)
        
        return intervals, transitions if ret_transitions else intervals

