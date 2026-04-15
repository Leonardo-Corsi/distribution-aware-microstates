import argparse
import json
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import ConnectionPatch, Circle
from mne.filter import filter_data
from scipy.signal import find_peaks
import s2_microstates_utils as ms_utils
from _config_loader import load_configs

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create S2 graphical abstract from one preprocessed EEG window.")
    parser.add_argument("--config", default="configs.json", help="Path to configs.json")
    parser.add_argument("--subject-id", type=str, default=None, help="Optional subject/id substring filter.")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed override (default from config).")
    return parser.parse_args()


def _norm_ch(ch_name):
    token = ch_name.upper().strip()
    token = token.replace("EEG", "")
    token = token.replace("-REF", "")
    token = token.replace("REF", "")
    token = token.replace("-LE", "")
    token = token.replace("LE", "")
    token = token.replace("-", "")
    token = token.replace("_", "")
    token = token.replace(" ", "")
    return token


def _map_alias(ch_token):
    aliases = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6",
    }
    return aliases.get(ch_token, ch_token)


def load_metamaps(json_path, n_ms):
    reorder_str = ["CBAD", "DCBAE", "DBCFEA", "DGAEFBC", "FACEGHBD"]
    if not os.path.exists(json_path):
        raise FileNotFoundError("Metamaps json not found: " + json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        all_solutions = json.load(f)

    parsed = []
    for idx, sol in enumerate(all_solutions):
        maps = np.array(sol["Maps"], dtype=float)
        labels = [str(lbl) for lbl in sol["Labels"]]
        colors = np.array(sol.get("ColorMap", []), dtype=float)
        chan_labels = [str(ch["labels"]) for ch in sol["chanlocs"]]
        if idx < len(reorder_str):
            order = np.array(list(reorder_str[idx]))
            order_idx = np.argsort(order)
            maps = maps[order_idx]
            if len(colors) == maps.shape[0]:
                colors = colors[order_idx]
            if len(labels) == maps.shape[0]:
                labels = [labels[i] for i in order_idx]
        parsed.append(
            {
                "maps": maps,
                "labels": labels,
                "colors": colors,
                "chan_labels": chan_labels,
            }
        )

    lengths = [p["maps"].shape[0] for p in parsed]
    min_len = min(lengths)
    max_len = max(lengths)

    if n_ms > max_len:
        raise ValueError(f"Requested n_ms={n_ms} but max available is {max_len}")

    if n_ms < min_len:
        target = next(p for p in parsed if p["maps"].shape[0] == min_len)
        maps = target["maps"][:n_ms]
        labels = target["labels"][:n_ms]
        colors = target["colors"][:n_ms] if len(target["colors"]) else np.array([])
        return maps, labels, colors, target["chan_labels"]

    target = next(p for p in parsed if p["maps"].shape[0] == n_ms)
    return target["maps"], target["labels"], target["colors"], target["chan_labels"]


def _resolve_path(base_dir, value):
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(base_dir, value))


def _load_gabstract_runtime(cfg, seed_override=None):
    s2_cfg = cfg.get("s2", {})
    gab_cfg = s2_cfg.get("gabstract", {})
    if not isinstance(gab_cfg, dict):
        raise KeyError("Expected cfg['s2']['gabstract'] object.")

    if "s2_parent" not in s2_cfg:
        raise KeyError("Missing required cfg['s2']['s2_parent'].")

    defaults = {
        "input_stage": "ica",
        "output_subdir": "gabstract",
        "window_sec": 1.0,
        "flank_sec": 4.0,
        "start_max_sec": 300.0,
        "topomap_size": 0.35,
        "link_lines": True,
        "n_ms": 4,
        "metamaps_json": "assets/microstates/metamaps_export.json",
        "cleanup": True,
        "show": False,
    }

    s2_parent = _resolve_path(cfg["output_parent"], s2_cfg["s2_parent"])
    output_subdir = str(gab_cfg.get("output_subdir", defaults["output_subdir"]))
    output_dir = _resolve_path(s2_parent, output_subdir)

    input_stage = str(gab_cfg.get("input_stage", defaults["input_stage"])).strip().lower()
    if input_stage not in {"ica", "segmented"}:
        raise ValueError("cfg['s2']['gabstract']['input_stage'] must be 'ica' or 'segmented'.")
    if input_stage == "ica":
        input_dir = os.path.join(cfg["s1"]["s1_parent"], "preprocessed_segmented_ICA")
    else:
        input_dir = os.path.join(cfg["s1"]["s1_parent"], "preprocessed_segmented")

    metamaps_json = _resolve_path(
        cfg["meta"]["project_root"],
        str(gab_cfg.get("metamaps_json", defaults["metamaps_json"])),
    )
    seed = int(seed_override if seed_override is not None else gab_cfg.get("seed", cfg["s1"]["seed"]))

    return {
        "seed": seed,
        "window_sec": float(gab_cfg.get("window_sec", defaults["window_sec"])),
        "flank_sec": float(gab_cfg.get("flank_sec", defaults["flank_sec"])),
        "start_max_sec": float(gab_cfg.get("start_max_sec", defaults["start_max_sec"])),
        "topomap_size": float(gab_cfg.get("topomap_size", defaults["topomap_size"])),
        "link_lines": bool(gab_cfg.get("link_lines", defaults["link_lines"])),
        "n_ms": int(gab_cfg.get("n_ms", defaults["n_ms"])),
        "metamaps_json": metamaps_json,
        "cleanup": bool(gab_cfg.get("cleanup", defaults["cleanup"])),
        "show": bool(gab_cfg.get("show", defaults["show"])),
        "input_dir": os.path.normpath(input_dir),
        "output_dir": os.path.normpath(output_dir),
    }


def discover_input_records(input_dir, subject_id=None):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError("Input directory not found: " + input_dir)

    metadata_paths = sorted(
        [
            os.path.join(input_dir, d, "metadata.json")
            for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
            and os.path.exists(os.path.join(input_dir, d, "metadata.json"))
        ]
    )

    records = []
    for metadata_path in metadata_paths:
        record_dir = os.path.dirname(metadata_path)
        record_stem = os.path.basename(record_dir)
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        edf_files = meta.get("edf_files")
        if not isinstance(edf_files, list) or len(edf_files) == 0:
            continue

        edf_paths = [os.path.join(record_dir, rel) for rel in edf_files]
        edf_paths = [p for p in edf_paths if os.path.exists(p)]
        if len(edf_paths) == 0:
            continue

        if subject_id:
            sid = subject_id.lower()
            meta_name = str(meta.get("filename", "")).lower()
            if sid not in record_stem.lower() and sid not in meta_name:
                continue

        records.append(
            {
                "record_stem": record_stem,
                "record_dir": record_dir,
                "metadata_path": metadata_path,
                "metadata": meta,
                "edf_paths": edf_paths,
            }
        )

    if len(records) == 0:
        raise FileNotFoundError("No valid metadata+EDF records found (after optional subject filter).")
    return records


def detach_raw_from_source(raw):
    """
    Convert any loaded Raw into an in-memory RawArray so downstream ops
    never depend on original EDF file paths embedded in the object.
    """
    raw_loaded = raw.copy().load_data()
    data = raw_loaded.get_data()
    info = raw_loaded.info.copy()
    detached = mne.io.RawArray(data, info, verbose="ERROR")
    if raw_loaded.annotations is not None and len(raw_loaded.annotations):
        detached.set_annotations(raw_loaded.annotations.copy())
    return detached


def _safe_edf_duration_sec(edf_path):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        return float(raw.n_times / raw.info["sfreq"])
    except Exception:
        return None


def choose_record_and_segment(records, rng, required_sec):
    if len(records) == 0:
        raise RuntimeError("No input records available for selection.")

    rec_order = rng.permutation(len(records))
    for rec_idx in rec_order:
        rec = records[int(rec_idx)]
        seg_order = rng.permutation(len(rec["edf_paths"]))
        for seg_idx in seg_order:
            edf_path = rec["edf_paths"][int(seg_idx)]
            duration_sec = _safe_edf_duration_sec(edf_path)
            if duration_sec is None:
                continue
            if duration_sec < required_sec:
                continue
            return rec, edf_path, float(duration_sec)

    raise RuntimeError(
        f"No eligible EDF segment long enough for required window+flanks ({required_sec:.2f}s)."
    )


def _build_ms_cfg(cfg):
    s2_extraction_params = (
        cfg.get("s2", {})
        .get("extraction", {})
        .get("microstates", {})
        .get("pycrostates_params")
    )
    if s2_extraction_params is not None:
        return {"pycrostates_params": s2_extraction_params}

    raise KeyError("Missing required cfg['s2']['extraction']['microstates']['pycrostates_params'].")


def compute_gfp_and_mtmi(raw_window):
    data = raw_window.get_data(picks="eeg")
    sfreq = float(raw_window.info["sfreq"])
    gfp = np.nanstd(data, axis=0)
    gfp_flt = filter_data(gfp[np.newaxis, :], sfreq=sfreq, l_freq=None, h_freq=32.0, verbose="ERROR").squeeze()

    min_distance = max(1, int(round(0.03 * sfreq)))
    prominence = float(np.std(gfp_flt))
    minima, _ = find_peaks(-gfp_flt, distance=min_distance, prominence=prominence)
    if minima.size < 2:
        minima, _ = find_peaks(-gfp_flt, distance=min_distance)

    mtmi_ms = np.diff(minima) * 1000.0 / sfreq if minima.size >= 2 else np.array([])
    return gfp, gfp_flt, minima, mtmi_ms


def contiguous_runs(labels):
    if len(labels) == 0:
        return []
    runs = []
    start = 0
    current = labels[0]
    for idx in range(1, len(labels)):
        if labels[idx] != current:
            runs.append((start, idx, current))
            start = idx
            current = labels[idx]
    runs.append((start, len(labels), current))
    return runs

def channel_alignment(raw_eeg, meta_chan_labels):
    raw_names = list(raw_eeg.ch_names)
    raw_idx = {_norm_ch(ch): i for i, ch in enumerate(raw_names)}

    montage = mne.channels.make_standard_montage("standard_1020")
    ch_pos = montage.get_positions()["ch_pos"]
    pos_idx = {_norm_ch(ch): np.array(pos[:2], dtype=float) for ch, pos in ch_pos.items()}

    aligned_raw_idx = []
    aligned_meta_idx = []
    aligned_pos = []
    aligned_names = []

    for mi, ch in enumerate(meta_chan_labels):
        token = _norm_ch(ch)
        candidates = [token, _map_alias(token)]

        ridx = None
        for cand in candidates:
            if cand in raw_idx:
                ridx = raw_idx[cand]
                break
        if ridx is None:
            continue

        pos = None
        for cand in candidates:
            if cand in pos_idx:
                pos = pos_idx[cand]
                break
        if pos is None:
            continue

        aligned_raw_idx.append(ridx)
        aligned_meta_idx.append(mi)
        aligned_pos.append(pos)
        aligned_names.append(ch)

    if len(aligned_raw_idx) < 8:
        raise RuntimeError(f"Only {len(aligned_raw_idx)} channels could be aligned between raw and metamaps.")

    return np.array(aligned_raw_idx), np.array(aligned_meta_idx), np.array(aligned_pos), aligned_names


def plot_metamaps_figure(maps, pos2d, out_prefix, show=False):
    n_ms = maps.shape[0]
    fig, axes = plt.subplots(1, n_ms, figsize=(1.5 * n_ms, 1.8))
    axes = np.atleast_1d(axes)
    vmax = 2.0
    for m in range(n_ms):
        mne.viz.plot_topomap(
            maps[m] / max(np.std(maps[m]), 1e-12),
            pos2d,
            axes=axes[m],
            show=False,
            contours=0,
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
        )
        axes[m].set_title(f"Map {chr(ord('A') + m)}", fontsize=10)
    fig.suptitle("Meta-microstates", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(out_prefix + ".svg", bbox_inches="tight")
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def create_graphical_abstract(
    raw_segment,
    topo_data_aligned,
    selected_path,
    selected_start_sec,
    window_sec,
    flank_sec,
    sequence,
    maps,
    pos2d,
    gfp,
    gfp_flt,
    minima_idx,
    mtmi_ms,
    output_prefix,
    topomap_size=0.35,
    link_lines=False,
    cleanup=False,
    show=False,
):
    def _zscore_map(vec):
        v = np.asarray(vec, dtype=float).copy()
        v -= np.mean(v)
        s = np.std(v)
        if s <= 0:
            s = 1.0
        return v / s

    def _add_topomap_outline(ax, edge_color, lw=3.0):
        # Keep circle in axes coordinates to preserve circular geometry.
        circ = Circle((0.5, 0.5), 0.48, transform=ax.transAxes, fill=False, edgecolor=edge_color, linewidth=lw, zorder=10)
        circ.set_clip_on(False)
        ax.add_patch(circ)

    sfreq = float(raw_segment.info["sfreq"])
    n_samples = raw_segment.n_times
    times = np.arange(n_samples, dtype=float) / sfreq - flank_sec
    data_uv = raw_segment.get_data(picks="eeg") * 1e6
    gfp_uv = gfp * 1e6
    ch_names = list(raw_segment.ch_names)
    n_ch = data_uv.shape[0]

    fig = plt.figure(figsize=(5.83, 4))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.2, 0.20, 1.4], hspace=0.0)
    ax_eeg = fig.add_subplot(gs[0, 0])
    ax_gfp = fig.add_subplot(gs[1, 0], sharex=ax_eeg)
    ax_strip = fig.add_subplot(gs[2, 0], sharex=ax_eeg)
    ax_mtmi = fig.add_subplot(gs[3, 0], sharex=ax_eeg)

    amp = np.percentile(np.abs(data_uv), 90)
    if amp <= 0:
        amp = 1.0
    spacing = 1.6 * amp
    offsets = (np.arange(n_ch)[::-1] * spacing).astype(float)
    for i in range(n_ch):
        ax_eeg.plot(times, data_uv[i] + offsets[i], color="#6e73b0", linewidth=0.8)
    ax_eeg.set_yticks(offsets)
    ax_eeg.set_yticklabels(ch_names, fontsize=7)
    ax_eeg.set_xlim(0.0, window_sec)
    ax_eeg.set_ylabel("EEG")
    ax_eeg.set_xlabel("")

    palette = ["tab:blue", "tab:orange", "tab:olive", "tab:purple", "tab:green", "tab:red", "tab:brown"]
    runs = contiguous_runs(sequence)
    for start, stop, label in runs:
        if label < 0 or label >= maps.shape[0]:
            continue
        ax_gfp.fill_between(
            times[start:stop],
            gfp_uv[start:stop],
            0,
            color=palette[label % len(palette)],
            alpha=0.35,
            linewidth=0.0,
        )
    ax_gfp.plot(times, gfp_uv, color="k", linewidth=1.2, alpha=0.9)
    if minima_idx.size > 0:
        ax_gfp.scatter(times[minima_idx], gfp_uv[minima_idx], s=12, color="k", zorder=3)
    ax_gfp.set_xlim(0.0, window_sec)
    ax_gfp.set_ylabel(r"GFP $\mu$V")
    ax_gfp.set_xlabel("")
    
    ax_strip.set_xlim(0.0, window_sec)
    ax_strip.set_ylim(0.0, 1.0)
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("")

    for start, stop, label in runs:
        if label < 0 or label >= maps.shape[0]:
            continue
        x0 = max(0.0, float(times[start]))
        x1 = min(window_sec, float(times[stop - 1]) if stop > start else float(times[start]))
        if x1 <= x0:
            continue
        ax_strip.axvspan(x0, x1, ymin=0.1, ymax=0.9, color=palette[label % len(palette)], alpha=0.85, linewidth=0.0, zorder=2)
        letter = chr(ord("A") + int(label)) if 0 <= int(label) < 26 else str(int(label))
        ax_strip.text((x0 + x1) * 0.5, 0.5, letter, ha="center", va="center", fontsize=8, color="k", zorder=3)

    mtmi_t = times[minima_idx[1:]] if minima_idx.size >= 2 else np.array([])
    if mtmi_ms.size > 0 and mtmi_t.size > 0:
        ax_mtmi.plot(mtmi_t, mtmi_ms, color="tab:red", marker="o", linewidth=1.2, markersize=4)
    else:
        ax_mtmi.text(0.5, 0.5, "Not enough minima for MTMI", ha="center", va="center", transform=ax_mtmi.transAxes)
    ax_mtmi.set_xlim(0.0, window_sec)
    ax_mtmi.set_ylabel("MTMI [ms]")
    ax_mtmi.set_xlabel("Time (s)")
    if mtmi_ms.size > 0 and mtmi_t.size > 0:
        in_window_mask = (mtmi_t >= 0.0) & (mtmi_t <= window_sec)
        in_window_vals = mtmi_ms[in_window_mask]
        range_source = in_window_vals.astype(float) if in_window_vals.size > 0 else mtmi_ms.astype(float)
        y_min = float(np.min(range_source))
        y_max = float(np.max(range_source))
        span = y_max - y_min
        if span <= 0:
            margin = max(1e-6, 0.05 * max(abs(y_min), 1.0))
        else:
            margin = 0.05 * span
        ax_mtmi.set_ylim(y_min - margin, y_max + margin)
    else:
        ax_mtmi.set_ylim(0.0, 1.0)

    for ax in [ax_eeg, ax_gfp, ax_strip]:
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.margins(x=0.0, y=0.0)

    ax_mtmi.spines["top"].set_visible(False)
    ax_mtmi.spines["right"].set_visible(False)
    ax_mtmi.margins(x=0.0)
    ax_mtmi.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
    ax_mtmi.set_xticks(np.linspace(0.0, window_sec, 5))

    in_view = np.where((times >= 0.0) & (times <= window_sec))[0]
    if in_view.size == 0:
        raise RuntimeError("No samples in plotting window after time alignment.")
    view_start = int(in_view[0])
    view_stop = int(in_view[-1]) + 1

    interval_argmax = []
    runs = contiguous_runs(sequence)
    for start, stop, label in runs:
        # valid microstate intervals only
        if label < 0 or label >= maps.shape[0]:
            continue
        left = max(int(start), view_start)
        right = min(int(stop), view_stop)
        if right <= left:
            continue
        local = np.argmax(gfp[left:right])
        idx = left + int(local)
        interval_argmax.append((idx, int(label)))

    minima_t = times[minima_idx] if minima_idx.size > 0 else np.array([])
    minima_t = minima_t[(minima_t >= 0.0) & (minima_t <= window_sec)]
    minima_in_window = minima_idx[(times[minima_idx] >= 0.0) & (times[minima_idx] <= window_sec)] if minima_idx.size > 0 else np.array([], dtype=int)

    if minima_in_window.size > 0:
        min_vals = gfp_uv[minima_in_window]
        arrow_y = max(float(np.min(min_vals)) / 2.0, float(ax_gfp.get_ylim()[1]) / 20.0)
        label_tokens = ["i"] + [f"i+{k}" for k in range(1, len(minima_in_window))]

        first_x = times[minima_in_window[0]]
        left_anchor = ax_gfp.get_xlim()[0]
        ax_gfp.annotate(
            "",
            xy=(first_x, arrow_y),
            xytext=(left_anchor, arrow_y),
            arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
            zorder=8,
        )
        ax_gfp.text((first_x + left_anchor) * 0.5, arrow_y, label_tokens[0], fontsize=8, va="bottom", ha="center", zorder=8)

        for k in range(len(minima_in_window) - 1):
            x0 = times[minima_in_window[k]]
            x1 = times[minima_in_window[k + 1]]
            ax_gfp.annotate(
                "",
                xy=(x1, arrow_y),
                xytext=(x0, arrow_y),
                arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
                zorder=8,
            )
            lbl = f"i+{k+1}"
            ax_gfp.text((x0 + x1) * 0.5, arrow_y, lbl, fontsize=8, va="bottom", ha="center", zorder=8)

        if mtmi_t.size > 0 and mtmi_ms.size > 0:
            mtmi_in_window = np.where((mtmi_t >= 0.0) & (mtmi_t <= window_sec))[0]
            mtmi_tokens = ["i"] + [f"i+{k}" for k in range(1, len(mtmi_in_window))]
            y0, y1 = ax_mtmi.get_ylim()
            y_target = 0.5 * (y1 - y0)
            y_span = max(1e-12, y1 - y0)
            x_shift = 0.015 * window_sec
            y_shift_abs = 0.085 * y_span
            for j, mi in enumerate(mtmi_in_window):
                lbl = mtmi_tokens[j]
                yv = mtmi_ms[mi]
                y_sign = 1.0 if (y_target - yv) >= 0 else -1.0
                ax_mtmi.text(
                    mtmi_t[mi] + x_shift,
                    yv + y_sign * y_shift_abs,
                    lbl,
                    fontsize=8,
                    va="bottom",
                    ha="left",
                    zorder=8,
                )

    if len(interval_argmax) > 0:
        fig.canvas.draw()
        eeg_pos = ax_eeg.get_position()
        fig_w, fig_h = fig.get_size_inches()
        box_w = float(topomap_size) / fig_w
        box_h = float(topomap_size) / fig_h

        # z-scored topomaps: value scale is comparable across samples/maps
        vmax_data = 1.0
        vmax_maps = 1.0
        gfp_pos = ax_gfp.get_position()
        map_top_y_center_base = gfp_pos.y1 + 0.006
        if map_top_y_center_base + box_h / 2.0 > 0.995:
            map_top_y_center_base = 0.995 - box_h / 2.0
        topo_t = np.array([times[idx] for idx, _ in interval_argmax], dtype=float)
        # minima lines: full-height only on MTMI; on GFP only from 0 to minima value
        for tmin in minima_t:
            ax_mtmi.axvline(tmin, color="k", linewidth=1.0, alpha=0.7, zorder=3)
            ax_strip.axvline(tmin, color="k", linewidth=1.0, alpha=0.7, zorder=3)
            idx_min = int(np.argmin(np.abs(times - tmin)))
            y_top = float(gfp_uv[idx_min])
            ax_gfp.vlines(tmin, ymin=0.0, ymax=y_top, color="k", linewidth=1.0, alpha=0.7, zorder=3)
        for idx, _ in interval_argmax:
            tt = times[idx]
            y_tt = float(gfp_uv[idx])
            y_guard_fig = map_top_y_center_base - box_h / 2.0 - 0.25 * box_h
            _, y_guard_data = ax_gfp.transData.inverted().transform(fig.transFigure.transform((tt, y_guard_fig)))
            y_cap = min(y_tt, float(y_guard_data))
            ax_gfp.vlines(tt, ymin=0.0, ymax=y_cap, color="0.45", linewidth=2.0, alpha=0.7, zorder=3)

        x_centers = []
        for idx, _ in interval_argmax:
            x_data = times[idx]
            x_center, _ = fig.transFigure.inverted().transform(ax_eeg.transData.transform((x_data, 0.0)))
            x_centers.append(x_center)
        x_centers = np.asarray(x_centers, dtype=float)
        if x_centers.size > 1:
            x0 = eeg_pos.x0
            x1 = eeg_pos.x0 + eeg_pos.width
            u = (x_centers - x0) / max((x1 - x0), 1e-12)
            u = np.clip(u, 0.0, 1.0)
            # hack: we displace them evenly so that we minimize overlap
            n = len(x_centers)
            u = np.linspace(0.5/n, 1.0-0.5/n, n)
            
            x_centers = x0 + u * (x1 - x0)

        for (idx, lbl), x_center in zip(interval_argmax, x_centers):
            x_data = times[idx]
            left = x_center - box_w / 2.0
            min_left = eeg_pos.x0
            max_left = eeg_pos.x0 + eeg_pos.width - box_w
            left = min(max(left, min_left), max_left)

            map_top_y_center = map_top_y_center_base - 0.004
            # Keep raw topomaps just above template row and slightly overlapping EEG.
            raw_y_center = max(map_top_y_center + 0.95 * box_h, eeg_pos.y0 + 0.22 * eeg_pos.height)
            if raw_y_center + box_h / 2.0 > eeg_pos.y1 + 0.02:
                raw_y_center = eeg_pos.y1 + 0.02 - box_h / 2.0

            ax_topo_data = fig.add_axes([left, raw_y_center - box_h / 2.0, box_w, box_h])
            template_vec = _zscore_map(maps[lbl])
            sample_vec = _zscore_map(topo_data_aligned[:, idx])
            corr = float(np.dot(sample_vec, template_vec) / (np.linalg.norm(sample_vec) * np.linalg.norm(template_vec) + 1e-12))
            if corr < 0:
                sample_vec = -sample_vec
            mne.viz.plot_topomap(
                sample_vec,
                pos2d,
                axes=ax_topo_data,
                show=False,
                sensors=False,
                contours=0,
                cmap="RdBu_r",
                vlim=(-vmax_data, vmax_data),
            )
            ax_topo_data.set_xticks([])
            ax_topo_data.set_yticks([])
            if link_lines:
                # raw policy: from max(0, topo_top + 0.25h) to top of EEG, then to top of raw topomap
                raw_top_fig = raw_y_center + box_h / 2.0
                raw_src_fig = raw_top_fig + 0.25 * box_h
                _, raw_src_data = ax_eeg.transData.inverted().transform(fig.transFigure.transform((x_data, raw_src_fig)))
                raw_src_data = max(0.0, float(raw_src_data))
                y_eeg_top = float(ax_eeg.get_ylim()[1])
                ax_eeg.vlines(x_data, ymin=raw_src_data, ymax=y_eeg_top, color="0.45", linewidth=2.0, alpha=0.7, zorder=3)
                conn_up = ConnectionPatch(
                    xyA=(x_data, raw_src_data),
                    coordsA=ax_eeg.transData,
                    xyB=(0.5, 1.0),
                    coordsB=ax_topo_data.transAxes,
                    color="0.45",
                    linewidth=2.0,
                    alpha=0.7,
                    zorder=4,
                    clip_on=False,
                )
                ax_eeg.add_artist(conn_up)
            _add_topomap_outline(ax_topo_data, edge_color="0.65", lw=3.0)

            ax_topo_map_top = fig.add_axes([left, map_top_y_center - box_h / 2.0, box_w, box_h])
            mne.viz.plot_topomap(
                template_vec,
                pos2d,
                axes=ax_topo_map_top,
                show=False,
                sensors=False,
                contours=0,
                cmap="RdBu_r",
                vlim=(-vmax_maps, vmax_maps),
            )
            ax_topo_map_top.set_xticks([])
            ax_topo_map_top.set_yticks([])
            if link_lines:
                # template policy: start at min(gfp(x), topo_bottom-0.25h in GFP-data), then connect to topo bottom
                y_gfp_here = float(gfp_uv[idx])
                topo_bottom_fig = map_top_y_center - box_h / 2.0
                y_guard_fig = topo_bottom_fig - 0.25 * box_h
                _, y_guard_data = ax_gfp.transData.inverted().transform(fig.transFigure.transform((x_data, y_guard_fig)))
                y_anchor = min(y_gfp_here, float(y_guard_data))
                conn_mid_top = ConnectionPatch(
                    xyA=(x_data, y_anchor),
                    coordsA=ax_gfp.transData,
                    xyB=(0.5, 0.0),
                    coordsB=ax_topo_map_top.transAxes,
                    color="0.45",
                    linewidth=2.0,
                    alpha=0.7,
                    zorder=4,
                    clip_on=False,
                )
                ax_gfp.add_artist(conn_mid_top)
            _add_topomap_outline(ax_topo_map_top, edge_color=palette[lbl % len(palette)], lw=3.0)
    else:
        for tmin in minima_t:
            ax_mtmi.axvline(tmin, color="k", linewidth=1.0, alpha=0.7, zorder=3)
            ax_strip.axvline(tmin, color="k", linewidth=1.0, alpha=0.7, zorder=3)
            idx_min = int(np.argmin(np.abs(times - tmin)))
            y_top = float(gfp_uv[idx_min])
            ax_gfp.vlines(tmin, ymin=0.0, ymax=y_top, color="k", linewidth=1.0, alpha=0.7, zorder=3)

    if cleanup:
        for ax in [ax_eeg, ax_gfp, ax_strip, ax_mtmi]:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis="both", which="both", length=0)
            ax.spines["left"].set_visible(False)
        ax_mtmi.spines["bottom"].set_visible(False)

    fig.subplots_adjust(hspace=0.0)
    fig.savefig(output_prefix + ".png", dpi=260, bbox_inches="tight")
    fig.savefig(output_prefix + ".svg", bbox_inches="tight")
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def main():
    args = parse_args()
    cfg = load_configs(config_path=args.config)
    runtime = _load_gabstract_runtime(cfg, seed_override=args.seed)
    rng = np.random.default_rng(runtime["seed"])

    window_sec = runtime["window_sec"]
    flank_sec = runtime["flank_sec"]
    if window_sec <= 0 or flank_sec < 0:
        raise ValueError("window-sec must be > 0 and flank-sec must be >= 0.")
    required_sec = window_sec + 2.0 * flank_sec

    input_dir = runtime["input_dir"]
    records = discover_input_records(input_dir, subject_id=args.subject_id)
    selected_record, selected_edf_path, duration_sec = choose_record_and_segment(records, rng, required_sec)
    selected_path = selected_edf_path

    raw = mne.io.read_raw_edf(selected_edf_path, preload=True, verbose="ERROR").pick("eeg")
    raw = detach_raw_from_source(raw)
    if raw.n_times < 10:
        raise RuntimeError("Selected raw has too few samples.")

    sfreq = float(raw.info["sfreq"])
    if duration_sec < required_sec:
        raise RuntimeError(
            f"Recording too short ({duration_sec:.2f}s) for required window+flanks ({required_sec:.2f}s)."
        )
    target_start_min = flank_sec
    target_start_max = min(runtime["start_max_sec"], duration_sec - flank_sec - window_sec)
    if target_start_max <= target_start_min:
        raise RuntimeError("No valid start time for requested window/flanks within recording.")

    start_sec = float(rng.uniform(target_start_min, target_start_max))
    seg_start = start_sec - flank_sec
    seg_stop = start_sec + window_sec + flank_sec
    raw_segment = raw.copy().crop(tmin=seg_start, tmax=seg_stop, include_tmax=False).load_data()

    maps, _, _, meta_chan_labels = load_metamaps(runtime["metamaps_json"], runtime["n_ms"])
    ridx, midx, pos2d, _ = channel_alignment(raw_segment, meta_chan_labels)

    raw_aligned = raw_segment.get_data(picks="eeg")[ridx, :]
    maps_aligned = maps[:, midx]
    ms_maps, ms_gev, ms_lbl = ms_utils.LoadMetamaps(filename=runtime["metamaps_json"], plot=False, n_ms=runtime["n_ms"])
    ms_model = ms_utils.PycroModKMeans(ms_maps, ms_gev, ms_lbl, plot=False)
    ms_cfg = _build_ms_cfg(cfg)
    ms_result = ms_utils.microstates_extraction(raw_segment.copy(), ms_model, ms_cfg)
    sequence = np.asarray(ms_result["sequence"], dtype=int)

    gfp, gfp_flt, minima_idx, mtmi_ms = compute_gfp_and_mtmi(raw_segment.copy().pick(raw_segment.ch_names))

    os.makedirs(runtime["output_dir"], exist_ok=True)
    base_name = selected_record["record_stem"]
    out_prefix_main = os.path.join(runtime["output_dir"], f"gabstract_{base_name}_seed{runtime['seed']}")
    out_prefix_maps = os.path.join(runtime["output_dir"], f"metamaps_n{runtime['n_ms']}")

    plot_metamaps_figure(maps_aligned, pos2d, out_prefix_maps, show=runtime["show"])
    create_graphical_abstract(
        raw_segment=raw_segment,
        topo_data_aligned=raw_aligned,
        selected_path=selected_path,
        selected_start_sec=start_sec,
        window_sec=window_sec,
        flank_sec=flank_sec,
        sequence=sequence,
        maps=maps_aligned,
        pos2d=pos2d,
        gfp=gfp,
        gfp_flt=gfp_flt,
        minima_idx=minima_idx,
        mtmi_ms=mtmi_ms,
        output_prefix=out_prefix_main,
        topomap_size=runtime["topomap_size"],
        link_lines=runtime["link_lines"],
        cleanup=runtime["cleanup"],
        show=runtime["show"],
    )

    print("Config:", args.config)
    print("Input dir:", input_dir)
    print("Output dir:", runtime["output_dir"])
    print("Selected record:", selected_record["record_stem"])
    print("Selected EDF:", selected_path)
    print("Start time [s]:", round(start_sec, 6))
    print("Flank [s]:", flank_sec)
    print("Window [s]:", window_sec)
    print("Output main:", out_prefix_main + ".png/.svg")
    print("Output maps:", out_prefix_maps + ".png/.svg")


if __name__ == "__main__":
    main()
