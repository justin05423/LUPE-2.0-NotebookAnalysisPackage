import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if os.path.join(os.path.abspath(''), '..') not in sys.path:
    sys.path.append(os.path.join(os.path.abspath(''), '..'))

from utils.classification import load_behaviors
from utils.meta import behavior_names
from utils.timing import constant_fps, project_dir
from utils.analysis_scripts.per_mouse_stats import (
    bout_metrics, describe, save_per_mouse_csv, save_figure, safe_name
)


def behavior_binned_mouse_screening(project_name, output_analysis_dir=None, heatmap_max_value=None,
                                    selected_groups=None, selected_conditions=None, label_max_chars=40,
                                    fps_lookup=None):
    """
    Per-mouse screening: frames per 1-minute bin for every behavior, as one heatmap per
    behavior per group-condition (rows = mice, columns = minutes).

    Inputs are the per-frame classification CSVs written by "Behavior CSV Classification"
    (figures/behaviors_csv_raw-classification/frames/<group>/<condition>/<file>.csv).

    Outputs (figures/behavior_individual-mouse_screening/):
      - <group>/<condition>/<behavior>_heatmap.svg (+ .png): mice x minute heatmap.
      - <group>/<condition>/<behavior>_data.csv: the matrix behind that heatmap (full file names).
      - mouse_id_key.csv: group, condition, full file name, and the shortened label used on figures.
      - per_mouse/<group>/<condition>/<file>.csv: the five indicators per behavior
        (percent_time, total_frames, bout_count, mean_bout_duration_s, bouts_per_min) and
        descriptives of frames-per-minute across bins (mean, sd, sem, median, min, max).

    Color scale: if heatmap_max_value is given, every heatmap uses it as vmax. Otherwise a shared
    vmax per behavior (max across all mice in the project) is used so conditions are comparable.

    Parameters:
        project_name (str): Project name.
        output_analysis_dir (str): Optional override for the output directory.
        heatmap_max_value (int or float): Optional fixed vmax for all heatmaps.
        selected_groups / selected_conditions (list): Restrict to these; None = all found.
        label_max_chars (int): Y-axis labels are truncated to this many characters (full names stay in CSVs).
        fps_lookup (callable): fps(group, condition); defaults to 60 fps everywhere.

    Returns:
        heatmap_files (dict): {group: {condition: {behavior: svg_path}}}
    """
    base_dir = project_dir(project_name)
    behaviors = load_behaviors(os.path.join(base_dir, f"behaviors_{project_name}.pkl"))
    if behaviors is None or not behaviors:
        raise ValueError("Failed to load behaviors or the dataset is empty.")

    frames_dir = os.path.join(base_dir, "figures", "behaviors_csv_raw-classification", "frames")
    csv_files = sorted(glob.glob(os.path.join(frames_dir, "*", "*", "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            "No per-frame classification CSVs found. Run 'Behavior CSV Classification' first."
        )

    analysis_dir = output_analysis_dir or os.path.join(base_dir, "figures", "behavior_individual-mouse_screening")
    os.makedirs(analysis_dir, exist_ok=True)

    fps_lookup = fps_lookup or constant_fps()
    stats = ['mean', 'sd', 'sem', 'median', 'min', 'max']

    # ---- Pass 1: per-mouse matrices (behavior x minute) and per-mouse CSVs
    # records[(group, condition)] = list of (mouse_id, matrix[n_behaviors, n_bins])
    records = {}
    key_rows = []
    for file in csv_files:
        condition = os.path.basename(os.path.dirname(file))
        group = os.path.basename(os.path.dirname(os.path.dirname(file)))
        if selected_groups is not None and group not in selected_groups:
            continue
        if selected_conditions is not None and condition not in selected_conditions:
            continue
        mouse_id = os.path.splitext(os.path.basename(file))[0]
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        if df.empty:
            print(f"Warning: {file} is empty. Skipping.")
            continue

        predict = df['behavior'].astype(int).values
        fps = fps_lookup(group, condition)
        # Minute index per frame at this group/condition's frame rate (works for non-integer fps)
        minute_bin = np.floor(np.arange(len(predict)) / (fps * 60.0)).astype(int)
        n_bins = int(minute_bin.max()) + 1
        matrix = np.zeros((len(behavior_names), n_bins), dtype=int)
        for b in range(len(behavior_names)):
            matrix[b] = np.bincount(minute_bin[predict == b], minlength=n_bins)
        records.setdefault((group, condition), []).append((mouse_id, matrix))

        short = mouse_id if len(mouse_id) <= label_max_chars else mouse_id[:label_max_chars] + "…"
        key_rows.append({'group': group, 'condition': condition, 'file': mouse_id, 'figure_label': short})

        # Per-mouse CSV: five indicators + frames-per-minute descriptives across bins
        pm = bout_metrics(predict, fps=fps)
        for stat in stats:
            pm[f'frames_per_min_{stat}'] = [describe(matrix[b])[stat] for b in range(len(behavior_names))]
        save_per_mouse_csv(pm, analysis_dir, group, condition, mouse_id, fps=fps)

    if not records:
        raise ValueError("No mice matched the selected groups/conditions.")

    key_df = pd.DataFrame(key_rows)
    key_df.to_csv(os.path.join(analysis_dir, "mouse_id_key.csv"), index=False)

    # Shared color max per behavior across the whole project (unless a fixed value is given)
    max_bins = max(m.shape[1] for recs in records.values() for _, m in recs)
    if heatmap_max_value is not None:
        vmax_by_behavior = {b: heatmap_max_value for b in range(len(behavior_names))}
    else:
        vmax_by_behavior = {
            b: max(int(m[b].max()) for recs in records.values() for _, m in recs) or 1
            for b in range(len(behavior_names))
        }

    # ---- Pass 2: one heatmap per behavior per group-condition
    heatmap_files = {}
    label_map = dict(zip(key_df['file'], key_df['figure_label']))
    for (group, condition), recs in records.items():
        out_dir = os.path.join(analysis_dir, safe_name(group), safe_name(condition))
        os.makedirs(out_dir, exist_ok=True)
        mice = [m for m, _ in recs]
        for b, behavior in enumerate(behavior_names):
            data = pd.DataFrame(
                [np.pad(mat[b], (0, max_bins - mat.shape[1])) for _, mat in recs],
                index=mice, columns=range(max_bins)
            )
            data.to_csv(os.path.join(out_dir, f"{behavior.replace(' ', '_')}_data.csv"))

            fig, ax = plt.subplots(figsize=(12, 0.35 * len(mice) + 2))
            sns.heatmap(
                data.rename(index=label_map), cmap='Reds', vmin=0, vmax=vmax_by_behavior[b],
                cbar_kws={'label': f'{behavior} frames per minute'}, ax=ax
            )
            ax.set_xlabel("Time Bin (minutes)")
            ax.set_ylabel("Mouse")
            ax.set_title(f"{behavior[0].upper() + behavior[1:]}  |  {group} - {condition}  (n = {len(mice)})")
            ax.tick_params(axis='y', labelrotation=0, labelsize=8)
            fig.tight_layout()

            svg_path, _ = save_figure(fig, os.path.join(out_dir, f"{behavior.replace(' ', '_')}_heatmap.svg"))
            plt.close(fig)
            heatmap_files.setdefault(group, {}).setdefault(condition, {})[behavior] = svg_path
            print(f"Saved heatmap: {svg_path}")

    return heatmap_files
