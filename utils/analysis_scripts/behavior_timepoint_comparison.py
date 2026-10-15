import os
import numpy as np
import pandas as pd

from utils.meta import behavior_names
from utils.timing import constant_fps, project_dir
from utils.analysis_scripts.per_mouse_stats import (
    describe, save_per_mouse_csv, save_group_summary
)


def behavior_timepoint_comparison(project_name, selected_groups, selected_conditions, time_ranges, fps_lookup=None):
    """
    Compare behavior metrics across user-defined time windows.

    Inputs are the per-second classification CSVs written by "Behavior CSV Classification"
    (figures/behaviors_csv_raw-classification/seconds/<group>/<condition>/<file>.csv).

    Outputs (figures/behavior_timepoint_comparison/):
      - per_mouse/<group>/<condition>/<file>.csv: one row per (time window, behavior) with the
        five indicators: Fraction Time, Total Frames, Bout Count, Bouts per Minute,
        Mean Bout Duration (s), plus bout-duration SD / median / min / max.
      - per_mouse/summary_across_mice_<group>-<condition>.csv: mean, SD, SEM, median, min, max
        and n (mice) of each indicator per (time window, behavior).

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): Groups to analyze.
        selected_conditions (list): Conditions to analyze.
        time_ranges (list): Tuples of (start_s, end_s), e.g. [(0, 600), (600, 1800)]. Windows are [start, end).
        fps_lookup (callable): fps(group, condition); defaults to 60 fps everywhere.
    """
    fps_lookup = fps_lookup or constant_fps()
    if len(time_ranges) < 2:
        raise ValueError("At least two time ranges are required for comparison.")

    time_labels = [f"{start // 60}-{end // 60} min" for start, end in time_ranges]
    bins = [start for start, _ in time_ranges] + [time_ranges[-1][1]]

    base_dir = project_dir(project_name)
    input_dir = os.path.join(base_dir, "figures", "behaviors_csv_raw-classification", "seconds")
    analysis_dir = os.path.join(base_dir, "figures", "behavior_timepoint_comparison")
    os.makedirs(analysis_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            "Per-second classification CSVs not found. Run 'Behavior CSV Classification' first."
        )

    indicator_cols = ['Fraction Time', 'Total Frames', 'Bout Count',
                      'Bouts per Minute', 'Mean Bout Duration (s)']

    def window_metrics(data, frame_rate):
        """Five indicators per behavior for one mouse within one time window."""
        rows = []
        n_rows = len(data)
        minutes = n_rows / frame_rate / 60 if n_rows else np.nan
        for b, label in enumerate(behavior_names):
            bdata = data[data['behavior'] == b]
            total_frames = int(len(bdata))
            if total_frames:
                bout_id = (bdata.index.to_series().diff() > 1).cumsum()
                bout_sizes = bdata.groupby(bout_id).size().values / frame_rate
            else:
                bout_sizes = np.array([])
            d = describe(bout_sizes)
            rows.append({
                'Behavior': b,
                'Behavior Label': label,
                'Fraction Time': total_frames / n_rows if n_rows else np.nan,
                'Total Frames': total_frames,
                'Bout Count': int(d['n']),
                'Bouts per Minute': d['n'] / minutes if minutes else np.nan,
                'Mean Bout Duration (s)': d['mean'],
                'Bout Duration SD (s)': d['sd'],
                'Bout Duration Median (s)': d['median'],
                'Bout Duration Min (s)': d['min'],
                'Bout Duration Max (s)': d['max'],
            })
        return rows

    for group in selected_groups:
        for condition in selected_conditions:
            group_cond_dir = os.path.join(input_dir, group, condition)
            if not os.path.isdir(group_cond_dir):
                print(f"No directory found for group '{group}' and condition '{condition}'")
                continue

            per_mouse_frames = []
            frame_rate = fps_lookup(group, condition)
            for file_name in sorted(os.listdir(group_cond_dir)):
                if not file_name.endswith('.csv'):
                    continue
                mouse = os.path.splitext(file_name)[0]
                df = pd.read_csv(os.path.join(group_cond_dir, file_name))
                # Time is recomputed from the row index at the CURRENT frame rate, so a stale
                # time_seconds column (CSV written at an earlier rate) cannot shift the windows.
                df['time_seconds'] = np.arange(len(df)) / frame_rate

                file_bins = list(bins)
                max_time = df['time_seconds'].max()
                if max_time < file_bins[-1]:
                    print(f"Warning: max time ({max_time}s) in {file_name} is below the final window end "
                          f"({file_bins[-1]}s); last window is truncated.")
                    file_bins[-1] = max_time + 1.0 / frame_rate

                try:
                    df['time_group'] = pd.cut(df['time_seconds'], bins=file_bins, labels=time_labels, right=False)
                except ValueError as e:
                    print(f"Error binning {file_name}: {e}")
                    continue

                rows = []
                for tg, gdata in df.groupby('time_group', observed=False):
                    if gdata.empty:
                        continue
                    for r in window_metrics(gdata, frame_rate):
                        rows.append({'Time Group': str(tg), **r})
                pm = pd.DataFrame(rows)
                save_per_mouse_csv(pm, analysis_dir, group, condition, mouse, fps=frame_rate)
                pm['file'] = mouse
                per_mouse_frames.append(pm)

            if per_mouse_frames:
                save_group_summary(pd.concat(per_mouse_frames, ignore_index=True), indicator_cols,
                                   analysis_dir, group, condition, by=('Time Group', 'Behavior Label'))
                print(f"Saved per-mouse timepoint CSVs and summary for {group} - {condition}")

    print('Behavior timepoint comparison completed.')
