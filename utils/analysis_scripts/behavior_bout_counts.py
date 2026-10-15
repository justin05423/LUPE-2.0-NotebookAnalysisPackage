import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.classification import load_behaviors
from utils.meta import behavior_names, behavior_colors  # Assumes meta.py defines these
from utils.timing import constant_fps, project_dir
from utils.analysis_scripts.per_mouse_stats import (
    group_figures, safe_name,
    save_figure,
    bout_metrics, describe, save_per_mouse_csv, save_group_summary, INDICATOR_COLS
)

def behavior_bout_counts(project_name, selected_groups, selected_conditions, fps_lookup=None):
    """
    Generate a figure showing the bout rate (bouts per minute) for each behavior as a horizontal
    bar chart (mean ± SD across mice) for each selected group and condition. Each file's bout count
    is divided by that file's own duration, so videos of different lengths are comparable. The
    group CSV holds descriptives of both bouts/min and raw counts across mice.

    Additionally, it performs a raw frequency analysis of behavioral bouts and saves CSVs:
      - One CSV per group/condition (raw bout counts per file)
      - One combined CSV aggregating all bout counts.
      - One CSV per mouse/file (per_mouse/<group>/<condition>/<file>.csv) with the five
        indicator stats per behavior (percent_time, total_frames, bout_count,
        mean_bout_duration_s, bouts_per_min) and bout-duration descriptives.
      - One across-mice summary per group/condition (mean, SD, SEM, median, min, max, n)
        built from the per-mouse values.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of groups to analyze.
        selected_conditions (list): List of conditions to analyze.

    Returns:
        figs (list of matplotlib.figure.Figure), one per group: The generated figure containing the bar charts.
    """

    base_dir = project_dir(project_name)
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")
    fps_lookup = fps_lookup or constant_fps()
    behaviors = load_behaviors(behaviors_file)

    directory_path = os.path.join(base_dir, "figures", "behavior_instance-counts")
    os.makedirs(directory_path, exist_ok=True)

    # Helper function: compute number of bouts for each behavior from a prediction vector.
    def get_num_bouts(predict, behavior_classes):
        bout_counts = []
        # Find indices where the predicted label changes
        bout_start_idx = np.where(np.diff(np.hstack([-1, predict])) != 0)[0]
        bout_start_label = predict[bout_start_idx]
        for b, _ in enumerate(behavior_classes):
            idx_b = np.where(bout_start_label == int(b))[0]
            if len(idx_b) > 0:
                bout_counts.append(len(idx_b))
            else:
                bout_counts.append(np.nan)
        return bout_counts

    ### Part 1: Main Analysis – Horizontal Bar Charts ###
    figs = []
    for selected_group, fig, ax_by_pair, layout in group_figures(
            selected_groups, selected_conditions, panel_w=6.0, panel_h=3.0, max_cols=1, extra_h=0.5,
            sharex=True, sharey=True):
        for selected_condition in selected_conditions:
            a = ax_by_pair[(selected_group, selected_condition)]
            is_bottom = layout['row_of'][(selected_group, selected_condition)] == layout['rows'] - 1

            bout_counts = []
            bout_rates = []
            if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                file_keys = list(behaviors[selected_group][selected_condition].keys())

                fps = fps_lookup(selected_group, selected_condition)
                for file_name in file_keys:
                    predict = behaviors[selected_group][selected_condition][file_name]
                    counts = get_num_bouts(predict, behavior_names)
                    bout_counts.append(counts)
                    # Normalize by each file's own duration so videos of any length are comparable
                    minutes = len(predict) / fps / 60.0
                    bout_rates.append([c / minutes if minutes else np.nan for c in counts])

                # Descriptives across mice (n = files) for each behavior
                bout_arr = np.asarray(bout_counts, dtype=float)
                rate_arr = np.asarray(bout_rates, dtype=float)
                desc_counts = [describe(bout_arr[:, b]) for b in range(len(behavior_names))]
                desc_rates = [describe(rate_arr[:, b]) for b in range(len(behavior_names))]

                behavior_instance_dict = {
                    'labels': behavior_names,
                    'colors': behavior_colors,
                    'n_mice': [d['n'] for d in desc_rates],
                    'mean_bouts_per_min': [d['mean'] for d in desc_rates],
                    'std_bouts_per_min': [d['sd'] for d in desc_rates],
                    'sem_bouts_per_min': [d['sem'] for d in desc_rates],
                    'median_bouts_per_min': [d['median'] for d in desc_rates],
                    'min_bouts_per_min': [d['min'] for d in desc_rates],
                    'max_bouts_per_min': [d['max'] for d in desc_rates],
                    'mean_counts': [d['mean'] for d in desc_counts],
                    'std_counts': [d['sd'] for d in desc_counts],
                    'sem_counts': [d['sem'] for d in desc_counts],
                    'median_counts': [d['median'] for d in desc_counts],
                    'min_counts': [d['min'] for d in desc_counts],
                    'max_counts': [d['max'] for d in desc_counts],
                }
                behavior_instance_df = pd.DataFrame(behavior_instance_dict)

                csv_filename = os.path.join(
                    directory_path,
                    f"behavior_instance_counts_{selected_group}-{selected_condition}.csv"
                )
                behavior_instance_df.to_csv(csv_filename, index=False)

                y_pos = np.arange(len(behavior_names))
                a.barh(y_pos, behavior_instance_df['mean_bouts_per_min'],
                       xerr=np.nan_to_num(behavior_instance_df['std_bouts_per_min']),
                       color=behavior_colors, height=0.6, zorder=3,
                       error_kw={'ecolor': 'black', 'capsize': 2, 'lw': 1})
                a.set_yticks(y_pos)
                a.set_yticklabels(behavior_names)
                a.set_title(f'{selected_condition}  (n = {len(file_keys)} mice)', fontsize=10)
                a.set_ylabel('')
                if selected_condition == selected_conditions[0]:
                    a.invert_yaxis()  # behaviors read top-down in behavior_names order (shared y-axis)
                a.grid(True, axis='x', zorder=0, color='#E6E6E6')
                a.set_axisbelow(True)
                for spine in a.spines.values():
                    spine.set_color('#D3D3D3')
                a.set_xlabel('Bouts / min')
                a.set_ylabel('Behavior')
                a.tick_params(labelbottom=True, labelleft=True)
            else:
                a.text(0.5, 0.5, f"Data not found for\n{selected_group} - {selected_condition}",
                       horizontalalignment='center', verticalalignment='center', transform=a.transAxes)
                a.set_title(f'{selected_condition}', fontsize=10)

        fig.suptitle(f'Group: {selected_group}\nBout rate per behavior (mean ± SD across mice)', fontsize=11)
        save_figure(fig, os.path.join(directory_path, f"behavior_counts_{safe_name(selected_group)}.svg"))
        figs.append(fig)

    ### Part 2: Additional Analysis – Raw Frequency CSVs ###
    raw_directory_path = os.path.join(directory_path, "behavior_instance-counts_raw")
    os.makedirs(raw_directory_path, exist_ok=True)

    all_data = []
    for selected_group in selected_groups:
        for selected_condition in selected_conditions:
            if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                file_keys = list(behaviors[selected_group][selected_condition].keys())
                bout_counts = []
                per_mouse_frames = []
                fps = fps_lookup(selected_group, selected_condition)
                for file_name in file_keys:
                    predict = behaviors[selected_group][selected_condition][file_name]
                    per_file_bout_counts = get_num_bouts(predict, behavior_names)
                    bout_counts.append(per_file_bout_counts)

                    # Per-mouse indicator stats: one CSV per file
                    pm = bout_metrics(predict, fps=fps)
                    save_per_mouse_csv(pm, directory_path, selected_group, selected_condition, file_name, fps=fps)
                    pm['file'] = file_name
                    per_mouse_frames.append(pm)
                    for behavior, count in zip(behavior_names, per_file_bout_counts):
                        all_data.append({
                            'Group': selected_group,
                            'Condition': selected_condition,
                            'File': file_name,
                            'Behavior': behavior,
                            'Count': count
                        })

                raw_bout_counts_df = pd.DataFrame(bout_counts, columns=behavior_names, index=file_keys)
                raw_csv_filename = os.path.join(
                    raw_directory_path,
                    f"behavior_instance_counts_raw_{selected_group}-{selected_condition}.csv"
                )
                raw_bout_counts_df.to_csv(raw_csv_filename)

                # Across-mice summary (n = mice) of the five indicators
                if per_mouse_frames:
                    save_group_summary(pd.concat(per_mouse_frames, ignore_index=True),
                                       INDICATOR_COLS, directory_path,
                                       selected_group, selected_condition)

    all_data_df = pd.DataFrame(all_data)
    all_data_csv = os.path.join(raw_directory_path, "behavior_instance_counts_raw_all.csv")
    all_data_df.to_csv(all_data_csv, index=False)

    return figs
