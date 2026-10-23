import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import platform
from pathlib import Path

from utils.classification import load_behaviors
from utils.timing import constant_fps, project_dir
from utils.analysis_scripts.per_mouse_stats import (
    save_figure,
    describe, save_per_mouse_csv, save_group_summary
)

def _safe_component(s: str) -> str:
    if s is None:
        return "None"
    s = str(s)

    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, "_")

    s = s.rstrip(" .")

    return s if s else "unnamed"

def _win_long_path(p: Path) -> str:
    s = str(p.resolve())

    if platform.system().lower().startswith("win"):
        if s.startswith("\\\\?\\"):
            return s

        if s.startswith("\\\\"):
            return "\\\\?\\UNC\\" + s.lstrip("\\")
        else:
            return "\\\\?\\" + s
    else:
        return s

def behavior_binned_ratio_timeline(project_name, selected_groups, selected_conditions, num_min, fps_lookup=None):
    """
    Generate a binned-ratio timeline plot and save the results as CSV and SVG files.

    Per-mouse outputs (per_mouse/<group>/<condition>/<file>.csv): one row per time bin with
    the fraction of frames in each behavior, followed by summary rows (row_type = mean, sd,
    sem, median, min, max) describing that mouse across bins. An across-mice summary
    (n = files) of each mouse's mean bin ratio is saved per group-condition.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of groups to analyze.
        selected_conditions (list): List of conditions to analyze.
        num_min (int): Number of minutes per time bin.

    Returns:
        figs (list): A list of matplotlib Figure objects (one for each group).
    """
    # Define the base directory (relative to the notebooks/ folder)
    base_dir = Path(project_dir(str(project_name)))
    behaviors_file = base_dir / f"behaviors_{project_name}.pkl"

    # Load behaviors
    if not behaviors_file.exists():
        raise FileNotFoundError(
            f"Behaviors file not found: {behaviors_file}\n"
            "Make sure preprocessing has generated behaviors_<project>.pkl"
        )

    behaviors = load_behaviors(str(behaviors_file))

    # Parameters
    fps_lookup = fps_lookup or constant_fps()

    # Define the directory path for saving figures and CSVs
    directory_path = base_dir / "figures" / "behavior_binned-ratio-timeline"
    directory_path.mkdir(parents=True, exist_ok=True)

    # Check if behaviors is None or empty
    if behaviors is None or not behaviors:
        raise ValueError("Failed to load behaviors or empty dataset.")

    # Behavior names and colors
    behavior_names = ['still', 'walking', 'rearing', 'grooming', 'licking hindpaw L', 'licking hindpaw R']
    behavior_colors = ['crimson', 'darkcyan', 'goldenrod', 'royalblue', 'rebeccapurple', 'mediumorchid']

    # List to store figures
    figs = []

    # Generate plots and save data
    for selected_group in selected_groups:
        fig, axes = plt.subplots(nrows=len(selected_conditions), figsize=(8, 2.5 * len(selected_conditions) + 1))
        fig.suptitle(f'Group: {selected_group}', fontsize=16)

        for idx, selected_condition in enumerate(selected_conditions):
            ax = axes[idx] if len(selected_conditions) > 1 else axes  # Handle single subplot case

            if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                file_keys = list(behaviors[selected_group][selected_condition].keys())
                if len(file_keys) == 0:
                    raise ValueError(f"No files found for group '{selected_group}' condition '{selected_condition}'.")

                # Bin size in frames for this group/condition's frame rate (rounded to whole frames)
                fps = fps_lookup(selected_group, selected_condition)
                time_bin_size = int(round(fps * 60 * float(num_min)))
                n_bins = len(behaviors[selected_group][selected_condition][file_keys[0]]) // time_bin_size
                if int(n_bins) <= 0:
                    raise ValueError(
                        f"Not enough data to bin for {selected_group}/{selected_condition}. "
                        f"time_bin_size={time_bin_size}, series_len={len(behaviors[selected_group][selected_condition][file_keys[0]])}"
                    )

                behavior_ratios_files = {key: np.nan for key in file_keys}

                for file_name in file_keys:
                    binned_behaviors = []
                    for bin_n in range(int(n_bins)):
                        behavior_ratios = {key: 0 for key in range(len(behavior_names))}
                        chunk = behaviors[selected_group][selected_condition][file_name][time_bin_size * bin_n:time_bin_size * (bin_n + 1)]
                        values, counts = np.unique(chunk, return_counts=True)
                        denom = sum(counts) if sum(counts) > 0 else 1
                        for i, value in enumerate(values):
                            # Guard for unexpected values
                            try:
                                v = int(value)
                            except Exception:
                                continue
                            if 0 <= v < len(behavior_names):
                                behavior_ratios[v] = counts[i] / denom
                        binned_behaviors.append(behavior_ratios)
                    behavior_ratios_files[file_name] = binned_behaviors

                # Per-mouse CSVs: bin-by-bin ratios + descriptives across bins
                per_mouse_frames = []
                for file_name in file_keys:
                    bins = behavior_ratios_files[file_name]
                    pm = pd.DataFrame({
                        'row_type': ['bin'] * len(bins),
                        'time_bin': np.arange(len(bins)),
                    })
                    for b in range(len(behavior_names)):
                        pm[behavior_names[b]] = [bins[k][b] for k in range(len(bins))]
                    summary_rows = []
                    for stat in ['mean', 'sd', 'sem', 'median', 'min', 'max']:
                        rec = {'row_type': stat, 'time_bin': np.nan}
                        for b in range(len(behavior_names)):
                            rec[behavior_names[b]] = describe(pm[behavior_names[b]].values)[stat]
                        summary_rows.append(rec)
                    pm = pd.concat([pm, pd.DataFrame(summary_rows)], ignore_index=True)
                    save_per_mouse_csv(pm, str(directory_path), selected_group, selected_condition, file_name, fps=fps)
                    per_mouse_frames.append(pd.DataFrame({
                        'behavior': behavior_names,
                        'mean_bin_ratio': [describe(pm.loc[pm.row_type == 'bin', bn].values)['mean']
                                           for bn in behavior_names],
                        'file': file_name,
                    }))
                if per_mouse_frames:
                    save_group_summary(pd.concat(per_mouse_frames, ignore_index=True), ['mean_bin_ratio'],
                                       str(directory_path), selected_group, selected_condition)

                data_to_save = {'Time_bin': np.arange(int(n_bins))}

                for b in range(len(behavior_names)):
                    y_files = []
                    for file_name in file_keys:
                        y_files.append(np.hstack([behavior_ratios_files[file_name][bin][b] for bin in range(len(behavior_ratios_files[file_name]))]))
                    y = np.mean(y_files, axis=0)
                    x = np.arange(int(n_bins))
                    y_sem = np.std(y_files, axis=0) / np.sqrt(len(behavior_ratios_files))

                    ax.plot(x, y, color=behavior_colors[b], label=behavior_names[b])
                    ax.fill_between(x, y - y_sem, y + y_sem, color=behavior_colors[b], alpha=0.2)

                    data_to_save[behavior_names[b]] = y
                    data_to_save[f'{behavior_names[b]}_SEM'] = y_sem

                ax.set_title(f'{selected_condition}')
                ax.set_xlabel(f'Time bin = {num_min} min')
                ax.set_ylabel('Percent')
                ax.spines[['top', 'right']].set_visible(False)

                df = pd.DataFrame(data_to_save)

                safe_project = _safe_component(project_name)
                safe_group = _safe_component(selected_group)
                safe_cond = _safe_component(selected_condition)

                output_filename = directory_path / (
                    f"behavior_binned-ratio-timeline__{safe_project}_{safe_group}-{safe_cond}.csv"
                )

                output_path_for_write = _win_long_path(output_filename)

                with open(output_path_for_write, "w", newline="", encoding="utf-8") as f:
                    df.to_csv(f, index=False)

                print(f"Data saved to {output_filename} (len={len(str(output_filename))}).")

            else:
                raise ValueError(
                    f"Selected group '{selected_group}' or condition '{selected_condition}' not found in the dataset."
                )

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title

        handles, labels = [], []
        for color, name in zip(behavior_colors, behavior_names):
            handles.append(plt.Line2D([0], [0], color=color, lw=4))
            labels.append(name)

        fig.legend(handles, labels, loc='lower center', ncol=len(behavior_names), bbox_to_anchor=(0.5, -0.05))

        safe_project = _safe_component(project_name)
        safe_group = _safe_component(selected_group)

        save_path_svg = directory_path / f"behavior_binned-ratio-timeline_{safe_project}_{safe_group}.svg"
        save_path_svg_for_write = _win_long_path(save_path_svg)

        save_figure(fig, save_path_svg_for_write)
        print(f"SVG + PNG saved to {save_path_svg.with_suffix('')}.[svg|png]")

        figs.append(fig)

    return figs
