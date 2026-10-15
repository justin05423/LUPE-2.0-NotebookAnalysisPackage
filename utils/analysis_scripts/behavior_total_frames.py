import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.classification import load_behaviors
from utils.meta import behavior_names, behavior_colors
from utils.timing import constant_fps, project_dir
from utils.analysis_scripts.per_mouse_stats import (
    group_figures, safe_name,
    save_figure,
    bout_metrics, save_per_mouse_csv, save_group_summary, INDICATOR_COLS
)

def behavior_total_frames(project_name, selected_groups, selected_conditions, fps_lookup=None):
    """
    Create a pie chart showing the total number of frames per behavior for each
    selected group and condition. For each group-condition combination, the function:
      - Aggregates the behavior predictions from all files.
      - Saves a CSV file summarizing the counts.
      - Saves one CSV per mouse/file (per_mouse/<group>/<condition>/<file>.csv) with
        percent_time, total_frames, bout_count, mean_bout_duration_s and bouts_per_min
        per behavior, plus an across-mice summary (mean, SD, SEM, median, min, max, n).
      - Creates a pie chart subplot.
    Finally, the overall figure is saved as an SVG and returned.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of group names.
        selected_conditions (list): List of condition names.

    Returns:
        figs (list): one donut figure per group.
    """

    base_dir = project_dir(project_name)
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")
    fps_lookup = fps_lookup or constant_fps()
    behaviors = load_behaviors(behaviors_file)

    directory_path = os.path.join(base_dir, "figures", "behavior_total-frames")
    os.makedirs(directory_path, exist_ok=True)

    # One donut per group-condition, wrapped into rows of at most 3 panels
    min_pct_label = 3.0  # wedges smaller than this get no on-plot label (values are in the CSV)
    figs = []
    for selected_group, fig, ax_by_pair, layout in group_figures(
              selected_groups, selected_conditions, panel_w=3.8, panel_h=3.8, max_cols=4, extra_h=1.2):
        for (selected_group, selected_condition), a in ax_by_pair.items():
            a.set_aspect('equal')
            if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                file_keys = list(behaviors[selected_group][selected_condition].keys())

                # Count frames per behavior across all files
                total_frames = np.zeros(len(behavior_names), dtype=int)
                for file_name in file_keys:
                    predict = np.asarray(behaviors[selected_group][selected_condition][file_name]).astype(int)
                    total_frames += np.bincount(predict, minlength=len(behavior_names))[:len(behavior_names)]

                df = pd.DataFrame({
                    'behavior': behavior_names,
                    'total_frames': total_frames,
                    'percent': 100.0 * total_frames / total_frames.sum() if total_frames.sum() else 0.0,
                    'colors': behavior_colors,
                })
                csv_filename = os.path.join(
                    directory_path,
                    f"behavior_total_frames_{selected_group}-{selected_condition}.csv"
                )
                df.drop(columns='colors').to_csv(csv_filename, index=False)

                # Per-mouse indicator stats (one CSV per file) + across-mice summary
                per_mouse_frames = []
                fps = fps_lookup(selected_group, selected_condition)
                for file_name in file_keys:
                    pm = bout_metrics(behaviors[selected_group][selected_condition][file_name], fps=fps)
                    save_per_mouse_csv(pm, directory_path, selected_group, selected_condition, file_name, fps=fps)
                    pm['file'] = file_name
                    per_mouse_frames.append(pm)
                if per_mouse_frames:
                    save_group_summary(pd.concat(per_mouse_frames, ignore_index=True),
                                       INDICATOR_COLS, directory_path,
                                       selected_group, selected_condition)

                a.pie(df['total_frames'], colors=df['colors'], startangle=90, counterclock=False,
                      autopct=lambda p: f'{p:.1f}%' if p >= min_pct_label else '',
                      pctdistance=0.78, textprops={'fontsize': 8},
                      wedgeprops={'width': 0.45, 'edgecolor': 'white', 'linewidth': 1})
                a.text(0, 0, f'n = {len(file_keys)}\nmice', ha='center', va='center', fontsize=9, color='#444444')
                a.set_title(f'{selected_condition}', fontsize=10)
            else:
                a.text(0.5, 0.5, f"Data not found for\n{selected_group} - {selected_condition}",
                       ha='center', va='center', transform=a.transAxes)
                a.set_title(f'{selected_condition}', fontsize=10)
                a.axis('off')

        # One shared legend per figure
        handles = [plt.matplotlib.patches.Patch(color=col, label=name) for name, col in zip(behavior_names, behavior_colors)]
        fig.legend(handles=handles, loc='lower center', ncol=min(6, len(behavior_names)), frameon=False, fontsize=9)
        fig.suptitle(f'Group: {selected_group}\nPercent of total frames per behavior (all files pooled per condition)', fontsize=11)
        save_figure(fig, os.path.join(directory_path, f"behavior_total-frames_{safe_name(selected_group)}.svg"))
        figs.append(fig)
        plt.close(fig)

    return figs
