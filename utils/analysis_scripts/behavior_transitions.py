import os
from utils.timing import project_dir
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.classification import load_behaviors
from utils.meta import behavior_names, behavior_colors
from utils.analysis_scripts.per_mouse_stats import (
    group_figures, safe_name,
    save_figure,
    save_per_mouse_csv, save_group_summary, per_mouse_dir, safe_name
)

if not os.path.join(os.path.abspath(''), '..') in sys.path:
    sys.path.append(os.path.join(os.path.abspath(''), '..'))


def behavior_transitions(project_name, selected_groups, selected_conditions, per_animal_images=False):
    """
    Generate CSV files and figures for behavior transitions (Part 1 only).

    For each group–condition combination, the function:
      - Computes the transition matrix (counts and normalized probabilities)
        using a helper function.
      - Saves a CSV file for the normalized transition matrix.
      - Saves one CSV per mouse/file (per_mouse/<group>/<condition>/<file>.csv) holding that
        mouse's transition count matrix and row-normalized probability matrix
        (self-transitions zeroed), plus total_transitions per current behavior.
      - Saves an across-mice summary (mean, SD, SEM, median, min, max, n) of every
        current->next transition probability, with n = mice.
      - Generates two heatmap figures (one with annotations and one without).
      - If per_animal_images is True, one annotated transition heatmap per mouse (SVG + PNG)
        in per_mouse/<group>/<condition>/<file>_transitions.svg.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of group names.
        selected_conditions (list): List of condition names.

    Returns:
        figs (list): annotated heatmaps (one figure per group) followed by unannotated ones.
    """
    # Set base directory for the app with cross-platform path handling
    base_dir = project_dir(project_name)
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")
    behaviors = load_behaviors(behaviors_file)

    # Directory for saving heatmap CSVs and figures
    heat_dir = os.path.join(base_dir, "figures", "behavior_transitions")
    os.makedirs(heat_dir, exist_ok=True)

    # Helper: Compute Transition Matrices
    def get_transitions(predict, behavior_classes):
        # Create a transition matrix (counts)
        tm = [[0] * len(behavior_classes) for _ in behavior_classes]
        for (i, j) in zip(predict, predict[1:]):
            tm[int(i)][int(j)] += 1
        tm_array = np.array(tm)
        # Normalize each row to get probabilities
        tm_norm = tm_array / tm_array.sum(axis=1, keepdims=True)
        return tm_array, tm_norm

    # Part 1: Heatmaps with Transition Motifs
    def plot_heatmaps(annot, fmt, save_stem):
        figs_out = []
        for group, fig, ax_by_pair, layout in group_figures(
                selected_groups, selected_conditions, panel_w=5.2, panel_h=4.2, max_cols=4, extra_h=0.5):
            for condition in selected_conditions:
                axp = ax_by_pair[(group, condition)]
                r = layout['row_of'][(group, condition)]
                c = layout['col_of'][(group, condition)]
                rows_num = layout['rows']
                all_count_tm = np.zeros((len(behavior_names), len(behavior_names)))
                if group in behaviors and condition in behaviors[group]:
                    file_keys = list(behaviors[group][condition].keys())
                    per_mouse_rows = []
                    for file_name in file_keys:
                        count_tm, _ = get_transitions(behaviors[group][condition][file_name], behavior_names)
                        np.fill_diagonal(count_tm, 0)
                        all_count_tm += count_tm

                        # Per-mouse matrices are only written once (first heatmap pass)
                        if annot:
                            with np.errstate(divide='ignore', invalid='ignore'):
                                prob_tm = np.nan_to_num(count_tm / count_tm.sum(axis=1, keepdims=True))
                            pm_rows = []
                            for i, cur in enumerate(behavior_names):
                                for j, nxt in enumerate(behavior_names):
                                    pm_rows.append({
                                        'current_behavior': cur,
                                        'next_behavior': nxt,
                                        'transition_count': int(count_tm[i, j]),
                                        'transition_probability': float(prob_tm[i, j]),
                                        'total_transitions_from_current': int(count_tm[i].sum()),
                                    })
                            pm_df = pd.DataFrame(pm_rows)
                            save_per_mouse_csv(pm_df, heat_dir, group, condition, file_name)

                            if per_animal_images:
                                pm_fig, pm_ax = plt.subplots(figsize=(6, 4.5))
                                sns.heatmap(pd.DataFrame(prob_tm, index=behavior_names, columns=behavior_names),
                                            annot=True, fmt='.2f', cmap='Blues', cbar=True, vmin=0, vmax=1, ax=pm_ax)
                                pm_ax.set_ylabel('Current behavior')
                                pm_ax.set_xlabel('Next behavior')
                                pm_ax.set_yticklabels(behavior_names, rotation=0)
                                pm_ax.set_title(f'{group} - {condition}\n{file_name}', fontsize=9)
                                pm_fig.tight_layout()
                                save_figure(pm_fig, os.path.join(per_mouse_dir(heat_dir, group, condition),
                                                                 f"{safe_name(file_name)}_transitions.svg"))
                                plt.close(pm_fig)
                            pm_df['file'] = file_name
                            per_mouse_rows.append(pm_df)
                    if annot and per_mouse_rows:
                        save_group_summary(pd.concat(per_mouse_rows, ignore_index=True),
                                           ['transition_probability', 'transition_count'],
                                           heat_dir, group, condition,
                                           by=('current_behavior', 'next_behavior'))
                    all_prob_tm = all_count_tm / all_count_tm.sum(axis=1, keepdims=True)
                    all_prob_tm = np.nan_to_num(all_prob_tm)
                    transmat_df = pd.DataFrame(all_prob_tm, index=behavior_names, columns=behavior_names)
                    transmat_df = transmat_df.fillna(0)
                    csv_filename = os.path.join(heat_dir, f"behavior_transitions_{group}_{condition}.csv")
                    transmat_df.to_csv(csv_filename)

                    sns.heatmap(
                        transmat_df,
                        annot=annot,
                        fmt=fmt,
                        cmap='Blues',
                        cbar=True,
                        vmin=0,
                        vmax=1,
                        ax=axp,
                        xticklabels=transmat_df.columns.tolist(),
                        yticklabels=transmat_df.index.tolist()
                    )

                    axp.tick_params(axis='y', labelrotation=0, labelleft=True)
                    axp.set_yticklabels(transmat_df.index.tolist(), rotation=0, ha='right', va='center', rotation_mode='anchor')
                    axp.set_ylabel('Current behavior')
                    axp.set_xlabel('Next behavior')
                    axp.set_title(f'{condition}  (n = {len(file_keys)} mice)', fontsize=10)
                else:
                    axp.text(0.5, 0.5, f"Data not found for\n{group} - {condition}",
                             horizontalalignment='center', verticalalignment='center', transform=axp.transAxes)
                    axp.set_title(f'{condition}', fontsize=10)
            fig.suptitle(f'Group: {group}\nBehavior transition probabilities (self-transitions zeroed)', fontsize=11)
            save_figure(fig, f"{save_stem}_{safe_name(group)}.svg")
            figs_out.append(fig)
        return figs_out

    figs_annot = plot_heatmaps(True, ".2f", os.path.join(heat_dir, "behavior_transitions_annot_true"))
    figs_noannot = plot_heatmaps(False, ".2f", os.path.join(heat_dir, "behavior_transitions_annot_false"))

    return figs_annot + figs_noannot