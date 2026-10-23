import os
from utils.timing import project_dir
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from utils.classification import load_behaviors, load_data
from utils.meta import behavior_names, behavior_colors  # Make sure these are defined in meta.py
from utils.analysis_scripts.per_mouse_stats import (
    group_figures,
    save_figure,
    describe, save_per_mouse_csv, save_group_summary, per_mouse_dir, safe_name
)


def _per_animal_location_figure(predict, xy, file_name, group, condition, out_svg, center, radius):
    """One figure per mouse: six panels (one per behavior) of tail-base density."""
    rgb_val = (0, 254 / 255, 1)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7), facecolor='#000000')
    fig.suptitle(f"{group} - {condition}\n{file_name}", color='white', fontsize=10)
    for b, behav_name in enumerate(behavior_names):
        ax = axes.flat[b]
        ax.set_facecolor('#000000')
        idx_b = np.where(predict == b)[0]
        cm = LinearSegmentedColormap.from_list("Custom", ['#000000', behavior_colors[b]], N=20)
        if idx_b.size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                heatmap, xedges, yedges = np.histogram2d(
                    xy[idx_b, 0], xy[idx_b, 1],
                    bins=[np.arange(0, 768, 20), np.arange(0, 770, 20)], density=True)
            heatmap[heatmap == 0] = np.nan
            ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      origin='lower', cmap=cm)
        ax.add_patch(Circle(center, radius, color=rgb_val, linewidth=2, fill=False))
        ax.set_aspect('equal')
        ax.set_xlim(center[0] - radius - 10, center[0] + radius + 10)
        ax.set_ylim(center[1] + radius + 10, center[1] - radius - 10)
        ax.axis('off')
        ax.set_title(f"{behav_name} (n={idx_b.size} frames)", color=behavior_colors[b], fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.subplots_adjust(hspace=0.35)
    save_figure(fig, out_svg, facecolor='#000000')
    plt.close(fig)


def behavior_location(project_name, selected_groups, selected_conditions, per_animal_images=False):
    """
    Generate figures showing the arena location of a specific behavior performed,
    with one figure per behavior. Each figure contains subplots for each combination
    of selected groups and conditions.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of group names.
        selected_conditions (list): List of condition names.

    Per-mouse outputs (per_mouse/<group>/<condition>/<file>.csv): one row per behavior with
    n_frames, mean tail-base x/y (px), and descriptives of the tail-base distance from the
    arena center (cm: mean, sd, sem, median, min, max) while that behavior was performed.
    An across-mice summary (n = files) of mean distance from center is saved per group-condition.
    If per_animal_images is True, one SVG+PNG per mouse (six panels, one per behavior) is also
    written to per_mouse/<group>/<condition>/<file>_location.svg.

    Returns:
        figs (list): A list of matplotlib Figure objects (one per behavior).
    """
    # Update file paths to use the app's base directory with cross-platform path handling
    base_dir = project_dir(project_name)
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")
    poses_file = os.path.join(base_dir, f"raw_data_{project_name}.pkl")

    behaviors = load_behaviors(behaviors_file)
    poses = load_data(poses_file)

    directory_path = os.path.join(base_dir, "figures", "behavior_location")
    os.makedirs(directory_path, exist_ok=True)

    bodypart_idx = 38  # tail-base as position indicator
    center = (768 / 2, 770 / 2)
    radius = 768 / 2 + 20
    h = '00FEFF'  # cyan-like color for the circle

    rows = len(selected_groups)
    cols = len(selected_conditions)

    figs = []  # list to hold the figures per behavior

    # Per-mouse location stats (one CSV per file) + across-mice summary
    pixels_to_cm = 0.0330828
    for selected_group in selected_groups:
        for selected_condition in selected_conditions:
            if not (selected_group in behaviors and selected_condition in behaviors[selected_group]):
                continue
            per_mouse_frames = []
            for file_name in behaviors[selected_group][selected_condition].keys():
                predict = np.asarray(behaviors[selected_group][selected_condition][file_name]).astype(int)
                pose = np.asarray(poses[selected_group][selected_condition][file_name], dtype=float)
                xy = pose[:, bodypart_idx:bodypart_idx + 2]
                dist_center = np.linalg.norm(xy - np.array(center), axis=1) * pixels_to_cm
                pm_rows = []
                for b, behav_name in enumerate(behavior_names):
                    idx_b = np.where(predict == b)[0]
                    d = describe(dist_center[idx_b])
                    pm_rows.append({
                        'behavior': behav_name,
                        'n_frames': int(idx_b.size),
                        'mean_x_px': float(np.mean(xy[idx_b, 0])) if idx_b.size else np.nan,
                        'mean_y_px': float(np.mean(xy[idx_b, 1])) if idx_b.size else np.nan,
                        'dist_from_center_mean_cm': d['mean'],
                        'dist_from_center_sd_cm': d['sd'],
                        'dist_from_center_sem_cm': d['sem'],
                        'dist_from_center_median_cm': d['median'],
                        'dist_from_center_min_cm': d['min'],
                        'dist_from_center_max_cm': d['max'],
                    })
                pm_df = pd.DataFrame(pm_rows)
                save_per_mouse_csv(pm_df, directory_path, selected_group, selected_condition, file_name)
                pm_df['file'] = file_name
                per_mouse_frames.append(pm_df)

                if per_animal_images:
                    _per_animal_location_figure(
                        predict, xy, file_name, selected_group, selected_condition,
                        os.path.join(per_mouse_dir(directory_path, selected_group, selected_condition),
                                     f"{safe_name(file_name)}_location.svg"),
                        center, radius)
            if per_mouse_frames:
                save_group_summary(pd.concat(per_mouse_frames, ignore_index=True),
                                   ['dist_from_center_mean_cm', 'dist_from_center_median_cm', 'n_frames'],
                                   directory_path, selected_group, selected_condition)

    for b, behav_name in enumerate(behavior_names):
        count = 0

        for selected_group, fig, ax_by_pair, layout in group_figures(
                selected_groups, selected_conditions, panel_w=3.2, panel_h=3.2, max_cols=4, extra_h=0.8):
            fig.patch.set_facecolor('#000000')
            fig.suptitle(f'{behav_name}\nGroup: {selected_group}', color=behavior_colors[b], fontsize=14, fontweight='bold')

            for selected_condition in selected_conditions:
                ax = ax_by_pair[(selected_group, selected_condition)]
                ax.set_facecolor('#000000')

                # Convert the hex string to an RGB tuple
                rgb_val = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
                # Create a circle patch to indicate arena border
                circle = Circle(center, radius, color=rgb_val, linewidth=3, fill=False)
                hist2d_all = []
                # Create a colormap from black to the behavior color
                colors = ['#000000', behavior_colors[b]]
                cm = LinearSegmentedColormap.from_list("Custom", colors, N=20)
                heatmaps = np.empty((38, 38))

                if selected_group in behaviors and selected_condition in behaviors[selected_group]:
                    file_keys = list(behaviors[selected_group][selected_condition].keys())

                    # Loop over each file to compute a heatmap of positions when the behavior occurred
                    for file_name in file_keys:
                        idx_b = np.where(behaviors[selected_group][selected_condition][file_name] == b)[0]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            heatmap, xedges, yedges = np.histogram2d(
                                poses[selected_group][selected_condition][file_name][idx_b, bodypart_idx],
                                poses[selected_group][selected_condition][file_name][idx_b, bodypart_idx + 1],
                                bins=[np.arange(0, 768, 20), np.arange(0, 770, 20)],
                                density=True)
                        heatmap[heatmap == 0] = np.nan
                        hist2d_all.append(heatmap)

                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        # Plot the mean heatmap (transposed for correct orientation)
                        ax.imshow(np.nanmean(hist2d_all, axis=0).T,
                                  extent=extent, origin='lower', cmap=cm)

                ax.add_patch(circle)
                ax.set_aspect('equal')
                ax.set_xlim(center[0] - radius - 10, center[0] + radius + 10)
                ax.set_ylim(center[1] + radius + 10, center[1] - radius - 10)  # inverted y for image orientation
                ax.axis('off')
                ax.set_title(f'{selected_condition}', color='white', fontsize=10)

            save_path_svg = os.path.join(directory_path, f"behavior_location_{behav_name}_{safe_name(selected_group)}.svg")
            save_figure(fig, save_path_svg, facecolor='#000000')
            figs.append(fig)

    return figs
