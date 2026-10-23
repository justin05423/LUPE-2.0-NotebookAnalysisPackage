import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils.classification import load_data
from utils.timing import constant_fps, project_dir
from utils.analysis_scripts.per_mouse_stats import (
    save_figure,
    describe, save_per_mouse_csv, save_group_summary
)

def behavior_distance_traveled_heatmaps(project_name, selected_groups, selected_conditions, fps_lookup=None):
    """
    Generate distance-traveled statistics and heatmaps for each group and condition.

    Per-mouse outputs (per_mouse/<group>/<condition>/<file>.csv): total distance (cm),
    n frames, duration (min), distance per minute, and descriptives of instantaneous speed
    (cm/s: mean, sd, sem, median, min, max). The group CSV reports mean, SD, SEM, median,
    min, max across mice (n = files) of total distance, and the cumulative distance.

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list): List of groups to analyze.
        selected_conditions (list): List of conditions to analyze.

    Returns:
        figs (list): A list of matplotlib Figure objects (one for each group-condition combination).
    """

    # Define the base directory using os.path.join for cross-platform compatibility
    fps_lookup = fps_lookup or constant_fps()
    base_dir = project_dir(project_name)
    poses_file = os.path.join(base_dir, f"raw_data_{project_name}.pkl")

    poses = load_data(poses_file)

    # Define the directory path for saving figures and CSVs
    directory_path = os.path.join(base_dir, "figures", "behavior_distance-traveled")

    # Conversion factor from pixels to units
    pixels_to_units = 0.0330828  # meters = 0.000330708, cm = 0.0330828
    unit = 'cm'
    bodypart_idx = 38  # Index of the body part to track (e.g., center of mass)

    # Fixed max_count for heatmap scaling
    max_count = 5000  # Arbitrary number to reflect pixel intensity differences

    # List to store figures
    figs = []

    # Calculate distances and generate heatmaps
    for group in selected_groups:
        for condition in selected_conditions:
            poses_selected = poses[group][condition]

            distances_traveled = []
            cumulative_distance_traveled = 0.0
            fps = fps_lookup(group, condition)
            per_mouse_rows = []

            # Ensure directory exists before any write
            os.makedirs(directory_path, exist_ok=True)

            for file_key in poses_selected:
                pose_data = np.asarray(poses_selected[file_key], dtype=float)
                xy = pose_data[:, bodypart_idx:bodypart_idx + 2]
                # Euclidean distance between consecutive frames (pixels -> units)
                step = np.linalg.norm(np.diff(xy, axis=0), axis=1) * pixels_to_units
                total_distance = float(np.sum(step))
                # Append to list and update cumulative distance traveled
                distances_traveled.append(total_distance)
                cumulative_distance_traveled += total_distance

                # Per-mouse indicator stats
                n_frames = int(len(pose_data))
                duration_min = n_frames / fps / 60.0
                speed = step * fps  # units per second
                sp = describe(speed)
                pm_df = pd.DataFrame([{
                    'total_distance_' + unit: total_distance,
                    'n_frames': n_frames,
                    'duration_min': duration_min,
                    'distance_per_min_' + unit: total_distance / duration_min if duration_min else np.nan,
                    f'speed_mean_{unit}_per_s': sp['mean'],
                    f'speed_sd_{unit}_per_s': sp['sd'],
                    f'speed_sem_{unit}_per_s': sp['sem'],
                    f'speed_median_{unit}_per_s': sp['median'],
                    f'speed_min_{unit}_per_s': sp['min'],
                    f'speed_max_{unit}_per_s': sp['max'],
                }])
                save_per_mouse_csv(pm_df, directory_path, group, condition, file_key, fps=fps)
                pm_df['file'] = file_key
                per_mouse_rows.append(pm_df)

            distances_traveled = np.array(distances_traveled)

            # Across-mice summary (n = files)
            if per_mouse_rows:
                save_group_summary(pd.concat(per_mouse_rows, ignore_index=True),
                                   ['total_distance_' + unit, 'distance_per_min_' + unit,
                                    f'speed_mean_{unit}_per_s'],
                                   directory_path, group, condition, by=())

            # Calculate statistics across mice
            d = describe(distances_traveled)

            # Save statistics to CSV using pandas (cross-platform compatible)
            stats_data = {
                'Statistic': [
                    'Average distance traveled',
                    'Standard deviation',
                    'Standard error of the mean (SEM)',
                    'Median distance traveled',
                    'Minimum distance traveled',
                    'Maximum distance traveled',
                    'Number of mice',
                    'Cumulative distance traveled'
                ],
                'Value': [
                    f"{d['mean']:.2f} {unit}",
                    f"{d['sd']:.2f} {unit}",
                    f"{d['sem']:.2f} {unit}",
                    f"{d['median']:.2f} {unit}",
                    f"{d['min']:.2f} {unit}",
                    f"{d['max']:.2f} {unit}",
                    f"{d['n']}",
                    f'{cumulative_distance_traveled:.2f} {unit}'
                ]
            }
            df = pd.DataFrame(stats_data)

            output_filename = os.path.join(directory_path, f"behavior_distance_stats-{unit}_{group}_{condition}.csv")
            df.to_csv(output_filename, index=False)

            # Generate heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            heatmap, xedges, yedges = np.histogram2d(
                np.hstack([poses_selected[file_key][:, bodypart_idx] for file_key in poses_selected]),
                np.hstack([poses_selected[file_key][:, bodypart_idx + 1] for file_key in poses_selected]),
                bins=50
            )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', interpolation='nearest',
                           vmax=max_count)
            plt.colorbar(im, ax=ax, label='Counts')
            ax.set_xlabel('X-coordinate')
            ax.set_ylabel('Y-coordinate')
            ax.set_title(f'{group}_{condition}')

            # Save the figure
            save_path = os.path.join(directory_path, f"behavior_distance-heatmap_{group}_{condition}.svg")
            save_figure(fig, save_path)

            # Add the figure to the list
            figs.append(fig)

    return figs
