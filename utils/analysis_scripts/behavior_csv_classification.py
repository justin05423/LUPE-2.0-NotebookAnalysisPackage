import os
import pandas as pd
from utils.classification import load_behaviors
from utils.timing import constant_fps, project_dir
from utils.analysis_scripts.per_mouse_stats import bout_metrics, save_per_mouse_csv

def behavior_csv_classification(project_name, selected_groups=None, selected_conditions=None, fps_lookup=None):
    """
    Generate CSV files for behavior classifications (per frame and per second).

    Also writes one summary CSV per mouse/file (per_mouse/<group>/<condition>/<file>.csv)
    with the five indicators per behavior: percent_time, total_frames, bout_count,
    mean_bout_duration_s, bouts_per_min (plus bout-duration descriptives).

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list, optional): List of groups to process. Process all if None.
        selected_conditions (list, optional): List of conditions to process. Process all if None.
    """

    # Define the base directory using os.path.join for cross-platform compatibility
    base_dir = project_dir(project_name)
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")

    # Load behaviors
    behaviors = load_behaviors(behaviors_file)
    behavior_labels = ['still', 'walking', 'rearing', 'grooming', 'licking hindpaw L', 'licking hindpaw R']

    # Create CSVs for Per-Frame Behavior Classification
    base_output_dir_frames = os.path.join(base_dir, "figures", "behaviors_csv_raw-classification", "frames")
    for group, conditions in behaviors.items():
        if selected_groups is not None and group not in selected_groups:
            continue
        for condition, files in conditions.items():
            if selected_conditions is not None and condition not in selected_conditions:
                continue
            output_dir_frames = os.path.join(base_output_dir_frames, group, condition)
            os.makedirs(output_dir_frames, exist_ok=True)
            for file_key, data in files.items():
                df = pd.DataFrame({
                    'frame': range(1, len(data) + 1),
                    'behavior': data,
                    'behavior_label': [behavior_labels[i] for i in data]
                })
                csv_filename = f'{file_key}.csv'
                df.to_csv(os.path.join(output_dir_frames, csv_filename), index=False)
                print(f'Saved {csv_filename} in {output_dir_frames}')

    # Create CSVs for Per-Second Behavior Classification
    base_output_dir_seconds = os.path.join(base_dir, "figures", "behaviors_csv_raw-classification", "seconds")
    fps_lookup = fps_lookup or constant_fps()
    for group, conditions in behaviors.items():
        if selected_groups is not None and group not in selected_groups:
            continue
        for condition, files in conditions.items():
            if selected_conditions is not None and condition not in selected_conditions:
                continue
            # Create a subdirectory for the group and condition
            output_dir_seconds = os.path.join(base_output_dir_seconds, group, condition)
            os.makedirs(output_dir_seconds, exist_ok=True)
            frame_rate = fps_lookup(group, condition)
            for file_key, data in files.items():
                df = pd.DataFrame({
                    'time_seconds': [i / frame_rate for i in range(len(data))],
                    'behavior': data,
                    'behavior_label': [behavior_labels[i] for i in data]
                })
                csv_filename = f'{file_key}.csv'
                df.to_csv(os.path.join(output_dir_seconds, csv_filename), index=False)
                print(f'Saved {csv_filename} in {output_dir_seconds}')

    # Per-mouse indicator summary (one CSV per file)
    summary_dir = os.path.join(base_dir, "figures", "behaviors_csv_raw-classification")
    for group, conditions in behaviors.items():
        if selected_groups is not None and group not in selected_groups:
            continue
        for condition, files in conditions.items():
            if selected_conditions is not None and condition not in selected_conditions:
                continue
            frame_rate = fps_lookup(group, condition)
            for file_key, data in files.items():
                save_per_mouse_csv(bout_metrics(data, fps=frame_rate), summary_dir, group, condition, file_key, fps=frame_rate)

    print('All files saved.')
