import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.classification import load_behaviors
from utils.meta import behavior_names, behavior_colors  # Assumes these are defined in meta.py
from utils.timing import constant_fps, project_dir
from utils.analysis_scripts.per_mouse_stats import (
    group_figures, safe_name,
    save_figure, describe, save_per_mouse_csv, save_group_summary
)

SPINE_COLOR = '#D3D3D3'


def get_duration_bouts(predict, behavior_classes, framerate=60):
    """Per-behavior arrays of bout durations (s) for one prediction vector."""
    predict = np.asarray(predict).astype(int)
    bout_start_idx = np.where(np.diff(np.hstack([-1, predict])) != 0)[0]
    bout_durations = np.hstack([np.diff(bout_start_idx), len(predict) - np.max(bout_start_idx)])
    bout_start_label = predict[bout_start_idx]
    behav_durations = []
    for b, _ in enumerate(behavior_classes):
        idx_b = np.where(bout_start_label == int(b))[0]
        behav_durations.append(bout_durations[idx_b] / framerate if len(idx_b) else np.array([]))
    return behav_durations


def _upper_whisker(x):
    """Upper whisker end as drawn by a Tukey boxplot (Q3 + 1.5 IQR, capped at data max)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    q1, q3 = np.percentile(x, [25, 75])
    hi = q3 + 1.5 * (q3 - q1)
    inside = x[x <= hi]
    return float(inside.max()) if inside.size else float(x.max())


def _style_axis(ax):
    ax.grid(True, axis='x', zorder=0, color='#E6E6E6')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)




def _draw_box_panels(ax_by_pair, layout, selected_groups, selected_conditions, durations, directory_path,
                     y_pos, log_scale, x_min, x_max, write_csv=True):
    """Fill a panel grid with pooled-bout boxplots plus per-mouse median dots."""
    rows = layout['rows']
    for selected_group in selected_groups:
        for selected_condition in selected_conditions:
            key = (selected_group, selected_condition)
            a = ax_by_pair[key]
            row = layout['row_of'][key]
            col = layout['col_of'][key]
            if key not in durations:
                a.text(0.5, 0.5, f"Data not found for\n{selected_group} - {selected_condition}",
                       ha='center', va='center', transform=a.transAxes)
                a.set_title(f'{selected_condition}')
                _style_axis(a)
                continue

            files = durations[key]
            file_keys = list(files.keys())

            # All bouts, long format (also the per-group-condition CSV)
            recs = []
            for fn in file_keys:
                for b, bname in enumerate(behavior_names):
                    for v in files[fn][b]:
                        recs.append((bname, float(v), fn))
            durations_df = pd.DataFrame(recs, columns=['behavior', 'duration', 'file'])
            if write_csv:
                durations_df.to_csv(os.path.join(
                    directory_path, f"behavior_durations_{selected_group}_{selected_condition}.csv"), index=False)

            # Matplotlib boxplot (version-independent): one box per behavior, colored by behavior
            pooled = [np.hstack([files[fn][b] for fn in file_keys]) if file_keys else np.array([])
                      for b in range(len(behavior_names))]
            pooled = [x if x.size else np.array([np.nan]) for x in pooled]
            box_kw = dict(positions=y_pos, widths=0.6, showfliers=False, patch_artist=True, zorder=3,
                          manage_ticks=False, medianprops={'color': 'black', 'lw': 1.2},
                          whiskerprops={'color': '#333333'}, capprops={'color': '#333333'})
            try:
                bp = a.boxplot(pooled, orientation='horizontal', **box_kw)   # matplotlib >= 3.10
            except TypeError:
                bp = a.boxplot(pooled, vert=False, **box_kw)                 # older matplotlib
            for patch, color in zip(bp['boxes'], behavior_colors):
                patch.set_facecolor(color)
                patch.set_edgecolor('#333333')
            a.set_yticks(y_pos)
            a.set_yticklabels(behavior_names)
            if key == (selected_groups[0], selected_conditions[0]):
                a.invert_yaxis()

            # One dot per mouse: that mouse's median bout duration, slight vertical jitter
            rng = np.random.default_rng(0)
            for b in range(len(behavior_names)):
                meds = [np.median(files[fn][b]) for fn in file_keys if files[fn][b].size]
                if meds:
                    jitter = rng.uniform(-0.15, 0.15, size=len(meds))
                    a.scatter(meds, y_pos[b] + jitter, s=18, color='black', edgecolor='white',
                              linewidth=0.5, zorder=4)

            _style_axis(a)
            a.set_ylabel('Behavior')
            a.set_xlabel('Behavior duration (s)')
            a.tick_params(labelbottom=True, labelleft=True)
            a.set_title(f'{selected_condition}  (n = {len(file_keys)} mice)')
            if log_scale:
                a.set_xscale('log')
            a.set_xlim(x_min, x_max)



def behavior_bout_durations(project_name, selected_groups, selected_conditions, fps_lookup=None):
    """
    Bout-duration analysis for each selected group and condition.

    Figures (SVG + PNG, in figures/behavior_instance-durations/):
      - behavior_durations: horizontal boxplots of all bouts pooled per group-condition, with one
        black dot per mouse (that mouse's median bout duration) overlaid. The x-axis is shared and
        scaled to the largest upper whisker in the selection so no visible data is clipped.
      - log_scale/behavior_durations_log: the same figure with a log x-axis, for right-skewed durations.
      - behavior_durations_mean-per-mouse: horizontal bars of mean bout duration (s), mean ± SEM
        across mice, matching the bouts/min chart from Behavior Bout Counts.

    CSVs:
      - behavior_durations_<group>_<condition>.csv: every bout (behavior, duration, file).
      - per_mouse/<group>/<condition>/<file>.csv: per behavior, that mouse's bout-duration
        descriptives (n_bouts, mean, sd, sem, median, min, max, total_duration_s).
      - per_mouse/summary_across_mice_<group>-<condition>.csv: descriptives across mice (n = files).
      - total_average_std_durations_per_file.csv: wide, one row per file (kept for compatibility).

    Returns:
        list of figures: linear boxplots, log boxplots, then mean-per-mouse bar charts (one per group each).
    """
    fps_lookup = fps_lookup or constant_fps()
    base_dir = project_dir(project_name)
    behaviors = load_behaviors(os.path.join(base_dir, f"behaviors_{project_name}.pkl"))

    directory_path = os.path.join(base_dir, "figures", "behavior_instance-durations")
    os.makedirs(directory_path, exist_ok=True)

    # ---- Pass 1: durations per file, for every selected group-condition
    # durations[(group, condition)] = {file_name: [array per behavior]}
    durations = {}
    for g in selected_groups:
        for c in selected_conditions:
            if g in behaviors and c in behaviors[g]:
                durations[(g, c)] = {fn: get_duration_bouts(behaviors[g][c][fn], behavior_names, fps_lookup(g, c))
                                     for fn in behaviors[g][c]}

    # Shared x-limit: largest upper whisker across all panels and behaviors
    whiskers = [_upper_whisker(d[b]) for files in durations.values() for d in files.values()
                for b in range(len(behavior_names))]
    # Whiskers are drawn on the pooled distribution, so compute on pooled data per panel/behavior
    pooled_whiskers = []
    for files in durations.values():
        for b in range(len(behavior_names)):
            pooled = np.hstack([d[b] for d in files.values()]) if files else np.array([])
            pooled_whiskers.append(_upper_whisker(pooled))
    x_max = max(pooled_whiskers) * 1.08 if pooled_whiskers and max(pooled_whiskers) > 0 else 6.0
    positives = [d[b][d[b] > 0].min() for files in durations.values() for d in files.values()
                 for b in range(len(behavior_names)) if (d[b] > 0).any()]
    x_min_log = min(positives) * 0.8 if positives else 1.0 / 60.0

    y_pos = np.arange(len(behavior_names))
    log_dir = os.path.join(directory_path, "log_scale")
    os.makedirs(log_dir, exist_ok=True)

    # ---- Figure 1 (linear) and Figure 1b (log): boxplots with per-mouse median dots
    box_figs = {False: [], True: []}
    for log_scale in (False, True):
        x_min = x_min_log if log_scale else 0.0
        for g, fig_box, ax_by_pair, layout in group_figures(
                selected_groups, selected_conditions, panel_w=6.0, panel_h=3.0, max_cols=1, extra_h=0.6,
                sharex=True, sharey=True):
            _draw_box_panels(ax_by_pair, layout, [g], selected_conditions, durations, directory_path,
                             y_pos, log_scale, x_min, x_max, write_csv=not log_scale)
            scale_note = ' (log scale)' if log_scale else ''
            fig_box.suptitle(f'Group: {g}\nBout durations{scale_note}: boxes = all bouts pooled, '
                             f'dots = per-mouse median', fontsize=11)
            out = os.path.join(log_dir, f"behavior_durations_log_{safe_name(g)}.svg") if log_scale \
                else os.path.join(directory_path, f"behavior_durations_{safe_name(g)}.svg")
            save_figure(fig_box, out)
            box_figs[log_scale].append(fig_box)

    # ---- Per-mouse CSVs, across-mice summaries, and the legacy wide CSV
    all_file_durations = []
    per_mouse_by_key = {}
    for (selected_group, selected_condition), files in durations.items():
        per_mouse_frames = []
        for file_name, file_durations in files.items():
            total_durations = [float(np.sum(d)) for d in file_durations]
            record = {'group': selected_group, 'condition': selected_condition, 'file_name': file_name}
            pm_rows = []
            for i, bname in enumerate(behavior_names):
                desc = describe(file_durations[i])
                record[f'{bname}_total_duration_s'] = total_durations[i]
                record[f'{bname}_average_duration_s'] = desc['mean'] if desc['n'] else 0.0
                record[f'{bname}_std_duration_s'] = desc['sd'] if desc['n'] else 0.0
                pm_rows.append({
                    'behavior': bname,
                    'n_bouts': desc['n'],
                    'mean_duration_s': desc['mean'],
                    'sd_duration_s': desc['sd'],
                    'sem_duration_s': desc['sem'],
                    'median_duration_s': desc['median'],
                    'min_duration_s': desc['min'],
                    'max_duration_s': desc['max'],
                    'total_duration_s': total_durations[i],
                })
            all_file_durations.append(record)
            pm_df = pd.DataFrame(pm_rows)
            save_per_mouse_csv(pm_df, directory_path, selected_group, selected_condition, file_name,
                               fps=fps_lookup(selected_group, selected_condition))
            pm_df['file'] = file_name
            per_mouse_frames.append(pm_df)
        if per_mouse_frames:
            per_mouse_by_key[(selected_group, selected_condition)] = pd.concat(per_mouse_frames, ignore_index=True)
            save_group_summary(per_mouse_by_key[(selected_group, selected_condition)],
                               ['n_bouts', 'mean_duration_s', 'median_duration_s', 'total_duration_s'],
                               directory_path, selected_group, selected_condition)

    pd.DataFrame(all_file_durations).to_csv(
        os.path.join(directory_path, "total_average_std_durations_per_file.csv"), index=False)

    # ---- Figure 2: mean bout duration, mean ± SEM across mice (companion to bouts/min chart)
    bar_figs = []
    for selected_group, fig_bar, ax2_by_pair, layout2 in group_figures(
            selected_groups, selected_conditions, panel_w=6.0, panel_h=3.0, max_cols=1, extra_h=0.6,
            sharex=True, sharey=True):
        rows = layout2['rows']
        for selected_condition in selected_conditions:
            key = (selected_group, selected_condition)
            a = ax2_by_pair[key]
            row = layout2['row_of'][key]
            col = layout2['col_of'][key]
            _style_axis(a)
            if key not in per_mouse_by_key:
                a.text(0.5, 0.5, f"Data not found for\n{selected_group} - {selected_condition}",
                       ha='center', va='center', transform=a.transAxes)
                a.set_title(f'{selected_condition}')
                continue
            pm = per_mouse_by_key[key]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stats = [describe(pm.loc[pm.behavior == b, 'mean_duration_s'].values) for b in behavior_names]
            means = [s['mean'] for s in stats]
            sems = [np.nan_to_num(s['sem']) for s in stats]
            a.barh(y_pos, means, xerr=sems, color=behavior_colors, zorder=3, height=0.6,
                   error_kw={'ecolor': 'black', 'capsize': 2, 'lw': 1})
            a.set_yticks(y_pos)
            a.set_yticklabels(behavior_names)
            if key == (selected_group, selected_conditions[0]):
                a.invert_yaxis()
            a.set_title(f'{selected_condition}  (n = {pm["file"].nunique()} mice)')
            a.set_xlabel('Mean bout duration (s)')
            a.set_ylabel('Behavior')
            a.tick_params(labelbottom=True, labelleft=True)
        fig_bar.suptitle(f'Group: {selected_group}\nMean bout duration per behavior (mean ± SEM across mice)',
                         fontsize=11)
        save_figure(fig_bar, os.path.join(directory_path, f"behavior_durations_mean-per-mouse_{safe_name(selected_group)}.svg"))
        bar_figs.append(fig_bar)

    # Order: all linear boxplots, then log versions, then bar charts (one figure per group in each)
    return box_figs[False] + box_figs[True] + bar_figs
