import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import pickle
from scipy.spatial.distance import cdist
from utils.timing import project_dir

try:
    import seaborn as sns
except Exception:
    sns = None

# Helper Functions
def find_bouts(binary_data: np.ndarray, min_frames: int):
    # Edge bouts (running at frame 0 or at the last frame) are counted, with durations censored at the recording edge.
    padded = np.concatenate(([0], np.asarray(binary_data, dtype=int), [0]))

    # diff: +1 indicates 0->1 transition; -1 indicates 1->0 transition
    diff = np.diff(padded)
    potential_starts = np.where(diff == 1)[0]
    potential_ends = np.where(diff == -1)[0] - 1

    keep = (potential_ends - potential_starts + 1) >= min_frames
    return potential_starts[keep], potential_ends[keep]

def calculate_behavior_statistics(
    behavior_vectors: list,
    sampling_rate: float,
    n_behaviors: int,
    min_frames: int,
):
    """
    Behavior statistics over each animal's full recording (or window), as in the original MATLAB
    (plain mean over the whole recording, no bins).

    behavior_vectors: list of 1-D int label arrays [0..n_behaviors-1], one per animal, each at its
      own length (no padding).
    Returns arrays shaped (n_animals, n_behaviors):
      fraction_time, n_bouts, mean_duration_s
    and n_frames: (n_animals,) frames per animal.
    """
    n_animals = len(behavior_vectors)
    fraction_time = np.zeros((n_animals, n_behaviors), dtype=float)
    n_bouts = np.zeros((n_animals, n_behaviors), dtype=float)
    mean_duration_s = np.zeros((n_animals, n_behaviors), dtype=float)
    n_frames = np.array([len(v) for v in behavior_vectors], dtype=int)

    for k, vec in enumerate(behavior_vectors):
        if len(vec) == 0:
            fraction_time[k, :] = np.nan
            n_bouts[k, :] = np.nan
            mean_duration_s[k, :] = np.nan
            continue

        for b in range(n_behaviors):
            binary_data = (np.asarray(vec) == b).astype(int)

            if np.sum(binary_data) > 0:
                bout_starts, bout_ends = find_bouts(binary_data, min_frames)

                if len(bout_starts) > 0 and len(bout_ends) > 0:
                    bout_ends = bout_ends[bout_ends >= bout_starts[0]]
                    if len(bout_ends) > 0:
                        bout_starts = bout_starts[bout_starts <= bout_ends[-1]]

                    if len(bout_starts) > 0 and len(bout_ends) > 0:
                        n_valid = min(len(bout_starts), len(bout_ends))
                        durations = (bout_ends[:n_valid] - bout_starts[:n_valid]) / sampling_rate

                        fraction_time[k, b] = np.mean(binary_data)
                        n_bouts[k, b] = float(len(bout_starts))
                        mean_duration_s[k, b] = float(np.mean(durations)) if len(durations) > 0 else 0.0

    return fraction_time, n_bouts, mean_duration_s, n_frames



def _min_to_slice(start_min: float, end_min: float, fps: float, n_frames: int):
    """
    Convert a [start_min, end_min) time window (minutes) to a frame slice.
    Clips indices to [0, n_frames].
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    start_frame = int(round(start_min * 60.0 * fps))
    end_frame = int(round(end_min * 60.0 * fps))

    start_frame = max(0, min(start_frame, n_frames))
    end_frame = max(0, min(end_frame, n_frames))

    return slice(start_frame, end_frame)

# Section 5: GENERATE TRANSITION MATRICES
def generate_transition_matrices(
    behavior_ds: np.ndarray,
    n_behaviors: int,
    window_size: int,
    window_slide: int,
):
    """Generate unfolded transition matrices for each animal.

    Parameters
    ----------
    behavior_ds : np.ndarray
        Downsampled behavior matrix of shape (n_frames_ds, n_animals) with 0-indexed labels.
    n_behaviors : int
        Number of behavior labels.
    window_size : int
        Window length in frames (downsampled frame-rate).
    window_slide : int
        Window stride in frames (downsampled frame-rate).

    Returns
    -------
    trans_unfolded : np.ndarray
        Array of shape (n_wins, n_behaviors**2, n_animals) containing flattened transition matrices.
    wins : np.ndarray
        Window start indices (frame indices in downsampled space).
    n_wins : int
        Number of windows.
    """
    if window_size <= 1:
        raise ValueError(f"window_size must be > 1, got {window_size}")
    if window_slide <= 0:
        raise ValueError(f"window_slide must be > 0, got {window_slide}")

    n_frames_ds, n_animals = behavior_ds.shape

    wins = np.arange(0, n_frames_ds, window_slide)
    wins = wins[wins + window_size < n_frames_ds]
    n_wins = int(len(wins))

    print(f"[LUPE-AMPS] Generating {n_wins} transition matrices per animal...")

    trans_unfolded = np.zeros((n_wins, n_behaviors ** 2, n_animals), dtype=float)

    for a in range(n_animals):
        # +1 to convert from 0-indexed labels to MATLAB-style 1-indexed labels
        data = behavior_ds[:, a].astype(int) + 1

        for t in range(n_wins):
            start_idx = int(wins[t])
            end_idx = int(start_idx + window_size)
            window = data[start_idx:end_idx]

            T = np.zeros((n_behaviors, n_behaviors), dtype=float)

            for n in range(1, n_behaviors + 1):
                idx = np.where(window == n)[0]
                idx = idx[idx < (len(window) - 1)]

                if idx.size == 0:
                    continue

                next_vals = window[idx + 1]
                for l in range(1, n_behaviors + 1):
                    T[n - 1, l - 1] = float(np.mean(next_vals == l))

            trans_unfolded[t, :, a] = T.flatten()

    # Replace NaNs with zeros
    trans_unfolded = np.nan_to_num(trans_unfolded, nan=0.0)

    return trans_unfolded, wins, n_wins

# Section 6: APPLY STATE MODEL
def load_state_centroids(model_path: str) -> np.ndarray:
    """Load pre-trained state centroids from file."""
    if model_path is None or str(model_path).strip() == "":
        raise ValueError("model_path is empty")

    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                for key in ['centroids', 'c', 'centers']:
                    if key in data:
                        return np.array(data[key])
            return np.array(data)
    elif model_path.endswith('.npy'):
        return np.load(model_path)
    elif model_path.endswith('.mat'):
        from scipy.io import loadmat
        data = loadmat(model_path)
        for key in ['c', 'centroids', 'centers']:
            if key in data:
                return np.array(data[key]).squeeze()
        raise KeyError(f"Could not find centroids in .mat file. Keys: {list(data.keys())}")
    else:
        raise ValueError(f"Unsupported file format: {model_path}")

def assign_states(trans_unfolded: np.ndarray, centroids: np.ndarray):
    """Assign states based on minimum distance to centroids."""
    n_wins, n_features, n_animals = trans_unfolded.shape

    state_assignments = np.zeros((n_wins, n_animals), dtype=int)
    min_distances = np.zeros((n_wins, n_animals), dtype=float)

    for a in range(n_animals):
        data = trans_unfolded[:, :, a]
        distances = cdist(data, centroids, metric='euclidean')
        state_assignments[:, a] = np.argmin(distances, axis=1) + 1  # 1-indexed
        min_distances[:, a] = np.min(distances, axis=1)

    return state_assignments, min_distances

def resample_states(state_assignments: np.ndarray, window_slide: int, window_size: int, target_length: int) -> np.ndarray:
    """Resample state assignments from window-level to frame-level."""
    n_wins, n_animals = state_assignments.shape
    behav_state = np.zeros((target_length, n_animals), dtype=int)

    for a in range(n_animals):
        for t in range(n_wins):
            start_idx = int(window_slide * t)
            end_idx = int(min(start_idx + window_size, target_length))
            behav_state[start_idx:end_idx, a] = int(state_assignments[t, a])

    return behav_state

def calculate_model_fit(trans_unfolded: np.ndarray, centroids: np.ndarray, n_shuffles: int = 100):
    """Calculate model fit compared to shuffled centroids."""
    n_wins, n_features, n_animals = trans_unfolded.shape

    _, real_distances = assign_states(trans_unfolded, centroids)
    real_fit = np.mean(real_distances, axis=0)

    print(f"[LUPE-AMPS] Calculating shuffled model fits ({n_shuffles} iterations)...")
    shuffled_fit = np.zeros((n_animals, n_shuffles), dtype=float)

    for v in range(n_shuffles):
        perm = np.random.permutation(n_features)
        shuffled_centroids = centroids[:, perm]

        for a in range(n_animals):
            data = trans_unfolded[:, :, a]
            distances = cdist(data, shuffled_centroids, metric='euclidean')
            shuffled_fit[a, v] = np.mean(np.min(distances, axis=1))

    return real_fit, shuffled_fit

def calculate_behavior_transitions_per_state(
    behavior_ds: np.ndarray,
    behav_state: np.ndarray,
    n_behaviors: int,
    n_states: int,
):
    """Calculate behavior→behavior transition matrices WITHIN each state.

    Returns
    -------
    state_transitions : dict[int, np.ndarray]
        dict[state] = (n_animals, n_behaviors, n_behaviors) transition probabilities
    """
    n_frames, n_animals = behavior_ds.shape

    state_transitions = {s: np.zeros((n_animals, n_behaviors, n_behaviors), dtype=float) for s in range(1, n_states + 1)}

    for a in range(n_animals):
        for t in range(n_frames - 1):
            current_state = int(behav_state[t, a])
            current_behavior = int(behavior_ds[t, a])
            next_behavior = int(behavior_ds[t + 1, a])

            if 1 <= current_state <= n_states:
                if current_behavior < n_behaviors and next_behavior < n_behaviors:
                    state_transitions[current_state][a, current_behavior, next_behavior] += 1.0

    for s in range(1, n_states + 1):
        for a in range(n_animals):
            for b in range(n_behaviors):
                row_sum = float(np.sum(state_transitions[s][a, b, :]))
                if row_sum > 0:
                    state_transitions[s][a, b, :] /= row_sum

    return state_transitions

def calculate_state_transitions(behav_state: np.ndarray, n_states: int) -> np.ndarray:
    """Calculate state→state transition matrices.

    Returns
    -------
    state_trans : np.ndarray
        (n_animals, n_states, n_states) transition probabilities
    """
    n_frames, n_animals = behav_state.shape
    state_trans = np.zeros((n_animals, n_states, n_states), dtype=float)

    for a in range(n_animals):
        for t in range(n_frames - 1):
            current_state = int(behav_state[t, a])
            next_state = int(behav_state[t + 1, a])

            if 1 <= current_state <= n_states and 1 <= next_state <= n_states:
                state_trans[a, current_state - 1, next_state - 1] += 1.0

    for a in range(n_animals):
        for s in range(n_states):
            row_sum = float(np.sum(state_trans[a, s, :]))
            if row_sum > 0:
                state_trans[a, s, :] /= row_sum

    return state_trans

# Section 7: STATE STATISTICS
def calculate_state_statistics(
    behav_state: np.ndarray,
    n_states: int,
    fps: float,
    min_frames: int,
):
    """Calculate fraction time, bout number, and bout duration for states.

    Parameters
    ----------
    behav_state : np.ndarray
        Frame-level state labels, shape (n_frames, n_animals), 1-indexed states.
    n_states : int
        Number of states.
    fps : float
        Frame-rate of `behav_state`.
    min_frames : int
        Minimum consecutive frames required to count a bout.

    Returns
    -------
    total_fraction_state : np.ndarray
        (n_animals, n_states) fraction time in each state.
    n_bouts_state : np.ndarray
        (n_animals, n_states) number of bouts for each state.
    bout_duration_state : np.ndarray
        (n_animals, n_states) mean bout duration (seconds) for each state.
    """
    if behav_state is None:
        raise ValueError("behav_state is None")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    n_frames, n_animals = behav_state.shape

    total_fraction_state = np.zeros((n_animals, n_states), dtype=float)
    n_bouts_state = np.zeros((n_animals, n_states), dtype=float)
    bout_duration_state = np.zeros((n_animals, n_states), dtype=float)

    for k in range(n_animals):
        data = behav_state[:, k]

        for b in range(1, n_states + 1):
            binary_data = (data == b).astype(int)

            if np.sum(binary_data) > 0:
                bout_starts, bout_ends = find_bouts(binary_data, int(min_frames))

                if len(bout_starts) > 0 and len(bout_ends) > 0:
                    # Align starts/ends
                    bout_ends = bout_ends[bout_ends >= bout_starts[0]]
                    if len(bout_ends) > 0:
                        bout_starts = bout_starts[bout_starts <= bout_ends[-1]]

                    if len(bout_starts) > 0 and len(bout_ends) > 0:
                        n_valid = min(len(bout_starts), len(bout_ends))
                        durations = (bout_ends[:n_valid] - bout_starts[:n_valid]) / float(fps)

                        total_fraction_state[k, b - 1] = float(np.mean(binary_data))
                        n_bouts_state[k, b - 1] = float(len(bout_starts))
                        bout_duration_state[k, b - 1] = float(np.mean(durations)) if len(durations) > 0 else 0.0


    return total_fraction_state, n_bouts_state, bout_duration_state

# Section 9 helpers: PCA params + projection
def load_pca_params(pca_path: str):
    """Load PCA parameters (coeff/loadings and mean) from .pkl or .mat.

    Returns
    -------
    coeff : np.ndarray
        (n_features, n_components)
    mu : np.ndarray
        (n_features,)

    Notes
    -----
    - For sklearn PCA objects: components_ is (n_components, n_features) so we transpose.
    - For MATLAB exports we support:
        * pain_scale_params struct containing coeff and mu
        * direct variables coeff and mu
    - We defensively squeeze/reshape to ensure coeff is 2D and mu is 1D.
    """
    if not isinstance(pca_path, str):
        raise TypeError(f"pca_path must be a string, got {type(pca_path)}")

    if pca_path.endswith('.pkl'):
        with open(pca_path, 'rb') as f:
            data = pickle.load(f)

        if hasattr(data, 'components_') and hasattr(data, 'mean_'):
            coeff = np.asarray(data.components_).T
            mu = np.asarray(data.mean_).squeeze()
            return coeff, mu

        if isinstance(data, dict):
            coeff = data.get('coeff', None)
            if coeff is None:
                coeff = data.get('components_', None)
            mu = data.get('mu', None)
            if mu is None:
                mu = data.get('mean_', None)

            if coeff is None:
                raise KeyError("PCA .pkl dict missing 'coeff'/'components_' key")

            coeff = np.asarray(coeff)
            if mu is None:
                n_features = int(coeff.shape[0]) if coeff.ndim == 2 else int(coeff.size)
                mu = np.zeros(n_features)

            mu = np.asarray(mu).squeeze()

            if coeff.ndim != 2:
                raise ValueError(f"PCA coeff must be 2D, got shape {coeff.shape}")

            if mu.ndim == 1 and coeff.shape[0] != mu.shape[0] and coeff.shape[1] == mu.shape[0]:
                coeff = coeff.T

            if mu.ndim != 1:
                mu = mu.reshape(-1)
            if coeff.shape[0] != mu.shape[0]:
                raise ValueError(f"Shape mismatch: coeff {coeff.shape} vs mu {mu.shape}")

            return coeff, mu

        raise ValueError("Unrecognized PCA format in .pkl")

    if pca_path.endswith('.mat'):
        from scipy.io import loadmat
        data = loadmat(pca_path)

        # Handle MATLAB struct format (pain_scale_params.coeff, pain_scale_params.mu)
        if 'pain_scale_params' in data:
            params = data['pain_scale_params']
            try:
                coeff = np.asarray(params['coeff'][0, 0])
            except Exception:
                coeff = np.asarray(params['coeff']).squeeze()
            try:
                mu = np.asarray(params['mu'][0, 0]).squeeze()
            except Exception:
                mu = np.asarray(params['mu']).squeeze()

            if mu.ndim != 1:
                mu = mu.reshape(-1)
            if coeff.ndim != 2:
                raise ValueError(f"PCA coeff must be 2D, got shape {coeff.shape}")

            if coeff.shape[0] != mu.shape[0] and coeff.shape[1] == mu.shape[0]:
                coeff = coeff.T

            if coeff.shape[0] != mu.shape[0]:
                raise ValueError(f"Shape mismatch: coeff {coeff.shape} vs mu {mu.shape}")

            return coeff, mu

        # Handle direct variable format (coeff, mu as separate variables)
        if 'coeff' in data and ('mu' in data or 'mean' in data):
            coeff = np.asarray(data['coeff']).squeeze()
            mu = np.asarray(data['mu'] if 'mu' in data else data['mean']).squeeze()

            if mu.ndim != 1:
                mu = mu.reshape(-1)
            if coeff.ndim != 2:
                raise ValueError(f"PCA coeff must be 2D, got shape {coeff.shape}")

            if coeff.shape[0] != mu.shape[0] and coeff.shape[1] == mu.shape[0]:
                coeff = coeff.T

            if coeff.shape[0] != mu.shape[0]:
                raise ValueError(f"Shape mismatch: coeff {coeff.shape} vs mu {mu.shape}")

            return coeff, mu

        keys = [k for k in data.keys() if not k.startswith('__')]
        raise KeyError(f"Could not find PCA params in .mat file. Keys: {keys}")

    raise ValueError(f"Unsupported file format: {pca_path}")

def project_to_pc_space(state_occupancy: np.ndarray, coeff: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Project state occupancy data into PC space.

    Parameters
    ----------
    state_occupancy : np.ndarray
        (n_samples, n_features)
    coeff : np.ndarray
        (n_features, n_components)
    mu : np.ndarray
        (n_features,)

    Returns
    -------
    projected : np.ndarray
        (n_samples, 2) for PC1/PC2
    """
    occ = np.asarray(state_occupancy, dtype=float)
    coeff = np.asarray(coeff, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)

    if occ.ndim != 2:
        raise ValueError(f"state_occupancy must be 2D, got shape {occ.shape}")
    if coeff.ndim != 2:
        raise ValueError(f"coeff must be 2D, got shape {coeff.shape}")
    if mu.ndim != 1:
        raise ValueError(f"mu must be 1D, got shape {mu.shape}")
    if occ.shape[1] != mu.shape[0]:
        raise ValueError(f"Shape mismatch: occupancy {occ.shape} vs mu {mu.shape}")
    if coeff.shape[0] != mu.shape[0]:
        raise ValueError(f"Shape mismatch: coeff {coeff.shape} vs mu {mu.shape}")

    centered = occ - mu
    projected = centered @ coeff[:, :2]
    return projected

# Section 4: PREPROCESSING - DOWNSAMPLING
def flip_injury_site(behavior_matrix: np.ndarray, injury_site: int) -> np.ndarray:
    """Flip left/right lick labels if injury was on right paw.

    LUPE-AMPS expects the injured paw lick to be treated consistently.
    Default convention: label 4 = injured paw lick, label 5 = uninjured paw lick.

    If injury_site == 1 (RIGHT hindpaw injured), swap labels 4 and 5.
    """
    if int(injury_site) == 0:
        return behavior_matrix  # No change needed (LEFT paw injury is default)

    flipped = behavior_matrix.copy()

    # Right paw injury - swap labels 4 and 5
    mask_4 = flipped == 4
    mask_5 = flipped == 5
    flipped[mask_4] = 5
    flipped[mask_5] = 4

    return flipped

def downsample_behavior(behavior_matrix: np.ndarray, sampling_rate: float, target_fps: float) -> np.ndarray:
    """Downsample behavior from original fps to target fps using mode.

    This converts a (n_frames, n_animals) integer label matrix at `sampling_rate`
    to a lower frame rate `target_fps` by taking the modal label within each
    downsampled window.
    """
    if sampling_rate <= 0 or target_fps <= 0:
        raise ValueError(f"sampling_rate and target_fps must be > 0 (got {sampling_rate}, {target_fps})")

    rate_ratio = float(target_fps) / float(sampling_rate)
    if rate_ratio <= 0:
        raise ValueError(f"Invalid rate_ratio: {rate_ratio}")

    n_frames, n_animals = behavior_matrix.shape
    n_downsampled = int(np.floor(n_frames * rate_ratio))
    n_downsampled = max(1, n_downsampled)

    downsampled = np.zeros((n_downsampled, n_animals), dtype=int)

    for n in range(n_downsampled):
        start_idx = int(np.floor(n / rate_ratio))
        end_idx = int(np.floor((n + 1) / rate_ratio))

        start_idx = max(0, min(start_idx, n_frames))
        end_idx = max(0, min(end_idx, n_frames))
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, n_frames)

        window = behavior_matrix[start_idx:end_idx, :]
        if window.size == 0:
            continue

        m = mode(window, axis=0, keepdims=False)
        downsampled[n, :] = m.mode.astype(int)

    return downsampled

def load_lupe_data(base_dir: str, groups: list, conditions: list):
    """
    Load LUPE behavior CSVs organized as:
      base_dir/<group>/<condition>/*.csv

    Returns:
      behavior_matrix: (expected_length, n_animals) int
      animal_names: list[str]
      group_labels: list[str]
      condition_labels: list[str]
      expected_length: int (max frames across animals)
      behavior_vectors: list[np.ndarray], each animal's labels at its own length (no padding),
        used for the per-animal behavior statistics (Sections 2/2b)
    """
    rows = []  # each entry: dict with file info + vector
    for group in groups:
        group_dir = os.path.join(base_dir, group)
        if not os.path.exists(group_dir):
            print(f"[LUPE-AMPS] Warning: Group directory not found: {group_dir}")
            continue

        for condition in conditions:
            condition_dir = os.path.join(group_dir, condition)
            if not os.path.exists(condition_dir):
                print(f"[LUPE-AMPS] Warning: Condition directory not found: {condition_dir}")
                continue

            csv_files = sorted(glob.glob(os.path.join(condition_dir, "*.csv")))
            print(f"[LUPE-AMPS] Found {len(csv_files)} files in {group}/{condition}")

            for file_path in csv_files:
                try:
                    data = pd.read_csv(file_path)

                    behav_col = pd.to_numeric(data.iloc[:, 1], errors="coerce").fillna(0).to_numpy()

                    rows.append(
                        {
                            "file_path": file_path,
                            "animal": os.path.splitext(os.path.basename(file_path))[0],
                            "group": group,
                            "condition": condition,
                            "vec": behav_col.astype(int),
                            "n_frames": int(len(behav_col)),
                        }
                    )
                except Exception as e:
                    print(f"[LUPE-AMPS] Error loading {file_path}: {e}")
                    continue

    if len(rows) == 0:
        raise ValueError("No data files were successfully loaded! Check base_dir/groups/conditions paths.")

    expected_length = max(r["n_frames"] for r in rows)
    if expected_length <= 0:
        raise ValueError("Loaded files but frame counts are zero.")

    # Pad/truncate all to expected_length (used by the state model, Sections 4-9)
    behavior_data = []
    behavior_vectors = []
    animal_names = []
    group_labels = []
    condition_labels = []

    for r in rows:
        vec = r["vec"]
        if len(vec) < expected_length:
            padded = np.zeros(expected_length, dtype=int)
            padded[: len(vec)] = vec
            vec_use = padded
        else:
            vec_use = vec[:expected_length]

        behavior_data.append(vec_use)
        behavior_vectors.append(vec)
        animal_names.append(r["animal"])
        group_labels.append(r["group"])
        condition_labels.append(r["condition"])

    behavior_matrix = np.column_stack(behavior_data)  # (expected_length, n_animals)

    print(f"[LUPE-AMPS] Loaded {behavior_matrix.shape[1]} animals, {behavior_matrix.shape[0]} frames each")
    return behavior_matrix, animal_names, group_labels, condition_labels, expected_length, behavior_vectors


# Main entrypoint called by Streamlit
def behavior_LUPE_AMPS(
    project_name: str,
    model_path: str | None = None,
    pca_model_path: str | None = None,
    model_dir: str | None = None,
    selected_groups: list = None,
    selected_conditions: list = None,
    injury_site: int = 0,
    use_timepoint_comparison: bool = False,
    time_ranges_min: list | None = None,
    time_labels: list | None = None,
    # defaults matching your notebook expectations
    sampling_rate: float = 60.0,
    target_fps: float = 20.0,
    min_frames_true_positive: int = 6,
    make_timepoint_plots: bool = True,
    make_behavior_plots: bool = True,
    window_length_sec: int = 30,
    window_slide_sec: int = 10,
    model_fit_shuffles: int = 100,
):
    """
    Backend LUPE-AMPS runner.

    Model file expectations (LUPE-AMPS):
      - State model centroids (required for Section 6):
          model/panpainmodel30.mat
      - Pain-scale / PCA parameters (used for later LUPE-AMPS sections):
          model/pain_scale_params.mat

    You may pass explicit paths:
      - model_path: direct path to `model/LUPE-AMPS/panpainmodel30.mat`
      - pca_model_path: direct path to `model/LUPE-AMPS/pain_scale_params.mat`

    Or pass model_dir and the runner will auto-resolve:
      - model_dir/<expected filenames>

    NOTE:
    - This function currently implements:
        * Section 2: Behavior stats export
        * Section 2b: Optional timepoint behavior stats export (long format CSV)
        * Section 2c: Optional timepoint behavior plots per group
        * Section 6: Apply state model (if `model_path` can be resolved)
    - Later sections (AMPS PCA projection, etc.) will be added next.
    """

    if project_name is None:
        raise ValueError("project_name is required")
    project_name = str(project_name).strip().rstrip("/\\")

    # Paths / model inputs (notebook project structure, relative to notebooks/)
    base_data_dir = os.path.join(
        project_dir(project_name), "figures",
        "behaviors_csv_raw-classification", "frames"
    )

    output_base_dir = os.path.join(
        project_dir(project_name), "figures",
        "behaviors_LUPE-AMPS"
    )
    os.makedirs(output_base_dir, exist_ok=True)

    # Resolve model_path / model_dir (if not provided)
    # Normalize provided model paths (expanduser + allow relative paths)
    if model_path is not None and str(model_path).strip() != "":
        model_path = os.path.expanduser(str(model_path).strip())
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)

    if pca_model_path is not None and str(pca_model_path).strip() != "":
        pca_model_path = os.path.expanduser(str(pca_model_path).strip())
        if not os.path.isabs(pca_model_path):
            pca_model_path = os.path.abspath(pca_model_path)

    if model_dir is not None and str(model_dir).strip() != "":
        model_dir = os.path.expanduser(str(model_dir).strip())
        if not os.path.isabs(model_dir):
            model_dir = os.path.abspath(model_dir)
    else:
        model_dir = os.path.abspath(os.path.join("..", "model"))

    # Helpful debug context
    try:
        print(f"[LUPE-AMPS] CWD: {os.getcwd()}")
    except Exception:
        pass
    print(f"[LUPE-AMPS] model_dir resolved to: {model_dir} (exists={os.path.isdir(model_dir)})")
    # Helpful debug: list contents when present
    try:
        if os.path.isdir(model_dir):
            print(f"[LUPE-AMPS] model_dir contents: {sorted(os.listdir(model_dir))}")
    except Exception:
        pass

    # If model_path wasn't provided, try to find it inside model_dir
    if model_path is None or str(model_path).strip() == "":
        candidate_paths: list[str] = []

        # Prefer the known LUPE-AMPS filenames first
        for fname in [
            "panpainmodel30.mat",
            "panpainmodel.mat",
        ]:
            candidate_paths.append(os.path.join(model_dir, fname))

        try:
            candidate_paths.extend(sorted(glob.glob(os.path.join(model_dir, "*.mat"))))
            candidate_paths.extend(sorted(glob.glob(os.path.join(model_dir, "*.pkl"))))
            candidate_paths.extend(sorted(glob.glob(os.path.join(model_dir, "*.npy"))))
        except Exception:
            pass

        for p in candidate_paths:
            if os.path.exists(p):
                model_path = p
                break

        if pca_model_path is None or str(pca_model_path).strip() == "":
            pca_candidate = os.path.join(model_dir, "pain_scale_params.mat")
            if os.path.exists(pca_candidate):
                pca_model_path = pca_candidate

        # Last-resort: recursive search from repo root (useful if working dir changes)
        if model_path is None or str(model_path).strip() == "":
            try:
                hits = glob.glob(os.path.join(".", "**", "model", "LUPE-AMPS", "panpainmodel30.mat"), recursive=True)
                hits = sorted(set(hits))
                if len(hits) >= 1:
                    model_path = os.path.abspath(hits[0])

                if pca_model_path is None or str(pca_model_path).strip() == "":
                    pca_hits = glob.glob(os.path.join(".", "**", "model", "LUPE-AMPS", "pain_scale_params.mat"), recursive=True)
                    pca_hits = sorted(set(pca_hits))
                    if len(pca_hits) >= 1:
                        pca_model_path = os.path.abspath(pca_hits[0])
            except Exception:
                pass

    # Final validation + debug prints
    if model_path is not None and str(model_path).strip() != "":
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        if os.path.exists(model_path):
            print(f"[LUPE-AMPS] State model centroids file resolved to: {model_path}")
        else:
            print(f"[LUPE-AMPS] model_path was set but does not exist: {model_path}")
            model_path = None
    else:
        print(
            "[LUPE-AMPS] No model_path provided and no default centroids file found. "
            f"Checked model_dir={model_dir}. "
            "Provide model_path directly (relative or absolute), or set model_dir to the folder containing panpainmodel30.mat."
        )
        try:
            if os.path.isdir(model_dir):
                print(f"[LUPE-AMPS] Contents of {model_dir}: {sorted(os.listdir(model_dir))}")
        except Exception:
            pass

    # Validate PCA params path (optional for now)
    if pca_model_path is not None and str(pca_model_path).strip() != "":
        if not os.path.isabs(pca_model_path):
            pca_model_path = os.path.abspath(pca_model_path)
        if os.path.exists(pca_model_path):
            print(f"[LUPE-AMPS] PCA params file resolved to: {pca_model_path}")
        else:
            print(f"[LUPE-AMPS] pca_model_path was set but does not exist: {pca_model_path}")
            pca_model_path = None
    else:
        print(
            "[LUPE-AMPS] No pca_model_path provided and no default PCA params file found. "
            f"Checked model_dir={model_dir}. "
            "(This is optional for now; it will be required for later LUPE-AMPS PCA/pain-scale sections.)"
        )

    # Static labels used by the notebook
    behaviors = ["Still", "Walking", "Rearing", "Grooming", "LeftLick", "RightLick"]
    n_behaviors = len(behaviors)

    print("[LUPE-AMPS] Configuration loaded!")
    print(f"[LUPE-AMPS] Project: {project_name}")
    print(f"[LUPE-AMPS] Groups: {selected_groups}")
    print(f"[LUPE-AMPS] Conditions: {selected_conditions}")
    print(f"[LUPE-AMPS] Base data dir: {base_data_dir}")
    print(f"[LUPE-AMPS] Output base dir: {output_base_dir}")

    # Debug: confirm timepoint args coming from Streamlit
    try:
        tr_len = 0 if time_ranges_min is None else len(time_ranges_min)
    except Exception:
        tr_len = -1
    print(
        f"[LUPE-AMPS] Timepoint args → use_timepoint_comparison={use_timepoint_comparison}, "
        f"time_ranges_min_len={tr_len}, time_labels_len={(0 if time_labels is None else len(time_labels))}"
    )
    if time_ranges_min:
        print(f"[LUPE-AMPS] Time windows: {time_ranges_min}")

    # Section 2: DATA LOADING AND BEHAVIOR STATISTICS
    behavior_matrix, animal_names, group_labels, condition_labels, expected_length, behavior_vectors = load_lupe_data(
        base_data_dir, selected_groups, selected_conditions
    )

    # Section 4: PREPROCESSING - DOWNSAMPLING
    try:
        behavior_matrix_flipped = flip_injury_site(behavior_matrix, injury_site)
        behavior_ds = downsample_behavior(behavior_matrix_flipped, sampling_rate, target_fps)

        print("=" * 60)
        print("[LUPE-AMPS] SECTION 4: PREPROCESSING COMPLETE ✅")
        print("=" * 60)
        print(f"  Injury site: {'LEFT' if int(injury_site) == 0 else 'RIGHT'} hindpaw")
        print(f"  Downsampled: {sampling_rate} fps → {target_fps} fps")
        print(f"  Original shape: {behavior_matrix.shape}")
        print(f"  Downsampled shape: {behavior_ds.shape}")
        print("=" * 60)

        # Section 5: GENERATE TRANSITION MATRICES
        window_size = int(round(target_fps * float(window_length_sec)))
        window_slide = int(round(target_fps * float(window_slide_sec)))

        # Guardrails
        window_size = max(2, window_size)
        window_slide = max(1, window_slide)

        trans_unfolded, wins, n_wins = generate_transition_matrices(
            behavior_ds=behavior_ds,
            n_behaviors=n_behaviors,
            window_size=window_size,
            window_slide=window_slide,
        )

        print(f"[LUPE-AMPS] Transition matrices shape: {trans_unfolded.shape}")
        print(f"[LUPE-AMPS]   - {n_wins} time windows")
        print(f"[LUPE-AMPS]   - {n_behaviors**2} features (unfolded {n_behaviors}x{n_behaviors} matrix)")
        print(f"[LUPE-AMPS]   - {trans_unfolded.shape[2]} animals")

        # Section 6: APPLY STATE MODEL
        if model_path is None or str(model_path).strip() == "":
            print("[LUPE-AMPS] No model_path available — skipping Section 6 (state model).")
        elif not os.path.exists(str(model_path)):
            print(f"[LUPE-AMPS] model_path was set but does not exist: {model_path} — skipping Section 6 (state model).")
        else:
            print(f"[LUPE-AMPS] Loading state centroids from: {model_path}")
            centroids = load_state_centroids(model_path)
            centroids = np.array(centroids)
            if centroids.ndim != 2:
                raise ValueError(f"Centroids must be 2D (n_states x n_features). Got shape: {centroids.shape}")
            print(f"[LUPE-AMPS] Centroids shape: {centroids.shape}")

            # Assign states to each animal (window-level)
            state_assignments, min_distances = assign_states(trans_unfolded, centroids)
            print(f"[LUPE-AMPS] State assignments shape: {state_assignments.shape}")

            # Resample to frame-level (downsampled frame space)
            behav_state = resample_states(state_assignments, window_slide, window_size, behavior_ds.shape[0])
            print(f"[LUPE-AMPS] Frame-level states shape: {behav_state.shape}")

            # Calculate model fit
            real_fit, shuffled_fit = calculate_model_fit(trans_unfolded, centroids, n_shuffles=int(model_fit_shuffles))
            shuffled_mean = np.mean(shuffled_fit, axis=1)
            shuffled_sem = np.std(shuffled_fit, axis=1, ddof=1) / np.sqrt(shuffled_fit.shape[1])

            # CREATE OUTPUT DIRECTORY STRUCTURE
            output_dir_section6 = os.path.join(output_base_dir, 'Section6_StateModel')
            output_dir_behavior_trans = os.path.join(output_dir_section6, 'behavior_transitions_per_state')
            output_dir_state_trans = os.path.join(output_dir_section6, 'state_to_state_transitions')
            output_dir_modelfit = os.path.join(output_dir_section6, 'model_fit_validation')
            output_dir_statefractions = os.path.join(output_dir_section6, 'individual_state_fractions')

            os.makedirs(output_dir_behavior_trans, exist_ok=True)
            os.makedirs(output_dir_state_trans, exist_ok=True)
            os.makedirs(output_dir_modelfit, exist_ok=True)
            os.makedirs(output_dir_statefractions, exist_ok=True)

            # State column names (descriptive)
            state_column_names = {
                1: 'State1_PainSuppressed',
                2: 'State2_NonPain',
                3: 'State3_PainEnhanced',
                4: 'State4_PainEnhanced',
                5: 'State5_NonPain',
                6: 'State6_NonPain'
            }

            # State descriptions (short form for plot labels)
            state_short_names = {
                1: 'S1\nPainSupp',
                2: 'S2\nNonPain',
                3: 'S3\nPainEnh',
                4: 'S4\nPainEnh',
                5: 'S5\nNonPain',
                6: 'S6\nNonPain'
            }

            # Behavior labels
            behaviors_redefined = ['Still', 'Walk', 'Rear', 'Groom', 'Injured\nLick', 'Uninjured\nLick']
            behaviors_csv = ['Still', 'Walk', 'Rear', 'Groom', 'Injured_Lick', 'Uninjured_Lick']

            # Get unique groups and conditions (sorted)
            unique_groups = sorted(list(set(group_labels)))
            unique_conditions = sorted(list(set(condition_labels)))

            n_states = 6

            # CALCULATE TRANSITIONS
            print("\n[LUPE-AMPS] Calculating behavior transitions per state...")
            behavior_trans_per_state = calculate_behavior_transitions_per_state(
                behavior_ds, behav_state, n_behaviors, n_states
            )

            print("[LUPE-AMPS] Calculating state-to-state transitions...")
            state_trans = calculate_state_transitions(behav_state, n_states)

            # FIGURE 1: BEHAVIOR→BEHAVIOR TRANSITIONS PER STATE
            print("\n[LUPE-AMPS] Generating behavior transition matrices per state...")

            all_behavior_trans_data = []

            for group in unique_groups:
                n_conds = len(unique_conditions)

                fig, axes = plt.subplots(n_states, n_conds, figsize=(5 * n_conds, 4 * n_states))
                if n_conds == 1:
                    axes = axes.reshape(-1, 1)

                for state_idx in range(n_states):
                    state = state_idx + 1

                    for cond_idx, cond in enumerate(unique_conditions):
                        ax = axes[state_idx, cond_idx]

                        mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                        n_animals_subset = int(np.sum(mask))

                        if n_animals_subset == 0:
                            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)
                            ax.set_title(f'{cond}', fontsize=9)
                            ax.axis('off')
                            continue

                        animal_indices = np.where(mask)[0]
                        trans_matrices = behavior_trans_per_state[state][animal_indices, :, :]
                        mean_trans = np.mean(trans_matrices, axis=0)

                        trans_log = np.log10(mean_trans + 1e-10)
                        ax.imshow(trans_log, cmap='magma', aspect='equal', vmin=-3, vmax=0)

                        for i in range(n_behaviors):
                            for j in range(n_behaviors):
                                value = float(mean_trans[i, j])
                                text_color = 'white' if float(trans_log[i, j]) < -1.5 else 'black'
                                if value >= 0.01:
                                    text = f'{value:.2f}'
                                elif value >= 0.001:
                                    text = f'{value:.3f}'
                                else:
                                    text = ''
                                ax.text(j, i, text, ha='center', va='center', fontsize=6, color=text_color)

                        ax.set_xticks(np.arange(-0.5, n_behaviors, 1), minor=True)
                        ax.set_yticks(np.arange(-0.5, n_behaviors, 1), minor=True)
                        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

                        ax.set_xticks(range(n_behaviors))
                        ax.set_yticks(range(n_behaviors))

                        if state_idx == n_states - 1:
                            ax.set_xticklabels(behaviors_redefined, rotation=45, ha='right', fontsize=7)
                        else:
                            ax.set_xticklabels([])

                        if cond_idx == 0:
                            ax.set_yticklabels(behaviors_redefined, fontsize=7)
                            ax.set_ylabel(f'{state_column_names[state]}', fontsize=9, fontweight='bold')
                        else:
                            ax.set_yticklabels([])

                        if state_idx == 0:
                            ax.set_title(f'{cond}\n(n={n_animals_subset})', fontsize=9)

                        # Store for CSV export
                        for i in range(n_behaviors):
                            for j in range(n_behaviors):
                                all_behavior_trans_data.append({
                                    'Group': group,
                                    'Condition': cond,
                                    'State': state_column_names[state],
                                    'FROM_Behavior': behaviors_csv[i],
                                    'TO_Behavior': behaviors_csv[j],
                                    'Transition_Probability': float(mean_trans[i, j]),
                                    'N_Animals': int(n_animals_subset)
                                })

                        df_matrix = pd.DataFrame(mean_trans, index=behaviors_csv, columns=behaviors_csv)
                        df_matrix.index.name = 'FROM_Behavior'
                        # SHORTENED FILENAME
                        filename_clean = f'behavior_trans_{group}_{cond}_State{state}'.replace('/', '_').replace(' ', '_')
                        df_matrix.to_csv(os.path.join(output_dir_behavior_trans, f'{filename_clean}.csv'))

                fig.subplots_adjust(right=0.92)
                cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
                sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=-3, vmax=0))
                sm.set_array([])
                cbar = fig.colorbar(sm, cax=cbar_ax)
                cbar.set_label('log₁₀(P)', fontsize=10)

                plt.suptitle(
                    f'Behavior→Behavior Transitions Per State: {group}\n(Rows=States, Columns=Conditions)',
                    fontsize=14,
                    fontweight='bold',
                    y=1.01,
                )

                # SHORTENED FILENAME
                fig_filename = f'behavior_transitions_per_state_{group}'
                plt.savefig(os.path.join(output_dir_behavior_trans, f'{fig_filename}.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(output_dir_behavior_trans, f'{fig_filename}.svg'), bbox_inches='tight')
                plt.close(fig)

                print(f"[LUPE-AMPS]   - Saved behavior transitions for {group}")

            df_all_behavior_trans = pd.DataFrame(all_behavior_trans_data)
            # SHORTENED FILENAME
            df_all_behavior_trans.to_csv(
                os.path.join(output_dir_behavior_trans, 'behavior_transitions_per_state_all.csv'),
                index=False,
            )

            print(f"\n[LUPE-AMPS] Behavior transitions per state saved to: {output_dir_behavior_trans}/")

            # FIGURE 2: STATE→STATE TRANSITION
            print("\n[LUPE-AMPS] Generating state-to-state transition matrices...")

            all_state_trans_data = []
            state_labels_full = [state_column_names[i] for i in range(1, 7)]

            for group in unique_groups:
                n_conds = len(unique_conditions)
                n_cols = min(3, n_conds)
                n_rows = int(np.ceil(n_conds / n_cols))

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
                if n_conds == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)

                for idx, cond in enumerate(unique_conditions):
                    row = idx // n_cols
                    col = idx % n_cols
                    ax = axes[row, col]

                    mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                    n_animals_subset = int(np.sum(mask))

                    if n_animals_subset == 0:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                        ax.set_title(f'{cond}\n(n=0)', fontsize=10, fontweight='bold')
                        ax.axis('off')
                        continue

                    animal_indices = np.where(mask)[0]
                    trans_matrices = state_trans[animal_indices, :, :]
                    mean_trans = np.mean(trans_matrices, axis=0)

                    trans_log = np.log10(mean_trans + 1e-10)
                    im = ax.imshow(trans_log, cmap='viridis', aspect='equal', vmin=-3, vmax=0)

                    for i in range(n_states):
                        for j in range(n_states):
                            value = float(mean_trans[i, j])
                            text_color = 'white' if float(trans_log[i, j]) < -1.5 else 'black'
                            if value >= 0.01:
                                text = f'{value:.2f}'
                            elif value >= 0.001:
                                text = f'{value:.3f}'
                            else:
                                text = ''
                            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=text_color)

                    ax.set_xticks(np.arange(-0.5, n_states, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, n_states, 1), minor=True)
                    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

                    ax.set_xticks(range(n_states))
                    ax.set_yticks(range(n_states))
                    ax.set_xticklabels([state_short_names[i] for i in range(1, 7)], rotation=45, ha='right', fontsize=8)
                    ax.set_yticklabels([state_short_names[i] for i in range(1, 7)], fontsize=8)
                    ax.set_xlabel('TO State', fontsize=9)
                    ax.set_ylabel('FROM State', fontsize=9)
                    ax.set_title(f'{cond}\n(n={n_animals_subset})', fontsize=10, fontweight='bold')

                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('log₁₀(P)', fontsize=8)

                    for i in range(n_states):
                        for j in range(n_states):
                            all_state_trans_data.append({
                                'Group': group,
                                'Condition': cond,
                                'FROM_State': state_column_names[i + 1],
                                'TO_State': state_column_names[j + 1],
                                'Transition_Probability': float(mean_trans[i, j]),
                                'N_Animals': int(n_animals_subset)
                            })

                    df_matrix = pd.DataFrame(mean_trans, index=state_labels_full, columns=state_labels_full)
                    df_matrix.index.name = 'FROM_State'
                    # SHORTENED FILENAME
                    filename_clean = f'state_trans_{group}_{cond}'.replace('/', '_').replace(' ', '_')
                    df_matrix.to_csv(os.path.join(output_dir_state_trans, f'{filename_clean}.csv'))

                for idx in range(n_conds, n_rows * n_cols):
                    row = idx // n_cols
                    col = idx % n_cols
                    axes[row, col].axis('off')

                plt.suptitle(
                    f'State→State Transitions: {group}\n(P(TO State | FROM State))',
                    fontsize=14,
                    fontweight='bold',
                    y=1.02,
                )
                plt.tight_layout()

                # SHORTENED FILENAME
                fig_filename = f'state_transitions_{group}'
                plt.savefig(os.path.join(output_dir_state_trans, f'{fig_filename}.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(output_dir_state_trans, f'{fig_filename}.svg'), bbox_inches='tight')
                plt.close(fig)

                print(f"[LUPE-AMPS]   - Saved state transitions for {group}")

            df_all_state_trans = pd.DataFrame(all_state_trans_data)
            # SHORTENED FILENAME
            df_all_state_trans.to_csv(
                os.path.join(output_dir_state_trans, 'state_transitions_all.csv'),
                index=False,
            )

            print(f"\n[LUPE-AMPS] State-to-state transitions saved to: {output_dir_state_trans}/")

            # FIGURE 3: Model Fit Validation (Real vs Shuffled)
            print("\n[LUPE-AMPS] Generating model fit validation...")

            animal_labels = [f'{g}_{c}_{a}' for g, c, a in zip(group_labels, condition_labels, animal_names)]

            fig, ax = plt.subplots(figsize=(max(12, len(animal_labels) * 0.3), 6))
            x = np.arange(len(real_fit))

            ax.bar(x, real_fit, color='red', alpha=0.4, label='Real data', edgecolor='darkred')
            ax.errorbar(
                x,
                shuffled_mean,
                yerr=shuffled_sem,
                fmt='o',
                color='gray',
                markersize=4,
                linewidth=2,
                capsize=3,
                label='Shuffled (mean ± SEM)',
            )

            ax.set_xlabel('Animal', fontsize=11)
            ax.set_ylabel('Mean Euclidean Distance (A.U.)', fontsize=11)
            ax.set_title(
                'Model Fit Validation: Real Data vs Shuffled Centroids\n(Lower = better fit)',
                fontsize=12,
                fontweight='bold',
            )
            ax.legend(loc='upper right')
            ax.set_xlim(-0.5, len(real_fit) - 0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(animal_labels, rotation=90, ha='center', fontsize=7)

            plt.tight_layout()
            # SHORTENED FILENAME
            plt.savefig(os.path.join(output_dir_modelfit, 'model_fit_validation.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir_modelfit, 'model_fit_validation.svg'), bbox_inches='tight')
            plt.close(fig)

            df_model_fit = pd.DataFrame({
                'Animal': animal_names,
                'Group': group_labels,
                'Condition': condition_labels,
                'RealModelFit': real_fit,
                'ShuffledModelFit_Mean': shuffled_mean,
                'ShuffledModelFit_SEM': shuffled_sem
            })
            # SHORTENED FILENAME
            df_model_fit.to_csv(os.path.join(output_dir_modelfit, 'model_fit.csv'), index=False)

            print(f"[LUPE-AMPS] Model fit validation saved to: {output_dir_modelfit}/")

            # EXPORT: Individual Animal State Fractions
            print("\n[LUPE-AMPS] Exporting individual state fractions...")

            state_fractions = np.zeros((len(animal_names), 6), dtype=float)
            for a in range(len(animal_names)):
                for s in range(1, 7):
                    state_fractions[a, s - 1] = float(np.mean(behav_state[:, a] == s))

            df_state_fractions = pd.DataFrame({
                'Animal': animal_names,
                'Group': group_labels,
                'Condition': condition_labels,
                state_column_names[1]: state_fractions[:, 0],
                state_column_names[2]: state_fractions[:, 1],
                state_column_names[3]: state_fractions[:, 2],
                state_column_names[4]: state_fractions[:, 3],
                state_column_names[5]: state_fractions[:, 4],
                state_column_names[6]: state_fractions[:, 5]
            })
            # SHORTENED FILENAME
            df_state_fractions.to_csv(
                os.path.join(output_dir_statefractions, 'individual_state_fractions.csv'),
                index=False,
            )

            summary_data = []
            for group in unique_groups:
                for cond in unique_conditions:
                    mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                    n_animals_subset = int(np.sum(mask))
                    if n_animals_subset == 0:
                        continue

                    cond_fractions = state_fractions[mask, :]
                    for s in range(6):
                        summary_data.append({
                            'Group': group,
                            'Condition': cond,
                            'State': state_column_names[s + 1],
                            'Mean_Fraction': float(np.mean(cond_fractions[:, s])),
                            'SEM_Fraction': float(np.std(cond_fractions[:, s], ddof=1) / np.sqrt(n_animals_subset)) if n_animals_subset > 1 else 0.0,
                        })

            df_summary = pd.DataFrame(summary_data)
            # SHORTENED FILENAME
            df_summary.to_csv(os.path.join(output_dir_statefractions, 'state_fractions_summary.csv'), index=False)

            print(f"[LUPE-AMPS] Individual state fractions saved to: {output_dir_statefractions}/")

            print("\n" + "=" * 60)
            print("[LUPE-AMPS] SECTION 6: STATE MODEL COMPLETE ✅")
            print("=" * 60)
            print(f"  States assigned for {len(animal_names)} animals")
            print(f"  Groups: {unique_groups}")
            print(f"  Conditions: {unique_conditions}")
            print(f"  Real model fit (mean ± SEM): {np.mean(real_fit):.4f} ± {np.std(real_fit, ddof=1) / np.sqrt(len(real_fit)):.4f}")
            print(f"  Shuffled model fit (mean): {np.mean(shuffled_mean):.4f}")
            print(f"\nOutput directory: {output_dir_section6}/")
            print("=" * 60)

            # Section 7: STATE STATISTICS
            print("\n[LUPE-AMPS] Calculating state statistics...")

            # Convert min_frames threshold from original fps space -> downsampled fps space
            min_frames_state_true_positive = max(
                1,
                int(round((float(min_frames_true_positive) / float(sampling_rate)) * float(target_fps))),
            )

            total_fraction_state, n_bouts_state, bout_duration_state = calculate_state_statistics(
                behav_state=behav_state,
                n_states=n_states,
                fps=float(target_fps),
                min_frames=min_frames_state_true_positive,
            )

            print(f"[LUPE-AMPS] State statistics calculated for {total_fraction_state.shape[0]} animals")

            # Descriptive state column names (consistent with Section 6)
            state_cols_descriptive = [state_column_names[i] for i in range(1, n_states + 1)]

            # Export state statistics
            output_dir_section7 = os.path.join(output_base_dir, 'Section7_StateStats')
            os.makedirs(output_dir_section7, exist_ok=True)

            df_frac = pd.DataFrame(total_fraction_state, columns=state_cols_descriptive, index=animal_names)
            df_frac['Group'] = group_labels
            df_frac['Condition'] = condition_labels
            # SHORTENED FILENAME
            df_frac.to_csv(os.path.join(output_dir_section7, 'state_fraction_time.csv'))

            df_bouts = pd.DataFrame(n_bouts_state, columns=state_cols_descriptive, index=animal_names)
            df_bouts['Group'] = group_labels
            df_bouts['Condition'] = condition_labels
            # SHORTENED FILENAME
            df_bouts.to_csv(os.path.join(output_dir_section7, 'state_bout_number.csv'))

            df_dur = pd.DataFrame(bout_duration_state, columns=state_cols_descriptive, index=animal_names)
            df_dur['Group'] = group_labels
            df_dur['Condition'] = condition_labels
            # SHORTENED FILENAME
            df_dur.to_csv(os.path.join(output_dir_section7, 'state_bout_duration.csv'))

            print(f"[LUPE-AMPS] State statistics exported to: {output_dir_section7}")

            # Section 7b/7c (OPTIONAL): TIMEPOINT COMPARISON — STATE STATS + VIZ
            if (not use_timepoint_comparison) or (not time_ranges_min) or (len(time_ranges_min) == 0):
                print("[LUPE-AMPS] Timepoint comparison disabled/empty — skipping Section 7b/7c (state stats).")
            else:
                if (not time_labels) or (len(time_labels) != len(time_ranges_min)):
                    time_labels = [f"{a:g}-{b:g} min" for (a, b) in time_ranges_min]

                # Timepoint output root (shared with behavior timepoint exports)
                output_dir_timepoints = os.path.join(output_base_dir, "Timepoint_Comparison")
                os.makedirs(output_dir_timepoints, exist_ok=True)

                print("[LUPE-AMPS] Computing timepoint state statistics...")
                state_timepoint_rows = []

                for label, (start_min, end_min) in zip(time_labels, time_ranges_min):
                    sl = _min_to_slice(start_min, end_min, float(target_fps), behav_state.shape[0])
                    if sl.stop <= sl.start:
                        print(f"[LUPE-AMPS] Skipping empty window: {label}")
                        continue

                    behav_state_win = behav_state[sl, :]

                    frac_s, bouts_s, dur_s = calculate_state_statistics(
                        behav_state=behav_state_win,
                        n_states=n_states,
                        fps=float(target_fps),
                        min_frames=min_frames_state_true_positive,
                    )

                    for a_idx, animal in enumerate(animal_names):
                        for s_idx in range(n_states):
                            state_name = state_column_names[s_idx + 1]
                            state_timepoint_rows.append({
                                'Time_Group': label,
                                'Start_min': float(start_min),
                                'End_min': float(end_min),
                                'Animal': animal,
                                'Group': group_labels[a_idx],
                                'Condition': condition_labels[a_idx],
                                'State': state_name,
                                'FractionTime': float(frac_s[a_idx, s_idx]),
                                'NumBouts': float(bouts_s[a_idx, s_idx]),
                                'MeanDuration_s': float(dur_s[a_idx, s_idx]),
                            })

                if len(state_timepoint_rows) == 0:
                    print("[LUPE-AMPS] No timepoint state rows to export — skipping Section 7b/7c.")
                else:
                    df_state_timepoints = pd.DataFrame(state_timepoint_rows)
                    # SHORTENED FILENAME
                    df_state_timepoints.to_csv(
                        os.path.join(output_dir_timepoints, 'state_timepoint_comparison.csv'),
                        index=False,
                    )
                    print(f"[LUPE-AMPS] Timepoint state comparison saved to: {output_dir_timepoints}")

                    # Section 7c: TIMEPOINT VISUALIZATION — STATE STATS
                    output_dir_timepoints_stateviz = os.path.join(output_dir_timepoints, 'State_Viz')
                    os.makedirs(output_dir_timepoints_stateviz, exist_ok=True)

                    metric_dirs = {
                        'FractionTime': os.path.join(output_dir_timepoints_stateviz, 'FractionTime'),
                        'NumBouts': os.path.join(output_dir_timepoints_stateviz, 'NumBouts'),
                        'MeanDuration_s': os.path.join(output_dir_timepoints_stateviz, 'MeanDuration'),
                    }
                    for d in metric_dirs.values():
                        os.makedirs(d, exist_ok=True)

                    df_state_timepoints['Time_Group'] = pd.Categorical(
                        df_state_timepoints['Time_Group'],
                        categories=time_labels,
                        ordered=True,
                    )

                    metrics_to_plot = [
                        ('FractionTime', 'Fraction Time'),
                        ('NumBouts', 'Number of Bouts'),
                        ('MeanDuration_s', 'Mean Bout Duration (s)'),
                    ]

                    def _sem(x):
                        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

                    unique_groups_tp = list(selected_groups)
                    unique_conditions_tp = list(selected_conditions)
                    unique_states = [state_column_names[i] for i in range(1, n_states + 1)]

                    for st in unique_states:
                        df_s = df_state_timepoints[df_state_timepoints['State'] == st].copy()
                        if df_s.empty:
                            continue

                        summary = (
                            df_s.groupby(['Group', 'Condition', 'Time_Group'], observed=True)
                            .agg(
                                mean_FractionTime=('FractionTime', 'mean'),
                                sem_FractionTime=('FractionTime', _sem),
                                mean_NumBouts=('NumBouts', 'mean'),
                                sem_NumBouts=('NumBouts', _sem),
                                mean_MeanDuration_s=('MeanDuration_s', 'mean'),
                                sem_MeanDuration_s=('MeanDuration_s', _sem),
                                n=('FractionTime', 'size'),
                            )
                            .reset_index()
                        )

                        # Plot PER GROUP
                        for group in unique_groups_tp:
                            summary_g = summary[summary['Group'] == group].copy()
                            if summary_g.empty:
                                continue

                            for metric_key, metric_label in metrics_to_plot:
                                mean_col = f"mean_{metric_key}"
                                sem_col = f"sem_{metric_key}"

                                fig = plt.figure(figsize=(10, 5))

                                for cond in unique_conditions_tp:
                                    sdf = summary_g[summary_g['Condition'] == cond].sort_values('Time_Group')
                                    if sdf.empty:
                                        continue

                                    # Align to full time_labels order (handles missing windows)
                                    sdf = sdf.set_index('Time_Group').reindex(time_labels).reset_index()

                                    x = np.arange(len(time_labels))
                                    y = sdf[mean_col].to_numpy(dtype=float)
                                    e = sdf[sem_col].to_numpy(dtype=float)

                                    n_line = int(np.nanmax(sdf['n'].to_numpy(dtype=float))) if np.any(~np.isnan(sdf['n'])) else 0
                                    line_label = f"{cond} (n={n_line})"

                                    plt.plot(x, y, marker='o', label=line_label)
                                    plt.fill_between(x, y - e, y + e, alpha=0.2)

                                plt.xticks(np.arange(len(time_labels)), time_labels, rotation=45, ha='right')
                                plt.xlabel('Time Window')
                                plt.ylabel(metric_label)
                                plt.title(f'{st} — {metric_label} by Time Window\nGroup: {group}')
                                plt.legend(frameon=False)
                                plt.tight_layout()

                                # SHORTENED FILENAME
                                fname = f"timepoint_{group}_{st}_{metric_key}".replace('/', '_').replace(' ', '_')
                                plt.savefig(os.path.join(metric_dirs[metric_key], f"{fname}.png"), dpi=300)
                                plt.savefig(os.path.join(metric_dirs[metric_key], f"{fname}.svg"))
                                plt.close(fig)

                    print(f"[LUPE-AMPS] Timepoint state visualizations saved to: {output_dir_timepoints_stateviz}")

            # Section 8: STATE VISUALIZATION
            # NOTE: This section depends on Section 7 outputs:
            #   total_fraction_state, n_bouts_state, bout_duration_state
            # and on Section 6 labels:
            #   state_column_names, unique_groups, unique_conditions, n_states

            print("\n[LUPE-AMPS] Generating state visualizations (Section 8)...")

            output_dir_section8 = os.path.join(output_base_dir, 'Section8_StateViz')
            os.makedirs(output_dir_section8, exist_ok=True)

            state_categories = {
                1: 'PainSuppressed',
                2: 'NonPain',
                3: 'PainEnhanced',
                4: 'PainEnhanced',
                5: 'NonPain',
                6: 'NonPain',
            }

            states_display = [f"State {i + 1}\n({state_categories[i + 1]})" for i in range(n_states)]

            try:
                if sns is not None:
                    palette = sns.color_palette("tab10", len(unique_conditions))
                    condition_colors = dict(zip(unique_conditions, palette))
                else:
                    raise RuntimeError("seaborn not available")
            except Exception:
                prop_cycle_colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
                if len(prop_cycle_colors) < len(unique_conditions):
                    # repeat if needed
                    prop_cycle_colors = (prop_cycle_colors * (len(unique_conditions) // max(1, len(prop_cycle_colors)) + 1))
                condition_colors = dict(zip(unique_conditions, prop_cycle_colors[: len(unique_conditions)]))

            # 8a. State Occupancy by Condition (per group)
            occupancy_summary = []

            for group in unique_groups:
                fig, ax = plt.subplots(figsize=(12, 6))

                for cond in unique_conditions:
                    mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                    n_animals_g = int(np.sum(mask))

                    if n_animals_g == 0:
                        continue

                    cond_data = total_fraction_state[mask, :]
                    means = np.mean(cond_data, axis=0)
                    sem = (
                        np.std(cond_data, axis=0, ddof=1) / np.sqrt(n_animals_g)
                        if n_animals_g > 1
                        else np.zeros_like(means)
                    )

                    x = np.arange(n_states)
                    ax.plot(x, means, marker='o', label=f'{cond} (n={n_animals_g})', color=condition_colors.get(cond, None))
                    ax.fill_between(x, means - sem, means + sem, alpha=0.2, color=condition_colors.get(cond, None))

                    # Store for CSV (ONE State column only)
                    for s in range(n_states):
                        occupancy_summary.append({
                            'Group': group,
                            'Condition': cond,
                            'State': state_column_names[s + 1],
                            'Mean': float(means[s]),
                            'SEM': float(sem[s]),
                        })

                ax.set_xticks(range(n_states))
                ax.set_xticklabels(states_display, rotation=45, ha='right')
                ax.set_xlabel('State')
                ax.set_ylabel('Fraction Occupancy')
                ax.set_title(f'State Occupancy by Condition — {group}')
                ax.legend(fontsize=8, frameon=False)

                plt.tight_layout()
                # SHORTENED FILENAME
                plt.savefig(os.path.join(output_dir_section8, f'state_occupancy_line_{group}.png'), dpi=300)
                plt.savefig(os.path.join(output_dir_section8, f'state_occupancy_line_{group}.svg'))
                plt.close(fig)

            # SHORTENED FILENAME
            pd.DataFrame(occupancy_summary).to_csv(
                os.path.join(output_dir_section8, 'state_occupancy_summary.csv'),
                index=False,
            )

            # 8b. Number of Bouts by Condition (per group)
            bouts_summary = []

            for group in unique_groups:
                fig, ax = plt.subplots(figsize=(12, 6))

                for cond in unique_conditions:
                    mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                    n_animals_g = int(np.sum(mask))

                    if n_animals_g == 0:
                        continue

                    cond_data = n_bouts_state[mask, :]
                    means = np.mean(cond_data, axis=0)
                    sem = (
                        np.std(cond_data, axis=0, ddof=1) / np.sqrt(n_animals_g)
                        if n_animals_g > 1
                        else np.zeros_like(means)
                    )

                    x = np.arange(n_states)
                    ax.plot(x, means, marker='o', label=f'{cond} (n={n_animals_g})', color=condition_colors.get(cond, None))
                    ax.fill_between(x, means - sem, means + sem, alpha=0.2, color=condition_colors.get(cond, None))

                    for s in range(n_states):
                        bouts_summary.append({
                            'Group': group,
                            'Condition': cond,
                            'State': state_column_names[s + 1],
                            'Mean': float(means[s]),
                            'SEM': float(sem[s]),
                        })

                ax.set_xticks(range(n_states))
                ax.set_xticklabels(states_display, rotation=45, ha='right')
                ax.set_xlabel('State')
                ax.set_ylabel('Number of Bouts')
                ax.set_title(f'State Bout Number by Condition — {group}')
                ax.legend(fontsize=8, frameon=False)

                plt.tight_layout()
                # SHORTENED FILENAME
                plt.savefig(os.path.join(output_dir_section8, f'state_bouts_line_{group}.png'), dpi=300)
                plt.savefig(os.path.join(output_dir_section8, f'state_bouts_line_{group}.svg'))
                plt.close(fig)

            # SHORTENED FILENAME
            pd.DataFrame(bouts_summary).to_csv(
                os.path.join(output_dir_section8, 'state_bouts_summary.csv'),
                index=False,
            )

            # 8c. Bout Duration by Condition (per group)
            duration_summary = []

            for group in unique_groups:
                fig, ax = plt.subplots(figsize=(12, 6))

                for cond in unique_conditions:
                    mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                    n_animals_g = int(np.sum(mask))

                    if n_animals_g == 0:
                        continue

                    cond_data = bout_duration_state[mask, :]
                    means = np.mean(cond_data, axis=0)
                    sem = (
                        np.std(cond_data, axis=0, ddof=1) / np.sqrt(n_animals_g)
                        if n_animals_g > 1
                        else np.zeros_like(means)
                    )

                    x = np.arange(n_states)
                    ax.plot(x, means, marker='o', label=f'{cond} (n={n_animals_g})', color=condition_colors.get(cond, None))
                    ax.fill_between(x, means - sem, means + sem, alpha=0.2, color=condition_colors.get(cond, None))

                    for s in range(n_states):
                        duration_summary.append({
                            'Group': group,
                            'Condition': cond,
                            'State': state_column_names[s + 1],
                            'Mean': float(means[s]),
                            'SEM': float(sem[s]),
                        })

                ax.set_xticks(range(n_states))
                ax.set_xticklabels(states_display, rotation=45, ha='right')
                ax.set_xlabel('State')
                ax.set_ylabel('Bout Duration (s)')
                ax.set_title(f'State Bout Duration by Condition — {group}')
                ax.legend(fontsize=8, frameon=False)

                plt.tight_layout()
                # SHORTENED FILENAME
                plt.savefig(os.path.join(output_dir_section8, f'state_duration_line_{group}.png'), dpi=300)
                plt.savefig(os.path.join(output_dir_section8, f'state_duration_line_{group}.svg'))
                plt.close(fig)

            # SHORTENED FILENAME
            pd.DataFrame(duration_summary).to_csv(
                os.path.join(output_dir_section8, 'state_duration_summary.csv'),
                index=False,
            )


            print(f"[LUPE-AMPS] State visualizations saved to: {output_dir_section8}")
            print(f"[LUPE-AMPS] Summary CSVs exported:")
            print(f"  - state_occupancy_summary.csv")
            print(f"  - state_bouts_summary.csv")
            print(f"  - state_duration_summary.csv")
            print(f"[LUPE-AMPS] Per-group figures exported:")
            print(f"  - state_occupancy_line_{{Group}}.png/svg")
            print(f"  - state_bouts_line_{{Group}}.png/svg")
            print(f"  - state_duration_line_{{Group}}.png/svg")

            # Section 9: PCA PROJECTION - AMPS CALCULATION
            print("\n[LUPE-AMPS] Computing AMPS via PCA projection (Section 9)...")

            output_dir_section9 = os.path.join(output_base_dir, 'Section9_AMPS')
            os.makedirs(output_dir_section9, exist_ok=True)

            # Define condition colors FIRST (before try/except)
            unique_conditions_s9 = sorted(list(set(condition_labels)))
            try:
                if sns is not None:
                    palette = sns.color_palette("tab10", len(unique_conditions_s9))
                    condition_colors_s9 = dict(zip(unique_conditions_s9, palette))
                else:
                    raise RuntimeError("seaborn not available")
            except Exception:
                prop_cycle_colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
                if len(prop_cycle_colors) < len(unique_conditions_s9):
                    prop_cycle_colors = (
                                prop_cycle_colors * (len(unique_conditions_s9) // max(1, len(prop_cycle_colors)) + 1))
                condition_colors_s9 = dict(zip(unique_conditions_s9, prop_cycle_colors[: len(unique_conditions_s9)]))

            # NOW load PCA parameters
            print(f"[LUPE-AMPS] Loading PCA parameters from: {pca_model_path}")
            try:
                if pca_model_path is None or str(pca_model_path).strip() == "":
                    raise ValueError("pca_model_path is empty")
                pca_coeff, pca_mu = load_pca_params(str(pca_model_path))
                print(f"[LUPE-AMPS] PCA coeff shape: {pca_coeff.shape}")
                print(f"[LUPE-AMPS] PCA mean shape: {pca_mu.shape}")
            except Exception as e:
                print(f"[LUPE-AMPS] Error loading PCA params: {e}")
                print("[LUPE-AMPS] Fitting PCA on current data as fallback...")
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    pca.fit(total_fraction_state)
                    pca_coeff = pca.components_.T
                    pca_mu = pca.mean_
                except Exception as e2:
                    print(f"[LUPE-AMPS] PCA fallback fit failed: {e2}")
                    pca_coeff, pca_mu = None, None

            if pca_coeff is None or pca_mu is None:
                print("[LUPE-AMPS] Could not compute PCA projection — skipping Section 9.")
            else:
                # ALL OF YOUR AMPS CODE MUST BE INSIDE THIS ELSE BLOCK

                # Project to PC space
                pca_projection = project_to_pc_space(total_fraction_state, pca_coeff, pca_mu)
                general_behavior_scale = pca_projection[:, 0]
                amps = pca_projection[:, 1]

                # Save all AMPS scores - SHORTENED FILENAME
                df_amps = pd.DataFrame({
                    'Animal': animal_names,
                    'Group': group_labels,
                    'Condition': condition_labels,
                    'PC1_GeneralBehaviorScale': general_behavior_scale,
                    'PC2_AMPS': amps,
                })
                df_amps.to_csv(os.path.join(output_dir_section9, 'amps_scores.csv'), index=False)

                # Generate figures PER GROUP
                for group in unique_groups:
                    group_mask = np.array([g == group for g in group_labels])
                    n_animals_group = int(np.sum(group_mask))

                    if n_animals_group == 0:
                        continue

                    group_conditions = sorted(
                        list(set([c for c, g in zip(condition_labels, group_labels) if g == group])))

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    # 9a) Scatter by condition
                    ax = axes[0]
                    for cond in group_conditions:
                        mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                        if np.sum(mask) == 0:
                            continue
                        ax.scatter(
                            general_behavior_scale[mask],
                            amps[mask],
                            c=[condition_colors_s9.get(cond, 'gray')],
                            label=f'{cond} (n={int(np.sum(mask))})',
                            s=50,
                            alpha=0.7,
                        )
                    ax.set_xlabel('PC1: General Behavior Scale')
                    ax.set_ylabel('PC2: Affective-Motivational Pain Scale')
                    ax.set_title(f'PCA Projection — {group}')
                    ax.legend(frameon=False)

                    # 9b) Bar plot for PC1
                    ax = axes[1]
                    means, sems = [], []
                    for cond in group_conditions:
                        mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                        data = general_behavior_scale[mask]
                        means.append(float(np.mean(data)) if len(data) else np.nan)
                        sems.append(float(np.std(data, ddof=1) / np.sqrt(len(data))) if len(data) > 1 else 0.0)

                    x = np.arange(len(group_conditions))
                    ax.bar(x, means, yerr=sems, capsize=5, color='lightgray', edgecolor='black')
                    for i, cond in enumerate(group_conditions):
                        mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                        data = general_behavior_scale[mask]
                        if len(data) > 0:
                            ax.scatter(np.ones(len(data)) * i + 0.15, data, c=[condition_colors_s9.get(cond, 'gray')],
                                       s=30, alpha=0.7)
                    ax.set_xticks(x)
                    ax.set_xticklabels(group_conditions, rotation=45, ha='right')
                    ax.set_ylabel('Score')
                    ax.set_title(f'PC1: General Behavior Scale — {group}')

                    # 9c) Bar plot for AMPS (PC2)
                    ax = axes[2]
                    means, sems = [], []
                    for cond in group_conditions:
                        mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                        data = amps[mask]
                        means.append(float(np.mean(data)) if len(data) else np.nan)
                        sems.append(float(np.std(data, ddof=1) / np.sqrt(len(data))) if len(data) > 1 else 0.0)

                    x = np.arange(len(group_conditions))
                    ax.bar(x, means, yerr=sems, capsize=5, color='lightgray', edgecolor='black')
                    for i, cond in enumerate(group_conditions):
                        mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                        data = amps[mask]
                        if len(data) > 0:
                            ax.scatter(np.ones(len(data)) * i + 0.15, data, c=[condition_colors_s9.get(cond, 'gray')],
                                       s=30, alpha=0.7)
                    ax.set_xticks(x)
                    ax.set_xticklabels(group_conditions, rotation=45, ha='right')
                    ax.set_ylabel('Score')
                    ax.set_title(f'PC2: AMPS (Pain Scale) — {group}')

                    fig.suptitle(f'AMPS Analysis — {group}', fontsize=14, fontweight='bold', y=1.02)
                    fig.tight_layout()

                    # SHORTENED FILENAME
                    fig.savefig(os.path.join(output_dir_section9, f'amps_projection_{group}.png'),
                                dpi=300, bbox_inches='tight')
                    fig.savefig(os.path.join(output_dir_section9, f'amps_projection_{group}.svg'),
                                bbox_inches='tight')
                    plt.close(fig)

                    print(f"[LUPE-AMPS]   - Saved AMPS projection for {group}")

                # Summary CSV - SHORTENED FILENAME
                amps_summary = []
                for group in unique_groups:
                    for cond in unique_conditions_s9:
                        mask = np.array([(g == group and c == cond) for g, c in zip(group_labels, condition_labels)])
                        n_animals_subset = int(np.sum(mask))
                        if n_animals_subset == 0:
                            continue
                        amps_summary.append({
                            'Group': group,
                            'Condition': cond,
                            'PC1_Mean': float(np.mean(general_behavior_scale[mask])),
                            'PC1_SEM': float(np.std(general_behavior_scale[mask], ddof=1) / np.sqrt(
                                n_animals_subset)) if n_animals_subset > 1 else 0.0,
                            'PC2_AMPS_Mean': float(np.mean(amps[mask])),
                            'PC2_AMPS_SEM': float(np.std(amps[mask], ddof=1) / np.sqrt(
                                n_animals_subset)) if n_animals_subset > 1 else 0.0,
                            'N_Animals': n_animals_subset,
                        })

                pd.DataFrame(amps_summary).to_csv(
                    os.path.join(output_dir_section9, 'amps_summary_by_group.csv'),
                    index=False,
                )

                print(f"[LUPE-AMPS] AMPS results saved to: {output_dir_section9}")

                # Section 9b (OPTIONAL): TIMEPOINT COMPARISON — AMPS / PC1
                if (not use_timepoint_comparison) or (not time_ranges_min) or (len(time_ranges_min) == 0):
                    print("[LUPE-AMPS] Timepoint comparison disabled/empty — skipping Section 9b (AMPS).")
                else:
                    if (not time_labels) or (len(time_labels) != len(time_ranges_min)):
                        time_labels = [f"{a:g}-{b:g} min" for (a, b) in time_ranges_min]

                    output_dir_timepoints = os.path.join(output_base_dir, "Timepoint_Comparison")
                    os.makedirs(output_dir_timepoints, exist_ok=True)

                    amps_timepoint_rows = []

                    for label, (start_min, end_min) in zip(time_labels, time_ranges_min):
                        sl = _min_to_slice(float(start_min), float(end_min), float(target_fps), behav_state.shape[0])
                        if sl.stop <= sl.start:
                            print(f"[LUPE-AMPS] Skipping empty window: {label}")
                            continue

                        behav_state_win = behav_state[sl, :]

                        # State occupancy per animal (n_animals x n_states)
                        occ = np.zeros((behav_state_win.shape[1], n_states), dtype=float)
                        for s in range(1, n_states + 1):
                            occ[:, s - 1] = np.mean(behav_state_win == s, axis=0)

                        proj = project_to_pc_space(occ, pca_coeff, pca_mu)
                        pc1_win = proj[:, 0]
                        pc2_win = proj[:, 1]

                        for a_idx, animal in enumerate(animal_names):
                            amps_timepoint_rows.append({
                                'Time_Group': label,
                                'Start_min': float(start_min),
                                'End_min': float(end_min),
                                'Animal': animal,
                                'Group': group_labels[a_idx],
                                'Condition': condition_labels[a_idx],
                                'PC1_GeneralBehaviorScale': float(pc1_win[a_idx]),
                                'PC2_AMPS': float(pc2_win[a_idx]),
                            })

                    if len(amps_timepoint_rows) > 0:
                        df_amps_timepoints = pd.DataFrame(amps_timepoint_rows)
                        # SHORTENED FILENAME
                        df_amps_timepoints.to_csv(
                            os.path.join(output_dir_timepoints, 'amps_timepoint_comparison.csv'),
                            index=False,
                        )
                        print(f"[LUPE-AMPS] Timepoint AMPS comparison saved to: {output_dir_timepoints}")

                        output_dir_timepoints_ampsviz = os.path.join(output_dir_timepoints, 'AMPS_Viz')
                        os.makedirs(output_dir_timepoints_ampsviz, exist_ok=True)

                        metric_dirs = {
                            'PC2_AMPS': os.path.join(output_dir_timepoints_ampsviz, 'PC2_AMPS'),
                            'PC1_GeneralBehaviorScale': os.path.join(output_dir_timepoints_ampsviz, 'PC1_GeneralBehaviorScale'),
                        }
                        for d in metric_dirs.values():
                            os.makedirs(d, exist_ok=True)

                        df_amps_timepoints['Time_Group'] = pd.Categorical(
                            df_amps_timepoints['Time_Group'],
                            categories=time_labels,
                            ordered=True,
                        )

                        def _sem(vals):
                            vals = np.asarray(vals, dtype=float)
                            return float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

                        def _plot_bar_timewindow(df_win, value_col, ylab, title, out_dir, fname_suffix):
                            if df_win is None or df_win.empty:
                                return

                            conds = list(unique_conditions_s9)
                            means, sems = [], []

                            for cond in conds:
                                vals = df_win.loc[df_win['Condition'] == cond, value_col].astype(float).values
                                means.append(float(np.mean(vals)) if len(vals) else np.nan)
                                sems.append(_sem(vals) if len(vals) else 0.0)

                            x = np.arange(len(conds))

                            plt.figure(figsize=(10, 5))
                            plt.bar(x, means, yerr=sems, capsize=5, color='lightgray', edgecolor='black', alpha=0.7)

                            # overlay individual points with jitter
                            for i, cond in enumerate(conds):
                                vals = df_win.loc[df_win['Condition'] == cond, value_col].astype(float).values
                                if len(vals) == 0:
                                    continue
                                jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
                                plt.scatter(
                                    np.ones(len(vals)) * i + jitter,
                                    vals,
                                    c=[condition_colors_s9[cond]],
                                    s=30,
                                    alpha=0.7,
                                    edgecolors='none',
                                )

                            plt.xticks(x, conds, rotation=45, ha='right')
                            plt.xlabel('Condition')
                            plt.ylabel(ylab)
                            plt.title(title)
                            plt.tight_layout()

                            # SHORTENED FILENAME
                            fname = f"timepoint_{fname_suffix}".replace('/', '_').replace(' ', '_').replace('-', '_')
                            plt.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=300)
                            plt.savefig(os.path.join(out_dir, f"{fname}.svg"))
                            plt.close()

                        for tg in time_labels:
                            df_win = df_amps_timepoints[df_amps_timepoints['Time_Group'] == tg].copy()
                            if df_win.empty:
                                continue

                            _plot_bar_timewindow(
                                df_win,
                                value_col='PC2_AMPS',
                                ylab='PC2: AMPS',
                                title=f"AMPS (PC2) — {tg}",
                                out_dir=metric_dirs['PC2_AMPS'],
                                fname_suffix=f"AMPS_PC2_{tg}",
                            )

                            _plot_bar_timewindow(
                                df_win,
                                value_col='PC1_GeneralBehaviorScale',
                                ylab='PC1: General Behavior Scale',
                                title=f"PC1 — {tg}",
                                out_dir=metric_dirs['PC1_GeneralBehaviorScale'],
                                fname_suffix=f"PC1_{tg}",
                            )

                        print(f"[LUPE-AMPS] Timepoint AMPS visualizations saved to: {output_dir_timepoints_ampsviz}")
                    else:
                        print("[LUPE-AMPS] No timepoint AMPS rows to export — skipping Section 9b plots.")

    except Exception as e:
        print("=" * 60)
        print("[LUPE-AMPS] SECTION 4: PREPROCESSING FAILED ❌")
        print("=" * 60)
        print(f"  Error: {e}")
        print("=" * 60)
        raise

    # Over each animal's full recording at its own length (no bins, no padding)
    print("\n[LUPE-AMPS] Calculating behavior statistics...")
    total_frac_time, total_num_bouts, total_mean_dur, animal_n_frames = calculate_behavior_statistics(
        behavior_vectors,
        sampling_rate=sampling_rate,
        n_behaviors=n_behaviors,
        min_frames=min_frames_true_positive,
    )
    animal_duration_min = animal_n_frames / sampling_rate / 60.0

    print(f"[LUPE-AMPS] Behavior statistics calculated for {total_frac_time.shape[0]} animals")

    # Export behavior statistics (per behavior CSV, same as notebook)
    output_dir_section2 = os.path.join(output_base_dir, "Section2_BehaviorStats")
    os.makedirs(output_dir_section2, exist_ok=True)

    for b, beh_name in enumerate(behaviors):
        df = pd.DataFrame(
            {
                "FractionTime": total_frac_time[:, b],
                "NumBouts": total_num_bouts[:, b],
                "MeanDuration_s": total_mean_dur[:, b],
                "n_frames": animal_n_frames,
                "duration_min": animal_duration_min,
                "bouts_per_min": total_num_bouts[:, b] / animal_duration_min,
            },
            index=animal_names,
        )
        df["Group"] = group_labels
        df["Condition"] = condition_labels
        df.to_csv(os.path.join(output_dir_section2, f"{beh_name}_statistics.csv"))

    print(f"[LUPE-AMPS] Behavior statistics exported to: {output_dir_section2}")

    # Section 3: BEHAVIOR VISUALIZATION
    if make_behavior_plots:
        output_dir_section3 = os.path.join(output_base_dir, "Section3_BehaviorViz")
        os.makedirs(output_dir_section3, exist_ok=True)

        behavior_colors = ["red", "orange", "yellow", "green", "blue", "purple"]

        unique_conditions = sorted(list(set(condition_labels)))
        if sns is not None:
            palette = sns.color_palette("tab10", len(unique_conditions))
            condition_colors = dict(zip(unique_conditions, palette))
        else:
            # fallback to matplotlib default cycle
            prop_cycle_colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
            condition_colors = dict(zip(unique_conditions, prop_cycle_colors[: len(unique_conditions)]))

        print(f"[LUPE-AMPS] Conditions (sorted): {unique_conditions}")

        # 3a. Raster: single animal + mean probability across animals
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        animal_of_interest = 0
        time_window = slice(0, min(10000, behavior_matrix.shape[0]))
        frames = np.arange(time_window.start, time_window.stop)

        ax = axes[0]
        for b in range(n_behaviors):
            beh_trace = (behavior_matrix[time_window, animal_of_interest] == b).astype(float)
            beh_trace_plot = beh_trace * 0.5 + b
            beh_trace_plot[beh_trace == 0] = np.nan
            ax.plot(frames, beh_trace_plot, color=behavior_colors[b], linewidth=0.5, label=behaviors[b])
            ax.axhline(y=b, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

        ax.set_ylim(-0.5, n_behaviors + 0.5)
        ax.set_yticks(np.arange(n_behaviors) + 0.25)
        ax.set_yticklabels(behaviors)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Behavior")
        ax.set_title(f"Behavior Raster: {animal_names[animal_of_interest]}")

        ax = axes[1]
        for b in range(n_behaviors):
            beh_prob = (behavior_matrix[time_window, :] == b).astype(float).mean(axis=1)
            ax.plot(frames, beh_prob + b, color=behavior_colors[b], linewidth=0.5, label=behaviors[b])
            ax.axhline(y=b, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
            ax.axhline(y=b + 1, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

        ax.set_ylim(-0.5, n_behaviors + 0.5)
        ax.set_yticks(np.arange(n_behaviors) + 0.5)
        ax.set_yticklabels(behaviors)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Behavior (P = probability)")
        ax.set_title("Mean Behavior Probability Across All Animals")

        plt.tight_layout()
        # SHORTENED FILENAME
        plt.savefig(os.path.join(output_dir_section3, "behavior_raster.png"), dpi=300)
        plt.savefig(os.path.join(output_dir_section3, "behavior_raster.svg"))
        plt.close(fig)

        # 3b. Probability over time by GROUP
        unique_groups = sorted(list(set(group_labels)))
        time_window_prob = slice(0, min(10000, behavior_matrix.shape[0]))

        fig, axes = plt.subplots(
            n_behaviors,
            max(1, len(unique_groups)),
            figsize=(5 * max(1, len(unique_groups)), 3 * n_behaviors),
        )
        if len(unique_groups) == 1:
            axes = np.array(axes).reshape(-1, 1)

        if sns is not None:
            group_colors = dict(zip(unique_groups, sns.color_palette("Set1", len(unique_groups))))
        else:
            prop_cycle_colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
            group_colors = dict(zip(unique_groups, prop_cycle_colors[: len(unique_groups)]))

        for b in range(n_behaviors):
            for g_idx, group in enumerate(unique_groups):
                ax = axes[b, g_idx]

                group_mask = np.array([gl == group for gl in group_labels])
                group_data = behavior_matrix[time_window_prob, :][:, group_mask]

                if group_data.size == 0:
                    ax.set_axis_off()
                    continue

                beh_binary = (group_data == b).astype(float)
                mean_beh = np.mean(beh_binary, axis=1)

                # SEM guarded for n=1
                if beh_binary.shape[1] > 1:
                    sem_beh = np.std(beh_binary, axis=1, ddof=1) / np.sqrt(beh_binary.shape[1])
                else:
                    sem_beh = np.zeros_like(mean_beh)

                fr = np.arange(len(mean_beh))
                ax.plot(fr, mean_beh, color=group_colors[group], linewidth=0.5)
                ax.fill_between(fr, mean_beh - sem_beh, mean_beh + sem_beh, color=group_colors[group], alpha=0.2)

                ax.set_ylim(0, 1)
                ax.set_xlabel("Frame")
                ax.set_ylabel("P(behavior)")
                ax.set_title(f"{group} - {behaviors[b]}")

        plt.tight_layout()
        # SHORTENED FILENAME
        plt.savefig(os.path.join(output_dir_section3, "behavior_probability_by_group.png"), dpi=300)
        plt.savefig(os.path.join(output_dir_section3, "behavior_probability_by_group.svg"))
        plt.close(fig)

        # 3c. Condition barplots (FractionTime / NumBouts / MeanDuration)
        def _barplot_metric(metric_matrix: np.ndarray, ylabel: str, out_stub: str):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for b, beh_name in enumerate(behaviors):
                ax = axes[b]

                means = []
                sems = []

                for cond in unique_conditions:
                    mask = np.array([c == cond for c in condition_labels])
                    data = metric_matrix[mask, b]
                    means.append(float(np.mean(data)) if len(data) else 0.0)
                    sems.append(float(np.std(data, ddof=1) / np.sqrt(len(data))) if len(data) > 1 else 0.0)

                x = np.arange(len(unique_conditions))
                ax.bar(x, means, yerr=sems, capsize=5, color="lightgray", edgecolor="black", alpha=0.7)

                for i, cond in enumerate(unique_conditions):
                    mask = np.array([c == cond for c in condition_labels])
                    data = metric_matrix[mask, b]
                    jitter = np.random.uniform(-0.15, 0.15, size=len(data)) if len(data) else np.array([])
                    if len(data):
                        ax.scatter(
                            np.ones(len(data)) * i + jitter,
                            data,
                            c=[condition_colors[cond]],
                            s=30,
                            alpha=0.7,
                            edgecolors="none",
                        )

                ax.set_xticks(x)
                ax.set_xticklabels(unique_conditions, rotation=45, ha="right")
                ax.set_ylabel(ylabel)
                ax.set_title(beh_name)

            plt.tight_layout()
            # SHORTENED FILENAME
            plt.savefig(os.path.join(output_dir_section3, f"{out_stub}.png"), dpi=300)
            plt.savefig(os.path.join(output_dir_section3, f"{out_stub}.svg"))
            plt.close(fig)

        _barplot_metric(total_frac_time, "Fraction Time", "behavior_fraction_by_condition")
        _barplot_metric(total_num_bouts, "Number of Bouts", "behavior_bouts_by_condition")
        _barplot_metric(total_mean_dur, "Bout Duration (s)", "behavior_duration_by_condition")

        print(f"[LUPE-AMPS] Behavior visualizations saved to: {output_dir_section3}")
    else:
        print("[LUPE-AMPS] Section 3 skipped (make_behavior_plots=False)")

    # Section 2b (OPTIONAL): TIMEPOINT COMPARISON — BEHAVIOR STATS
    output_dir_timepoints = os.path.join(output_base_dir, "Timepoint_Comparison")
    os.makedirs(output_dir_timepoints, exist_ok=True)

    df_behavior_timepoints = None

    if not use_timepoint_comparison:
        print("[LUPE-AMPS] Timepoint comparison disabled — skipping Section 2b/2c.")
    elif not time_ranges_min or len(time_ranges_min) == 0:
        print(
            "[LUPE-AMPS] Timepoint comparison was enabled, but no valid time windows were received "
            "(time_ranges_min is empty/None). Skipping Section 2b/2c."
        )
    else:
        # If labels not passed, generate them
        if not time_labels or len(time_labels) != len(time_ranges_min):
            time_labels = [f"{a:g}-{b:g} min" for (a, b) in time_ranges_min]

        behavior_timepoint_rows = []

        for label, (start_min, end_min) in zip(time_labels, time_ranges_min):
            # Each animal's own frames in the window (no padding); a window past the end of a
            # shorter recording covers only the frames that animal has
            vecs_win = [v[_min_to_slice(start_min, end_min, sampling_rate, len(v))] for v in behavior_vectors]
            if all(len(v) == 0 for v in vecs_win):
                print(f"[LUPE-AMPS] Skipping empty window: {label}")
                continue

            tf_win, nb_win, bd_win, nf_win = calculate_behavior_statistics(
                vecs_win,
                sampling_rate=sampling_rate,
                n_behaviors=n_behaviors,
                min_frames=min_frames_true_positive,
            )
            dur_win = nf_win / sampling_rate / 60.0

            for a_idx, animal in enumerate(animal_names):
                for b_idx, beh_name in enumerate(behaviors):
                    behavior_timepoint_rows.append(
                        {
                            "Time_Group": label,
                            "Start_min": float(start_min),
                            "End_min": float(end_min),
                            "Animal": animal,
                            "Group": group_labels[a_idx],
                            "Condition": condition_labels[a_idx],
                            "Behavior": beh_name,
                            "FractionTime": float(tf_win[a_idx, b_idx]),
                            "NumBouts": float(nb_win[a_idx, b_idx]),
                            "MeanDuration_s": float(bd_win[a_idx, b_idx]),
                            "n_frames": int(nf_win[a_idx]),
                            "duration_min": float(dur_win[a_idx]),
                            "bouts_per_min": (float(nb_win[a_idx, b_idx] / dur_win[a_idx])
                                              if nf_win[a_idx] > 0 else float("nan")),
                        }
                    )

        if len(behavior_timepoint_rows) > 0:
            df_behavior_timepoints = pd.DataFrame(behavior_timepoint_rows)
            # SHORTENED FILENAME
            df_behavior_timepoints.to_csv(
                os.path.join(output_dir_timepoints, "behavior_timepoint_comparison.csv"),
                index=False,
            )
            print(f"[LUPE-AMPS] Timepoint behavior comparison saved to: {output_dir_timepoints}")
        else:
            print("[LUPE-AMPS] No timepoint behavior rows to export.")

    # Section 2c (OPTIONAL): TIMEPOINT VISUALIZATION — BEHAVIOR STATS
    if make_timepoint_plots and df_behavior_timepoints is not None and len(df_behavior_timepoints) > 0:
        output_dir_timepoints_behviz = os.path.join(output_dir_timepoints, "Behavior_Viz")
        os.makedirs(output_dir_timepoints_behviz, exist_ok=True)

        metric_dirs = {
            "FractionTime": os.path.join(output_dir_timepoints_behviz, "FractionTime"),
            "NumBouts": os.path.join(output_dir_timepoints_behviz, "NumBouts"),
            "MeanDuration_s": os.path.join(output_dir_timepoints_behviz, "MeanDuration"),
        }
        for d in metric_dirs.values():
            os.makedirs(d, exist_ok=True)

        # enforce categorical ordering for x-axis
        df_behavior_timepoints["Time_Group"] = pd.Categorical(
            df_behavior_timepoints["Time_Group"], categories=time_labels, ordered=True
        )

        metrics_to_plot = [
            ("FractionTime", "Fraction Time"),
            ("NumBouts", "Number of Bouts"),
            ("MeanDuration_s", "Mean Bout Duration (s)"),
        ]

        def _sem(x):
            return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

        unique_groups = list(selected_groups)
        unique_conditions = list(selected_conditions)

        for beh in behaviors:
            df_b = df_behavior_timepoints[df_behavior_timepoints["Behavior"] == beh].copy()
            if df_b.empty:
                continue

            summary = (
                df_b.groupby(["Group", "Condition", "Time_Group"], observed=True)
                .agg(
                    mean_FractionTime=("FractionTime", "mean"),
                    sem_FractionTime=("FractionTime", _sem),
                    mean_NumBouts=("NumBouts", "mean"),
                    sem_NumBouts=("NumBouts", _sem),
                    mean_MeanDuration_s=("MeanDuration_s", "mean"),
                    sem_MeanDuration_s=("MeanDuration_s", _sem),
                    n=("FractionTime", "size"),
                )
                .reset_index()
            )

            for group in unique_groups:
                summary_g = summary[summary["Group"] == group].copy()
                if summary_g.empty:
                    continue

                for metric_key, metric_label in metrics_to_plot:
                    mean_col = f"mean_{metric_key}"
                    sem_col = f"sem_{metric_key}"

                    plt.figure(figsize=(10, 5))

                    for cond in unique_conditions:
                        sdf = summary_g[summary_g["Condition"] == cond].sort_values("Time_Group")
                        if sdf.empty:
                            continue

                        # Align to all time_labels
                        sdf = sdf.set_index("Time_Group").reindex(time_labels).reset_index()

                        x = np.arange(len(time_labels))
                        y = sdf[mean_col].to_numpy(dtype=float)
                        e = sdf[sem_col].to_numpy(dtype=float)

                        n_line = int(np.nanmax(sdf["n"].to_numpy(dtype=float))) if np.any(~np.isnan(sdf["n"])) else 0
                        line_label = f"{cond} (n={n_line})"

                        plt.plot(x, y, marker="o", label=line_label)
                        plt.fill_between(x, y - e, y + e, alpha=0.2)

                    plt.xticks(np.arange(len(time_labels)), time_labels, rotation=45, ha="right")
                    plt.xlabel("Time Window")
                    plt.ylabel(metric_label)
                    plt.title(f"{beh} — {metric_label} by Time Window\nGroup: {group}")
                    plt.legend(frameon=False)
                    plt.tight_layout()

                    # SHORTENED FILENAME
                    fname = f"timepoint_{group}_{beh}_{metric_key}".replace("/", "_").replace(" ", "_")
                    png_path = os.path.join(metric_dirs[metric_key], f"{fname}.png")
                    svg_path = os.path.join(metric_dirs[metric_key], f"{fname}.svg")

                    plt.savefig(png_path, dpi=300)
                    plt.savefig(svg_path)
                    plt.close()

        print(f"[LUPE-AMPS] Timepoint behavior visualizations saved to: {output_dir_timepoints_behviz}")
    else:
        print("[LUPE-AMPS] Skipping Section 2c: df_behavior_timepoints not available or plots disabled.")

    # Return something lightweight (optional)
    return {
        "output_base_dir": output_base_dir,
        "output_dir_section2": output_dir_section2,
        "output_dir_timepoints": output_dir_timepoints,
        "n_animals": int(behavior_matrix.shape[1]),
        "n_frames_expected": int(behavior_matrix.shape[0]),
        "n_frames_downsampled": int(behavior_ds.shape[0]),
        "transition_matrices": trans_unfolded,
        "transition_window_starts": wins,
        "n_transition_windows": int(n_wins),
        "transition_window_size_frames": int(window_size),
        "transition_window_slide_frames": int(window_slide),
        "model_path_used": (None if (model_path is None or str(model_path).strip() == "") else str(model_path)),
        "pca_model_path_used": (None if (pca_model_path is None or str(pca_model_path).strip() == "") else str(pca_model_path)),
        "state_model_ran": bool(model_path is not None and str(model_path).strip() != "" and os.path.exists(str(model_path))),
    }