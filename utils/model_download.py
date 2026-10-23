"""
Download of the LUPE classifier model from the LUPE 2.0 App GitHub release.

The model files are too large for the git repository, so they are attached to a GitHub release
of LUPE-2.0-App and downloaded on first use into the model/ folder:

    model.pkl            the A-SOiD random forest, used by the App (needs scikit-learn 1.2.x)
    model_portable.npz   the same forest stored as plain arrays, used by the notebooks and
                         Google Colab (any scikit-learn version, see utils/model_compat.py
                         in LUPE-2.0-NotebookAnalysisPackage)

Each download is checked against the sha256 below before it replaces anything in model/.
This file is identical in LUPE-2.0-App and LUPE-2.0-NotebookAnalysisPackage.
"""

import hashlib
import os
import urllib.request

RELEASE_URL = "https://github.com/justin05423/LUPE-2.0-App/releases/download/v2.3.0"

# file name -> (sha256, size in bytes)
MODEL_FILES = {
    "model.pkl": ("200ee368c484f910a75cb472a19760d32e4f15ab9f6748dda1bddeb101d236e0", 308304039),
    "model_portable.npz": ("4cb57a5f1103e52b9d00db91363c185fcbcce4c96316161ae951b00e89250594", 56939186),
}


def model_url(name):
    return f"{RELEASE_URL}/{name}"


def is_lfs_pointer(path):
    """True if path is a Git LFS pointer (a few lines of text) instead of the real file."""
    if os.path.getsize(path) > 1024:
        return False
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model_file(name, model_dir, progress=None):
    """
    Download one model file from the release into model_dir and check its sha256.

    progress: optional callable(bytes_done, bytes_total), called as the download runs.
    The file is written to <name>.part and only moved into place once the sha256 matches.
    """
    expected_sha, expected_size = MODEL_FILES[name]
    url = model_url(name)
    dest = os.path.join(model_dir, name)
    tmp = dest + ".part"
    os.makedirs(model_dir, exist_ok=True)

    h = hashlib.sha256()
    done = 0
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length") or expected_size)
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total)

    if h.hexdigest() != expected_sha:
        os.remove(tmp)
        raise RuntimeError(
            f"Downloaded {name} failed the sha256 check (got {h.hexdigest()}, expected {expected_sha}). "
            f"Try again, or download it manually from {url} into {model_dir}."
        )
    os.replace(tmp, dest)
    return dest


def ensure_model_file(name, model_dir, progress=None):
    """
    Make sure model_dir/<name> is present, downloading it if it is missing or only a Git LFS pointer.

    Returns (path, status), status one of:
      "ok"          present and the sha256 matches
      "downloaded"  was missing (or an LFS pointer) and has been downloaded and checked
      "mismatch"    present but the sha256 differs (e.g. a retrained model); left as is
    """
    path = os.path.join(model_dir, name)
    if not os.path.exists(path) or is_lfs_pointer(path):
        download_model_file(name, model_dir, progress=progress)
        return path, "downloaded"
    if sha256_file(path) != MODEL_FILES[name][0]:
        return path, "mismatch"
    return path, "ok"


def print_progress(done, total):
    """Progress callback for scripts and notebooks: prints every 10%."""
    pct = int(100 * done / total) if total else 100
    last = getattr(print_progress, "_last", -10)
    if pct >= last + 10 or done >= total:
        print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB ({pct}%)")
        print_progress._last = pct if done < total else -10
