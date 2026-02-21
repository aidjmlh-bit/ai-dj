"""Build a metadata DataFrame for a folder of WAV audio files.

Each WAV file is analysed for:
  - Musical key         (get_key)
  - Structural sections (get_sections)
  - Global BPM          (get_bpm)
  - Per-section BPM     (computed from audio slices between section timestamps)

The result is a tidy, long-format DataFrame with one row per (track, section).

Note: get_key and get_sections are expected to accept a filepath argument
once their implementations are complete.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pandas as pd

from bpm import get_bpm
from sections import get_sections

# get_key lives in the get_key/ sub-folder; add it to the path so its
# internal imports (key_profiles, camelot) resolve correctly too.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "get_key"))
from get_key import get_key  # noqa: E402


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_COLUMNS = [
    "filename",
    "filepath",
    "key_code",
    "key_name",
    "global_bpm",
    "section",
    "section_start",
    "section_end",
    "section_bpm",
]


def _bpm_from_array(y: np.ndarray, sr: int, start_bpm: float = 120.0) -> float:
    """Estimate BPM directly from a mono audio array.

    Uses the same onset-strength + dynamic-programming algorithm as
    ``get_bpm`` but operates on an in-memory slice so section segments
    never need to be written to disk.

    Args:
        y: Mono audio time-series (numpy float32/64 array).
        sr: Sample rate in Hz.
        start_bpm: Weak BPM prior passed to the beat tracker.

    Returns:
        BPM rounded to 2 decimal places, or ``nan`` if the segment is
        too short or the estimate falls outside [60, 200].
    """
    # Need at least ~4 seconds for a reliable estimate.
    if y.size < sr * 4:
        return float("nan")

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, _ = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        start_bpm=start_bpm,
        units="frames",
    )
    bpm = round(float(np.atleast_1d(tempo)[0]), 2)
    return bpm if 60.0 <= bpm <= 200.0 else float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_metadata(
    folder_path: str,
    output_dir: str = "/Users/alishasrivastava/Desktop/ai-dj/data/results",
) -> pd.DataFrame:
    """Build a metadata DataFrame for all WAV files in *folder_path*.

    For every ``.wav`` file the function:

    1. Calls ``get_bpm(filepath)`` for the global tempo.
    2. Calls ``get_key(filepath)`` → ``(key_code: int, key_name: str)``.
    3. Calls ``get_sections(filepath)`` → sequence of
       ``(section_label: str, timestamp: float)`` tuples, where timestamps
       are in seconds from the start of the track.
    4. Slices the audio between consecutive section timestamps and estimates
       a per-section BPM for each slice.

    The returned DataFrame is in **long format**: one row per
    ``(track, section)`` pair.

    Args:
        folder_path: Path to a directory containing ``.wav`` files.
            Sub-directories are not searched.
        output_dir: Directory where ``metadata.csv`` is written. Created
            automatically if it does not exist. Pass ``None`` to skip
            saving.

    Returns:
        ``pd.DataFrame`` with columns:

        +-----------------+-----------------------------------------------+
        | Column          | Description                                   |
        +=================+===============================================+
        | filename        | Base name of the WAV file                     |
        | filepath        | Absolute path to the WAV file                 |
        | key_code        | Integer key identifier from ``get_key``       |
        | key_name        | Key string, e.g. ``"C Major"``                |
        | global_bpm      | BPM for the full track                        |
        | section         | Section label, e.g. ``"Intro"``, ``"Chorus"``|
        | section_start   | Section start time in seconds                 |
        | section_end     | Section end time in seconds (NaN for last)    |
        | section_bpm     | BPM estimated for this section's audio slice  |
        +-----------------+-----------------------------------------------+

        Failed per-file analyses are skipped with a warning; failed
        per-field analyses yield ``NaN`` / ``None`` for that field only.

    Raises:
        FileNotFoundError: If *folder_path* does not exist.
        NotADirectoryError: If *folder_path* is not a directory.

    Example:
        >>> df = create_metadata("/path/to/wav/folder")
        >>> print(df[["filename", "section", "section_bpm"]])
    """
    folder = Path(folder_path).resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path!r}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path!r}")

    wav_files = sorted(folder.glob("*.wav"))
    if not wav_files:
        warnings.warn(f"No .wav files found in {folder_path!r}", UserWarning, stacklevel=2)
        return pd.DataFrame(columns=_COLUMNS)

    rows: List[dict] = []

    for wav_path in wav_files:
        filepath_str = str(wav_path)
        filename = wav_path.name

        print(f"  Processing: {filename}")

        # ------------------------------------------------------------------ #
        # Global BPM                                                           #
        # ------------------------------------------------------------------ #
        try:
            global_bpm = get_bpm(filepath_str)
        except Exception as exc:
            warnings.warn(f"[{filename}] get_bpm failed: {exc}", stacklevel=2)
            global_bpm = float("nan")

        # ------------------------------------------------------------------ #
        # Musical key  —  expects get_key(filepath) -> (int, str)             #
        # ------------------------------------------------------------------ #
        try:
            key_code, key_name = get_key(filepath_str)
        except TypeError:
            # Stub not yet updated to accept filepath; call without argument.
            try:
                key_code, key_name = get_key()
            except Exception as exc:
                warnings.warn(f"[{filename}] get_key failed: {exc}", stacklevel=2)
                key_code, key_name = None, None
        except Exception as exc:
            warnings.warn(f"[{filename}] get_key failed: {exc}", stacklevel=2)
            key_code, key_name = None, None

        # ------------------------------------------------------------------ #
        # Structural sections  —  expects get_sections(filepath)              #
        #   -> sequence of (section_label: str, timestamp: float) tuples      #
        # ------------------------------------------------------------------ #
        try:
            sections = get_sections(filepath_str)
        except TypeError:
            # Stub not yet updated to accept filepath.
            try:
                sections = get_sections()
            except Exception as exc:
                warnings.warn(f"[{filename}] get_sections failed: {exc}", stacklevel=2)
                sections = ()
        except Exception as exc:
            warnings.warn(f"[{filename}] get_sections failed: {exc}", stacklevel=2)
            sections = ()

        # ------------------------------------------------------------------ #
        # Load audio once for per-section BPM slicing                        #
        # ------------------------------------------------------------------ #
        y, sr, track_duration = None, None, float("nan")
        try:
            y, sr = librosa.load(filepath_str, mono=True, sr=None)
            track_duration = len(y) / sr
        except Exception as exc:
            warnings.warn(
                f"[{filename}] librosa.load failed; section BPMs will be NaN: {exc}",
                stacklevel=2,
            )

        # ------------------------------------------------------------------ #
        # Emit one row per section                                            #
        # ------------------------------------------------------------------ #
        if not sections:
            rows.append(_make_row(
                filename, filepath_str, key_code, key_name, global_bpm,
                section=None,
                section_start=float("nan"),
                section_end=float("nan"),
                section_bpm=float("nan"),
            ))
            continue

        for i, (section_label, raw_start) in enumerate(sections):
            try:
                start_sec = float(raw_start)
            except (TypeError, ValueError):
                start_sec = float("nan")

            # End = start of next section, or track duration for the last one.
            if i + 1 < len(sections):
                try:
                    end_sec = float(sections[i + 1][1])
                except (TypeError, ValueError):
                    end_sec = float("nan")
            else:
                end_sec = track_duration

            # Per-section BPM from the sliced waveform.
            section_bpm = float("nan")
            if y is not None and not (np.isnan(start_sec) or np.isnan(end_sec)):
                start_sample = int(start_sec * sr)
                end_sample = int(end_sec * sr)
                y_slice = y[start_sample:end_sample]
                prior = global_bpm if not np.isnan(global_bpm) else 120.0
                section_bpm = _bpm_from_array(y_slice, sr, start_bpm=prior)

            rows.append(_make_row(
                filename, filepath_str, key_code, key_name, global_bpm,
                section=section_label,
                section_start=start_sec,
                section_end=end_sec,
                section_bpm=section_bpm,
            ))

    df = pd.DataFrame(rows, columns=_COLUMNS)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved metadata to: {csv_path}")

    return df


def _make_row(
    filename, filepath, key_code, key_name, global_bpm,
    section, section_start, section_end, section_bpm,
) -> dict:
    return {
        "filename": filename,
        "filepath": filepath,
        "key_code": key_code,
        "key_name": key_name,
        "global_bpm": global_bpm,
        "section": section,
        "section_start": section_start,
        "section_end": section_end,
        "section_bpm": section_bpm,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    folder = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/alishasrivastava/Desktop/ai-dj/data/wav"
    )
    print(f"Building metadata for: {folder}\n")
    df = create_metadata(folder)
    print(df.to_string(index=False))
    print(f"\nTotal rows: {len(df)} | Tracks: {df['filename'].nunique()}")
