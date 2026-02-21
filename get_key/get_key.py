# detect_key.py
# The brain of the key folder.
# Loads a .wav file, extracts chroma features, compares to key profiles,
# and returns the detected key + Camelot code.

import numpy as np
import librosa
from key_profiles import MAJOR_PROFILE, MINOR_PROFILE, PITCH_CLASSES
from camelot import get_camelot_code, parse_camelot


def cosine_similarity(vec_a, vec_b):
    """
    Measures how similar two vectors are. Then returns a value between -1 and 1.
    
    Rule:
    1.0 = identical direction (perfect match)
    0.0 = no relationship
    -1.0 = opposite
    """
    dot_product = np.dot(vec_a, vec_b)
    magnitude = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if magnitude == 0:
        return 0
    return dot_product / magnitude


def rotate_profile(profile, steps):
    """
    Shifts a key profile by a number of semitones.
    This lets us compare one profile against all 12 possible roots.
    Example: rotating C major profile by 2 steps gives D major profile.
    """
    return profile[steps:] + profile[:steps]


def detect_key(filepath):
    """
    Main function. Takes a path to a .wav file.
    Returns a dictionary with the detected key information.

    Steps:
    1. Load audio
    2. Extract chroma (12 pitch class energies)
    3. Average chroma across the whole song
    4. Compare against all 24 keys (12 major + 12 minor) using cosine similarity
    5. Pick the best matching key
    6. Convert to Camelot code
    """

    # Step 1: Load audio
    print(f"Loading: {filepath}")
    y, sr = librosa.load(filepath, mono=True)

    # Step 2: Extract chroma features
    # chroma is a 12 x time matrix
    # Each row = one pitch class (C, C#, D, etc.)
    # Each column = one time frame
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Step 3: Average across time to get one 12-value vector
    avg_chroma = np.mean(chroma, axis=1)

    # Step 4: Compare against 24 possible keys (all)
    best_key = None
    best_score = -1
    best_quality = None  # major or minor

    for i, pitch in enumerate(PITCH_CLASSES):

        # Rotate the profiles to match this pitch class as root
        major_profile = rotate_profile(MAJOR_PROFILE, i)
        minor_profile = rotate_profile(MINOR_PROFILE, i)

        # Compare chroma to major profile
        major_score = cosine_similarity(avg_chroma, major_profile)
        if major_score > best_score:
            best_score = major_score
            best_key = f"{pitch} major"
            best_quality = "major"

        # Compare chroma to minor profile
        minor_score = cosine_similarity(avg_chroma, minor_profile)
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{pitch} minor"
            best_quality = "minor"

    # Step 5: Convert to Camelot code
    camelot_code = get_camelot_code(best_key)
    camelot_number, camelot_letter = parse_camelot(camelot_code)

    # Step 6: Return result
    result = {
        "key_name": best_key,
        "quality": best_quality,      # major or minor
        "camelot_code": camelot_code, # e.g. "8A"
        "camelot_number": camelot_number,  # e.g. 8
        "camelot_letter": camelot_letter,  # e.g. "A"
        "confidence": round(best_score, 4)
    }

    print(f"Detected key: {best_key} ({camelot_code}) — confidence: {best_score:.4f}")
    return result


# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect_key.py path/to/song.wav")
    else:
        result = detect_key(sys.argv[1])
        print(result)

def get_key(filepath: str):
    """Return the Camelot number and key name for a WAV file.

    Args:
        filepath: Path to a ``.wav`` audio file.

    Returns:
        ``(camelot_number, key_name)`` — e.g. ``(8, "A minor")``.
    """
    result = detect_key(filepath)
    return result["camelot_number"], result["key_name"]

