import numpy as np
import librosa
from keyProfiles import MAJOR_PROFILE, MINOR_PROFILE, PITCH_CLASSES
from camelot import get_camelot_code, parse_camelot

def detect_key(filepath):
    # Load audio
    print(f"Loading: {filepath}")
    y, sr = librosa.load(filepath, mono=True, sr=None)

    # Extract chroma with higher resolution
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=24)

    # Sum across time (instead of mean)
    chroma_vals = [np.sum(chroma[i]) for i in range(12)]

    # Build frequency dictionary
    keyfreqs = {PITCH_CLASSES[i]: chroma_vals[i] for i in range(12)}

    # Compute correlations against all 24 keys
    maj_corrs = []
    min_corrs = []

    for i in range(12):
        key_test = [keyfreqs.get(PITCH_CLASSES[(i + m) % 12]) for m in range(12)]
        maj_corrs.append(round(np.corrcoef(MAJOR_PROFILE, key_test)[1, 0], 3))
        min_corrs.append(round(np.corrcoef(MINOR_PROFILE, key_test)[1, 0], 3))

    # Build key dictionary
    keys = [p + ' major' for p in PITCH_CLASSES] + \
           [p + ' minor' for p in PITCH_CLASSES]
    key_dict = {**{keys[i]: maj_corrs[i] for i in range(12)},
                **{keys[i+12]: min_corrs[i] for i in range(12)}}

    # Find best key
    best_key = max(key_dict, key=key_dict.get)
    best_corr = max(key_dict.values())

    # Find alternate key if close
    alt_key = None
    for key, corr in key_dict.items():
        if corr > best_corr * 0.9 and corr != best_corr:
            alt_key = key

    # Convert to Camelot
    camelot_code = get_camelot_code(best_key)
    camelot_number, camelot_letter = parse_camelot(camelot_code)

    result = {
        "key_name": best_key,
        "camelot_code": camelot_code,
        "camelot_number": camelot_number,
        "camelot_letter": camelot_letter,
        "confidence": best_corr,
        "alternate_key": alt_key
    }

    print(f"Detected key: {best_key} ({camelot_code}) — confidence: {best_corr}")
    if alt_key:
        print(f"Also possible: {alt_key}")

    return result


def get_key(filepath: str):
    result = detect_key(filepath)
    return result["camelot_number"], result["key_name"]