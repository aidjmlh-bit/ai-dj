# camelot.py
# Converts detected key names into Camelot wheel codes.
# Also computes compatibility distance between two songs.

# Full mapping of every key to its Camelot code
# Format: "Note Quality" -> "NumberLetter"
# A = minor, B = major
KEY_TO_CAMELOT = {
    "C major":  "8B",
    "C minor":  "5A",
    "C# major": "3B",
    "C# minor": "12A",
    "D major":  "10B",
    "D minor":  "7A",
    "D# major": "5B",
    "D# minor": "2A",
    "E major":  "12B",
    "E minor":  "9A",
    "F major":  "7B",
    "F minor":  "4A",
    "F# major": "2B",
    "F# minor": "11A",
    "G major":  "9B",
    "G minor":  "6A",
    "G# major": "4B",
    "G# minor": "1A",
    "A major":  "11B",
    "A minor":  "8A",
    "A# major": "6B",
    "A# minor": "3A",
    "B major":  "1B",
    "B minor":  "10A",
}


def get_camelot_code(key_name):
    """
    Takes a key name like "A minor" or "C major"
    Returns the Camelot code like "8A" or "8B"
    """
    return KEY_TO_CAMELOT.get(key_name, None)


def parse_camelot(code):
    """
    Splits a Camelot code like "8A" into its number and letter.
    Returns: (number as int, letter as string)
    Example: "8A" -> (8, "A")
    """
    number = int(code[:-1])
    letter = code[-1]
    return number, letter


def camelot_compatibility(code_a, code_b):
    """
    Computes how compatible two Camelot codes are.

    Rules:
    - Same code        -> distance 0  (perfect match)
    - Same number, different letter -> distance 0.5 (relative major/minor)
    - Number differs by 1, same letter -> distance 1  (adjacent on wheel)
    - Everything else  -> distance 2+ (not ideal)

    Returns a compatibility score between 0.0 and 1.0
    where 1.0 = perfect and 0.0 = very incompatible
    """
    num_a, let_a = parse_camelot(code_a)
    num_b, let_b = parse_camelot(code_b)

    # Circular distance on the wheel (1 through 12 wraps around)
    raw_diff = abs(num_a - num_b)
    circular_diff = min(raw_diff, 12 - raw_diff)

    same_letter = (let_a == let_b)

    # Score based on DJ Camelot mixing rules
    if circular_diff == 0 and same_letter:
        # Perfect: same key
        score = 1.0
    elif circular_diff == 0 and not same_letter:
        # Relative major/minor: same notes, different feel
        score = 0.9
    elif circular_diff == 1 and same_letter:
        # Adjacent on wheel: fifth relationship, very stable
        score = 0.8
    elif circular_diff == 1 and not same_letter:
        # Adjacent + letter switch: acceptable
        score = 0.6
    elif circular_diff == 2:
        # Two steps away: risky but sometimes done
        score = 0.3
    else:
        # Too far apart: not recommended
        score = 0.0

    return score


def get_transition_advice(code_a, code_b):
    """
    Returns a plain English description of how compatible two keys are
    and what transition strategy to use.
    """
    score = camelot_compatibility(code_a, code_b)

    if score >= 0.8:
        strategy = "Smooth harmonic crossfade recommended."
    elif score >= 0.6:
        strategy = "Harmonic blend possible. Use short crossfade."
    elif score >= 0.3:
        strategy = "Harmonic clash likely. Transition during percussion-only section."
    else:
        strategy = "Keys are incompatible. Use hard cut at drop or echo-out transition."

    return {
        "song_a_key": code_a,
        "song_b_key": code_b,
        "compatibility_score": score,
        "strategy": strategy
    }