# key_profiles.py
# These are the Krumhansl-Schmuckler key profiles.
# They are fixed 12-dimensional vectors representing how strongly
# each of the 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
# appears in major vs minor keys.

MAJOR_PROFILE = [
    6.35,  # C
    2.23,  # C#
    3.48,  # D
    2.33,  # D#
    4.38,  # E
    4.09,  # F
    2.52,  # F#
    5.19,  # G
    2.39,  # G#
    3.66,  # A
    2.29,  # A#
    2.88,  # B
]

MINOR_PROFILE = [
    6.33,  # C
    2.68,  # C#
    3.52,  # D
    5.38,  # D#
    2.60,  # E
    3.53,  # F
    2.54,  # F#
    4.75,  # G
    3.98,  # G#
    2.69,  # A
    3.34,  # A#
    3.17,  # B
]

# The 12 pitch class names in order
# Index 0 = C, Index 1 = C#, Index 2 = D, etc.
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]