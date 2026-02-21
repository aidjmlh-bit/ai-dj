# Given a song pair, of two wav files, we need to find the key moments in each song, and then find the best transition point between the two songs.
# we need to select the best transition type
    # reverb, crossfade, low cut echo
    # apply the transition effect to the two songs at the transition point, and save the result as a new wav file

# Transition Algorithm
# 1. Check if the two songs have compatible BPMs (constraint: within 15 BPM) and keys (either the same letter and number, or the same letter and different number (maximum difference is +- 1), or same number different letters camelot code logic) (functions provided by bpm.py and key.py)
# 2. Analyze the two songs to find their key moments, using tempo curves and structural segmentation (function provided by sections.py)