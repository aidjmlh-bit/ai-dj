# test_key.py
# Run this to verify all 3 key files work correctly

from keyProfiles import MAJOR_PROFILE, MINOR_PROFILE, PITCH_CLASSES
from camelot import get_camelot_code, camelot_compatibility, get_transition_advice
from get_key import get_key

# --- Test 1: keyProfiles.py ---
print("TEST 1: keyProfiles.py")
print(f"Major profile has {len(MAJOR_PROFILE)} values (should be 12): {len(MAJOR_PROFILE) == 12}")
print(f"Minor profile has {len(MINOR_PROFILE)} values (should be 12): {len(MINOR_PROFILE) == 12}")
print(f"Pitch classes has {len(PITCH_CLASSES)} values (should be 12): {len(PITCH_CLASSES) == 12}")
print()

# --- Test 2: camelot.py ---
print("TEST 2: camelot.py")
print(f"A minor should be 8A: {get_camelot_code('A minor')}")
print(f"C major should be 8B: {get_camelot_code('C major')}")
print(f"G major should be 9B: {get_camelot_code('G major')}")
print()

print("Compatibility tests:")
print(f"8A vs 8A (same key) should be 1.0: {camelot_compatibility('8A', '8A')}")
print(f"8A vs 8B (relative) should be 0.9: {camelot_compatibility('8A', '8B')}")
print(f"8A vs 9A (adjacent) should be 0.8: {camelot_compatibility('8A', '9A')}")
print(f"8A vs 3B (far away) should be 0.0: {camelot_compatibility('8A', '3B')}")
print()

print("Transition advice:")
advice = get_transition_advice("8A", "9A")
print(f"  Score: {advice['compatibility_score']}")
print(f"  Strategy: {advice['strategy']}")
print()

# --- Test 3: detect_key.py on real song --- (add song)
print("TEST 3: detect_key.py")
#filepath = r"C:\Users\rheam\OneDrive\Documents\ai-dj\"
#camelot_number, key_name = get_key(filepath)
#print(f"  Key: {key_name}")
#print(f"  Camelot Number: {camelot_number}")