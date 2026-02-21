# Transition Algorithm (explicit + implementable)

# 0) Inputs
#    - songA_wav_path, songB_wav_path
#    - sr (sample rate)
#    - bpm.py provides: get_bpm(wav_path) -> float
#    - key.py provides: get_camelot_key(wav_path) -> str  (e.g., "8A", "9B")
#    - sections.py provides: get_sections(wav_path) -> List[Segment]
#         where Segment has:
#           - start_sec, end_sec
#           - label in {"intro","verse","buildup","drop","chorus","break","outro"}
#           - confidence
#           - rms_mean, onset_mean, centroid_mean (or equivalent)
#         and optionally events:
#           - drop_onsets: List[float]  (seconds)
#           - buildup_peaks: List[float] (seconds)
#    - transition_effects.py provides:
#         apply_transition(audioA, audioB, sr, tA, tB, transition_type,
#                          duration_beats, bpm_ref, phase_offset) -> np.ndarray
#         save_wav(audio, sr, out_path)

# 1) Compute BPM + key compatibility (explicit definitions)
#    1.1 bpmA = get_bpm(songA)
#        bpmB = get_bpm(songB)
#
#    1.2 Compute "effective BPM difference" allowing half/double-time matching:
#        delta_eff = min(
#            abs(bpmA - bpmB),
#            abs(bpmA - 2*bpmB),
#            abs(bpmA - 0.5*bpmB)
#        )
#        # This prevents false "big BPM diff" when tracks are half/double-time compatible.
#
#    1.3 Enforce BPM constraint:
#        if delta_eff > 15:
#            # either reject the pair OR force "low_cut_echo_slam" only and warn.
#
#    1.4 Compute stretch percent (for deciding if time-stretching is safe):
#        # Normalise bpmB to the same time-scale as bpmA first:
#        bpmB_norm = (bpmB * 2 if bpmB < bpmA * 0.75
#                     else bpmB / 2 if bpmB > bpmA * 1.33
#                     else bpmB)
#        # True stretch ratio — how far the time-stretcher must move:
#        stretch_pct = abs(bpmA / bpmB_norm - 1.0)
#        # Rule of thumb:
#        # - <= 0.06 (6%) => stretching is usually acceptable
#        # - > 0.06      => prefer switch transitions (echo/hard cut/bridge), not long blends
#
#    1.5 Key extraction:
#        keyA = get_camelot_key(songA)  # e.g., "8A"
#        keyB = get_camelot_key(songB)  # e.g., "9A"
#
#    1.6 Define explicit Camelot compatibility + score:
#        # Parse Camelot codes:
#        numA, letA = int(keyA[:-1]), keyA[-1]   # letter in {"A","B"}
#        numB, letB = int(keyB[:-1]), keyB[-1]
#
#        # Adjacency on the wheel is CIRCULAR (1 and 12 are neighbours).
#        # Use modular distance instead of raw abs():
#        diff = abs(numA - numB)
#        wheel_adjacent = (min(diff, 12 - diff) <= 1) and (letA == letB)
#
#        # Compatible if ANY:
#        #   - exact match:              numA == numB and letA == letB
#        #   - relative major/minor:     numA == numB and letA != letB
#        #   - adjacent on wheel:        wheel_adjacent (modular, see above)
#
#        # Score (deterministic):
#        if   numA == numB and letA == letB:   key_score = 1.0   # exact
#        elif wheel_adjacent:                  key_score = 0.8   # +/-1 same letter
#        elif numA == numB and letA != letB:   key_score = 0.7   # relative maj/min
#        else:                                 key_score = 0.2   # clash

# 1.7) Load audio buffers NOW — once, reused in steps 2, 3, and 5:
#        audioA, sr = librosa.load(songA_wav_path, mono=True, sr=None)
#        audioB, sr = librosa.load(songB_wav_path, mono=True, sr=None)

# 2) Segment both songs + compute beat grids
#    2.1 sectionsA = get_sections(songA)  # list of labeled segments + stats
#        sectionsB = get_sections(songB)
#
#    2.2 Define "key moments" as:
#        - all segment boundaries: start_sec and end_sec
#        - important events (if available): drop_onsets, buildup_peaks
#        - phrase boundaries: every 8 or 16 beats (derived from beat grid below)
#
#    2.3 Snap candidate times to the nearest beat so transitions land musically.
#
#    2.4 Beat phase alignment (critical — BPM match alone is not enough):
#        # Get the actual beat grid for each track (not a synthetic BPM grid):
#        beatsA = librosa.beat.beat_track(y=audioA, sr=sr, units='time')[1]
#        beatsB = librosa.beat.beat_track(y=audioB, sr=sr, units='time')[1]
#
#        # Derive downbeat positions (assume 4/4: every 4th beat):
#        downbeatsA = beatsA[::4]
#        downbeatsB = beatsB[::4]
#
#        # After tA_best and tB_best are chosen (step 3), snap to nearest downbeat:
#        tA_aligned = downbeatsA[argmin(abs(downbeatsA - tA_best))]
#        tB_aligned = downbeatsB[argmin(abs(downbeatsB - tB_best))]
#
#        # Compute phase offset so bar 1 of B follows bar 1 of A:
#        beat_period_A = 60.0 / bpmA
#        beat_period_B = 60.0 / bpmB_norm
#        phase_offset = (tA_aligned % (4 * beat_period_A)) - (tB_aligned % (4 * beat_period_B))
#        # Pass phase_offset to apply_transition so it can shift B's start sample accordingly.

# 3) Find the best transition point (scoring problem)
#    Goal: choose (tA, tB) where:
#      - tA is a good EXIT in song A (end of chorus/drop or start of breakdown/outro)
#      - tB is a good ENTRY in song B (start of intro/buildup OR drop, depending on style)
#      - both times are beat-aligned
#
#    3.1 Generate candidate exit points for A:
#        # Prefer:
#        #   - end of chorus / end of drop
#        #   - start of break / breakdown
#        #   - phrase boundary near those moments (8/16 beats)
#        exit_candidates_A = [...]
#
#    3.2 Generate candidate entry points for B:
#        # Prefer:
#        #   - start of intro (for smooth blends)
#        #   - start of buildup (for rise into B)
#        #   - drop onset (for echo slam / hard drop)
#        entry_candidates_B = [...]
#
#    3.3 Define explicit scoring features:
#        # ExitScore(A,t):
#        #   +1.0 if t is end of chorus/drop
#        #   +0.8 if t is start of break/breakdown
#        #   +0.4 if t is end of verse
#        #
#        # EntryScore(B,t):
#        #   +1.0 if t is drop onset (ONLY if using slam-style)
#        #   +0.8 if t is start of intro (ONLY if smooth-style)
#        #   +0.8 if t is start of buildup (bridge-style)
#        #
#        # EnergyPenalty:
#        #   energyA = mean RMS over window [t-2bars, t] in A  (use audioA array)
#        #   energyB = mean RMS over window [t, t+2bars] in B  (use audioB array)
#        #   penalty = abs(energyA - energyB)
#        #
#        # BPMPenalty:
#        #   penalty proportional to stretch_pct
#
#    3.4 Choose one of these two selection strategies:
#        # Strategy S (Smooth): aims for minimal energy discontinuity
#        # Strategy H (Hype):   allows energy jump and targets B's drop
#        #
#        # SmoothScore(tA,tB) = w1*ExitScore + w2*EntryScore - w3*EnergyPenalty - w4*stretch_pct
#        # HypeScore(tA,tB)   = w1*ExitScore + w2*EntryScore                    - w4*stretch_pct
#
#    3.5 Evaluate all candidate pairs and pick argmax:
#        # best = argmax_{tA in exitA, tB in entryB} Score(tA,tB)
#        # Return:
#        #   tA_best, tB_best
#        #   exit_label_A (segment label around tA)
#        #   entry_label_B (segment label around tB)
#        #   energy_jump = energyB - energyA

# 4) Select transition type (explicit deterministic logic)
#    Inputs to decision:
#      - delta_eff
#      - stretch_pct
#      - key_score
#      - exit_label_A, entry_label_B
#      - energy_jump
#
#    4.1 Define thresholds (tunable constants):
#        GREAT_BPM    = 2
#        GOOD_BPM     = 6
#        SAFE_STRETCH = 0.06
#        GOOD_KEY     = 0.7
#        CLASH_KEY    = 0.6
#        BIG_JUMP     = 0.15  # RMS units; tune against your track library
#
#    4.2 Decision rules (order matters — each condition is mutually exclusive):
#
#        # (A) Clean crossfade — everything lines up well
#        if delta_eff <= GREAT_BPM and key_score >= GOOD_KEY and abs(energy_jump) <= BIG_JUMP:
#            transition_type = "crossfade"
#            duration_beats  = 16
#
#        # (B) Key clash — filter blend regardless of stretch amount.
#        #     Checked BEFORE stretch conditions so a clash always gets masking,
#        #     not an echo slam that exposes the harmonic collision.
#        elif key_score < CLASH_KEY:
#            transition_type = "low_cut_filter"
#            duration_beats  = 8
#
#        # (C) Reverb tail — good key + safe stretch + smooth entry point
#        elif stretch_pct <= SAFE_STRETCH and key_score >= GOOD_KEY and entry_label_B in {"intro","verse"}:
#            transition_type = "reverb_tail"
#            duration_beats  = 8
#
#        # (D) Echo out + slam in — large BPM gap, drop entry, or big energy jump
#        elif stretch_pct > SAFE_STRETCH or delta_eff > GOOD_BPM or entry_label_B == "drop" or abs(energy_jump) > BIG_JUMP:
#            transition_type = "low_cut_echo_slam"
#            duration_beats  = 4
#
#        # (E) Default fallback (reached when stretch is safe + key is ok but no other rule fires)
#        else:
#            transition_type = "reverb_tail"
#            duration_beats  = 8

# 5) Apply transition + export (explicit IO behavior)
#    5.1 audioA and audioB are already loaded from step 1.7 — do NOT reload.
#
#    5.2 Choose bpm_ref for beat-aligned effects:
#        bpm_ref = bpmA  # effects are rendered in A's tempo; B is stretched to match
#
#    5.3 Apply effect:
#        mixed = apply_transition(
#            audioA         = audioA,
#            audioB         = audioB,
#            sr             = sr,
#            tA             = tA_aligned,
#            tB             = tB_aligned,
#            transition_type= transition_type,
#            duration_beats = duration_beats,
#            bpm_ref        = bpm_ref,
#            phase_offset   = phase_offset,   # from step 2.4
#        )
#
#    5.4 Optional but recommended: loudness normalisation / peak limiting
#        # normalize_peak(mixed, target=-1.0 dBFS) or similar
#
#    5.5 Save output wav:
#        out_path = f"mix_{keyA}_{int(bpmA)}__to__{keyB}_{int(bpmB)}_{transition_type}.wav"
#        save_wav(mixed, sr, out_path)
#
#    5.6 Save metadata (so you can debug/tune):
#        # - bpmA, bpmB, bpmB_norm, delta_eff, stretch_pct
#        # - keyA, keyB, key_score
#        # - chosen (tA_aligned, tB_aligned), exit/entry labels, energy_jump, phase_offset
#        # - transition_type, duration_beats
