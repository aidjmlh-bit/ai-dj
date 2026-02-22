import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

def low_cut_filter(y, sr, cutoff_hz):
    """
    Removes frequencies below cutoff_hz (cuts the bass).
    cutoff_hz = 200 means everything below 200Hz gets removed.
    """
    sos = butter(4, cutoff_hz, btype='high', fs=sr, output='sos')
    return sosfilt(sos, y)


def add_echo(y, sr, delay_seconds=0.2, decay=0.6):
    """
    Adds an echo effect by mixing the signal with a delayed version of itself.
    delay_seconds = how long before the echo hits
    decay = how loud the echo is (0.4 = 40% of original volume)
    *Was 0.3 for delayed_seconds and 0.4 for decay*
    """
    delay_samples = int(sr * delay_seconds)
    echo = np.zeros_like(y)
    echo[delay_samples:] = y[:-delay_samples] * decay
    return y + echo


def low_cut_echo_transition(song_a, song_b, sr, transition_seconds=8.0):
    """
    Transitions from song_a to song_b using:
    1. Gradually cutting bass from song_a
    2. Adding echo to song_a's tail
    3. Bringing in song_b clean
    """
    transition_samples = int(sr * transition_seconds)

    # Get the transition section of song_a (last N seconds)
    end_a = song_a[-transition_samples:].copy()

    # Get the start of song_b (first N seconds)
    start_b = song_b[:transition_samples].copy()

    # Gradually increase the low cut on song_a
    # At start of transition: no cut
    # At end of transition: full bass cut at 200Hz
    result_a = np.zeros_like(end_a)
    num_steps = 8  # number of steps to gradually apply filter

    step_size = transition_samples // num_steps

    for i in range(num_steps):
        start = i * step_size
        end = start + step_size

        # Cutoff increases from 0 to 200Hz gradually
        cutoff = int((i / num_steps) * 200)

        chunk = end_a[start:end].copy()

        if cutoff > 20:  # only filter if cutoff is meaningful
            chunk = low_cut_filter(chunk, sr, cutoff)

        result_a[start:end] = chunk

    # Add echo to the filtered song_a tail
    result_a = add_echo(result_a, sr, delay_seconds=0.3, decay=0.4) #the numbers are good

    # Fade out song_a, fade in song_b using equal power
    t = np.linspace(0, np.pi/2, transition_samples)
    fade_out = np.cos(t)
    fade_in = np.sin(t)

    # Blend together
    blended = (result_a * fade_out) + (start_b * fade_in)

    # Build final output
    final = np.concatenate([
        song_a[:-transition_samples],   # song_a without transition section
        blended,                         # the low cut echo transition
        song_b[transition_samples:]      # song_b after its intro
    ])

    return final


def mix_with_low_cut_echo(filepath_a, filepath_b, output_path, transition_seconds=8.0):
    """
    Loads two songs and creates a mix with low cut echo transition.
    """
    print(f"Loading Song A: {filepath_a}")
    song_a, sr = librosa.load(filepath_a, mono=True, sr=None)

    print(f"Loading Song B: {filepath_b}")
    song_b, sr2 = librosa.load(filepath_b, mono=True, sr=None)

    # Make sure both songs have same sample rate
    if sr != sr2:
        print("Warning: sample rates differ, resampling song_b")
        song_b = librosa.resample(song_b, orig_sr=sr2, target_sr=sr)

    print("Applying low cut echo transition...")
    final = low_cut_echo_transition(song_a, song_b, sr, transition_seconds)

    sf.write(output_path, final, sr)
    print(f"Mix saved to: {output_path}")