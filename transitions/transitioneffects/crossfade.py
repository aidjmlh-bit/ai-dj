import soundfile as sf
import librosa
import numpy as np
import os
from sections import get_sections

def crossfade(wav1_path, wav2_path, output_path, fade_out_start, fade_in_end):
    """
    Args:
        wav1_path:      first song
        wav2_path:      second song
        output_path:    where to save
        fade_out_start: timestamp (in seconds) where song 1 starts fading out
        fade_in_end:    timestamp (in seconds) where song 2 finishes fading in (i.e. the beat drop)
    """
    y1, sr1 = librosa.load(wav1_path, sr=None, mono=False)
    y2, sr2 = librosa.load(wav2_path, sr=None, mono=False)

    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
    sr = sr1

    fade_out_sample = int(fade_out_start * sr)
    fade_in_sample  = int(fade_in_end * sr)

    fade_out_len = len(y1[0]) - fade_out_sample
    fade_in_len  = fade_in_sample

    fade_out_curve = np.linspace(1, 0, fade_out_len)
    fade_in_curve  = np.linspace(0, 1, fade_in_len)

    y1[..., fade_out_sample:] *= fade_out_curve
    y2[..., :fade_in_sample]  *= fade_in_curve

    min_len = min(fade_out_len, fade_in_len)
    overlap = y1[..., -min_len:] + y2[..., :min_len]

    output = np.concatenate([
        y1[..., :-min_len],
        overlap,
        y2[..., min_len:]
    ], axis=-1)

    output = output / np.max(np.abs(output))
    sf.write(output_path, output.T, sr)
    print(f"Saved crossfade to: {output_path}")
