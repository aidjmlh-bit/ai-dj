import soundfile as sf
import librosa
import numpy as np
import os
from sections import get_sections

def crossfade(wav1_path, wav2_path, output_path, fade_duration=8):
    y1, sr1 = librosa.load(wav1_path, sr=None, mono=False)
    y2, sr2 = librosa.load(wav2_path, sr=None, mono=False)

    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
    sr = sr1

    moments1 = dict(get_sections(wav1_path))  #{"Intro": "0:00.0", "Outro": "3:00.0", ...}
    moments2 = dict(get_sections(wav2_path))

    fade_out_start = parse_time(moments1.get('Outro') or moments1.get('Chorus'))
    fade_in_start  = parse_time(moments2.get('Buildup') or moments2.get('Intro'))

    fade_out_sample = int(fade_out_start * sr)
    fade_in_sample  = int(fade_in_start * sr)

    fade_out_len = len(y1[0]) - fade_out_sample
    fade_in_len  = fade_in_sample

    fade_out_curve = np.linspace(1, 0, fade_out_len) #change later if we do not want it to be linear
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


def parse_time(timestamp):
    #converts to seconds
    if timestamp is None:
        return 0
    m, s = timestamp.split(':')
    return int(m) * 60 + float(s)
