import numpy as np
import librosa
#import soundfile as sf
from scipy.io.wavfile import write

def crossfadesin(wav1_path, wav2_path, output_path, fade_out_start, fade_in_end, max_fade_seconds=8):
    
    y1, sr1 = librosa.load(wav1_path, sr=None, mono=False)
    y2, sr2 = librosa.load(wav2_path, sr=None, mono=False)

    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
    sr = sr1

    # Cut song 1 to only go max_fade_seconds past the fade out start
    cut_sample = int((fade_out_start + max_fade_seconds) * sr)
    y1 = y1[..., :cut_sample]

    # Cut song 2 to start max_fade_seconds before the fade in end
    start_sample = max(0, int((fade_in_end - max_fade_seconds) * sr))
    y2 = y2[..., start_sample:]

    fade_samples = int(max_fade_seconds * sr)
    
    t = np.linspace(0, 1, fade_samples)
    fade_out_curve = np.cos(t * np.pi / 2)
    fade_in_curve  = np.sin(t * np.pi / 2)

    y1[..., -fade_samples:] *= fade_out_curve
    y2[..., :fade_samples]  *= fade_in_curve

    overlap = y1[..., -fade_samples:] + y2[..., :fade_samples]

    output = np.concatenate([
        y1[..., :-fade_samples],
        overlap,
        y2[..., fade_samples:]
    ], axis=-1)

    output = output / np.max(np.abs(output))
    
    from scipy.io.wavfile import write
    write(output_path, sr, output.T.astype(np.float32))
    print(f"Saved crossfade to: {output_path}")
    
# example: 
# crossfadesin('/Users/siddarvind/Downloads/thinkingaboutyou.wav', 
#              '/Users/siddarvind/Downloads/fameisagun.wav', 
#              'outputSin1.wav', 
#              fade_out_start=116.0, fade_in_end=14.0)    
