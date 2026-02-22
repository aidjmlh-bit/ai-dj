import os
import glob
import librosa
import numpy as np
import msaf
from collections import Counter

def find_buildup(wav_path, drop_time, window=16):
    y, sr = librosa.load(wav_path)
    rms   = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)
    mask  = (times >= drop_time - window) & (times < drop_time)
    if mask.any():
        return float(times[mask][0])
    return float(max(0, drop_time - window))

def extract_key_moments(wav_path, boundaries, labels):
    boundaries = list(boundaries)
    labels     = list(labels)

    counts       = Counter(labels)
    chorus_label = counts.most_common(1)[0][0]
    intro_label  = labels[0]
    outro_label  = labels[-1]

    def first_of(target_label):
        for i, label in enumerate(labels):
            if label == target_label:
                return float(boundaries[i])
        return None

    def last_of(target_label):
        for i, label in reversed(list(enumerate(labels))):
            if label == target_label:
                return float(boundaries[i])
        return None

    intro  = first_of(intro_label)
    chorus = first_of(chorus_label)
    outro  = last_of(outro_label) if outro_label != intro_label else float(boundaries[-1])  # fix outro

    # verse = first segment that is not intro or chorus
    verse = None
    for i, label in enumerate(labels):
        if label not in (intro_label, chorus_label):
            verse = float(boundaries[i])
            break

    # beatdrop and buildup from energy
    y, sr = librosa.load(wav_path)
    rms   = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)
    diff  = np.diff(rms)
    drop_time    = float(times[np.argmax(diff)])
    buildup_time = find_buildup(wav_path, drop_time, window=8)  # tightened to 8s

    return (
        ("Intro",    intro),
        ("Verse",    verse),
        ("Buildup",  buildup_time),
        ("Beatdrop", drop_time),
        ("Chorus",   chorus),
        ("Outro",    outro),
    )

def get_sections(filepath: str):
    boundaries, labels = msaf.process(
        filepath,
        boundaries_id='foote',
        labels_id='fmc2d',
    )
    return extract_key_moments(filepath, boundaries, labels)

def analyze_songs(folder_path):
    wav_files = glob.glob(f'{folder_path}/*.wav')
    all_results = {}

    for wav_path in wav_files:
        if not os.path.exists(wav_path):
            print(f"File not found, skipping: {wav_path}")
            continue

        print(f"\nAnalyzing: {wav_path}")
        boundaries, labels = msaf.process(
            wav_path,
            boundaries_id='foote',
            labels_id='fmc2d',
        )

        print("Segments:")
        for label, start in zip(labels, boundaries):
            print(f"  {label:10} {start:.1f}s")

        moments = extract_key_moments(wav_path, boundaries, labels)

        print("\nKey Moments:")
        for label, timestamp in moments:
            print(f"  {label:10} {timestamp}")

        all_results[wav_path] = moments

    return all_results


# --- Run ---
# results = analyze_songs('/path/to/your/folder')
# for song, moments in results.items():
#     print(f"\n{song}:")
#     print(moments)

# Or single file
#moments = get_sections('/Users/siddarvind/Downloads/thinkingaboutyou.wav')
#print(moments)
