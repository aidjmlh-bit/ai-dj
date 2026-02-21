import allin1
import os
import glob
import librosa
import numpy as np

def find_beat_drop(result):
    segments = result.segments
    for i, seg in enumerate(segments):
        if seg.label == 'chorus':
            if i > 0 and segments[i-1].label in ('verse', 'pre-chorus', 'intro'):
                return seg.start
    return None

def find_buildup(wav_path, drop_time, window=16):
    y, sr = librosa.load(wav_path)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)
    mask = (times >= drop_time - window) & (times < drop_time)
    if mask.any():
        return times[mask][0]
    return max(0, drop_time - window)

def format_time(seconds):
    m = int(seconds) // 60
    s = seconds % 60
    return f"{m}:{s:04.1f}"

def extract_key_moments(wav_path, result):
    segments = result.segments
    drop_time = find_beat_drop(result)
    buildup_time = find_buildup(wav_path, drop_time) if drop_time else None

    def first_of(label):
        for seg in segments:
            if seg.label == label:
                return seg.start
        return None

    intro    = first_of('intro')
    verse    = first_of('verse')
    chorus   = first_of('chorus')
    outro    = first_of('outro')

    return (
        ("Intro",     format_time(intro)    if intro    else None),
        ("Verse",     format_time(verse)    if verse    else None),
        ("Buildup",   format_time(buildup_time) if buildup_time else None),
        ("Beatdrop",  format_time(drop_time) if drop_time else None),
        ("Chorus",    format_time(chorus)   if chorus   else None),
        ("Outro",     format_time(outro)    if outro    else None),
    )

def analyze_songs(folder_path):
    wav_files = glob.glob(f'{folder_path}/*.wav')
    all_results = {}

    for wav_path in wav_files:
        if not os.path.exists(wav_path):
            print(f"File not found, skipping: {wav_path}")
            continue

        print(f"\nAnalyzing: {wav_path}")
        result = allin1.analyze(wav_path)

        print("Segments:")
        for seg in result.segments:
            print(f"  {seg.label:10} {seg.start:.1f}s → {seg.end:.1f}s")

        moments = extract_key_moments(wav_path, result)
        print("\nKey Moments:")
        for label, timestamp in moments:
            print(f"  {label:10} {timestamp}")

        all_results[wav_path] = moments

    return all_results

##results = analyze_songs('/path/to/your/folder')

##for song, moments in results.items():
    print(f"\n{song}:")
    print(moments)



#def get_sections() -> ((str, str), (str, str), (str, str), (str, str), (str, str)):
#    return ("Intro", "timestamp"), ("Verse", "timestamp"), ("Chorus", "timestamp"), ("Beatdrop", "timestamp"), ("Outro", "timestamp")
