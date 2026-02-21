import allin1
import os
import glob
import librosa

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

    # Use `is not None` so that a timestamp of 0.0 (valid) is not treated as missing.
    return (
        ("Intro",    float(intro)        if intro    is not None else None),
        ("Verse",    float(verse)        if verse    is not None else None),
        ("Buildup",  float(buildup_time) if buildup_time is not None else None),
        ("Beatdrop", float(drop_time)    if drop_time    is not None else None),
        ("Chorus",   float(chorus)       if chorus   is not None else None),
        ("Outro",    float(outro)        if outro    is not None else None),
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
##    print(f"\n{song}:")
##    print(moments)


def get_sections(filepath: str):
    """Return key structural moments for a WAV file as float timestamps.

    Runs allin1 analysis on *filepath* and extracts the start time (in
    seconds) for Intro, Verse, Buildup, Beatdrop, Chorus, and Outro.

    Args:
        filepath: Path to a ``.wav`` audio file.

    Returns:
        Tuple of ``(section_label, start_seconds)`` pairs. ``start_seconds``
        is a float, or ``None`` if that section was not detected.
    """
    result = allin1.analyze(filepath)
    return extract_key_moments(filepath, result)