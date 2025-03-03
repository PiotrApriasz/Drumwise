import os
import librosa
import soundfile as sf
import re

from src.models.constants import DRUM_INSTRUMENTS

INPUT_FOLDER = "/Users/piotrek/DataSets/IDMT-SMT-DRUMS-V2/audio"
OUTPUT_FOLDER = "/Users/piotrek/DataSets/IDMT-SMT-DRUMS-V2/training"

SETS_INPUT_FOLDER = "/Users/piotrek/Developer/Drumwise/midiConverter/audio/sets"
SETS_OUTPUT_FOLDER = "/Users/piotrek/Developer/Drumwise/midiConverter/audio"

instrument_folders = {
    "KD": "kick",
    "SD": "snare",
    "HH": "hi-hat"
}

INSTRUMENT_ORDER = [
    "kick",
    "snare",
    "hi-hat",
    "crash",
    "rack_tom",
    "floor_tom",
    "ride"
]

def get_set_number(file_name: str) -> str:
    match = re.search(r"set(\d+)", file_name.lower())
    if match:
        return match.group(1)
    return ""


def get_instrument(file_path: str):
    file_name = os.path.basename(file_path)
    set_number = get_set_number(file_name)

    if not set_number:
        print(f"Didn't find set number in file: {file_name}")
        return

    y, sr = librosa.load(file_path, sr=None, mono=True)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_samples = librosa.frames_to_samples(onset_frames)

    onset_samples = list(onset_samples)
    onset_samples.append(len(y))

    if len(onset_samples) < 8:
        print(f"WARNING: In file {file_name} there is no enough strikes, to cut out all instruments.")
        print(f"    Found onsets: {len(onset_samples) - 1} (without end of the file).")
        return

    os.makedirs(SETS_OUTPUT_FOLDER, exist_ok=True)

    for i in range(len(INSTRUMENT_ORDER)):
        instrument_name = INSTRUMENT_ORDER[i]

        start_sample = onset_samples[i]
        end_sample = onset_samples[i + 1]

        snippet = y[start_sample:end_sample]

        out_file_name = f"{instrument_name}{set_number}.wav"
        out_file_path = os.path.join(SETS_OUTPUT_FOLDER, out_file_name)

        sf.write(out_file_path, snippet, sr)
        print(f"[{file_name}] Saved: {out_file_name}")

def create_folder():
    for inst_code, folder_name in instrument_folders.items():
        out_path = os.path.join(OUTPUT_FOLDER, folder_name)
        os.makedirs(out_path, exist_ok=True)

def extract_instrument_code(file_name: str):
    file_name_upper = file_name.upper()
    if "KD" in file_name_upper:
        return DRUM_INSTRUMENTS[4]
    elif "SD" in file_name_upper:
        return DRUM_INSTRUMENTS[0]
    elif "HH" in file_name_upper:
        return DRUM_INSTRUMENTS[3]
    else:
        return None


def process_file(file_path: str):
    file_name = os.path.basename(file_path)
    instrument_folder = extract_instrument_code(file_name)

    if not instrument_folder:
        print(f"Instrument not found in: {file_name}")
        return

    y, sr = librosa.load(file_path, sr=None, mono=True)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_samples = librosa.frames_to_samples(onset_frames)

    start_sample = onset_samples[0]

    offset_in_samples = int(0.5 * sr)

    end_sample = min(start_sample + offset_in_samples, len(y))

    snippet = y[start_sample:end_sample]

    out_dir = os.path.join(OUTPUT_FOLDER, instrument_folder)
    os.makedirs(out_dir, exist_ok=True)

    out_file_name = f"{os.path.splitext(file_name)[0]}_first_hit.wav"
    out_file_path = os.path.join(out_dir, out_file_name)

    sf.write(out_file_path, snippet, sr)
    print(f"Saved: {out_file_path}")

def main():
    create_folder()

    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.lower().endswith(('.wav', '.mp3', '.flac')):
            file_path = os.path.join(INPUT_FOLDER, file_name)
            process_file(file_path)

def get_instruments():
    for file_name in os.listdir(SETS_INPUT_FOLDER):
        if file_name.lower().endswith((".wav", ".mp3", ".flac")):
            file_path = os.path.join(SETS_INPUT_FOLDER, file_name)
            get_instrument(file_path)

if __name__ == "__main__":
    #main()
    get_instruments()
