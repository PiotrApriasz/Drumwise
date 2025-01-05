import os
import librosa
import numpy as np

BASIC_DATASET_PATH = "/Users/piotrek/DataSets/DrumInstrumentsDataSet"
BASIC_DRUM_INSTRUMENTS = ["kick", "snare", "overheads", "toms"]

FULL_DATASET_PATH = "/Users/piotrek/DataSets/MDLib2.2/Training"
FULL_DRUM_INSTRUMENTS = ["crash", "floor_tom", "hi-hat", "kick", "rack_tom", "ride", "snare"]

def load_basic_data_set(dataset_path: str = FULL_DATASET_PATH,
                        instruments: list[str] = FULL_DRUM_INSTRUMENTS,
                        sr: int = 22050) -> list[tuple[np.ndarray, str]]:

    data_set = []
    desired_length = sr * 2

    for drum_instrument in instruments:
        folder_path = os.path.join(dataset_path, drum_instrument)
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                audio, _ = librosa.load(file_path, sr=sr, mono=True)

                if len(audio) > desired_length:
                    audio = audio[:desired_length]
                else:
                    audio = np.pad(audio, (0, desired_length - len(audio)), mode='constant')

                data_set.append((audio, drum_instrument))

    return data_set
