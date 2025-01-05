import os
import librosa
import numpy as np

BASIC_DATASET_PATH = "/Users/piotrek/DataSets/DrumInstrumentsDataSet"
DRUM_INSTRUMENTS = ["kick", "snare", "overheads", "toms"]

def load_data_set(dataset_path: str = BASIC_DATASET_PATH,
                  instruments: list[str] = DRUM_INSTRUMENTS,
                  sr: int = 22050) -> list[tuple[np.ndarray, str]]:

    data_set = []
    for drum_instrument in instruments:
        folder_path = os.path.join(dataset_path, drum_instrument)
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                audio, _ = librosa.load(file_path, sr=sr, mono=True)
                data_set.append((audio, drum_instrument))

    return data_set
