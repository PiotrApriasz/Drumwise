import librosa
import numpy as np


def generate_mel_spectrogram(y, sr=22050, n_fft=2048, hop_length=256, n_mels=256):
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels)

    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db


def generate_cqt_spectrogram(y, sr=22050, hop_length=512):
    cqt = librosa.cqt(y,
                      sr=sr,
                      hop_length=hop_length)

    cqt_mag = np.abs(cqt)

    cqt_db = librosa.amplitude_to_db(np.abs(cqt_mag), ref=np.max)
    return cqt_db