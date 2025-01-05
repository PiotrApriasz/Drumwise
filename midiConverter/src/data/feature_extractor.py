import numpy as np
import librosa

def extract_features_from_audio(y: np.ndarray, sr: int) -> np.ndarray:

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr, axis=1)

    features = np.concatenate((mfcc_mean, spec_centroid_mean, zcr_mean), axis=0)

    return features


def extract_features(data: list[tuple[np.ndarray, str]],
                     sr: int = 22050) -> list[tuple[np.ndarray, str]]:

    output = []
    for audio, label in data:
        feat = extract_features_from_audio(audio, sr)
        output.append((feat, label))

    return output
