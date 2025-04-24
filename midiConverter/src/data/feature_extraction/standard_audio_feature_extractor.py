import numpy as np
import librosa
from scipy.stats import kurtosis, skew

def extract_features_from_audio(y: np.ndarray, sr: int) -> np.ndarray:

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_median = np.median(mfcc, axis=1)
    mfcc_kurt = kurtosis(mfcc, axis=1)
    mfcc_skew = skew(mfcc, axis=1)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid, axis=1)
    spec_centroid_std = np.std(spec_centroid, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr, axis=1)
    zcr_std = np.std(zcr, axis=1)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff_mean = np.mean(rolloff, axis=1)
    rolloff_std = np.std(rolloff, axis=1)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_bw_mean = np.mean(spec_bw, axis=1)
    spec_bw_std = np.std(spec_bw, axis=1)

    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)
    spec_contrast_std = np.std(spec_contrast, axis=1)

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms, axis=1)
    rms_std = np.std(rms, axis=1)

    features = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_median, mfcc_kurt, mfcc_skew,
        spec_centroid_mean, spec_centroid_std,
        zcr_mean, zcr_std,
        #rolloff_mean, rolloff_std,
        #spec_bw_mean, spec_bw_std,
        #spec_contrast_mean, spec_contrast_std,
        #rms_mean, rms_std
    ], axis=0)

    return features


def extract_features(data: list[tuple[np.ndarray, str]],
                     sr: int = 22050) -> list[tuple[np.ndarray, str]]:

    output = []
    for audio, label in data:
        feat = extract_features_from_audio(audio, sr)
        output.append((feat, label))

    return output
