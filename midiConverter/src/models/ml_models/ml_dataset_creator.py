import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.data.dataset_loader import load_dataset_with_splits
from src.data.feature_extractor import extract_features


def prepare_final_data() -> tuple[np.ndarray, np.ndarray]:

    train, val, test = load_dataset_with_splits()
    data = train + val + test
    feats = extract_features(data)

    X = [f[0] for f in feats]
    y = [f[1] for f in feats]

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    joblib.dump(scaler, 'models/trained_models/scaler.pkl')
    joblib.dump(encoder, 'models/trained_models/encoder.pkl')

    return X_scaled, y_encoded