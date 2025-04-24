import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.constants import MEL_CNN_MODEL_NAME, MEL_BEST_CNN_MODEL_NAME, DRUM_INSTRUMENTS, TRAINED_DL_MODELS_PATH
from src.models.dl_classifiers.classifiers.cnn_classifier import train_cnn_classifier

if __name__ == "__main__":
    metrics = train_cnn_classifier(os.path.join(TRAINED_DL_MODELS_PATH, MEL_CNN_MODEL_NAME),
                                   os.path.join(TRAINED_DL_MODELS_PATH, MEL_BEST_CNN_MODEL_NAME))
    
    print("\nModel training complete!")
    print(f"Final test accuracy: {metrics['test_acc']:.4f}")

    cm = metrics["confusion_matrix"]
    print("\nPer-class accuracy:")
    for i, instrument in enumerate(DRUM_INSTRUMENTS):
        class_correct = cm[i, i]
        class_total = np.sum(cm[i, :])
        if class_total > 0:
            print(f"{instrument}: {class_correct/class_total:.4f}")