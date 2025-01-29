from src.models.dl_models.cnn_classifier import train_cnn_classifier

if __name__ == "__main__":
    train_cnn_classifier("standard_cnn_model.pth", "best_cnn_model.pth")