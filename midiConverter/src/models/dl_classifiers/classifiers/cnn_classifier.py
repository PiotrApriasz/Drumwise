import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import DRUM_INSTRUMENTS, CNN_DROPOUT_RATE
from src.models.dl_classifiers.classifiers.base_classifier import BaseClassifier

label_map = {inst: i for i, inst in enumerate(DRUM_INSTRUMENTS)}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = x.mean((2, 3))
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(batch, channels, 1, 1)
        return x * y


class DrumCNN(nn.Module):
    def __init__(self, num_classes=7, num_filters1=16, num_filters2=32, dropout_rate=CNN_DROPOUT_RATE):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters1)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters2)
        self.se1 = SEBlock(num_filters2)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(num_filters2, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.se2 = SEBlock(128)

        self.adapool = nn.AdaptiveMaxPool2d((15, 7))
        self.fc1 = nn.Linear(128 * 15 * 7, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se1(x)
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se2(x)
        x = self.pool(x)

        x = self.adapool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class DrumCNNClassifier(BaseClassifier):
    """CNN Classifier for Drum Sounds, using the BaseClassifier for training."""

    def __init__(self, feature_type='cqt', sr=22050, batch_size=8, num_workers=4,
                 num_filters1=16, num_filters2=32, dropout_rate=CNN_DROPOUT_RATE):
        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2
        self.dropout_rate = dropout_rate

        super().__init__(num_classes=len(DRUM_INSTRUMENTS), sr=sr,
                         feature_type=feature_type, batch_size=batch_size,
                         num_workers=num_workers)

    def _build_model(self) -> nn.Module:
        print("Building DrumCNN model...")
        return DrumCNN(num_classes=self.num_classes,
                       num_filters1=self.num_filters1,
                       num_filters2=self.num_filters2,
                       dropout_rate=self.dropout_rate)


# def train_cnn_classifier(model_path, best_model_path, train_with_mel=False):
#     best_val_acc = 0
#
#     train_data, val_data, test_data = load_dataset_with_splits()
#     train_data_aug = create_augmented_dataset(train_data, sr=22050, augment_factor=2)
#
#     if train_with_mel:
#         train_dataset = MelDrumDataset(train_data_aug, label_map, sr=22050)
#         val_dataset = MelDrumDataset(val_data, label_map, sr=22050)
#         test_dataset = MelDrumDataset(test_data, label_map, sr=22050)
#     else:
#         train_dataset = CqtDrumDataset(train_data_aug, label_map, sr=22050)
#         val_dataset = CqtDrumDataset(val_data, label_map, sr=22050)
#         test_dataset = CqtDrumDataset(test_data, label_map, sr=22050)
#
#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
#
#     model = DrumCNN(num_classes=len(DRUM_INSTRUMENTS))
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.000103, weight_decay=1.98e-05)
#     scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
#
#     activation = {}
#     def get_activation(name):
#         def hook(model, input, output):
#             activation[name] = output.detach()
#         return hook
#
#     model.conv1.register_forward_hook(get_activation('conv1'))
#     model.conv4.register_forward_hook(get_activation('conv4'))
#     model.se1.register_forward_hook(get_activation('se1'))
#     if hasattr(model, 'se2'):
#         model.se2.register_forward_hook(get_activation('se2'))
#
#     patience = 6
#     best_val_loss = float('inf')
#     epochs_no_improve = 0
#     max_epochs = 50
#
#     train_losses = []
#     val_losses = []
#     val_accs = []
#
#     for epoch in range(max_epochs):
#         model.train()
#         train_loss = 0
#         for x_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(x_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#
#         train_loss /= len(train_loader)
#         train_losses.append(train_loss)
#
#         scheduler.step()
#         model.eval()
#         val_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for x_val, y_val in val_loader:
#                 val_out = model(x_val)
#                 val_loss += criterion(val_out, y_val).item()
#                 preds = val_out.argmax(dim=1)
#                 correct += (preds == y_val).sum().item()
#                 total += y_val.size(0)
#         val_loss /= len(val_loader)
#         val_losses.append(val_loss)
#
#         val_acc = correct / total
#         val_accs.append(val_acc)
#
#         print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
#
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), best_model_path)
#             print(f"Saved best model so far: {best_val_acc:.4f}")
#
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print("Early stopping triggered")
#                 break
#
#     model.load_state_dict(torch.load(best_model_path))
#     model.eval()
#
#     #visualize_cnn_feature_maps(activation, test_loader, model)
#
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for x_test, y_test in test_loader:
#             outputs = model(x_test)
#             preds = outputs.argmax(dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y_test.cpu().numpy())
#
#     test_acc = accuracy_score(all_labels, all_preds)
#     cm = confusion_matrix(all_labels, all_preds)
#     print("Test Accuracy:", test_acc)
#     print("Confusion Matrix:\n", cm)
#
#     class_names = list(DRUM_INSTRUMENTS)
#     print("\nPer-class accuracy:")
#     for i, instrument in enumerate(class_names):
#         class_correct = cm[i, i]
#         class_total = cm[i, :].sum()
#         if class_total > 0:
#             print(f"{instrument}: {class_correct/class_total:.4f}")
#
#     torch.save(model.state_dict(), model_path)
#
#     return {
#         "train_losses": train_losses,
#         "val_losses": val_losses,
#         "val_accs": val_accs,
#         "test_acc": test_acc,
#         "confusion_matrix": cm
#     }
