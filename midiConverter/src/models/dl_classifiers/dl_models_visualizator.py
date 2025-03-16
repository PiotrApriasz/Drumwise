# import os
#
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import torch
# from src.constants import DRUM_INSTRUMENTS, CNN_VISUALIZATIONS_PATH
#
#
# def visualize_cnn_feature_maps(activation, test_loader, model, sample_idx=0):
#     x_sample, y_sample = next(iter(test_loader))
#     x_sample, y_sample = (x_sample[sample_idx:sample_idx + 1], y_sample[sample_idx:sample_idx + 1])
#
#     with torch.no_grad():
#         output = model(x_sample)
#
#     plt.figure(figsize=(10, 6))
#     plt.imshow(x_sample[0, 0].cpu().numpy(), aspect='auto', cmap='viridis')
#     plt.colorbar()
#     plt.title(f"Input CQT - True Class: {DRUM_INSTRUMENTS[y_sample.item()]}")
#     plt.tight_layout()
#     plt.savefig('visualizations/input_cqt.png')
#     plt.savefig(os.path.join(CNN_VISUALIZATIONS_PATH, 'input_cqt.png'))
#     plt.close()
#
#     if 'conv1' in activation:
#         conv1_act = activation['conv1'][0].cpu().numpy()
#         fig, axes = plt.subplots(4, 4, figsize=(12, 12))
#         for i, ax in enumerate(axes.flat):
#             if i < conv1_act.shape[0]:
#                 ax.imshow(conv1_act[i], cmap='viridis', aspect='auto')
#                 ax.set_title(f'Conv1 Filter {i + 1}')
#                 ax.axis('off')
#         plt.tight_layout()
#         plt.savefig(os.path.join(CNN_VISUALIZATIONS_PATH, 'conv1_features.png'))
#         plt.close()
#
#     if 'conv4' in activation:
#         conv4_act = activation['conv4'][0].cpu().numpy()
#         fig, axes = plt.subplots(4, 4, figsize=(12, 12))
#         for i, ax in enumerate(axes.flat):
#             if i < 16:
#                 ax.imshow(conv4_act[i], cmap='viridis', aspect='auto')
#                 ax.set_title(f'Conv4 Filter {i + 1}')
#                 ax.axis('off')
#         plt.tight_layout()
#         plt.savefig(os.path.join(CNN_VISUALIZATIONS_PATH, 'conv4_features.png'))
#         plt.close()
#
#     if 'se1' in activation:
#         se1_act = activation['se1']
#         se1_shape = se1_act.shape
#
#         plt.figure(figsize=(10, 5))
#         if len(se1_shape) == 4:
#             se1_weights = se1_act[0].mean(dim=(1, 2)).cpu().numpy()
#             plt.bar(range(len(se1_weights)), se1_weights)
#             plt.title('SE1 Channel Attention Weights')
#         else:
#             plt.plot(se1_act[0].cpu().numpy().flatten())
#             plt.title('SE1 Activation Values')
#         plt.xlabel('Channel')
#         plt.ylabel('Weight')
#         plt.tight_layout()
#         plt.savefig(os.path.join(CNN_VISUALIZATIONS_PATH, 'se1_attention.png'))
#         plt.close()
#
#     if 'se2' in activation:
#         plt.figure(figsize=(10, 5))
#         se2_act = activation['se2']
#         se2_shape = se2_act.shape
#
#         if len(se2_shape) == 4:
#             se2_weights = se2_act[0].mean(dim=(1, 2)).cpu().numpy()
#             plt.bar(range(len(se2_weights)), se2_weights)
#             plt.title('SE2 Channel Attention Weights')
#         else:
#             plt.plot(se2_act[0].cpu().numpy().flatten())
#             plt.title('SE2 Activation Values')
#         plt.xlabel('Channel')
#         plt.ylabel('Weight')
#         plt.tight_layout()
#         plt.savefig(os.path.join(CNN_VISUALIZATIONS_PATH, 'se2_attention.png'))
#         plt.close()
#
#     if 'conv4' in activation:
#         class_weights = model.fc2.weight.data.cpu().numpy()
#         final_conv = activation['conv4'].cpu().numpy()
#
#         pred_class = output.argmax(dim=1).item()
#         pred_weights = class_weights[pred_class]
#
#         cam = np.zeros(final_conv.shape[2:], dtype=np.float32)
#         for i, w in enumerate(pred_weights[:final_conv.shape[1]]):
#             cam += w * final_conv[0, i].mean(axis=0)
#
#         plt.figure(figsize=(10, 6))
#         plt.imshow(cam, cmap='jet', aspect='auto')
#         plt.colorbar()
#         plt.title(f'Class Activation Map for {DRUM_INSTRUMENTS[pred_class]}')
#         plt.tight_layout()
#         plt.savefig(os.path.join(CNN_VISUALIZATIONS_PATH, 'class_activation_map.png'))
#         plt.close()
#
#     print(f"Feature visualizations saved to 'visualizations/' directory")
#
#
# def plot_cnn_training_metrics(metrics):
#     plt.figure(figsize=(15, 10))
#
#     plt.subplot(2, 2, 1)
#     plt.plot(metrics["train_losses"], label="Training Loss")
#     plt.plot(metrics["val_losses"], label="Validation Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#
#     plt.subplot(2, 2, 2)
#     plt.plot(metrics["val_accs"], label="Validation Accuracy", color="green")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Validation Accuracy")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#
#     plt.subplot(2, 2, 3)
#     cm = metrics["confusion_matrix"]
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=DRUM_INSTRUMENTS,
#                 yticklabels=DRUM_INSTRUMENTS)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title(f"Confusion Matrix (Test Acc: {metrics['test_acc']:.4f})")
#
#     plt.subplot(2, 2, 4)
#     class_acc = np.diag(cm) / np.sum(cm, axis=1)
#     sns.barplot(x=list(DRUM_INSTRUMENTS), y=class_acc)
#     plt.xlabel("Instrument")
#     plt.ylabel("Accuracy")
#     plt.title("Per-Class Accuracy")
#     plt.xticks(rotation=45)
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(CNN_VISUALIZATIONS_PATH, 'training_metrics.png'), dpi=300)
#     plt.close()
#
#     print(f"Training metrics visualization saved to 'training_metrics.png'")
#
#
# def plot_cnn_learning_curves(metrics):
#     plt.figure(figsize=(12, 6))
#
#     epochs = range(1, len(metrics["train_losses"]) + 1)
#
#     plt.plot(epochs, metrics["train_losses"], 'b-', label='Training Loss')
#     plt.plot(epochs, metrics["val_losses"], 'r-', label='Validation Loss')
#
#     ax2 = plt.gca().twinx()
#     ax2.plot(epochs, metrics["val_accs"], 'g-', label='Validation Accuracy')
#     ax2.set_ylabel('Accuracy', color='g')
#     ax2.tick_params(axis='y', labelcolor='g')
#     ax2.set_ylim([0, 1])
#
#     plt.grid(True, alpha=0.3)
#     plt.title('Training and Validation Metrics')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#
#     lines1, labels1 = plt.gca().get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     plt.legend(lines1 + lines2, labels1 + labels2, loc='center right')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(CNN_VISUALIZATIONS_PATH, 'learning_curves.png'), dpi=300)
#     plt.close()
#
#     print(f"Learning curves visualization saved to 'learning_curves.png'")
#
#
# def visualize_cnn_model_performance(metrics):
#     plot_cnn_training_metrics(metrics)
#
#     plot_cnn_learning_curves(metrics)
#
#     plt.figure(figsize=(10, 8))
#     cm = metrics["confusion_matrix"]
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="YlGnBu",
#                 xticklabels=DRUM_INSTRUMENTS,
#                 yticklabels=DRUM_INSTRUMENTS)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Normalized Confusion Matrix")
#     plt.tight_layout()
#     plt.savefig(os.path.join(CNN_VISUALIZATIONS_PATH, 'normalized_confusion_matrix.png'), dpi=300)
#     plt.close()
#
#     print(f"Normalized confusion matrix saved to 'normalized_confusion_matrix.png'")
#
#     plt.figure(figsize=(12, 6))
#     error_matrix = cm.copy()
#     np.fill_diagonal(error_matrix, 0)
#
#     for i in range(len(DRUM_INSTRUMENTS)):
#         if np.sum(error_matrix[i]) > 0:
#             error_dist = error_matrix[i] / np.sum(error_matrix[i])
#             plt.subplot(2, 4, i + 1)
#             sns.barplot(x=list(DRUM_INSTRUMENTS), y=error_dist)
#             plt.title(f"True: {DRUM_INSTRUMENTS[i]}")
#             plt.xticks(rotation=90, fontsize=8)
#             plt.ylim(0, 1)
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(CNN_VISUALIZATIONS_PATH, 'misclassification_analysis.png'), dpi=300)
#     plt.close()
#
#     print(f"Misclassification analysis saved to 'misclassification_analysis.png'")
