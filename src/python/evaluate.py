import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import os
import datetime
import logging

import config
import model

config = config.Config()

## Validation dataset
batch_size = 16

dataset_loader = model.dataset_loader(config.data_dir, model.DatasetType.Validation, batch_size, False)
val_loader = dataset_loader.dataloader
val_dataset = dataset_loader.dataset

## Validator 
num_classes = 2
# TODO: remove hardcoded trained model directory
trained_model_dir = f"{config.experiments_dir}/trained_20250302_183521" 

# validation files
validation_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
validation_file = f"{trained_model_dir}/validation_results_{validation_timestamp}"
confusion_matrix_file = f"{trained_model_dir}/confusion_matrix_{validation_timestamp}.jpg"

# trained model file
model_path = f"{trained_model_dir}/model" 
device = torch.device("cudu" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# tally results
correct    = 0
total      = 0
all_preds  = []
all_labels = []
misclassified = []
item_index = 0
class_names = val_dataset.classes


for images, labels in val_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    correct += (predicted == labels).sum().item()
    total += labels.size(0)

    for predicted_result, labeled_result in zip(predicted, labels):
        if predicted_result != labeled_result:
            image_path = val_dataset.samples[item_index][0]
            misclassified.append((image_path, class_names[labeled_result], class_names[predicted_result]))
        item_index += 1

    all_preds.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

accuracy = correct / total
validation_accuracy = f"Validation Accuracy: {accuracy:.4f}"
config.logger.debug(f"{validation_accuracy}")
with open(validation_file, "a") as file:
    file.write(f"{validation_accuracy}\n\n")

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(confusion_matrix_file, format="jpeg")
# plt.show() # comment to remove plot

# Log classification report
classification_report = f"\nClassification Report:\n{classification_report(all_labels, all_preds, target_names=class_names)}"
config.logger.debug(classification_report)
with open(validation_file, "a") as file:
    file.write(classification_report)

# Log misclassified images
misclassification_report = f"\nMisclassified Images:"
for path, actual, predicted in misclassified:
    misclassification_report += f"\nImage: {path}, Actual: {actual}, Predicted: {predicted}"
config.logger.debug(misclassification_report)
with open(validation_file, "a") as file:
    file.write(misclassification_report)