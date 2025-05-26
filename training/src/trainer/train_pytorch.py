import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import datetime
import os
import tempfile

from trainer_config import config as cfg
import trainer.dataset as ds

config = cfg.Config()

## training data loader definition
train_loader = ds.dataset_loader(f"{config.data_dir}", ds.DatasetType.Train).dataloader

num_classes = 2  # squirrel or not squirrel
num_epochs  = 15 # run the data 15 times

# import the pre-trained classification model
dataset = models.mobilenet_v2(pretrained=True)

# add a layer to make the model only classify squirrels and non-squirrels
dataset.classifier[1] = nn.Linear(dataset.classifier[1].in_features, num_classes)

# Freeze base layers of the model
for param in dataset.features.parameters():
    param.requires_grad = False

# Binary classification:  squirrel or no-squirrel
criterion  = nn.CrossEntropyLoss() 
optimizer = optim.Adam(dataset.classifier.parameters(), lr=0.001)
training_results = ""

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader: 
        # zero out gradients
        optimizer.zero_grad()

        # forward prop
        outputs = dataset(inputs)
        loss = criterion(outputs, labels)

        # backward prop
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_result = f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}"
    config.logger.debug(epoch_result)
    training_results += f"{epoch_result}\n"


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
trained_dir = f"{config.experiments_dir}/trained"
experiment_dir = f"{trained_dir}/{timestamp}"
os.mkdir(experiment_dir)
model_filename  = f"{experiment_dir}/model"
training_results_filename = f"{experiment_dir}/training_results"

# save the trained model
torch.save(dataset.state_dict(), model_filename)
config.logger.debug(f"Model saved to {model_filename}")

# os.symlink can't replace links, so do the swap-eroo
def _update_symlink(target, link):
    temp_link = tempfile.mktemp(prefix=link + "_tmp")
    os.symlink(target, temp_link)
    os.replace(temp_link, link)

# update "latest" symlink
_update_symlink(experiment_dir, f"{trained_dir}/latest")

# save the epoch output 
with open(training_results_filename, "w") as epoch_file:
    epoch_file.write(training_results)
config.logger.debug(f"Training results saved to {training_results_filename}")