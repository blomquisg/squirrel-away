from enum import Enum

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import config as cfg

config = cfg.Config()

class DatasetType(Enum):
    Train      = "train"
    Test       = "test"
    Validation = "validation"

class DatasetLoader():
    def __init__(self, dataset, dataloader):
        self.dataset = dataset
        self.dataloader = dataloader

def dataset_loader(data_dir, dataset_type=DatasetType.Train, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),      # size that the model expects
    transforms.ToTensor(),              # use PyTorch tensors
    transforms.Normalize(               # normalize based on ImageNet statistics
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
    ])

    # load the training dataset ... eventually parameterize the `root` param
    image_folder_path = f"{data_dir}/{dataset_type.value}/"
    dataset = datasets.ImageFolder(root=image_folder_path, transform=transform)
    dataloader = DataLoader(dataset=dataset, 
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      num_workers=num_workers)

    # return data set loader
    return DatasetLoader(dataset, dataloader)


def main():
    import os
    data_dir = config.data_dir
    dataset_type = DatasetType.Train
    loader = dataset_loader(data_dir, dataset_type, 16, False, 2)

if __name__ == "__main__":
    main()
