
import numpy as np
import glob
import torch
from typing import Self
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
from PIL import Image
from typing import Any
from galaxy_classification.utils import trim_file_list, img_label
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from galaxy_classification.training_utils import EPS, GAMMA



# --- Label preprocessing constants and groups ---

CLASS_GROUPS = {
    'Class 1': ['Class1.1', 'Class1.2', 'Class1.3'],
    'Class 2': ['Class2.1', 'Class2.2'],
    'Class 7': ['Class7.1', 'Class7.2', 'Class7.3'],
}

"""
Class to load images into list of strings and labels into a dataframe.
The images are discarded if they are not in the labels dataframe.
"""
@dataclass
class GalaxyDataset(Dataset):
    images: list[str]
    labels: pd.DataFrame

    @classmethod
    def load(cls,image_path:str,label_path:str) -> Self:
        """
        Load the galaxy dataset from the given path.
        """
        file_paths = glob.glob(f"{image_path}/*.jpg")
        labels_df = pd.read_csv(label_path).set_index("GalaxyID")
        file_paths = trim_file_list(file_paths, labels_df)
        return cls(images=file_paths, labels=labels_df)
    def __len__(self):
        return len(self.images)


#       ------------------------ CLASSIFICATION ------------------------
"""
Class to convert the images into tensors and apply the transformations.
The main transformations are:
- CenterCrop
- Resize
- ToTensor
The training transformations are:
- CenterCrop
- Resize
- RandomHorizontalFlip
- RandomRotation
- ColorJitter
- ToTensor
The images are converted to RGB.
"""
@dataclass
class PreparedGalaxyClassificationDataset(Dataset):
    images: list[str]
    labels: pd.DataFrame
    transform: Any

    @classmethod
    def from_unprepared(cls, dataset: GalaxyDataset, train:bool=True) -> Self:
        if train:
            # Apply transformations for training
            transform = transforms.Compose([
                transforms.CenterCrop(207),
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), #modify randomnly the brightness, contrast and saturation of the image
                transforms.ToTensor() #convert the pixel values to float between 0 and 1
            ])
        else:          
            transform = transforms.Compose([
            transforms.CenterCrop(207),
            transforms.Resize((64, 64)),
            transforms.ToTensor()  #convert the pixel values to float between 0 and 1
        ])
        return cls(images=dataset.images, labels=dataset.labels, transform=transform) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        # 
        img_path = self.images[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)   

        
        id = int(img_path.split("/")[-1].split(".")[0])
        label = int(self.labels.loc[id]["label"])
        return dict(images=image, labels=torch.tensor(label, dtype=torch.long))
"""
Class to split the dataset into training, validation and test sets.
It uses a GalaxyDataset as input and PreparedGalaxyDataset to transform the images.
It returns three dataloaders for training, validation and test sets.
"""    
@dataclass
class SplitGalaxyClassificationDataSet:
    training_dataloader: DataLoader
    validation_dataloader: DataLoader
    test_dataloader: DataLoader

    def __init__(self, dataset:GalaxyDataset, batch_size: int=256, validation_fraction:float=0.1, test_fraction:float=0.1):
        val_size = int(validation_fraction * len(dataset))
        test_size = int(test_fraction * len(dataset))
        train_size = len(dataset) - val_size - test_size
        # Ensure that the split sizes are valid
        if val_size+test_size+train_size != len(dataset):
            raise ValueError("The split sizes do not match the dataset size.")
        # Split the dataset into training, validation, and test sets
        train_ds, val_ds, test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        # Convert the split datasets back to PreparedGalaxyDataset
        train_ds=PreparedGalaxyClassificationDataset.from_unprepared(train_ds.dataset, train=True)
        val_ds=PreparedGalaxyClassificationDataset.from_unprepared(val_ds.dataset, train=False)
        test_ds=PreparedGalaxyClassificationDataset.from_unprepared(test_ds.dataset, train=False)

        # Create DataLoaders for each split
        self.training_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        self.validation_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) 
        self.test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        

#       ------------------------ REGRESSION ------------------------

"""
Class to convert the images into tensors and apply the transformations.
The main transformations are:
- CenterCrop
- Resize
- ToTensor
The training transformations are:
- CenterCrop
- Resize
- RandomHorizontalFlip
- RandomRotation
- ColorJitter
- ToTensor
The images are converted to RGB.

The labels are preprocessed by clipping the values to a minimum of EPS, raising them to the power of GAMMA, and normalizing them.
"""
@dataclass
class PreparedGalaxyRegressionDataset(Dataset):
    images: list[str]
    labels: pd.DataFrame
    transform: Any

    @classmethod
    def from_unprepared(cls, dataset: GalaxyDataset, train:bool=True) -> Self:
        if train:
            # Apply transformations for training
            transform = transforms.Compose([
                transforms.CenterCrop(207),
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), #modify randomnly the brightness, contrast and saturation of the image
                transforms.ToTensor() #convert the pixel values to float between 0 and 1
            ])
        else:          
            transform = transforms.Compose([
            transforms.CenterCrop(207),
            transforms.Resize((64, 64)),
            transforms.ToTensor()  #convert the pixel values to float between 0 and 1
        ])
        labels=dataset.labels.copy()
        for col in CLASS_GROUPS.values():
            labels[col] = labels[col].clip(lower=EPS)
            labels[col] = labels[col].pow(GAMMA)    
        return cls(images=dataset.images, labels=dataset.labels, transform=transform) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        
        img_path = self.images[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)   

        
        id = int(Path(img_path).stem)
        row = self.labels.loc[id]
        
        q1 = torch.tensor([row["Class1.1"], row["Class1.2"], row["Class1.3"]], dtype=torch.float32)
        q2 = torch.tensor([row["Class2.1"], row["Class2.2"]], dtype=torch.float32)
        q7 = torch.tensor([row["Class7.1"], row["Class7.2"], row["Class7.3"]], dtype=torch.float32)

        return {
            "images": image,
            "labels": {
                "q1": q1,
                "q2": q2,
                "q7": q7  
            }
        }
        
@dataclass
class SplitGalaxyRegressionDataset:
    training_dataloader: DataLoader
    validation_dataloader: DataLoader
    test_dataloader: DataLoader
    loss_weights:dict[str, float]= field(init=False)

    def __init__(self, dataset: GalaxyDataset, batch_size: int = 256, validation_fraction: float = 0.1, test_fraction: float = 0.1):
        val_size = int(validation_fraction * len(dataset))
        test_size = int(test_fraction * len(dataset))
        train_size = len(dataset) - val_size - test_size

        if val_size + test_size + train_size != len(dataset):
            raise ValueError("Dataset split sizes do not sum to total dataset size")

        # Split indices
        train_ds, val_ds, test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Prepare each split with appropriate transforms
        train_ds = PreparedGalaxyRegressionDataset.from_unprepared(train_ds.dataset, train=True)
        val_ds   = PreparedGalaxyRegressionDataset.from_unprepared(val_ds.dataset, train=False)
        test_ds  = PreparedGalaxyRegressionDataset.from_unprepared(test_ds.dataset, train=False)

        # Create DataLoaders
        self.training_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        self.validation_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        train_labels = train_ds.labels  
        means = train_labels.mean()  
        loss_weights = (1.0 / (means + EPS)).to_dict()
        self.loss_weights = loss_weights