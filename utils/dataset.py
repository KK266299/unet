"""
COVID-19 CT数据集加载器
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A


class COVID19Dataset(Dataset):
    """COVID-19"""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        self.images = sorted(os.listdir(images_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        mask = (mask > 127).astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask


def get_transforms(image_size=256, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size)
        ])
