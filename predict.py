"""
prediction
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from configs import Config
from models import UNet
from utils import COVID19Dataset, get_transforms, dice_coefficient, iou_score


def load_model(checkpoint_path, device):
    model = UNet(
        in_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        init_features=Config.INIT_FEATURES
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']+1}")
    print(f"Best Dice: {checkpoint['best_dice']:.4f}")
    
    return model


def predict(model, dataset, num_samples=8, save_path='./results/predictions.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    device = next(model.parameters()).device
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    dice_list = []
    iou_list = []
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            image_input = image.unsqueeze(0).to(device)
            
            output = model(image_input)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu()
            
            dice = dice_coefficient(output, mask.unsqueeze(0).to(device))
            iou = iou_score(output, mask.unsqueeze(0).to(device))
            
            dice_list.append(dice)
            iou_list.append(iou)
            
            axes[i, 0].imshow(image.squeeze(0).cpu().numpy(), cmap='gray')
            axes[i, 0].set_title('Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred.numpy(), cmap='jet', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Prediction Dice={dice:.3f}')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPredictions saved to {save_path}")
    print(f"Average Dice: {np.mean(dice_list):.4f}")
    print(f"Average IoU: {np.mean(iou_list):.4f}")


def main():
    print("Loading model...")
    model = load_model(Config.BEST_MODEL_PATH, Config.DEVICE)
    
    print("Loading dataset...")
    transform = get_transforms(Config.IMAGE_SIZE, is_train=False)
    dataset = COVID19Dataset(Config.IMAGES_DIR, Config.MASKS_DIR, transform=transform)
    
    train_size = int(Config.TRAIN_SPLIT * len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))
    val_dataset = Subset(dataset, indices[train_size:])
    
    print(f"Validation set: {len(val_dataset)} images")
    
    print("\nPredicting...")
    predict(model, val_dataset, num_samples=8)
    
    print("Done")


if __name__ == '__main__':
    main()