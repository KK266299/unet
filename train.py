"""
U-Net训练脚本
XINYAO LIU 2025-10-19
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from configs import Config
from models import UNet
from utils import COVID19Dataset, get_transforms, DiceLoss, dice_coefficient, iou_score, pixel_accuracy


def plot_training_history(history, save_path='./results/training_history.png'):
    """绘制训练历史曲线"""
    for key in history:
        if isinstance(history[key][0], torch.Tensor):
            history[key] = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in history[key]]
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    axes[0, 1].plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    axes[0, 1].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['val_iou'], 'g-', label='Val IoU', linewidth=2)
    axes[1, 0].set_title('IoU Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, history['val_acc'], 'm-', label='Val Accuracy', linewidth=2)
    axes[1, 1].set_title('Pixel Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += dice_coefficient(outputs, masks)

        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice_coefficient(outputs, masks):.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    return epoch_loss, epoch_dice


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="验证中"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            running_dice += dice_coefficient(outputs, masks)
            running_iou += iou_score(outputs, masks)
            running_acc += pixel_accuracy(outputs, masks)

    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    return epoch_loss, epoch_dice, epoch_iou, epoch_acc


def main():
    Config.create_dirs()

    print("="*60)
    print("U-Net COVID-19 CT分割训练")
    print("="*60)
    print(f"设备: {Config.DEVICE}")
    print(f"批量大小: {Config.BATCH_SIZE}")
    print(f"学习率: {Config.LEARNING_RATE}")
    print(f"训练轮数: {Config.NUM_EPOCHS}")
    print("="*60)

    transform_train = get_transforms(image_size=Config.IMAGE_SIZE, is_train=True)
    transform_val = get_transforms(image_size=Config.IMAGE_SIZE, is_train=False)

    train_full_dataset = COVID19Dataset(
        Config.IMAGES_DIR, 
        Config.MASKS_DIR, 
        transform=transform_train
    )
    
    val_full_dataset = COVID19Dataset(
        Config.IMAGES_DIR, 
        Config.MASKS_DIR, 
        transform=transform_val
    )

    train_size = int(Config.TRAIN_SPLIT * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size

    indices = torch.randperm(len(train_full_dataset), generator=torch.Generator().manual_seed(42))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)
    
    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")
    print("="*60)

    train_loader = DataLoader(    
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    model = UNet(
        in_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        init_features=Config.INIT_FEATURES
    ).to(Config.DEVICE)

    criterion = DiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )   

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    best_dice = 0.0
    
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_acc': []
    }

    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}]")
        print("-" * 60)
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_dice, val_iou, val_acc = validate(model, val_loader, criterion, Config.DEVICE)

        history['train_loss'].append(float(train_loss))
        history['train_dice'].append(float(train_dice))
        history['val_loss'].append(float(val_loss))
        history['val_dice'].append(float(val_dice))
        history['val_iou'].append(float(val_iou))
        history['val_acc'].append(float(val_acc))

        print(f"训练集 - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"验证集 - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, Acc: {val_acc:.4f}")

        scheduler.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, Config.BEST_MODEL_PATH)
            print(f"模型已保存，最佳 Dice: {best_dice:.4f}")
        
        if (epoch + 1) % 10 == 0:
            plot_training_history(history)
    
    print("\n" + "="*60)
    print(f"训练完成! 最佳Dice: {best_dice:.4f}")
    print("="*60)
    
    plot_training_history(history)


if __name__ == "__main__":
    main()