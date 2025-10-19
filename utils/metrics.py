"""
metrics
"""
import torch
import torch.nn.functional as F


def dice_coefficient(pred, target, smooth=1e-6):
    """dice coefficient"""
    if pred.dim() == 4:
        pred = F.softmax(pred, dim=1)
        pred = pred[:, 1, :, :]
    
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """IoU"""
    if pred.dim() == 4:
        pred = F.softmax(pred, dim=1)
        pred = pred[:, 1, :, :]
    
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def pixel_accuracy(pred, target):
    """Pixel Accuracy"""
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).sum()
    total = target.numel()
    
    return (correct / total).item()


class DiceLoss(torch.nn.Module):
    """Dice Loss"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        target = target.long()
        
        target_one_hot = F.one_hot(target, num_classes=pred.size(1))
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()
