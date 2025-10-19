"""
测试脚本
"""
import torch
from configs import Config
from models import UNet
from utils import COVID19Dataset, get_transforms, DiceLoss, dice_coefficient
from torch.utils.data import DataLoader


def test():
    print("测试开始...")
    
    dataset = COVID19Dataset(
        Config.IMAGES_DIR,
        Config.MASKS_DIR,
        transform=get_transforms(Config.IMAGE_SIZE, is_train=True)
    )
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    images, masks = next(iter(loader))
    print(f"数据: 图像{images.shape}, 掩码{masks.shape}")
    
    model = UNet(
        in_channels=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        init_features=Config.INIT_FEATURES
    ).to(Config.DEVICE)
    print(f"模型: {sum(p.numel() for p in model.parameters())/1e6:.1f}M参数")
    
    images = images.to(Config.DEVICE)
    masks = masks.to(Config.DEVICE)
    outputs = model(images)
    print(f"前向: 输出{outputs.shape}")
    
    criterion = DiceLoss()
    loss = criterion(outputs, masks)
    print(f"损失: {loss.item():.4f}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("反向: 完成")
    
    dice = dice_coefficient(outputs, masks)
    print(f"指标: Dice={dice:.4f}")
    
    print("\n测试通过")


if __name__ == '__main__':
    test()