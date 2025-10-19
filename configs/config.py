import torch
import os

class Config:

    DATASET_PATH = '/mri3/liuxy/unet/dataset'
    IMAGES_DIR = os.path.join(DATASET_PATH, 'frames')
    MASKS_DIR = os.path.join(DATASET_PATH, 'masks')
    CHECKPOINT_DIR = './checkpoints'
    RESULTS_DIR = './results'
    
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'last_model.pth')

    IMAGE_SIZE = 256
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    NUM_CLASSES = 2

    INPUT_CHANNELS = 1
    INIT_FEATURES = 64

    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    
    @staticmethod
    def create_dirs():
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
