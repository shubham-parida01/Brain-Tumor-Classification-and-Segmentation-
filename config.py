"""
Configuration parameters for brain tumor detection models.
"""
import os
from pathlib import Path
import torchvision.transforms as transforms
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class Config:
    def __init__(self):
        # Base paths
        self.BASE_DIR = Path(__file__).parent.absolute()
        self.DATA_DIR = self.BASE_DIR / "data"
        
        # Dataset configuration
        self.TRAIN_DIR = self.DATA_DIR / "training"
        self.TEST_DIR = self.DATA_DIR / "testing"
        self.CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
        self.NUM_CLASSES = len(self.CLASS_NAMES)
        
        # Image preprocessing
        self.IMG_SIZE = 224
        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 4
        
        # DWT preprocessing
        self.WAVELET = 'db4'
        self.DWT_LEVELS = 3
        self.DWT_MODE = 'soft'
        self.DWT_THRESHOLD = 'BayesShrink'
        
        # CSA feature selection
        self.CSA_N_CROWS = 50
        self.CSA_MAX_ITER = 100
        self.CSA_FLIGHT_LENGTH = 2.0
        self.CSA_AWARENESS_PROB = 0.1
        
        # Model configuration
        self.BACKBONE = "resnet50"
        self.PRETRAINED = True
        self.FEATURE_EXTRACT = False
        self.DROPOUT_RATE = 0.3
        
        # Attention configuration
        self.ATTENTION_REDUCTION_RATIO = 16
        self.ATTENTION_KERNEL_SIZE = 7
        
        # Training configuration
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-4
        self.MOMENTUM = 0.9
        self.EPOCHS = 200
        self.EARLY_STOPPING_PATIENCE = 20
        
        # Learning rate scheduling
        self.LR_SCHEDULER = CosineAnnealingWarmRestarts
        self.LR_SCHEDULER_PARAMS = {
            'T_0': 10,  # Number of iterations for the first restart
            'T_mult': 2,  # A factor increases T_i after a restart
            'eta_min': 1e-6  # Minimum learning rate
        }
        
        # Loss functions
        self.CLASSIFICATION_LOSS = 'focal'
        self.SEGMENTATION_LOSS = 'dice_bce'
        self.LOSS_WEIGHTS = {
            'classification': 0.6,
            'segmentation': 0.4
        }
        
        # Focal loss parameters
        self.FOCAL_LOSS_ALPHA = [0.25, 0.25, 0.25, 0.25]  # Class weights
        self.FOCAL_LOSS_GAMMA = 2.0
        
        # Data augmentation
        self.AUGMENTATION_PROB = 0.5
        self.AUGMENTATION_PARAMS = {
            'rotation': (-15, 15),
            'translation': (0.1, 0.1),
            'scale': (0.9, 1.1),
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        }
        
        # Evaluation metrics
        self.METRICS = [
            'accuracy',
            'precision',
            'recall',
            'f1',
            'auc',
            'dice',
            'iou'
        ]
        
        # Checkpoints and logs
        self.CHECKPOINT_DIR = self.BASE_DIR / "checkpoints"
        self.LOGS_DIR = self.BASE_DIR / "logs"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        
        # Create directories
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        
        # Class weights for handling imbalance
        self.CLASS_WEIGHTS = [1.0, 1.2, 1.2, 1.2]
        
        # Visualization
        self.COLORS = ['blue', 'red', 'green', 'yellow']
        self.GRADCAM_ALPHA = 0.5
        self.SEGMENTATION_THRESHOLD = 0.5

# For backward compatibility
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "training"
TEST_DIR = DATA_DIR / "testing"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4
BACKBONE = "resnet50"
PRETRAINED = True
FEATURE_EXTRACT = False
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
CLASS_WEIGHTS = [1.0, 1.2, 1.2, 1.2]

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)