import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
# from utils import (
#    load_checkpoint,
 #   save_checkpoint,
  #  get_loaders,
   # check_accuracy,
    #save_predictions_as_imgs,
#)

# Hyperparameters etc.
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
num_epochs = 3
num_workers = 2
image_height = 160
image_width = 240
pin_memory = True
load_model = True


