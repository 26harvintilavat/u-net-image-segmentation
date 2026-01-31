from pathlib import Path
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.model import UNET
from torch.amp import GradScaler, autocast
ROOT_DIR = Path(__file__).resolve().parent.parent
from src.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
scaler = GradScaler(enabled=use_amp)
batch_size = 16
num_epochs = 3
num_workers = 2
image_height = 160
image_width = 240
pin_memory = device == "cuda"
load_model = True
train_image_dir = ROOT_DIR/'data'/'small'/'train'/'images'
train_mask_dir = ROOT_DIR/'data'/'small'/'train'/'masks'
val_image_dir = ROOT_DIR/'data'/'small'/'val'/'images'
val_mask_dir = ROOT_DIR/'data'/'small'/'val'/'masks'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        optimizer.zero_grad()

        # forward
        with autocast(device_type=device, enabled=use_amp):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,       
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,            
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(device=device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_loaders(
        train_image_dir,
        train_mask_dir,
        val_image_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory,
    )

    use_amp = device == "cuda"
    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

    # save model
    checkpoint = {
        'state_dict' : model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    # check accuracy
    check_accuracy(val_loader, model, device)

    # print some example to a folder
    save_predictions_as_imgs(
         val_loader, model=model, folder=ROOT_DIR/"data"/"saved_images", device=device,
    )

if __name__ == "__main__":
    main()