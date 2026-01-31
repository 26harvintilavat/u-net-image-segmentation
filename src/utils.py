import torch
import torchvision
from data.dataset import OxfordIIITPetDataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers = 2,
        pin_memory = True,    
):
    train_ds = OxfordIIITPetDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        trasnform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = OxfordIIITPetDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        trasnform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            y = (y > 0).float()

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)

            intersection = (preds * y).sum().item()
            dice_score += (2 * intersection) / (
                preds.sum().item() + y.sum().item() + 1e-8
            )

    acc = num_correct / num_pixels * 100
    dice = dice_score / len(loader)

    print(f"Got {num_correct}/{num_pixels} with acc {acc:.2f}%")
    print(f"Dice score: {dice:.4f}")

    model.train()

def save_predictions_as_imgs(
        loader, model, folder="data/saved_images/", device=device
): 
    import os
    os.makedirs(folder, exist_ok=True)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/gt_{idx}.png")

    model.train()