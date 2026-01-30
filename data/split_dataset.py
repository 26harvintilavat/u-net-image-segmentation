import os
import shutil
import random

# paths
base_url = 'the-oxfordiiit-pet-dataset'
image_dir = os.path.join(base_url, "images")
mask_dir = os.path.join(base_url, 'annotations', 'trimaps')

out_dir = "data/processed"
train_ratio = 0.8
seed = 42

# create output directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(out_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, split, 'masks'), exist_ok=True)

# collect image names (without extension issues)
image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.endswith('.jpg')
])

random.seed(seed)
random.shuffle(image_files)

split_idx = int(len(image_files) * train_ratio)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def copy_files(file_list, split):
    for img_name in file_list:
        base_name = os.path.splitext(img_name)[0]

        img_src = os.path.join(image_dir, img_name)
        mask_src = os.path.join(mask_dir, base_name + ".png")

        img_dst = os.path.join(out_dir, split, "images", img_name)
        mask_dst = os.path.join(out_dir, split, 'masks', base_name + ".png")

        if not os.path.exists(mask_src):
            print(f"Mask missing for {img_name}, skipping")
            continue

        shutil.copy(img_src, img_dst)
        shutil.copy(mask_src, mask_dst)

copy_files(train_files, "train")
copy_files(val_files, "val")

print("Dataset split completed")
print(f"Train samples: {len(train_files)}")
print(f"Val samples: {len(val_files)}")