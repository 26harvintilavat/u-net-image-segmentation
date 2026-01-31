import shutil
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
full_dir = root_dir/'data'/'processed'
small_dir = root_dir/'data'/'small'

num_train = 100
num_val = 20

def copy_subset(split, num_samples):
    img_src = full_dir/split/"images"
    mask_src = full_dir/split/"masks"

    img_dst = small_dir/split/"images"
    mask_dst = small_dir/split/"masks"

    img_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)

    images = sorted(img_src.iterdir())[:num_samples]

    for img_path in images:
        mask_path = mask_src / (img_path.stem + ".png")

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path.name}")
        
        shutil.copy(img_path, img_dst/img_path.name)
        shutil.copy(mask_path, mask_dst/mask_path.name)

    print(f"Copied {len(images)} {split} samples")

def main():
    copy_subset("train", num_train)
    copy_subset("val", num_val)
    print("small dataset created successfully")

if __name__ == "__main__":
    main()