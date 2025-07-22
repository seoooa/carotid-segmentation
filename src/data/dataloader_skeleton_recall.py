import autorootcwd
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandFlipd,
    CropForegroundd,
    Compose,
    Spacingd,
)
from monai.data import CacheDataset, DataLoader
import os
import lightning.pytorch as pl
from pathlib import Path
import yaml
from src.data.transforms import AddSkeletonToDatad


class CarotidSkeletonDataModule(pl.LightningDataModule):
    """
    Data module that supports skeleton recall loss training by generating skeleton data
    """
    # define intensity range for each target
    INTENSITY_RANGES = {
        "carotid": {"a_min": -20, "a_max": 380},
        "mandible": {"a_min": -150, "a_max": 1900},
        "spinalcord": {"a_min": -100, "a_max": 150},
        "thyroid": {"a_min": -110, "a_max": 320},
    }

    def __init__(
        self,
        data_dir: str = "data/Han_Seg",
        batch_size: int = 4,
        patch_size: tuple = (96, 96, 96),
        num_workers: int = 4,
        cache_rate: float = 0.1,
        fold_number: int = 1,
        target: str = "carotid",
        use_skeleton: bool = True,
        skeleton_do_tube: bool = True
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.fold_number = fold_number
        self.target = target
        self.use_skeleton = use_skeleton
        self.skeleton_do_tube = skeleton_do_tube
        
        # set intensity range for each target
        if target not in self.INTENSITY_RANGES:
            raise ValueError(f"Unsupported target: {target}. Must be one of {list(self.INTENSITY_RANGES.keys())}")
        self.intensity_range = self.INTENSITY_RANGES[target]

    def load_data_splits(self, yaml_path, fold_number):
        # Read the YAML file
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)
        
        # Extract train, val, test splits from the specified fold
        fold_key = f"fold_{fold_number}"
        fold = data["cross_validation_splits"][fold_number-1][fold_key]
        train_split = fold["train"]
        val_split = fold["val"]
        test_split = fold["test"]

        # Add folder path to each entry in the splits and create dictionaries
        base_dir = os.path.dirname(yaml_path)
        train_split = [
            {
                "image": os.path.join(base_dir, entry, "CT.nii.gz"),
                "label": os.path.join(base_dir, entry, "label.nii.gz"),
            }
            for entry in train_split
        ]
        val_split = [
            {
                "image": os.path.join(base_dir, entry, "CT.nii.gz"),
                "label": os.path.join(base_dir, entry, "label.nii.gz"),
            }
            for entry in val_split
        ]
        test_split = [
            {
                "image": os.path.join(base_dir, entry, "CT.nii.gz"),
                "label": os.path.join(base_dir, entry, "label.nii.gz"),
            }
            for entry in test_split
        ]
        print(f"Loaded data splits from {yaml_path} for fold {fold_number}")

        return train_split, val_split, test_split

    def prepare_data(self):
        # Base transforms for all stages
        base_keys = ["image", "label"]
        
        # Training transforms
        train_transform_list = [
            LoadImaged(keys=base_keys),
            EnsureChannelFirstd(keys=base_keys),
            Orientationd(keys=base_keys, axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.intensity_range["a_min"],
                a_max=self.intensity_range["a_max"],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=base_keys, source_key="image"),
        ]
        
        # Add skeleton generation if enabled
        if self.use_skeleton:
            train_transform_list.append(
                AddSkeletonToDatad(
                    keys=["label"], 
                    do_tube=self.skeleton_do_tube,
                    allow_missing_keys=False
                )
            )
            # Update keys to include skeleton for subsequent transforms
            base_keys_with_skel = base_keys + ["skeleton"]
        else:
            base_keys_with_skel = base_keys
        
        # Continue with augmentation transforms
        train_transform_list.extend([
            RandCropByPosNegLabeld(
                keys=base_keys_with_skel,
                label_key="label",
                spatial_size=self.patch_size,
                pos=1,
                neg=1,
                num_samples=6,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=base_keys_with_skel,
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=base_keys_with_skel,
                spatial_axis=[1],
                prob=0.10,
            ),
            RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5),
        ])
        
        self.train_transforms = Compose(train_transform_list)

        # Validation transforms (no augmentation)
        val_transform_list = [
            LoadImaged(keys=base_keys),
            EnsureChannelFirstd(keys=base_keys),
            Orientationd(keys=base_keys, axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.intensity_range["a_min"],
                a_max=self.intensity_range["a_max"],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=base_keys, source_key="image"),
        ]
        
        # Add skeleton generation for validation too if enabled
        if self.use_skeleton:
            val_transform_list.append(
                AddSkeletonToDatad(
                    keys=["label"], 
                    do_tube=self.skeleton_do_tube,
                    allow_missing_keys=False
                )
            )
        
        self.val_transforms = Compose(val_transform_list)

    def setup(self, stage=None):
        # set up the correct data path
        train_files, val_files, test_files = self.load_data_splits(
            yaml_path=f"data/Han_Seg_{self.target.capitalize()}/data_splits.yaml", 
            fold_number=self.fold_number
        )

        print(f"Found {len(train_files)} training cases")
        print(f"Found {len(val_files)} validation cases")
        print(f"Found {len(test_files)} test cases")

        self.train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )

        self.val_ds = CacheDataset(
            data=val_files,
            transform=self.val_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )

        self.test_ds = CacheDataset(
            data=test_files,
            transform=self.val_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers
        ) 