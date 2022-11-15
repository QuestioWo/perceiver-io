import os
import sys
import copy
from typing import Optional

import pytorch_lightning as pl
import SimpleITK as sitk
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from perceiver.data.segmentation.common import channels_to_last, SegmentationPreprocessor, lift_transform

IMAGE_SIZE = (512, 512) # Adjustable meta parameter
IMAGE_HEIGHT = 298 # Adjustable meta parameter

class MICCAIPreprocessor(SegmentationPreprocessor):
    def __init__(self, normalize: bool = True, channels_last: bool = True):
        super().__init__(miccai_transform(normalize, channels_last))


class MICCAIDataset(Dataset) :
    def __init__(self, data, transforms=None) :
        self.data = data
        self.transforms = transforms

    def set_transform(self, transforms) :
        self.transforms = transforms
        
    def __getitem__(self, index):
        sample = self.data[index]
        if self.transforms:
            sample = self.transforms(sample)
        return sample
    
    def __len__(self) :
        return len(self.data)


class MICCAILoader() :
    IMAGES_TR = "imagesTr"
    LABELS_TR = "labelsTr"
    
    TRAIN_SIZE = 60
    TEST_SIZE = 20
    # VAL_SIZE = 20 # the rest
    
    BASIC_DATASET_ITEM = {'label' : None, 'image' : None}

    LIMIT_SCAN_COUNT = 10

    DATA_DTYPE = np.float64
    
    def __init__(self, root) :
        self.root = root

        if not os.path.exists(root) :
            print("Cannot find dataset at specified root")
            sys.exit(1)

        self.images_dir = os.path.join(self.root, self.IMAGES_TR)
        if not os.path.exists(self.images_dir) :
            print("Cannot find imageTr inside dataset")
            sys.exit(1)

        self.labels_dir = os.path.join(self.root, self.LABELS_TR)
        if not os.path.exists(self.labels_dir) :
            print("Cannot find labelsTr inside dataset")
            sys.exit(1)

        self._no_files = min(len(os.listdir(self.labels_dir)), self.LIMIT_SCAN_COUNT)

        self.data = []
        self._load_data()

    def _load_data(self) :
        for i in tqdm(range(self._no_files)) :
            filename = os.listdir(self.images_dir)[i]

            itk_img = sitk.ReadImage(os.path.join(self.images_dir, filename))
            # cast array as torch cannot convert arrays of dtype=np.uint16 and transformations require floating point data
            img = torch.from_numpy(sitk.GetArrayFromImage(itk_img).astype(self.DATA_DTYPE))

            itk_img_seg = sitk.ReadImage(os.path.join(self.labels_dir, filename))
            img_seg = torch.from_numpy(sitk.GetArrayFromImage(itk_img_seg).astype(self.DATA_DTYPE)) 

            image_object = copy.copy(self.BASIC_DATASET_ITEM)
            image_object['label'] = img_seg
            image_object['image'] = img
            self.data.append(image_object)

    def get_split(self, split) :
        _training_split = int(self._no_files * (100 / self.TRAIN_SIZE))
        _test_split = int(self._no_files * (100 / self.TEST_SIZE))
        _val_split = self._no_files - (_training_split + _test_split)

        start_location = 0
        split_size = self._no_files
        if split == "train" :
            start_location = 0
            split_size = _training_split
        elif split == "test" :
            start_location = _training_split
            split_size = _test_split
        elif split == "val" :
            start_location = _training_split + _test_split
            split_size = _val_split

        return MICCAIDataset(self.data[start_location:split_size])


class MICCAIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str = "AMOS22",
        normalize: bool = True,
        channels_last: bool = True,
        random_crop: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = True,
        shuffle: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.channels_last = channels_last

        self.tf_train = miccai_transform(normalize, channels_last, random_crop=random_crop)
        self.tf_valid = miccai_transform(normalize, channels_last, random_crop=None)

        self.ds_train = None
        self.ds_valid = None
        
        self.dataset_loader = None

    @property
    def num_classes(self):
        return 16

    @property
    def image_shape(self):
        if self.hparams.channels_last:
            return (IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_HEIGHT)
        else:
            return (IMAGE_HEIGHT, IMAGE_SIZE[0], IMAGE_SIZE[1])

    def load_dataset(self, split: Optional[str] = None):
        if self.dataset_loader == None :
            self.dataset_loader = MICCAILoader(self.hparams.dataset_dir)
        
        return self.dataset_loader.get_split(split)

    def prepare_data(self) -> None:
        self.load_dataset()

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = self.load_dataset(split="train")
        self.ds_train.set_transform(lift_transform(self.tf_train))

        self.ds_valid = self.load_dataset(split="test")
        self.ds_valid.set_transform(lift_transform(self.tf_valid))

        for i in range(len(self.ds_train)) :
            print(self.ds_train[i]['image'].shape)

        def showNii(img):
            for i in range(img.shape[0]):
                print(i)
                plt.imshow(img[i, :, :], cmap='gray')
                plt.show()

        # showNii(self.ds_train[0]['image'])

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            shuffle=self.hparams.shuffle,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


def miccai_transform(normalize: bool = True, channels_last: bool = True, random_crop: Optional[int] = None):
    transform_list = []

    transform_list.append(transforms.Resize(IMAGE_SIZE))
    
    def stack_to_same_size(img: torch.Tensor) :
        stack_images = torch.zeros((IMAGE_HEIGHT - img.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1]))
        return torch.vstack([img, stack_images])

    transform_list.append(stack_to_same_size)

    if random_crop is not None:
        transform_list.append(transforms.RandomCrop(random_crop))

    if normalize:
        transform_list.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))

    if channels_last:
        transform_list.append(channels_to_last)
    
    return transforms.Compose(transform_list)
