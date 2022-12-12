import json
import math
import os
import sys
import copy
from typing import Optional

import pytorch_lightning as pl
import SimpleITK as sitk
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from skimage.transform import rescale

from perceiver.data.segmentation.common import channels_to_last, SegmentationPreprocessor, lift_transform, coregister_scan

IMAGE_SIZE = (300, 512, 512)
SCAN_TO_COREGISTER_TO = None

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
	IMAGES_TR_PREPROCESSED = "imagesTr_preprocessed"
	LABELS_TR_PREPROCESSED = "labelsTr_preprocessed"
	
	TRAIN_SIZE = 60
	TEST_SIZE = 20
	# VAL_SIZE = 20 # the rest
	
	BASIC_DATASET_ITEM = {'label' : None, 'image' : None, 'filename' : None}

	LIMIT_SCAN_COUNT = 10

	DATA_DTYPE = np.float64
	
	def __init__(self, root) :
		self.root = root

		if not os.path.exists(root) :
			print("Cannot find dataset at specified root")
			sys.exit(1)

		self.images_dir = os.path.join(self.root, self.IMAGES_TR)
		self.images_preprocessed_dir = os.path.join(self.root, self.IMAGES_TR_PREPROCESSED)
		if not os.path.exists(self.images_dir) :
			print("Cannot find imageTr inside dataset")
			sys.exit(1)

		self.labels_dir = os.path.join(self.root, self.LABELS_TR)
		self.labels_preprocessed_dir = os.path.join(self.root, self.LABELS_TR_PREPROCESSED)
		if not os.path.exists(self.labels_dir) :
			print("Cannot find labelsTr inside dataset")
			sys.exit(1)

		self._no_files = min(len(os.listdir(self.labels_dir)), self.LIMIT_SCAN_COUNT)

		self.data = []
		self._load_data()

	def _load_data(self) :
		coregister_scans = False

		if not os.path.exists(self.images_preprocessed_dir) :
			os.mkdir(self.images_preprocessed_dir)
		else :
			for i in range(self._no_files) :
				filename = os.listdir(self.images_dir)[i]

				if not filename in os.listdir(self.images_preprocessed_dir) :
					coregister_scans = True
					break

		global IMAGE_SIZE, SCAN_TO_COREGISTER_TO
		if coregister_scans :
			largest_size = 0

			for i in tqdm(range(self._no_files)) :
				filename = os.listdir(self.images_dir)[i]

				itk_img = sitk.ReadImage(os.path.join(self.images_dir, filename))
				# cast array as torch cannot convert arrays of dtype=np.uint16 and transformations require floating point data
				img = torch.from_numpy(sitk.GetArrayFromImage(itk_img).astype(self.DATA_DTYPE))

				itk_img_seg = sitk.ReadImage(os.path.join(self.labels_dir, filename))
				img_seg = torch.from_numpy(sitk.GetArrayFromImage(itk_img_seg).astype(self.DATA_DTYPE)) 

				image_object = copy.copy(self.BASIC_DATASET_ITEM)
				image_object['label'] = None
				image_object['image'] = None
				image_object['filename'] = filename
				self.data.append(image_object)

				if (itk_img.GetWidth() * itk_img.GetHeight() * itk_img.GetDepth()) > largest_size :
					# IMAGE_SIZE = (itk_img.GetWidth(), itk_img.GetHeight(), itk_img.GetDepth())
					SCAN_TO_COREGISTER_TO = [filename, img.numpy(), img_seg.numpy()]
					largest_size = (itk_img.GetWidth() * itk_img.GetHeight() * itk_img.GetDepth())

					if (itk_img.GetWidth() * itk_img.GetHeight() * itk_img.GetDepth()) > (IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]) :
						raise ValueError("One of the files loaded is larger than the image size to be used")

			print("Preparing", SCAN_TO_COREGISTER_TO[0], "for coregistration")

			if SCAN_TO_COREGISTER_TO[1].shape[1] != IMAGE_SIZE[1] or SCAN_TO_COREGISTER_TO[1].shape[2] != IMAGE_SIZE[2] :
				scale_factor = (max(IMAGE_SIZE[1], IMAGE_SIZE[2]) / min(SCAN_TO_COREGISTER_TO[1].shape[1], SCAN_TO_COREGISTER_TO[1].shape[2]))
				scale_factor_width = int(scale_factor * SCAN_TO_COREGISTER_TO[1].shape[1])
				scale_factor_height = int(scale_factor * SCAN_TO_COREGISTER_TO[1].shape[2])
				scale_factor_depth = int(scale_factor * SCAN_TO_COREGISTER_TO[1].shape[0])
				SCAN_TO_COREGISTER_TO[1] = rescale(SCAN_TO_COREGISTER_TO[1], (scale_factor_depth, scale_factor_width, scale_factor_height))
				SCAN_TO_COREGISTER_TO[2] = rescale(SCAN_TO_COREGISTER_TO[2], (scale_factor_depth, scale_factor_width, scale_factor_height))

			if np.product(SCAN_TO_COREGISTER_TO[1].shape) > np.product(np.array(IMAGE_SIZE)) :
				raise ValueError("Scan to coregister to is too large")

			if SCAN_TO_COREGISTER_TO[1].shape[0] < IMAGE_SIZE[0] :
				diff = IMAGE_SIZE[0] - SCAN_TO_COREGISTER_TO[1].shape[0]
				diff_top = math.ceil(diff / 2.0)
				diff_bottom = math.floor(diff / 2.0)
				SCAN_TO_COREGISTER_TO[1] = np.pad(SCAN_TO_COREGISTER_TO[1], ((diff_top, diff_bottom)), mode="constant")
				SCAN_TO_COREGISTER_TO[2] = np.pad(SCAN_TO_COREGISTER_TO[2], ((diff_top, diff_bottom)), mode="constant")

			print("Coregistering scans to :", SCAN_TO_COREGISTER_TO[0], "of size", IMAGE_SIZE)

			for i in tqdm(range(self._no_files)) :
				image_object = self.data[i]
				img_np = img_label_np = transformation = scaling = None

				print(image_object['filename'])

				if image_object['filename'] != SCAN_TO_COREGISTER_TO[0] :
					itk_img = sitk.ReadImage(os.path.join(self.images_dir, image_object['filename']))
					img = torch.from_numpy(sitk.GetArrayFromImage(itk_img).astype(self.DATA_DTYPE))
					itk_img_seg = sitk.ReadImage(os.path.join(self.labels_dir, image_object['filename']))
					img_seg = torch.from_numpy(sitk.GetArrayFromImage(itk_img_seg).astype(self.DATA_DTYPE)) 

					img_np, img_label_np, transformation, scaling = coregister_scan(
						img,
						img_seg,
						SCAN_TO_COREGISTER_TO[1]
					)

					image_object['image'] = torch.from_numpy(img_np)
					image_object['label'] = torch.from_numpy(img_label_np)
				else :
					image_object['image'] = torch.from_numpy(SCAN_TO_COREGISTER_TO[1])
					image_object['label'] = torch.from_numpy(SCAN_TO_COREGISTER_TO[2])

				sitk.WriteImage(sitk.GetImageFromArray(image_object['image'].numpy()), os.path.join(self.images_preprocessed_dir, image_object['filename']))
				sitk.WriteImage(sitk.GetImageFromArray(image_object['label'].numpy()), os.path.join(self.labels_preprocessed_dir, image_object['filename']))
				if transformation != None :
					sitk.WriteTransform(transformation, os.path.join(self.images_preprocessed_dir, image_object['filename'].replace(".nii.gz", "_transformation.tfm")))
					with open(os.path.join(self.images_preprocessed_dir, image_object['filename'].replace(".nii.gz", "_metadata.json")), "w") as f :
						f.write(json.dumps({"scaling": scaling}))

				image_object['label'] = None
				image_object['image'] = None

		self.data = []

		for i in tqdm(range(self._no_files)) :
			filename = list(filter(lambda x: x.endswith(".nii.gz"), os.listdir(self.images_preprocessed_dir)))[i] # check the loading

			itk_img = sitk.ReadImage(os.path.join(self.images_preprocessed_dir, filename))
			# cast array as torch cannot convert arrays of dtype=np.uint16 and transformations require floating point data
			img = torch.from_numpy(sitk.GetArrayFromImage(itk_img).astype(self.DATA_DTYPE))

			itk_img_seg = sitk.ReadImage(os.path.join(self.labels_preprocessed_dir, filename))
			img_seg = torch.from_numpy(sitk.GetArrayFromImage(itk_img_seg).astype(self.DATA_DTYPE)) 

			image_object = copy.copy(self.BASIC_DATASET_ITEM)
			image_object['label'] = img_seg
			image_object['image'] = img
			image_object['filename'] = filename
			self.data.append(image_object)

			if (itk_img.GetWidth() * itk_img.GetHeight() * itk_img.GetDepth()) > (IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]) :
				IMAGE_SIZE = (itk_img.GetWidth(), itk_img.GetHeight(), itk_img.GetDepth())
			
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
			return (IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2])
		else:
			return (IMAGE_SIZE[2], IMAGE_SIZE[0], IMAGE_SIZE[1])

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

		# for i in range(len(self.ds_train)) :
		# 	print(self.ds_train[i]['image'].shape)

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

	if random_crop is not None:
		transform_list.append(transforms.RandomCrop(random_crop))

	if normalize:
		transform_list.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))

	if channels_last:
		transform_list.append(channels_to_last)
	
	return transforms.Compose(transform_list)
