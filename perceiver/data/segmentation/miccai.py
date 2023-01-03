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

from perceiver.data.segmentation.common import channels_to_last, SegmentationPreprocessor, interpolate_scan_by_scaling_factors, lift_transform, coregister_scan, ImageInfo

IMAGE_SIZE = (220, 256, 256)
IMAGE_SIZE = (165, 192, 192)
IMAGE_SIZE = (110, 128, 128)
IMAGE_SIZE = (55, 64, 64)
NUM_CLASSES = 16

class MICCAIPreprocessor(SegmentationPreprocessor):
	def __init__(self, channels_last: bool = True, normalize: bool = True):
		super().__init__(miccai_transform(channels_last, normalize))


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
	
	# NOTE: it appears that the train and test sets are combined for training epochs
	TRAIN_SIZE = 60
	TEST_SIZE = 20
	# VAL_SIZE = 20 # the rest
	
	BASIC_DATASET_ITEM = {'label' : None, 'image' : None, 'filename' : None}

	LIMIT_SCAN_COUNT = 10000

	SCAN_TO_COREGISTER_TO = None
	
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
			os.mkdir(self.labels_preprocessed_dir)
			coregister_scans = True
		else :
			for i in range(self._no_files) :
				filename = os.listdir(self.images_dir)[i]

				if not filename in os.listdir(self.images_preprocessed_dir) :
					coregister_scans = True
					break

		image_info = ImageInfo()
		if coregister_scans :
			largest_size = 0

			for i in tqdm(range(self._no_files)) :
				filename = os.listdir(self.images_dir)[i]

				itk_img = sitk.ReadImage(os.path.join(self.images_dir, filename), sitk.sitkFloat64)
				itk_img_seg = sitk.ReadImage(os.path.join(self.labels_dir, filename), sitk.sitkFloat64)

				itk_img = sitk.DICOMOrient(itk_img, "RPI")
				itk_img_seg = sitk.DICOMOrient(itk_img_seg, "RPI")

				image_object = copy.copy(self.BASIC_DATASET_ITEM)
				image_object['label'] = None
				image_object['image'] = None
				image_object['filename'] = filename
				self.data.append(image_object)

				world_image_width = itk_img.GetWidth() * itk_img.GetSpacing()[0]
				world_image_height = itk_img.GetHeight() * itk_img.GetSpacing()[1]
				world_image_depth = itk_img.GetDepth() * itk_img.GetSpacing()[2]
				image_world_volume = (world_image_width * world_image_height * world_image_depth)

				if image_world_volume > largest_size :
					self.SCAN_TO_COREGISTER_TO = [filename, itk_img, itk_img_seg]
					largest_size = image_world_volume

				image_info.size_width_height = max(itk_img.GetWidth(), itk_img.GetHeight(), image_info.size_width_height)
				image_info.size_depth = max(itk_img.GetDepth(), image_info.size_depth)
				image_info.spacing_width_height = min(itk_img.GetSpacing()[0], itk_img.GetSpacing()[1], image_info.spacing_width_height)
				image_info.spacing_depth = min(itk_img.GetSpacing()[2], image_info.spacing_depth)
				image_info.world_size_width_height = max(world_image_height, world_image_width, image_info.world_size_width_height)
				image_info.world_size_depth = max(world_image_depth, image_info.world_size_depth)

			print("Preparing largest scan - %s - for coregistration..." % self.SCAN_TO_COREGISTER_TO[0])
			
			# Resample for largest scalings
			spacing_scale_factor_width_height = (self.SCAN_TO_COREGISTER_TO[1].GetSpacing()[0] / image_info.spacing_width_height)
			spacing_scale_factor_depth = (self.SCAN_TO_COREGISTER_TO[1].GetSpacing()[2] / image_info.spacing_depth)
			scale_factors = (spacing_scale_factor_depth, spacing_scale_factor_width_height, spacing_scale_factor_width_height)

			self.SCAN_TO_COREGISTER_TO[1], self.SCAN_TO_COREGISTER_TO[2] = interpolate_scan_by_scaling_factors(
				self.SCAN_TO_COREGISTER_TO[1],
				self.SCAN_TO_COREGISTER_TO[2],
				scale_factors,
				(image_info.spacing_width_height, image_info.spacing_width_height, image_info.spacing_depth)
			)
			
			# Add padding to match largest world width/height/depth sizes
			current_world_width = self.SCAN_TO_COREGISTER_TO[1].GetWidth() * self.SCAN_TO_COREGISTER_TO[1].GetSpacing()[0]
			current_world_height = self.SCAN_TO_COREGISTER_TO[1].GetHeight() * self.SCAN_TO_COREGISTER_TO[1].GetSpacing()[1]
			current_world_depth = self.SCAN_TO_COREGISTER_TO[1].GetDepth() * self.SCAN_TO_COREGISTER_TO[1].GetSpacing()[2]
			
			world_width_diff = (image_info.world_size_width_height - current_world_width)
			world_height_diff = (image_info.world_size_width_height - current_world_height)
			world_depth_diff = (image_info.world_size_depth - current_world_depth)
			
			pixel_width_diff = (world_width_diff / self.SCAN_TO_COREGISTER_TO[1].GetSpacing()[0]) / 2
			pixel_height_diff = (world_height_diff / self.SCAN_TO_COREGISTER_TO[1].GetSpacing()[1]) / 2
			pixel_depth_diff = (world_depth_diff / self.SCAN_TO_COREGISTER_TO[1].GetSpacing()[2]) / 2
			if pixel_width_diff > 0 and pixel_height_diff > 0 :
				self.SCAN_TO_COREGISTER_TO[1] = sitk.ConstantPad(
					self.SCAN_TO_COREGISTER_TO[1],
					padLowerBound=(math.floor(pixel_width_diff), math.floor(pixel_height_diff), math.floor(pixel_depth_diff)),
					padUpperBound=(math.ceil(pixel_width_diff), math.ceil(pixel_height_diff), math.ceil(pixel_depth_diff)),
					constant=-2048
				)
				self.SCAN_TO_COREGISTER_TO[2] = sitk.ConstantPad(
					self.SCAN_TO_COREGISTER_TO[2],
					padLowerBound=(math.floor(pixel_width_diff), math.floor(pixel_height_diff), math.floor(pixel_depth_diff)),
					padUpperBound=(math.ceil(pixel_width_diff), math.ceil(pixel_height_diff), math.ceil(pixel_depth_diff)),
					constant=-2048
				)

			# Add padding to match a scaling of IMAGE_SIZE's depth so that all scans will be scaled version of IMAGE_SIZE
			# as coregistration makes all images the same size as what they are coregistering to - SCAN_TO_COREGISTER_TO
			scaling_factor = (self.SCAN_TO_COREGISTER_TO[1].GetWidth() / IMAGE_SIZE[1])
			required_depth = (IMAGE_SIZE[0] * scaling_factor)
			if required_depth < self.SCAN_TO_COREGISTER_TO[1].GetDepth() :
				raise ValueError("IMAGE_SIZE is too short for scans, cannot reshape without data loss: %d < %d" % (required_depth, self.SCAN_TO_COREGISTER_TO[1].GetDepth()))
			
			depth_diff = ((required_depth - self.SCAN_TO_COREGISTER_TO[1].GetDepth()) / 2)
			self.SCAN_TO_COREGISTER_TO[1] = sitk.ConstantPad(self.SCAN_TO_COREGISTER_TO[1], padLowerBound=(0, 0, math.floor(depth_diff)), padUpperBound=(0, 0, math.ceil(depth_diff)))
			self.SCAN_TO_COREGISTER_TO[2] = sitk.ConstantPad(self.SCAN_TO_COREGISTER_TO[2], padLowerBound=(0, 0, math.floor(depth_diff)), padUpperBound=(0, 0, math.ceil(depth_diff)))

			# Interpolate back down to IMAGE_SIZE to make data smaller and faster to coregister
			scale_factor_width_height = max(IMAGE_SIZE[1], IMAGE_SIZE[2]) / max(self.SCAN_TO_COREGISTER_TO[1].GetWidth(), self.SCAN_TO_COREGISTER_TO[1].GetHeight())
			scale_factor_depth = IMAGE_SIZE[0] / self.SCAN_TO_COREGISTER_TO[1].GetDepth() 
			to_image_size_scaling_factors = (scale_factor_depth, scale_factor_width_height, scale_factor_width_height)

			new_spacing = [
				sz * spc / nsz
				for nsz, sz, spc in zip(IMAGE_SIZE[::-1], self.SCAN_TO_COREGISTER_TO[1].GetSize(), self.SCAN_TO_COREGISTER_TO[1].GetSpacing())
			]

			self.SCAN_TO_COREGISTER_TO[1], self.SCAN_TO_COREGISTER_TO[2] = interpolate_scan_by_scaling_factors(
				self.SCAN_TO_COREGISTER_TO[1],
				self.SCAN_TO_COREGISTER_TO[2],
				to_image_size_scaling_factors,
				new_spacing
			)

			print(
				"Coregistering scans to:", self.SCAN_TO_COREGISTER_TO[0], "of size", self.SCAN_TO_COREGISTER_TO[1].GetSize(),
				"and spacings", self.SCAN_TO_COREGISTER_TO[1].GetSpacing()
			)

			for i in tqdm(range(self._no_files)) :
				image_object = self.data[i]

				itk_img = sitk.ReadImage(os.path.join(self.images_dir, image_object['filename']), sitk.sitkFloat64)
				itk_img_seg = sitk.ReadImage(os.path.join(self.labels_dir, image_object['filename']), sitk.sitkFloat64)

				itk_img = sitk.DICOMOrient(itk_img, "RPI")
				itk_img_seg = sitk.DICOMOrient(itk_img_seg, "RPI")

				# print(image_object['filename'])
				# print("orig", itk_img.GetSize())

				# One scan will coregister to itself, but will be preprocessed in the same way as everythign else
				itk_img, itk_img_seg, transformation, scaling = coregister_scan(
					itk_img,
					itk_img_seg,
					self.SCAN_TO_COREGISTER_TO[1]
				)

				# print("final size", itk_img.GetSize())

				# Save coregistered and correctly sized images
				sitk.WriteImage(itk_img, os.path.join(self.images_preprocessed_dir, image_object['filename']))
				sitk.WriteImage(sitk.Cast(itk_img_seg, sitk.sitkUInt8), os.path.join(self.labels_preprocessed_dir, image_object['filename']))
				if transformation != None :
					sitk.WriteTransform(transformation, os.path.join(self.images_preprocessed_dir, image_object['filename'].replace(".nii.gz", "_transformation.tfm")))
					with open(os.path.join(self.images_preprocessed_dir, image_object['filename'].replace(".nii.gz", "_metadata.json")), "w") as f :
						f.write(json.dumps({"scaling": scaling}))

				image_object['label'] = None
				image_object['image'] = None

		self.data = []

		# Load all images
		for i in tqdm(range(self._no_files)) :
			filename = list(filter(lambda x: x.endswith(".nii.gz"), os.listdir(self.images_preprocessed_dir)))[i] # check the loading

			itk_img = sitk.ReadImage(os.path.join(self.images_preprocessed_dir, filename), sitk.sitkFloat64)
			itk_img = sitk.DICOMOrient(itk_img, "RPI")
			# cast array as torch cannot convert arrays of dtype=np.uint16 and transformations require floating point data
			img = torch.from_numpy(sitk.GetArrayFromImage(itk_img))

			itk_img_seg = sitk.ReadImage(os.path.join(self.labels_preprocessed_dir, filename), sitk.sitkUInt8)
			itk_img_seg = sitk.DICOMOrient(itk_img_seg, "RPI")
			img_seg = torch.from_numpy(sitk.GetArrayFromImage(itk_img_seg))

			image_object = copy.copy(self.BASIC_DATASET_ITEM)
			image_object['image'] = img
			image_object['filename'] = filename
			image_object['label'] = img_seg
			self.data.append(image_object)
			
	def get_split(self, split) :
		_training_split = int(self._no_files * (self.TRAIN_SIZE / 100))
		_test_split = int(self._no_files * (self.TEST_SIZE / 100))
		_val_split = self._no_files - (_training_split + _test_split)

		start_location = 0
		split_size = self._no_files
		if split == "train" :
			start_location = 0
			split_size = _training_split
		elif split == "test" :
			start_location = _training_split
			split_size = start_location + _test_split
		elif split == "val" :
			start_location = _training_split + _test_split
			split_size = start_location + _val_split

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

		self.tf_train = miccai_transform(channels_last, random_crop=random_crop, normalize=normalize)
		self.tf_valid = miccai_transform(channels_last, random_crop=None, normalize=normalize)

		self.ds_train = None
		self.ds_valid = None
		
		self.dataset_loader = None

	@property
	def num_classes(self):
		return NUM_CLASSES

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
		self.ds_train.set_transform(lift_transform(*self.tf_train))

		self.ds_valid = self.load_dataset(split="test")
		self.ds_valid.set_transform(lift_transform(*self.tf_valid))

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


def miccai_transform(channels_last: bool = True, random_crop: Optional[int] = None, normalize:bool = True):
	image_transform_list = []
	label_transform_list = []

	image_transform_list.append(transforms.ConvertImageDtype(torch.float32))

	def convert_less_than_x_to_y(x, y) :
		def apply(t) :
			t[t < x] = y
			return t
		return apply

	def convert_greater_than_x_to_y(x, y) :
		def apply(t) :
			t[t > x] = y
			return t
		return apply

	# TODO: add random crop back in for 3d images

	image_transform_list.append(convert_less_than_x_to_y(-1000, -1000))
	label_transform_list.append(convert_less_than_x_to_y(0, 0))
	label_transform_list.append(convert_greater_than_x_to_y(NUM_CLASSES-1, 0))

	if normalize :
		image_transform_list.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))


	if channels_last:
		image_transform_list.append(channels_to_last)
		label_transform_list.append(channels_to_last)

	return transforms.Compose(image_transform_list), transforms.Compose(label_transform_list)
