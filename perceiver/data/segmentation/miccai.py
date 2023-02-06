from functools import partial
import json
import math
import os
import sys
import copy
from typing import Any, Callable, List, Optional

import pytorch_lightning as pl
import SimpleITK as sitk
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from skimage.transform import rescale

from perceiver.data.segmentation.common import channels_to_last, SegmentationPreprocessor, lift_transform, coregister_image_and_label, ImageInfo, zoom_sitk_image_image, zoom_sitk_image_label

IMAGE_SIZE = (220, 256, 256)
IMAGE_SIZE = (165, 192, 192)
IMAGE_SIZE = (110, 128, 128)
# IMAGE_SIZE = (55, 64, 64)
NUM_CLASSES = 16

COREGISTRATION_IMAGE_FILENAME = "coregistration_image" # + ".nii.gz"

SKIP_PREPROCESSED_CHECK = False

CT_ONLY = True

class MICCAIPreprocessor(SegmentationPreprocessor):
	def __init__(self, channels_last: bool = True, normalize: bool = True):
		super().__init__(miccai_transform(channels_last, normalize))


class MICCAIDataset(Dataset) :
	def __init__(self, data, images_preprocessed_dir, metadata, transforms=None) :
		self.data = [func() for func in tqdm(data)]
		self.transforms = transforms
		self.images_preprocessed_dir = images_preprocessed_dir
		self.metadata = metadata

	def set_transform(self, transforms) :
		self.transforms = transforms

	def get_coregistration_image(self) :
		return sitk.ReadImage(os.path.join(self.images_preprocessed_dir, COREGISTRATION_IMAGE_FILENAME + ".nii.gz"), sitk.sitkFloat64)

	def get_labels(self) :
		return self.metadata["labels"]

	def __getitem__(self, index):
		sample = self.data[index]
		if self.transforms:
			sample = self.transforms(sample)
		return sample
	
	def __len__(self) :
		return len(self.data)


def get_ct_only_dataset_files(l : List[Any], indexing_func: Callable[[Any], str] = lambda x: x['image']) :
	# scans 500 and lower are ct scans
	return list(filter(lambda x: int(os.path.basename(indexing_func(x)).split('.')[0].split("amos_")[-1]) <= 500, l))


class MICCAILoader() :
	IMAGES_TR = "imagesTr"
	LABELS_TR = "labelsTr"
	IMAGES_TR_PREPROCESSED = "imagesTr_preprocessed"
	LABELS_TR_PREPROCESSED = "labelsTr_preprocessed"
	
	IMAGES_VA = "imagesVa"
	LABELS_VA = "labelsVa"
	IMAGES_VA_PREPROCESSED = "imagesVa_preprocessed"
	LABELS_VA_PREPROCESSED = "labelsVa_preprocessed"
	
	TRAIN_SIZE = 70
	TEST_SIZE = 30
	
	BASIC_DATASET_ITEM = {'label' : None, 'image' : None, 'filename' : None}

	LIMIT_SCAN_COUNT = 10000

	SCAN_TO_COREGISTER_TO = None
	
	def __init__(self, root, load_raw_instead) :
		self.root = root

		if not os.path.exists(root) :
			print("Cannot find dataset at specified root")
			sys.exit(1)

		self.images_training_dir = os.path.join(self.root, self.IMAGES_TR)
		self.images_validation_dir = os.path.join(self.root, self.IMAGES_VA)
		self.images_training_preprocessed_dir = os.path.join(self.root, self.IMAGES_TR_PREPROCESSED)
		self.images_validation_preprocessed_dir = os.path.join(self.root, self.IMAGES_VA_PREPROCESSED)
		if not os.path.exists(self.images_training_dir) :
			print("Cannot find %s inside dataset" % self.IMAGES_TR)
			sys.exit(1)
		if not os.path.exists(self.images_validation_dir) :
			print("Cannot find %s inside dataset", self.IMAGES_VA)
			sys.exit(1)

		self.labels_training_dir = os.path.join(self.root, self.LABELS_TR)
		self.labels_validation_dir = os.path.join(self.root, self.LABELS_VA)
		self.labels_training_preprocessed_dir = os.path.join(self.root, self.LABELS_TR_PREPROCESSED)
		self.labels_validation_preprocessed_dir = os.path.join(self.root, self.LABELS_VA_PREPROCESSED)
		if not os.path.exists(self.labels_training_dir) :
			print("Cannot find %s inside dataset", self.LABELS_TR)
			sys.exit(1)
		if not os.path.exists(self.labels_validation_dir) :
			print("Cannot find %s inside dataset", self.LABELS_VA)
			sys.exit(1)
		
		self.metadata = {}
		with open(os.path.join(self.root, "dataset.json"), "r") as f :
			self.metadata = json.load(f)

		self._task_training_dataset = self.metadata['training']
		self._task_validation_dataset = self.metadata['validation']

		if CT_ONLY :
			self._task_training_dataset = get_ct_only_dataset_files(self._task_training_dataset)
			self._task_validation_dataset = get_ct_only_dataset_files(self._task_validation_dataset)
		
		self.data = []
		self._load_data(load_raw_instead)

	def _load_data(self, load_raw_instead) :
		coregister_image_and_labels = False

		if not os.path.exists(self.images_training_preprocessed_dir) :
			os.mkdir(self.images_training_preprocessed_dir)
			os.mkdir(self.labels_training_preprocessed_dir)
			os.mkdir(self.images_validation_preprocessed_dir)
			os.mkdir(self.labels_validation_preprocessed_dir)
			coregister_image_and_labels = True
		else :
			for file_obj in self._task_training_dataset :
				filename = os.path.basename(file_obj['image'])

				if not filename in os.listdir(self.images_training_preprocessed_dir) :
					coregister_image_and_labels = True
					break

			for file_obj in self._task_validation_dataset :
				filename = os.path.basename(file_obj['image'])

				if not filename in os.listdir(self.images_validation_preprocessed_dir) :
					coregister_image_and_labels = True
					break

		if coregister_image_and_labels and not SKIP_PREPROCESSED_CHECK :
			image_info = self._find_coregistration_scan()

			print("Preparing largest scan - %s - for coregistration..." % self.SCAN_TO_COREGISTER_TO[0])
			
			self._prepare_coregistration_image(image_info)

			for image_object in tqdm(self.data['train']) :
				itk_img = sitk.ReadImage(os.path.join(self.images_training_dir, image_object['filename']), sitk.sitkFloat64)
				itk_img_seg = sitk.ReadImage(os.path.join(self.labels_training_dir, image_object['filename']), sitk.sitkUInt8)

				original_image_size = itk_img.GetSize()

				itk_img = sitk.DICOMOrient(itk_img, "RPI")
				itk_img_seg = sitk.DICOMOrient(itk_img_seg, "RPI")

				# One scan will coregister to itself, but will be preprocessed in the same way as everythign else
				itk_img, itk_img_seg, transformation, scaling = coregister_image_and_label(
					itk_img,
					itk_img_seg,
					self.SCAN_TO_COREGISTER_TO[1]
				)

				# Save coregistered and correctly sized images
				sitk.WriteImage(itk_img, os.path.join(self.images_training_preprocessed_dir, image_object['filename']))
				sitk.WriteImage(sitk.Cast(itk_img_seg, sitk.sitkUInt8), os.path.join(self.labels_training_preprocessed_dir, image_object['filename']))
				if transformation != None :
					sitk.WriteTransform(transformation, os.path.join(self.images_training_preprocessed_dir, image_object['filename'].replace(".nii.gz", "_transformation.tfm")))
					with open(os.path.join(self.images_training_preprocessed_dir, image_object['filename'].replace(".nii.gz", "_metadata.json")), "w") as f :
						f.write(json.dumps({"scaling": scaling, "original_size" : original_image_size}))

				image_object['label'] = None
				image_object['image'] = None

			for image_object in tqdm(self.data['val']) :
				itk_img = sitk.ReadImage(os.path.join(self.images_validation_dir, image_object['filename']), sitk.sitkFloat64)
				itk_img_seg = sitk.ReadImage(os.path.join(self.labels_validation_dir, image_object['filename']), sitk.sitkUInt8)

				original_image_size = itk_img.GetSize()

				itk_img = sitk.DICOMOrient(itk_img, "RPI")
				itk_img_seg = sitk.DICOMOrient(itk_img_seg, "RPI")

				# One scan will coregister to itself, but will be preprocessed in the same way as everythign else
				itk_img, itk_img_seg, transformation, scaling = coregister_image_and_label(
					itk_img,
					itk_img_seg,
					self.SCAN_TO_COREGISTER_TO[1]
				)

				# Save coregistered and correctly sized images
				sitk.WriteImage(itk_img, os.path.join(self.images_validation_preprocessed_dir, image_object['filename']))
				sitk.WriteImage(sitk.Cast(itk_img_seg, sitk.sitkUInt8), os.path.join(self.labels_validation_preprocessed_dir, image_object['filename']))
				if transformation != None :
					sitk.WriteTransform(transformation, os.path.join(self.images_validation_preprocessed_dir, image_object['filename'].replace(".nii.gz", "_transformation.tfm")))
					with open(os.path.join(self.images_validation_preprocessed_dir, image_object['filename'].replace(".nii.gz", "_metadata.json")), "w") as f :
						f.write(json.dumps({"scaling": scaling, "original_size" : original_image_size}))

				image_object['label'] = None
				image_object['image'] = None

		self.data = {'val':[], 'train':[]}

		# Load all images
		training_filenames = [os.path.basename(f['image']) for f in self._task_training_dataset]
		for file_object in tqdm(self._task_training_dataset + self._task_validation_dataset) :
			filename = os.path.basename(file_object['image']) # check the loading

			def _get_image_from_filename(fname: str, images_dir:str, labels_dir:str, images_preprocessed_dir:str, labels_preprocessed_dir:str) :
				image_object = copy.copy(self.BASIC_DATASET_ITEM)
				image_object['filename'] = fname
				
				if load_raw_instead :
					itk_img = sitk.ReadImage(os.path.join(images_dir, image_object['filename']), sitk.sitkFloat64)
					itk_img = sitk.DICOMOrient(itk_img, "RPI")
					# cast array as torch cannot convert arrays of dtype=np.uint16 and transformations require floating point data
					# img = torch.from_numpy(sitk.GetArrayFromImage(itk_img))
					image_object['image'] = itk_img

					itk_img_seg = sitk.ReadImage(os.path.join(labels_dir, image_object['filename']), sitk.sitkUInt8)
					itk_img_seg = sitk.DICOMOrient(itk_img_seg, "RPI")
					img_seg = torch.from_numpy(sitk.GetArrayFromImage(itk_img_seg))
					image_object['label'] = img_seg
				
				else :
					itk_img = sitk.ReadImage(os.path.join(images_preprocessed_dir, image_object['filename']), sitk.sitkFloat64)
					itk_img = sitk.DICOMOrient(itk_img, "RPI")
					# cast array as torch cannot convert arrays of dtype=np.uint16 and transformations require floating point data
					img = torch.from_numpy(sitk.GetArrayFromImage(itk_img))

					itk_img_seg = sitk.ReadImage(os.path.join(labels_preprocessed_dir, image_object['filename']), sitk.sitkUInt8)
					itk_img_seg = sitk.DICOMOrient(itk_img_seg, "RPI")
					img_seg = torch.from_numpy(sitk.GetArrayFromImage(itk_img_seg))

					image_object['image'] = img
					image_object['label'] = img_seg
					image_object['transformation'] = sitk.ReadTransform(os.path.join(images_preprocessed_dir, image_object['filename']).replace(".nii.gz", "_transformation.tfm"))
					with open(os.path.join(images_preprocessed_dir, image_object['filename'].replace(".nii.gz", "_metadata.json")), "r") as f :
						for k,v in json.load(f).items() :
							image_object[k] = v

				return image_object
				
			dataset = 'train' if filename in training_filenames else 'val'
			is_in_training_dataset = (dataset == 'train')
			self.data[dataset].append(
				partial(
					_get_image_from_filename,
					filename,
					self.images_training_dir if is_in_training_dataset else self.images_validation_dir,
					self.labels_training_dir if is_in_training_dataset else self.labels_validation_dir,
					self.images_training_preprocessed_dir if is_in_training_dataset else self.images_validation_preprocessed_dir,
					self.labels_training_preprocessed_dir if is_in_training_dataset else self.labels_validation_preprocessed_dir,
				)
			)

	def _prepare_coregistration_image(self, image_info) :
		# Resample for largest scalings
		spacing_scale_factor_width_height = (self.SCAN_TO_COREGISTER_TO[1].GetSpacing()[0] / image_info.spacing_width_height)
		spacing_scale_factor_depth = (self.SCAN_TO_COREGISTER_TO[1].GetSpacing()[2] / image_info.spacing_depth)
		scale_factors = (spacing_scale_factor_depth, spacing_scale_factor_width_height, spacing_scale_factor_width_height)

		print("\tResampling to largest scalings - %s - and spacings..." % (str(scale_factors)))

		self.SCAN_TO_COREGISTER_TO[1] = zoom_sitk_image_image(
			self.SCAN_TO_COREGISTER_TO[1],
			scale_factors,
			(image_info.spacing_width_height, image_info.spacing_width_height, image_info.spacing_depth)
		)

		self.SCAN_TO_COREGISTER_TO[2] = zoom_sitk_image_label(
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
			print("\tAdding padding interaxial - width := %d, height := %d, depth := %d..." % (pixel_width_diff, pixel_height_diff, pixel_depth_diff))
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
				constant=0
			)

		# Add padding to match a scaling of IMAGE_SIZE's depth so that all scans will be scaled version of IMAGE_SIZE
		# as coregistration makes all images the same size as what they are coregistering to - SCAN_TO_COREGISTER_TO
		scaling_factor = (self.SCAN_TO_COREGISTER_TO[1].GetWidth() / IMAGE_SIZE[1])
		required_depth = (IMAGE_SIZE[0] * scaling_factor)
		if required_depth < self.SCAN_TO_COREGISTER_TO[1].GetDepth() :
			raise ValueError("IMAGE_SIZE is too short for scans, cannot reshape without data loss: %d < %d" % (required_depth, self.SCAN_TO_COREGISTER_TO[1].GetDepth()))
		
		depth_diff = ((required_depth - self.SCAN_TO_COREGISTER_TO[1].GetDepth()) / 2)
		print("\tAdding depth padding := %d..." % depth_diff)
		self.SCAN_TO_COREGISTER_TO[1] = sitk.ConstantPad(self.SCAN_TO_COREGISTER_TO[1], padLowerBound=(0, 0, math.floor(depth_diff)), padUpperBound=(0, 0, math.ceil(depth_diff)), constant=-2048)
		self.SCAN_TO_COREGISTER_TO[2] = sitk.ConstantPad(self.SCAN_TO_COREGISTER_TO[2], padLowerBound=(0, 0, math.floor(depth_diff)), padUpperBound=(0, 0, math.ceil(depth_diff)), constant=0)

		# Interpolate back down to IMAGE_SIZE to make data smaller and faster to coregister
		print("\tInterpolating to IMAGE_SIZE...")
		scale_factor_width_height = max(IMAGE_SIZE[1], IMAGE_SIZE[2]) / max(self.SCAN_TO_COREGISTER_TO[1].GetWidth(), self.SCAN_TO_COREGISTER_TO[1].GetHeight())
		scale_factor_depth = IMAGE_SIZE[0] / self.SCAN_TO_COREGISTER_TO[1].GetDepth() 
		to_image_size_scaling_factors = (scale_factor_depth, scale_factor_width_height, scale_factor_width_height)

		new_spacing = [
			sz * spc / nsz
			for nsz, sz, spc in zip(IMAGE_SIZE[::-1], self.SCAN_TO_COREGISTER_TO[1].GetSize(), self.SCAN_TO_COREGISTER_TO[1].GetSpacing())
		]

		self.SCAN_TO_COREGISTER_TO[1] = zoom_sitk_image_image(
			self.SCAN_TO_COREGISTER_TO[1],
			to_image_size_scaling_factors,
			new_spacing
		)

		self.SCAN_TO_COREGISTER_TO[2] = zoom_sitk_image_label(
			self.SCAN_TO_COREGISTER_TO[2],
			to_image_size_scaling_factors,
			new_spacing
		)

		self.SCAN_TO_COREGISTER_TO[1] = sitk.Cast(self.SCAN_TO_COREGISTER_TO[1], sitk.sitkFloat64)

		print(
			"Coregistering scans to:", self.SCAN_TO_COREGISTER_TO[0], "of size", self.SCAN_TO_COREGISTER_TO[1].GetSize(),
			"and spacings", self.SCAN_TO_COREGISTER_TO[1].GetSpacing()
		)

		sitk.WriteImage(self.SCAN_TO_COREGISTER_TO[1], os.path.join(self.images_training_preprocessed_dir, COREGISTRATION_IMAGE_FILENAME + ".nii.gz"))

	def _find_coregistration_scan(self) :
		image_info = ImageInfo()
		largest_size = 0

		self.data = {'train':[], 'val':[]}

		for file_obj in tqdm(self._task_training_dataset) :
			filename = os.path.basename(file_obj['image'])

			itk_img = sitk.ReadImage(os.path.join(self.images_training_dir, filename), sitk.sitkFloat32)
			itk_img_seg = sitk.ReadImage(os.path.join(self.labels_training_dir, filename), sitk.sitkUInt8)

			itk_img = sitk.DICOMOrient(itk_img, "RPI")
			itk_img_seg = sitk.DICOMOrient(itk_img_seg, "RPI")

			image_object = copy.copy(self.BASIC_DATASET_ITEM)
			image_object['label'] = None
			image_object['image'] = None
			image_object['filename'] = filename
			self.data['train'].append(image_object)

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

		for file_obj in tqdm(self._task_validation_dataset) :
			filename = os.path.basename(file_obj['image'])

			image_object = copy.copy(self.BASIC_DATASET_ITEM)
			image_object['filename'] = filename
			self.data['val'].append(image_object)

		return image_info

	def get_split(self, split) :
		used_dataset = 'train'
		_no_files = 0
		if split == "train" or split == "test" :
			_no_files = len(self._task_training_dataset)
		else :
			used_dataset = "val"
			_no_files = len(self._task_validation_dataset)

		_training_split = int(_no_files * (self.TRAIN_SIZE / 100))
		_test_split = int(_no_files - _training_split)

		start_location = 0
		split_size = 1
		if split == "train" :
			start_location = 0
			split_size = _training_split
		elif split == "test" :
			start_location = _training_split
			split_size = start_location + _test_split
		elif split == "val" :
			start_location = 0
			split_size = len(self._task_validation_dataset)

		print("Loading", split, "dataset...")
		return MICCAIDataset(self.data[used_dataset][start_location:split_size], self.images_training_preprocessed_dir, self.metadata)


class MICCAIDataModule(pl.LightningDataModule):
	def __init__(
		self,
		dataset_dir: str = "/mnt/d/amos22",
		normalize: bool = True,
		channels_last: bool = True,
		random_crop: Optional[int] = None,
		batch_size: int = 64,
		num_workers: int = 3,
		pin_memory: bool = True,
		shuffle: bool = True,
		load_raw_instead: bool = False,
		**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		self.channels_last = channels_last
		self.load_raw_instead = load_raw_instead

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
			self.dataset_loader = MICCAILoader(self.hparams.dataset_dir, self.load_raw_instead)
		
		return self.dataset_loader.get_split(split)

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
