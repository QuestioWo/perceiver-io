from dataclasses import dataclass
import math

from typing import Tuple

import torch
import SimpleITK as sitk
import numpy as np

from scipy.ndimage import zoom


class SegmentationPreprocessor:
	def __init__(self, transform):
		self.image_transform, _ = transform

	def preprocess(self, img):
		return self.image_transform(img)

	def preprocess_batch(self, img_batch):
		return torch.stack([self.preprocess(img) for img in img_batch])


def lift_transform(image_transform, label_transform):
	def apply(sample):
		result = {}
		
		result["image"] = image_transform(sample['image'])
		result["label"] = label_transform(sample['label'])
		return result

	return apply


def _command_iteration(method):
	# if method.GetOptimizerIteration() == 0:
	# 	print(f"\tLevel: {method.GetCurrentLevel()}")
	# 	print(f"\tScales: {method.GetOptimizerScales()}")
	# print(f"#{method.GetOptimizerIteration()}")
	# print(f"\tMetric Value: {method.GetMetricValue():10.5f}")
	# print(f"\tLearningRate: {method.GetOptimizerLearningRate():10.5f}")
	# if method.GetOptimizerConvergenceValue() != sys.float_info.max:
	# 	print(
	# 		"\tConvergence Value: "
	# 		+ f"{method.GetOptimizerConvergenceValue():.5e}"
	# 	)
	pass


def _command_multiresolution_iteration(method):
	# print(f"\tStop Condition: {method.GetOptimizerStopConditionDescription()}")
	# print("============= Resolution Change =============")
	pass

@dataclass
class ImageInfo :
	size_width_height = 0
	size_depth = 0
	spacing_width_height = float("inf")
	spacing_depth = float("inf")
	world_size_width_height = 0
	world_size_depth = 0


def interpolate_scan_by_scaling_factors(moving: sitk.Image, moving_label: sitk.Image, scaling_factors, spacings: Tuple[float, float, float]) :
	moving_origin = moving.GetOrigin()
	moving_direction = moving.GetDirection()
	
	np_moving = sitk.GetArrayFromImage(moving)
	np_moving_label = sitk.GetArrayFromImage(moving_label)
	
	np_moving = zoom(np_moving, scaling_factors, mode='nearest', cval=np.min(np_moving))
	np_moving_label = zoom(np_moving_label, scaling_factors, mode='nearest', cval=0, order=0)

	moving = sitk.GetImageFromArray(np_moving)
	moving.SetSpacing(spacings)
	moving.SetDirection(moving_direction)
	moving.SetOrigin(moving_origin)

	moving_label = sitk.GetImageFromArray(np_moving_label)
	moving_label.SetSpacing(spacings)
	moving_label.SetDirection(moving_direction)
	moving_label.SetOrigin(moving_origin)
	return moving, moving_label


def coregister_scan(moving: sitk.Image, moving_label: sitk.Image, fixed: sitk.Image) -> Tuple[sitk.Image, sitk.Image, sitk.Transform, int] :
	# Interpolate/Resample to the same spacing and size aka worldsize
	## NOTE: assuming width and height have same spacing for simplicity
	fixed_spacing_width_height = min(fixed.GetSpacing()[0], fixed.GetSpacing()[1])
	spacing_scale_factor_width_height = (min(moving.GetSpacing()[0], moving.GetSpacing()[1]) / fixed_spacing_width_height)
	spacing_scale_factor_depth = (moving.GetSpacing()[2] / fixed.GetSpacing()[2])

	# Rescale using scale factors
	scaling_factors = (spacing_scale_factor_depth, spacing_scale_factor_width_height, spacing_scale_factor_width_height)
	moving, moving_label = interpolate_scan_by_scaling_factors(moving, moving_label, scaling_factors, (fixed_spacing_width_height, fixed_spacing_width_height, fixed.GetSpacing()[2]))

	# print("post interpolation size", moving.GetSize())

	# Coregister moving to fixed
	initialTx = sitk.CenteredTransformInitializer(
		fixed, moving, sitk.AffineTransform(fixed.GetDimension())
	)

	R = sitk.ImageRegistrationMethod()

	R.SetShrinkFactorsPerLevel([1, 1, 1])
	R.SetSmoothingSigmasPerLevel([1, 1, 1])

	R.SetMetricAsJointHistogramMutualInformation(20)
	R.MetricUseMovingImageGradientFilterOff()

	R.SetOptimizerAsGradientDescent(
		learningRate=2.0,
		numberOfIterations=100,
		estimateLearningRate=R.EachIteration,
	)
	R.SetOptimizerScalesFromPhysicalShift()

	R.SetInitialTransform(initialTx)

	R.SetInterpolator(sitk.sitkLinear)

	R.AddCommand(sitk.sitkIterationEvent, lambda: _command_iteration(R))
	R.AddCommand(
		sitk.sitkMultiResolutionIterationEvent,
		lambda: _command_multiresolution_iteration(R),
	)

	tx = R.Execute(fixed, moving)

	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(fixed)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetDefaultPixelValue(np.min(sitk.GetArrayFromImage(moving)))
	resampler.SetTransform(tx)

	out_moving = resampler.Execute(moving)
	resampler.SetDefaultPixelValue(0)
	resampler.SetInterpolator(sitk.sitkNearestNeighbor)
	out_moving_label = resampler.Execute(moving_label)

	return (out_moving, out_moving_label, tx, {"width_height": spacing_scale_factor_width_height, "depth": spacing_scale_factor_depth})


def channels_to_last(img: torch.Tensor):
	return img.permute(1, 2, 0).contiguous()
