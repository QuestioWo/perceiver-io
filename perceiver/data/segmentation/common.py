from dataclasses import dataclass

from typing import Callable, List, Tuple

import torch
import SimpleITK as sitk
import numpy as np

from scipy.ndimage import zoom

GRADIENT_DESCENT_ITERATIONS = 500

class SegmentationPreprocessor:
	def __init__(self, transform):
		self.image_transform, _ = transform

	def preprocess(self, img: torch.Tensor) -> torch.Tensor:
		return self.image_transform(img)

	def preprocess_batch(self, img_batch : List[torch.Tensor]) -> torch.Tensor:
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


def zoom_numpy_array(img: np.ndarray, scaling_factors: Tuple[float, float, float], min_func: Callable[[np.ndarray], int], order: int) -> np.ndarray :
	return zoom(img, scaling_factors, mode="nearest", cval=min_func(img), order=order)


def zoom_sitk_image(img: sitk.Image, scaling_factors: Tuple[float, float, float], spacings: Tuple[float, float, float], min_func: Callable[[np.ndarray], int], order:int) -> sitk.Image :
	moving_origin = img.GetOrigin()
	moving_direction = img.GetDirection()
	
	img = sitk.GetArrayFromImage(img)
	# print("\tArrayed := %s - dtype := %s - being scaled by %s" % (str(moving.shape), str(moving.dtype), str(scaling_factors)))
	
	img = zoom_numpy_array(img, scaling_factors, min_func, order)
	# print("\tImage zoomed, dtype := %s, size := %s" % (str(moving.dtype), str(moving.shape)))
	
	img = sitk.GetImageFromArray(img, isVector=False)
	# print("\tImage retrieved")
	img.SetSpacing(spacings)
	img.SetDirection(moving_direction)
	img.SetOrigin(moving_origin)

	return img


def zoom_sitk_image_image(moving: sitk.Image, scaling_factors: Tuple[float, float, float], spacings: Tuple[float, float, float]) -> sitk.Image :
	return zoom_sitk_image(moving, scaling_factors, spacings, np.min, 3)


def zoom_sitk_image_label(moving: sitk.Image, scaling_factors: Tuple[float, float, float], spacings: Tuple[float, float, float]) -> sitk.Image :
	return zoom_sitk_image(moving, scaling_factors, spacings, lambda _: 0, 0) # must be 3 otherwise edges linearly interpolate, which we dont want as they will produce incorrect labellings


def coregister_image_and_label(moving: sitk.Image, moving_label: sitk.Image, fixed: sitk.Image) -> Tuple[sitk.Image, sitk.Image, sitk.Transform, dict] :
	(moving, tx, scaling_factors, spacings) = coregister_image(moving, fixed)

	moving_label = zoom_sitk_image_label(moving_label, scaling_factors, spacings)
	
	moving_label = apply_transformation_to_image_label(moving_label, fixed, tx)

	return (moving, moving_label, tx, {"depth": scaling_factors[0], "width_height": scaling_factors[1]})


def coregister_image(moving: sitk.Image, fixed: sitk.Image) -> Tuple[sitk.Image, sitk.Transform, Tuple[float, float, float], Tuple[float, float, float]] :
	# Interpolate/Resample to the same spacing and size aka worldsize
	## NOTE: assuming width and height have same spacing for simplicity
	fixed_spacing_width_height = min(fixed.GetSpacing()[0], fixed.GetSpacing()[1])
	spacing_scale_factor_width_height = (min(moving.GetSpacing()[0], moving.GetSpacing()[1]) / fixed_spacing_width_height)
	spacing_scale_factor_depth = (moving.GetSpacing()[2] / fixed.GetSpacing()[2])

	# Rescale using scale factors
	scaling_factors = (spacing_scale_factor_depth, spacing_scale_factor_width_height, spacing_scale_factor_width_height)
	spacings = (fixed_spacing_width_height, fixed_spacing_width_height, fixed.GetSpacing()[2])
	
	moving = zoom_sitk_image_image(moving, scaling_factors, spacings)
	
	tx = computeCoregistrationTransformation(moving, fixed)

	moving = apply_transformation_to_image_image(moving, fixed, tx)

	return (moving, tx, scaling_factors, spacings)


def apply_transformation_to_image(moving: sitk.Image, fixed: sitk.Image, tx: sitk.Transform, default_pixel_value: Callable[[sitk.Image], float], interpolator = sitk.sitkLinear) :
	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(fixed)
	resampler.SetTransform(tx)
	resampler.SetInterpolator(interpolator)
	resampler.SetDefaultPixelValue(default_pixel_value(moving))
	moving = resampler.Execute(moving)
	return moving


def apply_transformation_to_image_image(moving: sitk.Image, fixed: sitk.Image, tx: sitk.Transform) :
	return apply_transformation_to_image(moving, fixed, tx, lambda i: np.min(sitk.GetArrayFromImage(i)), sitk.sitkLinear)


def apply_transformation_to_image_label(moving: sitk.Image, fixed: sitk.Image, tx: sitk.Transform) :
	return apply_transformation_to_image(moving, fixed, tx, lambda _: 0, sitk.sitkNearestNeighbor)


def computeCoregistrationTransformation(moving: sitk.Image, fixed: sitk.Image) :
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
		numberOfIterations=GRADIENT_DESCENT_ITERATIONS,
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

	return tx


def channels_to_last(img: torch.Tensor):
	return img.permute(1, 2, 0).contiguous()
