import torch
import SimpleITK as sitk
import skimage
import sys


class SegmentationPreprocessor:
    def __init__(self, transform):
        self.transform = transform

    def preprocess(self, img):
        return self.transform(img)

    def preprocess_batch(self, img_batch):
        return torch.stack([self.preprocess(img) for img in img_batch])


def lift_transform(transform):
    def apply(sample):
        sample["image"] = transform(sample['image'])
        sample["label"] = transform(sample['label'])
        return sample

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


def coregister_scan(np_moving, np_moving_label, np_fixed) :
	fixed = sitk.GetImageFromArray(np_fixed)
	moving = sitk.GetImageFromArray(np_moving)
	moving_label = sitk.GetImageFromArray(np_moving_label)

	scale_factor = (max(np_fixed.shape[0], np_fixed.shape[1]) / min(np_moving.shape[1], np_moving.shape[1]))
	scale_factor_width = int(scale_factor * moving.GetWidth())
	scale_factor_height = int(scale_factor * moving.GetHeight())
	scale_factor_depth = int(scale_factor * moving.GetDepth())

	# The spatial definition of the images we want to use in a deep learning framework (smaller than the original).
	new_size = (scale_factor_width, scale_factor_height, scale_factor_depth)
	reference_image = sitk.Image(new_size, moving.GetPixelIDValue())
	reference_image.SetOrigin(moving.GetOrigin())
	reference_image.SetDirection(moving.GetDirection())
	reference_image.SetSpacing(
		[
			sz * spc / nsz
			for nsz, sz, spc in zip(new_size, moving.GetSize(), moving.GetSpacing())
		]
	)

	moving = sitk.Resample(moving, reference_image)
	moving_label = sitk.Resample(moving_label, reference_image)

	initialTx = sitk.CenteredTransformInitializer(
		fixed, moving, sitk.AffineTransform(fixed.GetDimension())
	)

	R = sitk.ImageRegistrationMethod()

	R.SetShrinkFactorsPerLevel([1, 1, 1])
	R.SetSmoothingSigmasPerLevel([2, 1, 1])

	R.SetMetricAsJointHistogramMutualInformation(20)
	R.MetricUseMovingImageGradientFilterOff()

	R.SetOptimizerAsGradientDescent(
		learningRate=2.0,
		numberOfIterations=10,
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
	resampler.SetDefaultPixelValue(-2048)
	resampler.SetTransform(tx)

	out_moving = resampler.Execute(moving)
	out_moving_label = resampler.Execute(moving_label)

	new_spacing = [
		sz * spc / nsz
		for nsz, sz, spc in zip(fixed.GetSize(), out_moving.GetSize(), out_moving.GetSpacing())
	]

	out_moving.SetSpacing(new_spacing)
	out_moving_label.SetSpacing(new_spacing)

	return (sitk.GetArrayFromImage(out_moving), sitk.GetArrayFromImage(out_moving_label), tx, scale_factor)
	# simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
	# simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)




def channels_to_last(img: torch.Tensor):
    return img.permute(1, 2, 0).contiguous()
