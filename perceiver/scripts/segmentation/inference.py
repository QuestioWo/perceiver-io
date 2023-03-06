import os
import re
import time

from typing import List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from matplotlib.axes import Axes
from tqdm import tqdm
from medpy import metric

from perceiver.model.segmentation.segmentation import LitSegmentationMapper
from perceiver.data.segmentation.common import coregister_image
from perceiver.data.segmentation.miccai import CT_ONLY, IMAGE_SIZE, NUM_CLASSES, MICCAIDataModule, MICCAIPreprocessor, get_ct_only_dataset_files

DATASET_ROOT = "/volume/amos22"

BATCH_SIZE = 1
USE_CUDA = False
COLS, ROWS = 5, 5

LOAD_SPECIFIC_VERSION = 'version_0'
USE_LAST_CHECKPOINT = False

GENERATE_MICCAI_TEST_RESULTS = False
SAVE_PREDICTIONS = GENERATE_MICCAI_TEST_RESULTS or True
SAVE_PREDICTIONS_DIR = "results"
COMPUTE_METRICS = not GENERATE_MICCAI_TEST_RESULTS and True
COMPUTE_INFERENCE_TIMES = True
COREGISTER_IMAGES = not GENERATE_MICCAI_TEST_RESULTS and False

DISPLAY_DIFFS = not GENERATE_MICCAI_TEST_RESULTS and False
DEFAULT_SLICE = 42
DISPLAY_UPSCALED_INFERENCE_RESULTS = False


def atoi(text):
	return int(text) if text.isdigit() else text
def natural_keys(text):
	return [ atoi(c) for c in re.split('(\d+)',text) ]

class IndexTrackers:
	def __init__(self, axes: List[Axes], imgs, preds, masks, segmentation_dataset):
		self.axes = axes
		self.gt_objs = []

		self.ims = []
		self.im_masks = []

		self.imgs = []
		self.masks = []
		self.preds = []
		
		for i in range(len(axes)) :
			ax = self.axes[i]
			gt_obj = segmentation_dataset[i]

			ax.set_title('file: %s' % (str(gt_obj['filename'])))
			ax.set_aspect(1)

			self.imgs.append(imgs[i])
			self.preds.append(preds[i])
			if DISPLAY_DIFFS :
				self.masks.append(masks[i])

			self.ims.append(ax.imshow(self.imgs[i][:,:,0], cmap='gray', interpolation='nearest'))
			if DISPLAY_DIFFS :
				self.im_masks.append(ax.imshow(self.masks[i][:,:,0], cmap="Reds", alpha=0.5, interpolation="nearest", vmin=0, vmax=1))
			else :
				self.im_masks.append(ax.imshow(self.preds[i][:,:,0], cmap='jet', alpha=0.5, interpolation='nearest', vmin=0, vmax=NUM_CLASSES))

			self.gt_objs.append(gt_obj)

		self.slices = max([i.shape[-1] for i in imgs])
		self.ind = (self.slices // 2) if DEFAULT_SLICE == None else DEFAULT_SLICE

		self.update()

	def on_scroll(self, event):
		if event.button == 'up':
			self.ind = (self.ind + 1) % self.slices
		else:
			self.ind = (self.ind - 1) % self.slices
		self.update()

	def update(self):
		for i in range(len(self.axes)) :
			try: 
				# print(self.gt_objs[i]['filename'], ":")
				self.ims[i].set_data(self.imgs[i][:, :, self.ind])
				if DISPLAY_DIFFS :
					self.im_masks[i].set_data(self.masks[i][:, :, self.ind])
				else:
					self.im_masks[i].set_data(self.preds[i][:, :, self.ind])
				self.axes[i].set_ylabel('slice %s' % self.ind)
				self.ims[i].axes.figure.canvas.draw()
				self.im_masks[i].axes.figure.canvas.draw()

				# print("\texpected", np.bincount(self.gt_objs[i]['label'][SLICE_INDEX_FROM + self.ind,:,:].flatten()))
				# print("\treceived", np.bincount(self.preds[i][:, :, self.ind].flatten()))
				# print("\tdiffs, good vs bad", np.bincount(self.masks[i][:,:,self.ind].flatten()), "accuracy :=", 1 - (np.sum(self.masks[i][:,:,self.ind]) / self.masks[i][:,:,self.ind].flatten().shape[0]))

			except Exception :
				# print("\tfailed update")
				pass


def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 0, 0 # was originally 1, 0
    else:
        return 0, 0


def load_model(ckpt_filename:str=None, load_last_ckpt:bool=USE_LAST_CHECKPOINT, specific_version:str=LOAD_SPECIFIC_VERSION) :
	if ckpt_filename == None :
		base_logs = os.path.join('logs', 'miccai_seg')
		most_recent_version = os.path.join(base_logs, sorted(os.listdir(base_logs), key=natural_keys)[-1])
		most_recent_checkpoints = os.path.join(most_recent_version, 'checkpoints')
		if specific_version != None :
			most_recent_checkpoints = os.path.join(os.path.join(base_logs, specific_version), 'checkpoints')
		all_ckpts = list(filter(lambda x: x.startswith("epoch"), os.listdir(most_recent_checkpoints)))
		sorted_ckpts = sorted(all_ckpts, key=lambda x: float(x.split("val_dice=")[-1].split(".ckpt")[0]), reverse=True)
		
		if load_last_ckpt :
			ckpt = os.path.join(most_recent_checkpoints, "last.ckpt")
		else :
			best_ckpt = sorted_ckpts[0]
			ckpt = os.path.join(most_recent_checkpoints, best_ckpt)
	
	else :
		ckpt = ckpt_filename

	print("Loading %s checkpoint file..." % (ckpt))

	# Load the PyTorch Lightning module of the image classifier from a checkpoint
	model = LitSegmentationMapper.load_from_checkpoint(ckpt).eval()

	return model


def load_and_preprocess_data(generate_test_results:bool=GENERATE_MICCAI_TEST_RESULTS, coregister_loaded_images:bool=COREGISTER_IMAGES, dataset:str="val") :
	data_module = MICCAIDataModule(dataset_dir=DATASET_ROOT, load_raw_instead=True)
	print("Loading and preprocessing validation dataset...")
	segmentation_dataset = data_module.load_dataset(split=dataset)
	miccai_preproc = MICCAIPreprocessor()

	segmentation_objects = []
	if generate_test_results :
		file_list = segmentation_dataset.metadata['test']
		if CT_ONLY :
			file_list = [f['image'] for f in get_ct_only_dataset_files(file_list)]

		file_list = file_list[:len(file_list) // 2]
		# file_list = file_list[len(file_list) // 2:]

		segmentation_objects = [{'image': sitk.ReadImage(os.path.join(DATASET_ROOT, file_name), sitk.sitkFloat32), 'filename': os.path.basename(file_name)} for file_name in tqdm(file_list)]
	else :
		segmentation_objects = [segmentation_dataset[-i] for i in range(len(segmentation_dataset))]
		# segmentation_objects = [segmentation_dataset[-i] for i in range(COLS * ROWS)]

	if coregister_loaded_images :
		print("Coregistering scans for inference...")
		coregistered_images = [coregister_image(sitk.Cast(segmentation_objects[i]['image'], sitk.sitkFloat32), segmentation_dataset.get_coregistration_image()) for i in tqdm(range(len(segmentation_objects)))]
	else :
		print("Loading precoregistered scans for inference...")
	
		preproc_dir = "imagesVa_preprocessed"
		if dataset == 'test' :
			preproc_dir = "imagesTr_preprocessed"
	
		# segmentation_objects = [{"image" : None, "label" : obj['label'], "filename": obj['filename']} for obj in segmentation_objects]
		_accompanying_transformations = [sitk.ReadTransform(os.path.join(DATASET_ROOT, preproc_dir, f['filename'].replace(".nii.gz", "_transformation.tfm"))) for f in segmentation_objects]
		_coregistered_images_only = [sitk.ReadImage(os.path.join(DATASET_ROOT, preproc_dir, f['filename']), sitk.sitkFloat32) for f in tqdm(segmentation_objects)]
		coregistered_images = list(zip(_coregistered_images_only, _accompanying_transformations))

	coregistered_transformations = [obj[1] for obj in coregistered_images]
	imgs = [torch.from_numpy(sitk.GetArrayFromImage(obj[0])) for obj in coregistered_images]

	imgs = miccai_preproc.preprocess_batch(imgs)

	return (segmentation_dataset, segmentation_objects, imgs, coregistered_images, coregistered_transformations, miccai_preproc)


def perform_inferences(imgs, model, device, batch_size=BATCH_SIZE) :
	preds = []
	for i in tqdm(range(len(imgs) // batch_size)) : 
		with torch.no_grad():
			if batch_size == 1 :
				raw_imgs = [imgs[i]]
			else :
				raw_imgs = imgs[i:i+batch_size]
			inputs = {"image":torch.stack(raw_imgs).to(device=device),"label":None}

			logits, _ = model(inputs)
			predictions = logits.argmax(dim=1).int().numpy(force=True)

			preds.append(*predictions)

	return preds


def transform_and_upscale_predictions(model: LitSegmentationMapper, preds, coregistered_images, coregistered_transformations, segmentation_objects, save_predictions:bool=SAVE_PREDICTIONS, save_predictions_dir:str=SAVE_PREDICTIONS_DIR) :
	print("Upscaling predictions...")
	upscaled_preds = []
	prediction_sitk_imgs = []
	masked_labels = []
	for i, p in tqdm(enumerate(preds)) :
		# Convert to sitk.Image to perform transformation then back to np.ndarray for visualisation
		full_input = np.zeros(IMAGE_SIZE, dtype=np.int32)
		full_input[model.slabs_start:model.slabs_start+model.slabs_depth,:,:] = np.einsum("w h d -> d w h", p)

		mask = np.zeros(IMAGE_SIZE, dtype=np.int32)
		mask[model.slabs_start:model.slabs_start+model.slabs_depth,:,:] = 1

		mask_p = sitk.GetImageFromArray(mask)
		mask_p.CopyInformation(coregistered_images[i][0])
		mask_p = sitk.Resample(mask_p, segmentation_objects[i]['image'], coregistered_transformations[i].GetInverse(), sitk.sitkNearestNeighbor, 0)
		mask_p.SetDirection(coregistered_images[i][0].GetDirection())
		mask_p = sitk.Cast(mask_p, sitk.sitkUInt8)

		p = sitk.GetImageFromArray(full_input)
		p.CopyInformation(coregistered_images[i][0])
		p = sitk.Resample(p, segmentation_objects[i]['image'], coregistered_transformations[i].GetInverse(), sitk.sitkNearestNeighbor, 0)
		p.SetDirection(coregistered_images[i][0].GetDirection())
		p = sitk.Cast(p, sitk.sitkUInt8)
		
		prediction_sitk_imgs.append(p)
		
		p = sitk.GetArrayFromImage(p)
		mask_p = sitk.GetArrayFromImage(mask_p)

		p = np.einsum("d w h -> w h d", p)
		mask_p = np.einsum("d w h -> w h d", mask_p)

		masked_label = np.einsum("d w h -> w h d", segmentation_objects[i]['label'])
		masked_label[mask_p != 1] = 0 # set all other values to default/background to avoid their affect on the dice score
		
		upscaled_preds.append(p)
		masked_labels.append(masked_label)

	if save_predictions :
		if not os.path.exists(save_predictions_dir) :
			os.mkdir(save_predictions_dir)

		print("Saving prediction images...")
		for i, p in tqdm(enumerate(prediction_sitk_imgs)) :
			out_fname = os.path.join(save_predictions_dir, "%s.nii.gz" % (segmentation_objects[i]['filename']))
			sitk.WriteImage(p, out_fname)

	return upscaled_preds, masked_labels


def compute_prediction_diffs(upscaled_preds, masked_labels) :
	diffs = []
	if DISPLAY_DIFFS :
		print("Computing diffs...")
		for i in tqdm(range(len(masked_labels))) :
			gt_label = masked_labels[i]
			diffs.append(np.ones_like(upscaled_preds[i]) * (upscaled_preds[i] != gt_label))
		print(len(diffs))

	return diffs


def compute_and_print_metrics(segmentation_dataset, segmentation_objects, upscaled_preds, masked_labels) :
	metrics_list = [
		[["", ""], *[[v, v] for _,v in sorted(segmentation_dataset.get_labels().items(), key=lambda x: int(x[0]))[1:]]]
	]
	print("Computing metrics...")
	for i in tqdm(range(len(segmentation_objects))) :
		fname = segmentation_objects[i]['filename']
		metrics_list.append([[fname, fname]])
		for j in range(1, NUM_CLASSES) :
			curr_gt_label = masked_labels[i]
			curr_gt_label = np.ones_like(curr_gt_label) * (curr_gt_label == j)
			
			curr_slab_start = 0
			curr_slab_end = curr_gt_label.shape[2]
			
			curr_gt_label = curr_gt_label[:,:,curr_slab_start:curr_slab_end]
			# curr_gt_label = np.reshape(curr_gt_label, [1, *curr_gt_label.shape]) # fake batch
			# curr_gt_label = torch.from_numpy(curr_gt_label)

			curr_upscaled_preds = upscaled_preds[i][:,:,curr_slab_start:curr_slab_end]
			curr_upscaled_preds = np.ones_like(curr_upscaled_preds, dtype=np.float32) * (curr_upscaled_preds == j)
			curr_upscaled_preds = np.reshape(curr_upscaled_preds, curr_gt_label.shape)
			# curr_upscaled_preds = torch.from_numpy(curr_upscaled_preds)

			metrics_list[-1].append(calculate_metrics(curr_upscaled_preds, curr_gt_label))

	print("")
	for score_sets in metrics_list :
		for s in score_sets :
			print(s[0], end="\t")
		print("")
	print("")
	for score_sets in metrics_list :
		for s in score_sets :
			print(s[1], end="\t")
		print("")
	print("")

	total_dice = 0
	total_hd = 0
	for score_sets in metrics_list[1:] :
		for s in score_sets[1:] :
			total_dice += s[0]
			total_hd += s[1]
	total_values = (len(metrics_list) * len(metrics_list[0]))

	mean_dsc = (total_dice / total_values)
	mean_hd95 = (total_hd / total_values)

	print("Mean DSC := %.3f, Mean HD95 := %.3f" % (mean_dsc, mean_hd95))

	return mean_dsc


def main() :
	model = load_model()

	cuda = USE_CUDA and torch.cuda.is_available()
	dev = "cpu"
	if cuda :
		dev = "cuda"
		print("Using CUDA device for inferences!")
	else :
		print("Using CPU for inferences!")

	model.to(device=dev)

	segmentation_dataset, segmentation_objects, imgs, coregistered_images, coregistered_transformations, miccai_preproc =  load_and_preprocess_data(dataset='test')

	copy_of_imgs = [torch.clone(i[:,:,model.slabs_start:model.slabs_start+model.slabs_depth]) for i in imgs]

	# Perform inference and get predictions
	print("Performing inferences...")
	start = time.clock_gettime(0)

	preds = perform_inferences(imgs, model, dev)

	if COMPUTE_INFERENCE_TIMES :
		end = time.clock_gettime(0)
		print("Average inference time := %s" %(str((end - start) / len(imgs))))

	# Transform and scale labels back to original size
	upscaled_preds, masked_labels = transform_and_upscale_predictions(model, preds, coregistered_images, coregistered_transformations, segmentation_objects)

	fig, axes = plt.subplots(ROWS, COLS)
	axes = axes.flatten()

	diffs = compute_prediction_diffs(upscaled_preds, masked_labels)

	if COMPUTE_METRICS :
		compute_and_print_metrics(segmentation_dataset, segmentation_objects, upscaled_preds, masked_labels)

	print("Preprocessing raw images...")
	imgs_raw = []
	for i in tqdm(range(len(segmentation_objects))) :
		imgs_raw.append(
			miccai_preproc.preprocess_batch([
				torch.from_numpy(sitk.GetArrayFromImage(segmentation_objects[i]['image']))
			])[0]
		)
		segmentation_objects[i]['image'] = None

	tracker = None
	if DISPLAY_UPSCALED_INFERENCE_RESULTS :
		tracker = IndexTrackers(axes, imgs_raw, upscaled_preds, diffs, segmentation_objects)
	else :
		tracker = IndexTrackers(axes, copy_of_imgs, preds, diffs, segmentation_objects)
	
	fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)

	plt.show()

if __name__ == "__main__" :
	main()
