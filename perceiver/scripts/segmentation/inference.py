import math
import os
import re
import shutil
import time

from typing import List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from matplotlib.axes import Axes
from tqdm import tqdm
from medpy import metric

from perceiver.model.segmentation.segmentation import SLABS_DEPTH, SLABS_START, DiceLoss, LitSegmentationMapper, SLICE_INDEX_FROM, SLICE_INDEX_TO
from perceiver.data.segmentation.common import coregister_image, zoom_sitk_image_label
from perceiver.data.segmentation.miccai import IMAGE_SIZE, NUM_CLASSES, MICCAIDataModule, MICCAIPreprocessor

DEFAULT_SLICE = 42
DISPLAY_DIFFS = False
BATCH_SIZE = 1
USE_CUDA = True
COLS, ROWS = 3, 3
USE_LAST_CHECKPOINT = False
SAVE_PREDICTIONS = True
SAVE_PREDICTIONS_DIR = "labelsTs"
COMPUTE_METRICS = True
COMPUTE_INFERENCE_TIMES = True

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
			# plt.axis('off')
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
				print("\tfailed update")


def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def load_model() :
	base_logs = os.path.join('logs', 'miccai_seg')
	most_recent_version = os.path.join(base_logs, sorted(os.listdir(base_logs), key=natural_keys)[-1])
	most_recent_checkpoints = os.path.join(most_recent_version, 'checkpoints')
	# most_recent_checkpoints = os.path.join(os.path.join(base_logs, "version_12"), 'checkpoints')
	all_ckpts = list(filter(lambda x: x.startswith("epoch"), os.listdir(most_recent_checkpoints)))
	sorted_ckpts = sorted(all_ckpts, key=lambda x: float(x.split("val_loss=")[-1].split(".ckpt")[0]))
	best_ckpt = sorted_ckpts[0]
	ckpt = os.path.join(most_recent_checkpoints, best_ckpt)

	if USE_LAST_CHECKPOINT :
		ckpt = os.path.join(most_recent_checkpoints, "last.ckpt")

	print("Loading %s checkpoint file..." % (ckpt))

	# Load the PyTorch Lightning module of the image classifier from a checkpoint
	model = LitSegmentationMapper.load_from_checkpoint(ckpt).eval()

	return model


def load_and_preprocess_data() :
	data_module = MICCAIDataModule(root="AMOS22", load_raw_instead=True)
	print("Loading and preprocessing validation dataset...")
	segmentation_dataset = data_module.load_dataset("val")
	miccai_preproc = MICCAIPreprocessor()

	segmentation_objects = [segmentation_dataset[-i] for i in range(COLS * ROWS)]

	imgs_raw = [torch.from_numpy(sitk.GetArrayFromImage(segmentation_objects[i]['image'])) for i in range(len(segmentation_objects))]
	imgs_raw = [miccai_preproc.preprocess_batch([raw_img])[0] for raw_img in imgs_raw]
	imgs_raw = [raw_img for raw_img in imgs_raw]

	print("Coregistering scans for inference...")
	coregistered_images = [coregister_image(segmentation_objects[i]['image'], segmentation_dataset.get_coregistration_image()) for i in tqdm(range(len(segmentation_objects)))]

	coregistered_transformations = [obj[1] for obj in coregistered_images]
	imgs = [torch.from_numpy(sitk.GetArrayFromImage(obj[0])) for obj in coregistered_images]

	imgs = miccai_preproc.preprocess_batch(imgs)

	return (segmentation_dataset, segmentation_objects, imgs, coregistered_images, coregistered_transformations, imgs_raw)


def perform_inferences(total_inferences, imgs, model) :
	preds = []
	for i in tqdm(range(total_inferences)) : 
		with torch.no_grad():
			if BATCH_SIZE == 1 :
				raw_imgs = [imgs[i]]
			else :
				raw_imgs = imgs[i:i+BATCH_SIZE]
			inputs = {"image":torch.stack(raw_imgs).to(device=dev),"label":None}

			logits, _ = model(inputs)
			predictions = logits.argmax(dim=1).int().numpy(force=True)

			preds.append(*predictions)

	return preds


def transform_and_upscale_predictions(preds) :
	print("Upscaling predictions...")
	upscaled_preds = []
	prediction_sitk_imgs = []
	for i, p in tqdm(enumerate(preds)) :
		# Convert to sitk.Image to perform transformation then back to np.ndarray for visualisation
		full_input = np.zeros(IMAGE_SIZE, dtype=np.int32)
		full_input[SLABS_START:SLABS_START+SLABS_DEPTH,:,:] = np.einsum("w h d -> d w h", p)

		p = sitk.GetImageFromArray(full_input)

		p.CopyInformation(coregistered_images[i][0])

		p = sitk.Resample(p, segmentation_objects[i]['image'], coregistered_transformations[i].GetInverse(), sitk.sitkNearestNeighbor, 0)

		p = p.SetDirection(coregistered_images[i][0].GetDirection())

		p = sitk.Cast(p, sitk.sitkUInt8)
		
		prediction_sitk_imgs.append(p)
		
		p = sitk.GetArrayFromImage(p)

		p = np.einsum("d w h -> w h d", p)
		
		upscaled_preds.append(p)

	if SAVE_PREDICTIONS :
		if os.path.exists(SAVE_PREDICTIONS_DIR) :
			shutil.rmtree(SAVE_PREDICTIONS_DIR)
		os.mkdir(SAVE_PREDICTIONS_DIR)

		print("Saving prediction images...")
		for i, p in tqdm(enumerate(prediction_sitk_imgs)) :
			out_fname = os.path.join(SAVE_PREDICTIONS_DIR, "%s.nii.gz" % (segmentation_objects[i]['filename']))
			sitk.WriteImage(p, out_fname)

	return upscaled_preds


def compute_prediction_diffs(upscaled_preds, segmentation_objects) :
	diffs = []
	if DISPLAY_DIFFS :
		print("Computing diffs...")
		for i in tqdm(range(len(segmentation_objects))) :
			gt_label = np.einsum("d w h -> w h d", segmentation_objects[i]['label'].numpy(force=True))
			diffs.append(np.ones_like(upscaled_preds[i]) * (upscaled_preds[i] != gt_label))
		print(len(diffs))

	return diffs


def compute_and_print_metrics(segmentation_dataset, segmentation_objects, upscaled_preds) :
	metrics_list = [
		[["", ""], *[[v, v] for _,v in sorted(segmentation_dataset.get_labels().items(), key=lambda x: int(x[0]))]]
	]
	print("Computing metrics")
	for i in tqdm(range(len(segmentation_objects))) :
		fname = segmentation_objects[i]['filename']
		metrics_list.append([[fname, fname]])
		for j in range(NUM_CLASSES) :
			curr_gt_label = segmentation_objects[i]['label'].numpy(force=True)
			curr_gt_label = np.einsum("d w h -> w h d", curr_gt_label)
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


if __name__ == "__main__" :
	model = load_model()

	cuda = USE_CUDA and torch.cuda.is_available()
	dev = "cpu"
	if cuda :
		dev = "cuda"
		print("Using CUDA device for inferences!")
	else :
		print("Using CPU for inferences!")

	model.to(device=dev)

	segmentation_dataset, segmentation_objects, imgs, coregistered_images, coregistered_transformations, imgs_raw =  load_and_preprocess_data()

	copy_of_imgs = [torch.clone(i[:,:,SLICE_INDEX_FROM:SLICE_INDEX_TO]) for i in imgs]

	# Perform inference and get predictions
	print("Performing inferences...")
	start = time.clock_gettime(0)
	total_inferences = (COLS * ROWS) // BATCH_SIZE

	preds = perform_inferences(total_inferences, imgs, model)

	if COMPUTE_INFERENCE_TIMES :
		end = time.clock_gettime(0)
		print("Average inference time := %s" %(str((end - start) / total_inferences)))

	# Transform and scale labels back to original size
	upscaled_preds = transform_and_upscale_predictions(preds)

	fig, axes = plt.subplots(ROWS, COLS)
	axes = axes.flatten()

	# non-vectorised but for weird shapes
	diffs = compute_prediction_diffs(upscaled_preds, segmentation_objects)

	if COMPUTE_METRICS :
		compute_and_print_metrics(segmentation_dataset, segmentation_objects, upscaled_preds)

	# tracker = IndexTrackers(axes, copy_of_imgs, preds, diffs, segmentation_objects)
	tracker = IndexTrackers(axes, imgs_raw, upscaled_preds, diffs, segmentation_objects)
	fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)

	# plt.show()
