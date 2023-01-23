import os
import re
import shutil

from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from matplotlib.axes import Axes
from tqdm import tqdm

from perceiver.model.segmentation.segmentation import SLABS_DEPTH, SLABS_START, LitSegmentationMapper, SLICE_INDEX_FROM, SLICE_INDEX_TO
from perceiver.data.segmentation.common import coregister_image, zoom_sitk_image_label
from perceiver.data.segmentation.miccai import IMAGE_SIZE, NUM_CLASSES, MICCAIDataModule, MICCAIPreprocessor

DEFAULT_SLICE = 42
DISPLAY_DIFFS = False
BATCH_SIZE = 1
USE_CUDA = False
COLS, ROWS = 3, 3
USE_LAST_CHECKPOINT = False
SAVE_PREDICTIONS = True
SAVE_PREDICTIONS_DIR = "labelsTs"

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

cuda = USE_CUDA and torch.cuda.is_available()
dev = "cpu"
if cuda :
	dev = "cuda"
	print("Using CUDA device for inferences!")
else :
    print("Using CPU for inferences!")

model.to(device=dev)

data_module = MICCAIDataModule(root="AMOS22", load_raw_instead=True)
print("Loading and preprocessing validation dataset...")
segmentation_dataset = data_module.load_dataset("val")
miccai_preproc = MICCAIPreprocessor()

segmentation_objects = [segmentation_dataset[-i] for i in range(COLS * ROWS)]

original_sizes = [segmentation_objects[i]['image'].GetSize() for i in range(len(segmentation_objects))]

imgs_raw = [torch.from_numpy(sitk.GetArrayFromImage(segmentation_objects[i]['image'])) for i in range(len(segmentation_objects))]
# print(original_sizes)
imgs_raw = [miccai_preproc.preprocess_batch([raw_img])[0] for raw_img in imgs_raw]
imgs_raw = [raw_img for raw_img in imgs_raw]

print("Coregistering scans for inference...")
coregistered_images = [coregister_image(segmentation_objects[i]['image'], segmentation_dataset.get_coregistration_image()) for i in tqdm(range(len(segmentation_objects)))]

coregistered_scalings = [obj[2] for obj in coregistered_images]
coregistered_spacings = [obj[3] for obj in coregistered_images]
coregistered_transformations = [obj[1] for obj in coregistered_images]
imgs = [torch.from_numpy(sitk.GetArrayFromImage(obj[0])) for obj in coregistered_images]

imgs = miccai_preproc.preprocess_batch(imgs)
copy_of_imgs = [torch.clone(i[:,:,SLICE_INDEX_FROM:SLICE_INDEX_TO]) for i in imgs]
preds = []

# Perform inference and get predictions
print("Performing inferences...")
for i in tqdm(range((COLS * ROWS) // BATCH_SIZE)) : 
	with torch.no_grad():
		if BATCH_SIZE == 1 :
			raw_imgs = [imgs[i]]
		else :
			raw_imgs = imgs[i:i+BATCH_SIZE]
		inputs = {"image":torch.stack(raw_imgs).to(device=dev),"label":None}

		logits, _ = model(inputs)
		predictions = logits.argmax(dim=1).int().numpy(force=True)

		preds.append(*predictions)

# Transform and scale labels back to original size
upscaled_preds = []
prediction_sitk_imgs = []
print("Upscaling predictions...")
for i, p in tqdm(enumerate(preds)) :
	# Convert to sitk.Image to perform transformation then back to np.ndarray for visualisation
	full_input = np.zeros(IMAGE_SIZE, dtype=np.int32)
	full_input[SLABS_START:SLABS_START+SLABS_DEPTH,:,:] = np.einsum("w h d -> d w h", p)

	p = sitk.GetImageFromArray(full_input)

	p.SetDirection(coregistered_images[i][0].GetDirection())
	p.SetOrigin(coregistered_images[i][0].GetOrigin())
	p.SetSpacing(coregistered_images[i][0].GetSpacing())
	
	p = sitk.Resample(p, segmentation_objects[i]['image'], coregistered_transformations[i].GetInverse(), sitk.sitkNearestNeighbor, 0)

	p = sitk.DICOMOrient(p, "RPI")
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

fig, axes = plt.subplots(ROWS, COLS)
axes = axes.flatten()

# non-vectorised but for weird shapes
diffs = []
if DISPLAY_DIFFS :
	print("Computing diffs...")
	for i in tqdm(range(len(segmentation_objects))) :
		gt_label = np.einsum("d w h -> w h d", np.array(segmentation_objects[i]['label'].numpy(force=True)))
		diffs.append(np.ones_like(upscaled_preds[i]) * (upscaled_preds[i] != gt_label))
	print(len(diffs))

# tracker = IndexTrackers(axes, copy_of_imgs, preds, diffs, segmentation_objects)
tracker = IndexTrackers(axes, imgs_raw, upscaled_preds, diffs, segmentation_objects)
fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)

plt.show()
