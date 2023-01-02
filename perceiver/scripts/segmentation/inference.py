import os

from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
import re

from matplotlib.axes import Axes
from tqdm import tqdm

from perceiver.model.segmentation.segmentation import LitSegmentationMapper, SLICE_INDEX_FROM, SLICE_INDEX_TO
from perceiver.data.segmentation.miccai import NUM_CLASSES, MICCAIDataModule, MICCAIPreprocessor

DEFAULT_SLICE = 4
DISPLAY_DIFFS = False
BATCH_SIZE = 1
USE_CUDA = False

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

		self.slices = (SLICE_INDEX_TO - SLICE_INDEX_FROM)
		self.ind = (self.slices // 2) if DEFAULT_SLICE == None else DEFAULT_SLICE
		
		for i in range(len(axes)) :
			ax = self.axes[i]
			# plt.axis('off')
			gt_obj = segmentation_dataset[i]

			ax.set_title('file: %s' % (str(gt_obj['filename'])))

			self.imgs.append(imgs[i])
			self.preds.append(preds[i])
			self.masks.append(masks[i])

			self.ims.append(ax.imshow(np.array(self.imgs[i][:,:,SLICE_INDEX_FROM + self.ind]), cmap='gray', interpolation='nearest'))
			if DISPLAY_DIFFS :
				self.im_masks.append(ax.imshow(self.masks[i][:,:,self.ind], cmap="Reds", alpha=0.5, interpolation="nearest"))
			else :
				self.im_masks.append(ax.imshow(np.array(self.preds[i][:,:,self.ind]), cmap='jet', alpha=0.5, interpolation='nearest'))

			self.gt_objs.append(gt_obj)

		self.update()

	def on_scroll(self, event):
		# print("%s %s" % (event.button, event.step))
		if event.button == 'up':
			self.ind = (self.ind + 1) % self.slices
		else:
			self.ind = (self.ind - 1) % self.slices
		self.update()

	def update(self):
		for i in range(len(self.axes)) :
			self.ims[i].set_data(self.imgs[i][:, :, SLICE_INDEX_FROM + self.ind])
			if DISPLAY_DIFFS :
				self.im_masks[i].set_data(self.masks[i][:, :, self.ind])
			else:
				self.im_masks[i].set_data(self.preds[i][:, :, self.ind])
			self.axes[i].set_ylabel('slice %s' % self.ind)
			self.ims[i].axes.figure.canvas.draw()
			self.im_masks[i].axes.figure.canvas.draw()

			print(self.gt_objs[i]['filename'], ":")
			print("\texpected", np.bincount(self.gt_objs[i]['label'][SLICE_INDEX_FROM + self.ind,:,:].flatten()))
			print("\treceived", np.bincount(self.preds[i][:, :, self.ind].flatten()))
			print("\tdiffs, good vs bad", np.bincount(self.masks[i].flatten()))

base_logs = os.path.join('logs', 'miccai_seg')
most_recent_version = os.path.join(base_logs, sorted(os.listdir(base_logs), key=natural_keys)[-1])
most_recent_checkpoints = os.path.join(most_recent_version, 'checkpoints')
most_recent_ckpt = os.listdir(most_recent_checkpoints)[0]
ckpt = os.path.join(most_recent_checkpoints, most_recent_ckpt)

# Load the PyTorch Lightning module of the image classifier from a checkpoint
model = LitSegmentationMapper.load_from_checkpoint(ckpt).model.eval()

cuda = USE_CUDA and torch.cuda.is_available()
dev = "cpu"
if cuda :
	dev = "cuda"
	print("Using CUDA device")

model.to(device=dev)

data_module = MICCAIDataModule(root="AMOS22")
segmentation_dataset = data_module.load_dataset()
miccai_preproc = MICCAIPreprocessor()

cols, rows = 3, 3
imgs = [segmentation_dataset[i]['image'] for i in range(cols * rows)]
print(imgs[0].shape)
imgs = miccai_preproc.preprocess_batch(imgs)
print(imgs[0].shape)
preds = []

for i in tqdm(range((cols * rows) // BATCH_SIZE)) : 
	with torch.no_grad():
		if BATCH_SIZE == 1 :
			raw_imgs = [imgs[i]]
		else :
			raw_imgs = imgs[i:i+BATCH_SIZE]
		inputs = torch.stack(raw_imgs)

		logits = model(inputs.to(device=dev))
		logits = torch.reshape(logits, [BATCH_SIZE, *inputs[0].shape[:-1], (SLICE_INDEX_TO - SLICE_INDEX_FROM), NUM_CLASSES])
		logits = torch.einsum("b w h d c -> b c w h d", logits)
		predictions = logits.argmax(dim=1).int().numpy(force=True)
		preds.append(*predictions)
	
preds = np.array(preds)

fig, axes = plt.subplots(rows, cols)
axes = axes.flatten()
all_labels = np.einsum("b d w h -> b w h d", np.array([segmentation_dataset[i]['label'][SLICE_INDEX_FROM:SLICE_INDEX_TO,:,:].numpy(force=True) for i in range(cols*rows)]))
print(all_labels.shape)
diffs = np.ones_like(preds) * (preds != all_labels)
print(preds.shape)
print(diffs.shape)

tracker = IndexTrackers(axes, imgs, preds, diffs, segmentation_dataset)
fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)

plt.show()