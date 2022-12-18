import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import re

from perceiver.model.segmentation.segmentation import LitSegmentationMapper
from perceiver.data.segmentation.miccai import NUM_CLASSES, MICCAIDataModule, MICCAIPreprocessor

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

BATCH_SIZE = 1
use_cuda = True

base_logs = os.path.join('logs', 'miccai_seg')
most_recent_version = os.path.join(base_logs, sorted(os.listdir(base_logs), key=natural_keys)[-1])
most_recent_checkpoints = os.path.join(most_recent_version, 'checkpoints')
most_recent_ckpt = os.listdir(most_recent_checkpoints)[0]
ckpt = os.path.join(most_recent_checkpoints, most_recent_ckpt)

# Load the PyTorch Lightning module of the image classifier from a checkpoint
model = LitSegmentationMapper.load_from_checkpoint(ckpt).model.eval()

cuda = use_cuda and torch.cuda.is_available()
dev = "cpu"
if cuda :
	dev = "cuda"
	print("Using CUDA device")

model.to(device=dev)

data_module = MICCAIDataModule(root="AMOS22")
segmentation_dataset = data_module.load_dataset()
miccai_preproc = MICCAIPreprocessor()

cols, rows = 4, 5
imgs = [segmentation_dataset[i]['image'] for i in range(cols * rows)]
print(imgs[0].shape)
imgs = miccai_preproc.preprocess_batch(imgs)
print(imgs[0].shape)
preds = []

for i in range(len(imgs) // BATCH_SIZE) : 
	with torch.no_grad():
		if BATCH_SIZE == 1 :
			raw_imgs = [imgs[i]]
		else :
			raw_imgs = imgs[i:i+BATCH_SIZE]
		inputs = torch.stack(raw_imgs)

		logits = model(inputs.to(device=dev))
		logits = torch.reshape(logits, [BATCH_SIZE, *inputs[0].shape[:-1], 1, NUM_CLASSES]) # TODO: remove splicing and 1 when 3d
		logits = torch.einsum("b w h d c -> b c w h d", logits)
		predictions = logits.argmax(dim=1).int().numpy(force=True)
		print(predictions.shape)
		print(np.bincount(predictions.flatten()))
		print(segmentation_dataset[i]['label'][124,:,:].shape)
		print(np.bincount(segmentation_dataset[i]['label'][124,:,:].flatten()))
		preds.append(*predictions)
	
plt.figure(figsize=(8, 8))
for i, (img, pred) in enumerate(zip(imgs, preds)):
	plt.subplot(rows, cols, i + 1)
	plt.axis('off')
	plt.title('file: %s' % (str(segmentation_dataset[i]['filename'])))
	pred = np.array(pred)
	# TODO: remove indexing when 3d
	plt.imshow(np.array(img[:,:,124][:,:,None]), cmap='gray', interpolation='nearest')
	print(np.bincount(np.array(pred).flatten()))
	plt.imshow(np.array(pred), cmap='jet', alpha=0.5, interpolation='nearest') # TODO: maybe change colour map?

plt.show()