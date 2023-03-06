import copy

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

from tqdm import tqdm

from perceiver.model.segmentation.segmentation import DiceLoss
from perceiver.data.segmentation.miccai import MICCAIDataModule, MICCAIPreprocessor, NUM_CLASSES

data_module = MICCAIDataModule(dataset_dir="/dev/shm/amos22")
segmentation_dataset = data_module.load_dataset("val")
miccai_preproc = MICCAIPreprocessor()

cols, rows = 3, 3
labels = np.array([segmentation_dataset[i]['label'].numpy(force=True) for i in range(cols * rows)])
print("labels [0] shape", labels[0].shape)
labels = np.einsum("n d w h -> n w h d", labels)
print("labels [0] shape", labels[0].shape)

labels_classified = np.zeros((1, *labels[0].shape, NUM_CLASSES))
gt_label = labels[:1,]
t_gt_label = torch.from_numpy(gt_label)

print("labels classified shape", labels_classified.shape)
print("gtlabel shape", gt_label.shape)
print("tensor gtlabel shape", t_gt_label.shape)

for w in tqdm(range(labels[0].shape[0])) :
	for h in range(labels[0].shape[1]) :
		for d in range(labels[0].shape[2]) :
			labels_classified[0,w,h,d] = F.one_hot(torch.arange(NUM_CLASSES))[labels[0,w,h,d]]

tampered_labels_classified = copy.deepcopy(labels_classified)
tampered_labels_classified[0,10:20,10:20,3:4] = F.one_hot(torch.arange(NUM_CLASSES))[15]

labels_classified = np.einsum("b w h d c -> b c w h d", labels_classified)
tampered_labels_classified = np.einsum("b w h d c -> b c w h d", tampered_labels_classified)
t_labels_classified = torch.from_numpy(labels_classified)
t_tampered_labels_classified = torch.from_numpy(tampered_labels_classified)
print("tensor classificatoin shape", t_labels_classified.shape)

y_pred = t_labels_classified.argmax(dim=1)
print("y pred shape", y_pred.shape)
print("first diff val", (t_gt_label.flatten().long() - y_pred.flatten().int())[0])
print("diff bincount", np.bincount((t_gt_label.flatten().long() - y_pred.flatten().int()).abs()))

loss_function = DiceLoss(NUM_CLASSES)

t_labels_classified = t_labels_classified.float()#.flatten(2)
print("flattened labels classified shape", t_labels_classified.shape)
t_gt_label = t_gt_label.long()#.flatten(1)
print("flattened gtlabels classified shape", t_gt_label.shape)

result = loss_function.forward(t_labels_classified, t_gt_label)

print("result untampered", result)
print("result untampered shape", result.shape)
print("result untampered item", result.item())


result = loss_function.forward(t_tampered_labels_classified, t_gt_label)

print("result tampered", result)
print("result tampered shape", result.shape)
print("result tampered item", result.item())

