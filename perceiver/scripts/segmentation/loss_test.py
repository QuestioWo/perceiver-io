import numpy as np
import torch.nn.functional as F
import torch

from tqdm import tqdm

from perceiver.model.segmentation.segmentation import SegmentationClassificationLoss
from perceiver.data.segmentation.miccai import MICCAIDataModule, MICCAIPreprocessor, NUM_CLASSES

data_module = MICCAIDataModule(root="AMOS22")
segmentation_dataset = data_module.load_dataset()
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

labels_classified = np.einsum("b w h d c -> b c w h d", labels_classified)
t_labels_classified = torch.from_numpy(labels_classified)
print("tensor classificatoin shape", t_labels_classified.shape)

y_pred = t_labels_classified.argmax(dim=1)
print("y pred shape", y_pred.shape)
print("first diff val", (t_gt_label.flatten().long() - y_pred.flatten().int())[0])
print("diff bincount", np.bincount(t_gt_label.flatten().long() - y_pred.flatten().int()))

loss_function = SegmentationClassificationLoss()

t_labels_classified = t_labels_classified.flatten(2).float()
print("flattened labels classified shape", t_labels_classified.shape)
t_gt_label = t_gt_label.flatten(1).long()
print("flattened gtlabels classified shape", t_gt_label.shape)

result = loss_function.forward(t_labels_classified, t_gt_label)

print("result", result)
print("result shape", result.shape)
print("result item", result.item())
