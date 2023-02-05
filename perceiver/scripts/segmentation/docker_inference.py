import time
import os
from typing import List, Tuple

import torch
import SimpleITK as sitk

from tqdm import tqdm

from perceiver.data.segmentation.miccai import MICCAIPreprocessor
from perceiver.data.segmentation.common import coregister_image
from perceiver.scripts.segmentation.inference import load_model, perform_inferences, transform_and_upscale_predictions

# To build and run the docker image, the following commands can be used as starting point:
"""
sudo bash scripts/docker_build.sh \
	logs/miccai_seg/correct_11_recusive_2_overlap_5_ce_only/checkpoints/epoch\=264-val_loss\=0.102.ckpt \
	/mnt/d/amos22/imagesTr_preprocessed/coregistration_image.nii.gz &&
sudo docker run -it --rm --runtime=nvidia --ipc=host  -e NVIDIA_VISIBLE_DEVICES=all \
	--gpus 0 --user root -v/mnt/d/amos22/imagesVa/:/workspace/input/ \
	-v$(pwd)/current_results/:/workspace/output/ jimcarty-perceiver-io:latest \
	python run_inference.py
"""

# As defined in the AMOS docker image submission requirements: https://github.com/JiYuanFeng/AMOS/tree/docker
INPUT_IMAGES_DIR = '/workspace/input/'
SAVE_PREDICTIONS = True
SAVE_PREDICTIONS_DIR = "/workspace/output/"

# As defined in the Dockerfile
CHECKPOINT_FILE = os.path.join("/app", "best.ckpt")
COREGISTRATION_IMAGE_PATH = os.path.join("/app", "coregistration_image.nii.gz")

# Inference config
BATCH_SIZE = 1
USE_CUDA = True

def load_coregistraion_image() -> sitk.Image :
	return sitk.ReadImage(COREGISTRATION_IMAGE_PATH, sitk.sitkFloat64)

def load_and_preprocess_data_without_amos_dataset(coregistration_image: sitk.Image) -> Tuple[List[dict], torch.Tensor, List[Tuple[sitk.Image, sitk.Transform, Tuple[float, float, float], Tuple[float, float, float]]], List[sitk.Transform]] :
	miccai_preproc = MICCAIPreprocessor()
	
	segmentation_objects = [{'image': sitk.ReadImage(os.path.join(INPUT_IMAGES_DIR, file_name), sitk.sitkFloat32), 'filename': os.path.basename(file_name)} for file_name in tqdm(os.listdir(INPUT_IMAGES_DIR))]
	coregistered_images = [coregister_image(sitk.Cast(segmentation_objects[i]['image'], sitk.sitkFloat64), coregistration_image) for i in tqdm(range(len(segmentation_objects)))]
	coregistered_transformations = [obj[1] for obj in coregistered_images]
	imgs = [torch.from_numpy(sitk.GetArrayFromImage(obj[0])) for obj in coregistered_images]

	imgs = miccai_preproc.preprocess_batch(imgs)

	return (segmentation_objects, imgs, coregistered_images, coregistered_transformations)


def main() :
	model = load_model(CHECKPOINT_FILE)

	cuda = USE_CUDA and torch.cuda.is_available()
	dev = "cpu"
	if cuda :
		dev = "cuda"
		print("Using CUDA device for inferences!")
	else :
		print("Using CPU for inferences!")

	model.to(device=dev)

	coreg_image = load_coregistraion_image()

	segmentation_objects, imgs, coregistered_images, coregistered_transformations =  load_and_preprocess_data_without_amos_dataset(coreg_image)

	# Perform inference and get predictions
	print("Performing inferences...")
	start = time.clock_gettime(0)

	preds = perform_inferences(imgs, model, dev, BATCH_SIZE)

	end = time.clock_gettime(0)
	print("Average inference time := %s" %(str((end - start) / len(imgs))))

	# Transform and scale labels back to original size
	transform_and_upscale_predictions(preds, coregistered_images, coregistered_transformations, segmentation_objects, SAVE_PREDICTIONS, SAVE_PREDICTIONS_DIR)


if __name__ == "__main__" :
	main()