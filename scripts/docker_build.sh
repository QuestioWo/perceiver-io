#!/bin/bash

# See examples for running in perceiver/scripts/segmentation/docker_inference.py script

model_path=$1
coregistration_image_path=$2
tagname=$3

if [[ -z $model_path ]]; then
	echo "Please supply a path to the model"
	exit 1
fi

if [[ -z $coregistration_image_path ]]; then
	echo "Please supply a path to the image used for coregistration"
	exit 1
fi

if [[ -z $tagname ]]; then
	tagname="ghcr.io/questiowo/miccai-perceiver-io:latest"
fi

echo "NOTE: This script must be ran with sudo when used on Ubuntu"

cp $model_path ./best.ckpt
cp $coregistration_image_path ./coregistration_image.nii.gz

sudo docker build -t $tagname \
	--build-arg MODEL_PATH=best.ckpt \
	--build-arg COREGISTRATION_IMAGE_PATH=coregistration_image.nii.gz \
	--build-arg RUN_INFERENCE_PATH=run_inference.py .