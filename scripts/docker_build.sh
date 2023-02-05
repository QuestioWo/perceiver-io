model_path=$1
coregistration_image_path=$2

if [[ -z $model_path ]]; then
	echo "Please supply a path to the model"
	exit 1
fi

if [[ -z $coregistration_image_path ]]; then
	echo "Please supply a path to the image used for coregistration"
	exit 1
fi

echo "NOTE: This script must be ran with sudo when used on Ubuntu"

cp $model_path ./best.ckpt
cp $coregistration_image_path ./coregistration_image.nii.gz

sudo docker build -t jimcarty-perceiver-io:latest \
	--build-arg MODEL_PATH=best.ckpt \
	--build-arg COREGISTRATION_IMAGE_PATH=coregistration_image.nii.gz \
	--build-arg RUN_INFERENCE_PATH=run_inference.py .