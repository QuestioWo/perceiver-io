import nevergrad as ng
import torch
import traceback
import gc

from perceiver.scripts.segmentation.mapper import SegmentationMapperCLI
from perceiver.model.segmentation.segmentation import LitSegmentationMapper
from perceiver.scripts.segmentation.inference import compute_and_print_metrics, load_and_preprocess_data, load_model, perform_inferences, transform_and_upscale_predictions

USE_CUDA_FOR_LOSS_INFERENCES = True

ENCODER_DROPOUT = 0.3
DECODER_DROPOUT = 0.3
SLABS_START = 80
SLABS_DEPTH = 20
SLABS_SIZE = 4

OPTIMISATION_BUDGET = 50

ALL_MODEL_PARAMERTERS = [
	# NOTE: Omitted/set values:
	# - encoder/decoder.dropout
	# - slabs_start
	# - slabs_depth
	# - slabs_size
	# - encoder.num_cross_attention_qk_channels
	# - encoder.num_cross_attention_v_channels
	# - encoder.num_self_attention_qk_channels
	# - encoder.num_self_attention_v_channels
	# - encoder.num_cross_attention_qk_channels
	# - decoder.num_cross_attention_qk_channels
	# - decoder.num_cross_attention_v_channels
	# - decoder.cross_attention_residual
	
	"model.num_latents",
	"model.num_latent_channels",

	# Custom parameters for segmentation
	"model.overlap_slices",
	"model.recursive_slices",
	
	"model.encoder.num_frequency_bands",

	"model.encoder.num_cross_attention_layers",
	"model.encoder.num_cross_attention_heads",
	
	"model.encoder.num_self_attention_blocks",
	"model.encoder.num_self_attention_layers_per_block",
	
	"model.encoder.num_self_attention_heads",

	"model.decoder.num_cross_attention_heads",
]

def retreive_default_parameters(parameter_candidate) :
	parameter_values, *_ = parameter_candidate.args

	defaults = {k:parameter_values[i] for i,k in enumerate(ALL_MODEL_PARAMERTERS)}
	
	defaults["model.encoder.dropout"] = ENCODER_DROPOUT
	defaults["model.decoder.dropout"] = DECODER_DROPOUT
	
	defaults["model.slabs_start"] = SLABS_START
	defaults["model.slabs_depth"] = SLABS_DEPTH
	defaults["model.slabs_size"] = SLABS_SIZE

	defaults["model.encoder.num_cross_attention_qk_channels"] = defaults["model.encoder.num_cross_attention_heads"] * 2
	defaults["model.encoder.num_cross_attention_v_channels"] = defaults["model.encoder.num_cross_attention_heads"] * 2

	defaults["model.encoder.num_self_attention_qk_channels"] = defaults["model.encoder.num_self_attention_heads"] * 2
	defaults["model.encoder.num_self_attention_v_channels"] = defaults["model.encoder.num_self_attention_heads"] * 2
	
	defaults["model.decoder.num_cross_attention_qk_channels"] = defaults["model.decoder.num_cross_attention_heads"] * 2
	defaults["model.decoder.num_cross_attention_v_channels"] = defaults["model.decoder.num_cross_attention_heads"] * 2
	
	defaults["model.decoder.cross_attention_residual"] = True
	
	return defaults


def find_loss(index:int) :
	model = load_model(specific_version=None) # load most recent version

	cuda = torch.cuda.is_available() and USE_CUDA_FOR_LOSS_INFERENCES
	dev = "cpu"
	if cuda :
		dev = "cuda"
		print("Using CUDA device for inferences!")
	else :
		print("Using CPU for inferences!")

	model.to(device=dev)

	segmentation_dataset, segmentation_objects, imgs, coregistered_images, coregistered_transformations, _ =  load_and_preprocess_data(dataset='test')

	preds = perform_inferences(imgs, model, dev)
	
	upscaled_preds, masked_labels = transform_and_upscale_predictions(model, preds, coregistered_images, coregistered_transformations, segmentation_objects, save_predictions_dir="results_" + str(index))
	
	mean_dsc = compute_and_print_metrics(segmentation_dataset, segmentation_objects, upscaled_preds, masked_labels)

	return 1-mean_dsc # negation of mean dsc, meaning better models have lower loss

if __name__ == "__main__":
	current_candidate = (
		512,
		128,
		
		2,
		1,
		
		4,
		
		2,
		2,

		2,
		4,

		2,
		
		2,
	)

	num_latents = ng.p.Scalar(lower=4, upper=2048).set_integer_casting()
	num_latent_channels = ng.p.Scalar(lower=1, upper=1024).set_integer_casting()
	
	overlap_slices = ng.p.Scalar(lower=0, upper=4).set_integer_casting()
	recursive_slices = ng.p.Scalar(lower=0, upper=4).set_integer_casting()
	
	encoder_num_frequency_bands = ng.p.Scalar(lower=1, upper=128).set_integer_casting()

	encoder_num_cross_attention_layers = ng.p.Scalar(lower=1, upper=4).set_integer_casting()
	encoder_num_cross_attention_heads = ng.p.Scalar(lower=1, upper=4).set_integer_casting()

	encoder_num_self_attention_blocks = ng.p.Scalar(lower=1, upper=4).set_integer_casting()
	encoder_num_self_attention_layers_per_block = ng.p.Scalar(lower=1, upper=4).set_integer_casting()

	encoder_num_self_attention_heads = ng.p.Scalar(lower=1, upper=4).set_integer_casting()

	decoder_num_cross_attention_heads = ng.p.Scalar(lower=1, upper=4).set_integer_casting()

	instru = ng.p.Instrumentation(ng.p.Tuple(
		num_latents,
		num_latent_channels,
		
		overlap_slices,
		recursive_slices,
		
		encoder_num_frequency_bands,
		
		encoder_num_cross_attention_layers,
		encoder_num_cross_attention_heads,
		
		encoder_num_self_attention_blocks,
		encoder_num_self_attention_layers_per_block,
		
		encoder_num_self_attention_heads,
		
		decoder_num_cross_attention_heads
	))
	
	ng_optimizer = ng.optimizers.NGOpt(parametrization=instru, budget=OPTIMISATION_BUDGET, num_workers=1)

	# Initialise with current best values
	ng_optimizer.suggest(current_candidate)

	for i in range(OPTIMISATION_BUDGET) :
		current_candidate = ng_optimizer.ask()
		
		current_args = retreive_default_parameters(current_candidate)
		print("!!!!!!!!!!!!!!!!!!!!!!!!! Current args !!!!!!!!!!!!!!!!!!!!!!!!!")
		print(current_args)
		print("!!!!!!!!!!!!!!!!!!!!!! Beginning training !!!!!!!!!!!!!!!!!!!!!!")

		loss = 1.0

		try:
			SegmentationMapperCLI(LitSegmentationMapper, default_parameters=current_args, description="Segmentation map generator", run=True)

			gc.collect()

			loss = find_loss(i)
		except Exception :
			print("!!!!!!!!!!!!!!!!!!! Errored during training !!!!!!!!!!!!!!!!!!!!")
			traceback.print_exc()
			loss = 1.0

		gc.collect()

		print("!!!!!!!!!!!!! %dth loss (1-mean_dsc) := %.4f !!!!!!!!!!!!!" % (i, loss))

		ng_optimizer.tell(current_candidate, loss)

	final_candidate = ng_optimizer.provide_recommendation()

	recommended_args = retreive_default_parameters(final_candidate)
	print("!!!!!!!!!!!!!!!!!!!!!!! Recommended args !!!!!!!!!!!!!!!!!!!!!!!")
	print(recommended_args)
	print("!!!!!!!!!!!!!!!!!!!! Optimisation completed !!!!!!!!!!!!!!!!!!!!")




