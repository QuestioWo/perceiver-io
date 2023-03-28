import nevergrad as ng
import torch
import traceback
import gc
import os

from perceiver.scripts.segmentation.mapper import SegmentationMapperCLI
from perceiver.model.segmentation.segmentation import LitSegmentationMapper
from perceiver.scripts.segmentation.inference import compute_and_print_metrics, load_and_preprocess_data, load_model, natural_keys, perform_inferences, transform_and_upscale_predictions

USE_CUDA_FOR_LOSS_INFERENCES = True

ENCODER_DROPOUT = 0.3
DECODER_DROPOUT = 0.3
SLABS_START = 80
SLABS_DEPTH = 12
SLABS_SIZE = 4

OPTIMISATION_BUDGET = 50

OPTIMISER_PICKLE = "/volume/ng_optimiser.pkl"

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

PAST_CONFIG = [
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 12, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 1026, 'model.num_latent_channels': 512, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 64, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 2, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 12, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 409, 'model.overlap_slices': 2, 'model.recursive_slices': 3, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 12, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 409, 'model.overlap_slices': 2, 'model.recursive_slices': 3, 'model.encoder.num_frequency_bands': 123, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 12, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 251, 'model.overlap_slices': 2, 'model.recursive_slices': 3, 'model.encoder.num_frequency_bands': 123, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 3, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 12, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 409, 'model.overlap_slices': 2, 'model.recursive_slices': 3, 'model.encoder.num_frequency_bands': 123, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 2, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 12, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 1020, 'model.num_latent_channels': 409, 'model.overlap_slices': 1, 'model.recursive_slices': 3, 'model.encoder.num_frequency_bands': 74, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 3, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 12, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 6, 'model.encoder.num_self_attention_v_channels': 6, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 409, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 123, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 12, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 409, 'model.overlap_slices': 2, 'model.recursive_slices': 3, 'model.encoder.num_frequency_bands': 42, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 12, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
]
PAST_LOSS = [
	0.07569444, 0.09068072, 0.06783247, 0.06761980, 0.07806760, 0.07555628, 1.00000000, 0.07580739, 0.07397062
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
	model = load_model(specific_version=None, load_path=os.path.join("/volume", "logs_optimisation", "miccai_seg_optimisation")) # load most recent version

	cuda = torch.cuda.is_available() and USE_CUDA_FOR_LOSS_INFERENCES
	dev = "cpu"
	if cuda :
		dev = "cuda"
		print("Using CUDA device for inferences!")
	else :
		print("Using CPU for inferences!")

	model.to(device=dev)

	segmentation_dataset, segmentation_objects, imgs, coregistered_images, coregistered_transformations, _ =  load_and_preprocess_data(dataset='val')

	preds = perform_inferences(imgs, model, dev)
	
	upscaled_preds, masked_labels = transform_and_upscale_predictions(model, preds, coregistered_images, coregistered_transformations, segmentation_objects, save_predictions_dir="/volume/results/results_" + str(index))
	
	mean_dsc = compute_and_print_metrics(segmentation_dataset, segmentation_objects, upscaled_preds, masked_labels)

	return 1-mean_dsc # negation of mean dsc, meaning better models have lower loss

def find_loss_simple(index:int) :
	base_logs = os.path.join("/volume", "logs_optimisation", "miccai_seg_optimisation")
	most_recent_version = os.path.join(base_logs, sorted(os.listdir(base_logs), key=natural_keys)[-1])
	most_recent_checkpoints = os.path.join(most_recent_version, 'checkpoints')
	all_ckpts = list(filter(lambda x: x.startswith("epoch"), os.listdir(most_recent_checkpoints)))
	dice_scores = [float(x.split("val_dice=")[-1].split(".ckpt")[0]) for x in all_ckpts]
	return 1-max(dice_scores)

if __name__ == "__main__":
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = False

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
	ng_optimizer.enable_pickling()

	loaded = False
	try :
		ng_optimizer.load(OPTIMISER_PICKLE)
		loaded = True
		print("Successfully loaded optimiser!")
	except Exception :
		print("Could not load optimiser")

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
			if current_args in PAST_CONFIG :
				loss  = PAST_LOSS[PAST_CONFIG.index(current_args)]
			else :
				SegmentationMapperCLI(LitSegmentationMapper, default_parameters=current_args, description="Segmentation map generator", run=True)

				gc.collect()

				loss = find_loss_simple(i)

		except Exception :
			print("!!!!!!!!!!!!!!!!!!! Errored during training !!!!!!!!!!!!!!!!!!!!")
			traceback.print_exc()
			loss = 1.0

		gc.collect()

		print("!!!!!!!!!!! %dth loss (1-mean_dsc) := %.8f !!!!!!!!!!!" % (i, loss))

		ng_optimizer.tell(current_candidate, loss)

		ng_optimizer.dump(OPTIMISER_PICKLE)
		print("!!!!!!!!!!!!!!!!!!!!!!! Saved optimiser! !!!!!!!!!!!!!!!!!!!!!!!")

	final_candidate = ng_optimizer.provide_recommendation()

	recommended_args = retreive_default_parameters(final_candidate)
	print("!!!!!!!!!!!!!!!!!!!!!!! Recommended args !!!!!!!!!!!!!!!!!!!!!!!")
	print(recommended_args)
	print("!!!!!!!!!!!!!!!!!!!! Optimisation completed !!!!!!!!!!!!!!!!!!!!")
	print("Pareto front:", ng_optimizer.pareto_front())




