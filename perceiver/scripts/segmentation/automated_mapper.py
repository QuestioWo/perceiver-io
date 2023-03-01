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

OPTIMISER_PICKLE = "ng_optimiser.pkl"

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

PAST_CONFIGS = [
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 1026, 'model.num_latent_channels': 512, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 64, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 2, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 561, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 2, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 3, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 71, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 3, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 6, 'model.encoder.num_self_attention_v_channels': 6, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 3, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 915, 'model.num_latent_channels': 510, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 3, 'model.encoder.num_self_attention_layers_per_block': 3, 'model.encoder.num_self_attention_heads': 4, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 8, 'model.encoder.num_self_attention_v_channels': 8, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 60, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 3, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 266, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 61, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 3, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 3, 'model.encoder.num_self_attention_layers_per_block': 1, 'model.encoder.num_self_attention_heads': 3, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 6, 'model.encoder.num_self_attention_v_channels': 6, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 3, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 687, 'model.overlap_slices': 2, 'model.recursive_slices': 1, 'model.encoder.num_frequency_bands': 4, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 3, 'model.encoder.num_self_attention_layers_per_block': 3, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 1, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 1, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 448, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 1, 'model.encoder.num_self_attention_layers_per_block': 3, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 1277, 'model.num_latent_channels': 551, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 52, 'model.encoder.num_cross_attention_layers': 3, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 3, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 4, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 8, 'model.encoder.num_self_attention_v_channels': 8, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 771, 'model.num_latent_channels': 625, 'model.overlap_slices': 2, 'model.recursive_slices': 3, 'model.encoder.num_frequency_bands': 70, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 1, 'model.encoder.num_self_attention_layers_per_block': 2, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 1301, 'model.num_latent_channels': 500, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 37, 'model.encoder.num_cross_attention_layers': 3, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 541, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 1, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 1, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 3, 'model.encoder.num_self_attention_heads': 3, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 6, 'model.encoder.num_self_attention_v_channels': 6, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 1, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 1482, 'model.num_latent_channels': 478, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 1, 'model.encoder.num_self_attention_layers_per_block': 3, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 633, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 1, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 383, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 2, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 3, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 2, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 4, 'model.decoder.num_cross_attention_v_channels': 4, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 128, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 105, 'model.encoder.num_cross_attention_layers': 1, 'model.encoder.num_cross_attention_heads': 3, 'model.encoder.num_self_attention_blocks': 1, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 6, 'model.encoder.num_cross_attention_v_channels': 6, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True},
	{'model.num_latents': 512, 'model.num_latent_channels': 317, 'model.overlap_slices': 2, 'model.recursive_slices': 2, 'model.encoder.num_frequency_bands': 75, 'model.encoder.num_cross_attention_layers': 3, 'model.encoder.num_cross_attention_heads': 2, 'model.encoder.num_self_attention_blocks': 2, 'model.encoder.num_self_attention_layers_per_block': 4, 'model.encoder.num_self_attention_heads': 2, 'model.decoder.num_cross_attention_heads': 3, 'model.encoder.dropout': 0.3, 'model.decoder.dropout': 0.3, 'model.slabs_start': 80, 'model.slabs_depth': 20, 'model.slabs_size': 4, 'model.encoder.num_cross_attention_qk_channels': 4, 'model.encoder.num_cross_attention_v_channels': 4, 'model.encoder.num_self_attention_qk_channels': 4, 'model.encoder.num_self_attention_v_channels': 4, 'model.decoder.num_cross_attention_qk_channels': 6, 'model.decoder.num_cross_attention_v_channels': 6, 'model.decoder.cross_attention_residual': True}
]

PAST_LOSSES = [ 0.9774, 1.0000, 0.9751, 0.9751, 0.9773, 0.9760, 1.0000, 0.9754, 0.9757, 0.9799, 0.9755, 1.0000, 0.9790, 0.9801, 0.9751, 0.9798, 0.9837, 0.9744, 0.9750, 0.9744, 0.9768, 1.0000, 1.0000, 1.0000, 0.9750, 0.9825, 0.9744, 0.9814, 0.9744, 1.0000, 0.9807, 0.9789, 0.9745, 1.0000 ]

assert len(PAST_CONFIGS) == len(PAST_LOSSES), "configs and losses are not equal"

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
	ng_optimizer.enable_pickling()

	try :
		ng_optimizer.load(OPTIMISER_PICKLE)
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
			if i >= len(PAST_CONFIGS) or not current_args in PAST_CONFIGS :
				SegmentationMapperCLI(LitSegmentationMapper, default_parameters=current_args, description="Segmentation map generator", run=True)

				gc.collect()
	
				loss = find_loss(i)
			else :
				print("Loss already found") # TODO: remove
				loss = PAST_LOSSES[PAST_CONFIGS.index(current_args)]

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




