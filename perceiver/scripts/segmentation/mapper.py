from pytorch_lightning.cli import LightningArgumentParser

from perceiver.model.segmentation.segmentation import LitSegmentationMapper
from perceiver.scripts.cli import CLI
from perceiver.data.segmentation.miccai import IMAGE_SIZE, NUM_CLASSES

IMAGE_SIZE_XY = IMAGE_SIZE[1]

class SegmentationMapperCLI(CLI):
	def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
		super().add_arguments_to_parser(parser)
		# parser.link_arguments("data.image_shape", "model.encoder.image_shape", apply_on="instantiate")
		parser.link_arguments("data.num_classes", "model.decoder.num_classes", apply_on="instantiate")
		parser.set_defaults(
			# 64x64x1 config:
			# {
			#     "model.num_latents": ((IMAGE_SIZE_XY//4) * (IMAGE_SIZE_XY//4) * 1), # 256
			#     "model.num_latent_channels": ((IMAGE_SIZE_XY * 16) // 4), # 256
				
			# 	"model.encoder.num_frequency_bands": (NUM_CLASSES * 2),
			#     "model.encoder.num_cross_attention_layers": IMAGE_SIZE_XY//4,
			#     "model.encoder.num_cross_attention_heads": IMAGE_SIZE_XY//4,
			# 	"model.encoder.num_cross_attention_qk_channels": IMAGE_SIZE_XY//2,
			# 	"model.encoder.num_cross_attention_v_channels":IMAGE_SIZE_XY//2,
			#     "model.encoder.num_self_attention_blocks": IMAGE_SIZE_XY//4,
			#     "model.encoder.num_self_attention_layers_per_block": IMAGE_SIZE_XY//4,
			#     "model.encoder.num_self_attention_heads": IMAGE_SIZE_XY//4,
			# 	"model.encoder.num_self_attention_qk_channels": IMAGE_SIZE_XY//2,
			# 	"model.encoder.num_self_attention_v_channels":IMAGE_SIZE_XY//2,
			#     "model.encoder.dropout": 0.3,

			#     "model.decoder.num_cross_attention_heads": IMAGE_SIZE_XY//4,
			# 	"model.decoder.num_cross_attention_qk_channels": IMAGE_SIZE_XY//2,
			# 	"model.decoder.num_cross_attention_v_channels": IMAGE_SIZE_XY//2,
			# 	"model.decoder.cross_attention_residual": True,
			#     "model.decoder.dropout": 0.3,
			# }

			# 128x128x1 config, version_0:
			# {
			# 	"model.num_latents": 256, # ((IMAGE_SIZE_XY//8) * (IMAGE_SIZE_XY//8) * 1)
			# 	"model.num_latent_channels": 64, # ((IMAGE_SIZE_XY) // 2)
				
			# 	"model.encoder.num_frequency_bands": 16, # (NUM_CLASSES)
			# 	"model.encoder.num_cross_attention_layers": 8, # IMAGE_SIZE_XY//16
			# 	"model.encoder.num_cross_attention_heads": 8, # IMAGE_SIZE_XY//16
			# 	"model.encoder.num_cross_attention_qk_channels": 16, # IMAGE_SIZE_XY//8
			# 	"model.encoder.num_cross_attention_v_channels":16, # IMAGE_SIZE_XY//8
			# 	"model.encoder.num_self_attention_blocks": 8, # IMAGE_SIZE_XY//16
			# 	"model.encoder.num_self_attention_layers_per_block": 8, # IMAGE_SIZE_XY//16
			# 	"model.encoder.num_self_attention_heads": 8, # IMAGE_SIZE_XY//16
			# 	"model.encoder.num_self_attention_qk_channels": 16, # IMAGE_SIZE_XY//8
			# 	"model.encoder.num_self_attention_v_channels":16, # IMAGE_SIZE_XY//8
			# 	"model.encoder.dropout": 0.7,

			# 	"model.decoder.num_cross_attention_heads": 8, # IMAGE_SIZE_XY//16
			# 	"model.decoder.num_cross_attention_qk_channels": 16, # IMAGE_SIZE_XY//8
			# 	"model.decoder.num_cross_attention_v_channels": 16, # IMAGE_SIZE_XY//8
			# 	"model.decoder.cross_attention_residual": True,
			# 	"model.decoder.dropout": 0.7,
			# }

			# 128x128x1 config - version_1 - more self and cross attention
			# {
			# 	"model.num_latents": 256, # ((IMAGE_SIZE_XY//8) * (IMAGE_SIZE_XY//8) * 1)
			# 	"model.num_latent_channels": 64, # ((IMAGE_SIZE_XY) // 2)
				
			# 	"model.encoder.num_frequency_bands": 16, # (NUM_CLASSES)
			# 	"model.encoder.num_cross_attention_layers": 12,
			# 	"model.encoder.num_cross_attention_heads": 12,
			# 	"model.encoder.num_cross_attention_qk_channels": 24,
			# 	"model.encoder.num_cross_attention_v_channels": 24,
			# 	"model.encoder.num_self_attention_blocks": 12,
			# 	"model.encoder.num_self_attention_layers_per_block": 12,
			# 	"model.encoder.num_self_attention_heads": 12,
			# 	"model.encoder.num_self_attention_qk_channels": 24,
			# 	"model.encoder.num_self_attention_v_channels": 24,
			# 	"model.encoder.dropout": 0.7,

			# 	"model.decoder.num_cross_attention_heads": 12,
			# 	"model.decoder.num_cross_attention_qk_channels": 24,
			# 	"model.decoder.num_cross_attention_v_channels": 24,
			# 	"model.decoder.cross_attention_residual": True,
			# 	"model.decoder.dropout": 0.7,
			# }

			# 128x128x1 config - version_2 - more dropout 
			# {
			# 	"model.num_latents": 256, # ((IMAGE_SIZE_XY//8) * (IMAGE_SIZE_XY//8) * 1)
			# 	"model.num_latent_channels": 64, # ((IMAGE_SIZE_XY) // 2)
				
			# 	"model.encoder.num_frequency_bands": 16, # (NUM_CLASSES)
			# 	"model.encoder.num_cross_attention_layers": 12,
			# 	"model.encoder.num_cross_attention_heads": 12,
			# 	"model.encoder.num_cross_attention_qk_channels": 24,
			# 	"model.encoder.num_cross_attention_v_channels": 24,
			# 	"model.encoder.num_self_attention_blocks": 12,
			# 	"model.encoder.num_self_attention_layers_per_block": 12,
			# 	"model.encoder.num_self_attention_heads": 12,
			# 	"model.encoder.num_self_attention_qk_channels": 24,
			# 	"model.encoder.num_self_attention_v_channels": 24,
			# 	"model.encoder.dropout": 0.8,

			# 	"model.decoder.num_cross_attention_heads": 12,
			# 	"model.decoder.num_cross_attention_qk_channels": 24,
			# 	"model.decoder.num_cross_attention_v_channels": 24,
			# 	"model.decoder.cross_attention_residual": True,
			# 	"model.decoder.dropout": 0.8,
			# }

			# 128x128x1 config - version_3 - less dropout 
			# {
			# 	"model.num_latents": 256, # ((IMAGE_SIZE_XY//8) * (IMAGE_SIZE_XY//8) * 1)
			# 	"model.num_latent_channels": 64, # ((IMAGE_SIZE_XY) // 2)
				
			# 	"model.encoder.num_frequency_bands": 16, # (NUM_CLASSES)
			# 	"model.encoder.num_cross_attention_layers": 12,
			# 	"model.encoder.num_cross_attention_heads": 12,
			# 	"model.encoder.num_cross_attention_qk_channels": 24,
			# 	"model.encoder.num_cross_attention_v_channels": 24,
			# 	"model.encoder.num_self_attention_blocks": 12,
			# 	"model.encoder.num_self_attention_layers_per_block": 12,
			# 	"model.encoder.num_self_attention_heads": 12,
			# 	"model.encoder.num_self_attention_qk_channels": 24,
			# 	"model.encoder.num_self_attention_v_channels": 24,
			# 	"model.encoder.dropout": 0.5,

			# 	"model.decoder.num_cross_attention_heads": 12,
			# 	"model.decoder.num_cross_attention_qk_channels": 24,
			# 	"model.decoder.num_cross_attention_v_channels": 24,
			# 	"model.decoder.cross_attention_residual": True,
			# 	"model.decoder.dropout": 0.5,
			# }

			# 128x128x1 config - version_4 - less latents 
			# {
			# 	"model.num_latents": 128, # ((IMAGE_SIZE_XY//8) * (IMAGE_SIZE_XY//8) * 1)
			# 	"model.num_latent_channels": 64, # ((IMAGE_SIZE_XY) // 2)
				
			# 	"model.encoder.num_frequency_bands": 16, # (NUM_CLASSES)
			# 	"model.encoder.num_cross_attention_layers": 12,
			# 	"model.encoder.num_cross_attention_heads": 12,
			# 	"model.encoder.num_cross_attention_qk_channels": 24,
			# 	"model.encoder.num_cross_attention_v_channels": 24,
			# 	"model.encoder.num_self_attention_blocks": 12,
			# 	"model.encoder.num_self_attention_layers_per_block": 12,
			# 	"model.encoder.num_self_attention_heads": 12,
			# 	"model.encoder.num_self_attention_qk_channels": 24,
			# 	"model.encoder.num_self_attention_v_channels": 24,
			# 	"model.encoder.dropout": 0.5,

			# 	"model.decoder.num_cross_attention_heads": 12,
			# 	"model.decoder.num_cross_attention_qk_channels": 24,
			# 	"model.decoder.num_cross_attention_v_channels": 24,
			# 	"model.decoder.cross_attention_residual": True,
			# 	"model.decoder.dropout": 0.5,
			# }

			# 128x128x1 config - version_5 - less latents, more channels 
			# {
			# 	"model.num_latents": 128, # ((IMAGE_SIZE_XY//8) * (IMAGE_SIZE_XY//8) * 1)
			# 	"model.num_latent_channels": 128, # ((IMAGE_SIZE_XY) // 2)
				
			# 	"model.encoder.num_frequency_bands": 16, # (NUM_CLASSES)
			# 	"model.encoder.num_cross_attention_layers": 12,
			# 	"model.encoder.num_cross_attention_heads": 12,
			# 	"model.encoder.num_cross_attention_qk_channels": 24,
			# 	"model.encoder.num_cross_attention_v_channels": 24,
			# 	"model.encoder.num_self_attention_blocks": 12,
			# 	"model.encoder.num_self_attention_layers_per_block": 12,
			# 	"model.encoder.num_self_attention_heads": 12,
			# 	"model.encoder.num_self_attention_qk_channels": 24,
			# 	"model.encoder.num_self_attention_v_channels": 24,
			# 	"model.encoder.dropout": 0.5,

			# 	"model.decoder.num_cross_attention_heads": 12,
			# 	"model.decoder.num_cross_attention_qk_channels": 24,
			# 	"model.decoder.num_cross_attention_v_channels": 24,
			# 	"model.decoder.cross_attention_residual": True,
			# 	"model.decoder.dropout": 0.5,
			# }

			# 128x128x1 config - version_6 - less channels 
			# {
			# 	"model.num_latents": 256, # ((IMAGE_SIZE_XY//8) * (IMAGE_SIZE_XY//8) * 1)
			# 	"model.num_latent_channels": 32, # ((IMAGE_SIZE_XY) // 2)
				
			# 	"model.encoder.num_frequency_bands": 16, # (NUM_CLASSES)
			# 	"model.encoder.num_cross_attention_layers": 12,
			# 	"model.encoder.num_cross_attention_heads": 12,
			# 	"model.encoder.num_cross_attention_qk_channels": 24,
			# 	"model.encoder.num_cross_attention_v_channels": 24,
			# 	"model.encoder.num_self_attention_blocks": 12,
			# 	"model.encoder.num_self_attention_layers_per_block": 12,
			# 	"model.encoder.num_self_attention_heads": 12,
			# 	"model.encoder.num_self_attention_qk_channels": 24,
			# 	"model.encoder.num_self_attention_v_channels": 24,
			# 	"model.encoder.dropout": 0.5,

			# 	"model.decoder.num_cross_attention_heads": 12,
			# 	"model.decoder.num_cross_attention_qk_channels": 24,
			# 	"model.decoder.num_cross_attention_v_channels": 24,
			# 	"model.decoder.cross_attention_residual": True,
			# 	"model.decoder.dropout": 0.5,
			# }

			# 128x128x1 config - version_7 - less channels more latents
			# {
			# 	"model.num_latents": 300, # ((IMAGE_SIZE_XY//8) * (IMAGE_SIZE_XY//8) * 1)
			# 	"model.num_latent_channels": 32, # ((IMAGE_SIZE_XY) // 2)
				
			# 	"model.encoder.num_frequency_bands": 16, # (NUM_CLASSES)
			# 	"model.encoder.num_cross_attention_layers": 12,
			# 	"model.encoder.num_cross_attention_heads": 12,
			# 	"model.encoder.num_cross_attention_qk_channels": 24,
			# 	"model.encoder.num_cross_attention_v_channels": 24,
			# 	"model.encoder.num_self_attention_blocks": 12,
			# 	"model.encoder.num_self_attention_layers_per_block": 12,
			# 	"model.encoder.num_self_attention_heads": 12,
			# 	"model.encoder.num_self_attention_qk_channels": 24,
			# 	"model.encoder.num_self_attention_v_channels": 24,
			# 	"model.encoder.dropout": 0.5,

			# 	"model.decoder.num_cross_attention_heads": 12,
			# 	"model.decoder.num_cross_attention_qk_channels": 24,
			# 	"model.decoder.num_cross_attention_v_channels": 24,
			# 	"model.decoder.cross_attention_residual": True,
			# 	"model.decoder.dropout": 0.5,
			# }

			# 128x128x1 config - version_8 - more channels?
			{
				"model.num_latents": 256, # ((IMAGE_SIZE_XY//8) * (IMAGE_SIZE_XY//8) * 1)
				"model.num_latent_channels": 128, # ((IMAGE_SIZE_XY) // 2)
				
				"model.encoder.num_frequency_bands": 16, # (NUM_CLASSES)
				"model.encoder.num_cross_attention_layers": 12,
				"model.encoder.num_cross_attention_heads": 12,
				"model.encoder.num_cross_attention_qk_channels": 24,
				"model.encoder.num_cross_attention_v_channels": 24,
				"model.encoder.num_self_attention_blocks": 12,
				"model.encoder.num_self_attention_layers_per_block": 12,
				"model.encoder.num_self_attention_heads": 12,
				"model.encoder.num_self_attention_qk_channels": 24,
				"model.encoder.num_self_attention_v_channels": 24,
				"model.encoder.dropout": 0.5,

				"model.decoder.num_cross_attention_heads": 12,
				"model.decoder.num_cross_attention_qk_channels": 24,
				"model.decoder.num_cross_attention_v_channels": 24,
				"model.decoder.cross_attention_residual": True,
				"model.decoder.dropout": 0.5,
			}

			# 128x128x1 config
			# {
			# 	"model.num_latents": 256, # ((IMAGE_SIZE_XY//8) * (IMAGE_SIZE_XY//8) * 1)
			# 	"model.num_latent_channels": 64, # ((IMAGE_SIZE_XY) // 2)
				
			# 	"model.encoder.num_frequency_bands": 16, # (NUM_CLASSES)
			# 	"model.encoder.num_cross_attention_layers": 8, # IMAGE_SIZE_XY//16
			# 	"model.encoder.num_cross_attention_heads": 8, # IMAGE_SIZE_XY//16
			# 	"model.encoder.num_cross_attention_qk_channels": 16, # IMAGE_SIZE_XY//8
			# 	"model.encoder.num_cross_attention_v_channels":16, # IMAGE_SIZE_XY//8
			# 	"model.encoder.num_self_attention_blocks": 8, # IMAGE_SIZE_XY//16
			# 	"model.encoder.num_self_attention_layers_per_block": 8, # IMAGE_SIZE_XY//16
			# 	"model.encoder.num_self_attention_heads": 8, # IMAGE_SIZE_XY//16
			# 	"model.encoder.num_self_attention_qk_channels": 16, # IMAGE_SIZE_XY//8
			# 	"model.encoder.num_self_attention_v_channels":16, # IMAGE_SIZE_XY//8
			# 	"model.encoder.dropout": 0.7,

			# 	"model.decoder.num_cross_attention_heads": 8, # IMAGE_SIZE_XY//16
			# 	"model.decoder.num_cross_attention_qk_channels": 16, # IMAGE_SIZE_XY//8
			# 	"model.decoder.num_cross_attention_v_channels": 16, # IMAGE_SIZE_XY//8
			# 	"model.decoder.cross_attention_residual": True,
			# 	"model.decoder.dropout": 0.7,
			# }
		)


if __name__ == "__main__":
	SegmentationMapperCLI(LitSegmentationMapper, description="Segmentation map generator", run=True)
