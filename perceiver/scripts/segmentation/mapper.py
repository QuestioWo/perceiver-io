from pytorch_lightning.cli import LightningArgumentParser

from perceiver.model.segmentation.segmentation import LitSegmentationMapper
from perceiver.scripts.cli import CLI
from perceiver.data.segmentation.miccai import IMAGE_SIZE

class SegmentationMapperCLI(CLI):
	def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
		super().add_arguments_to_parser(parser)
		parser.link_arguments("data.num_classes", "model.decoder.num_classes", apply_on="instantiate")
		parser.set_defaults(
			# NOTE: attention is the largest drain on compute time, likely a
			# linear mapping of attention to increase in inference time
			{
				"model.num_latents": 512,
				"model.num_latent_channels": 128,

				# Custom parameters for segmentation
				"model.slabs_size" : 4,
				"model.slabs_start" : 80,
				"model.slabs_depth" : 20,
				"model.overlap_slices" : 2,
				"model.recursive_slices" : 2,
				
				"model.encoder.num_frequency_bands": 4,
				"model.encoder.num_cross_attention_layers": 2,
				"model.encoder.num_cross_attention_heads": 2,
				"model.encoder.num_cross_attention_qk_channels": 4,
				"model.encoder.num_cross_attention_v_channels": 4,
				"model.encoder.num_self_attention_blocks": 2,
				"model.encoder.num_self_attention_layers_per_block": 4,
				"model.encoder.num_self_attention_heads": 2,
				"model.encoder.num_self_attention_qk_channels": 4,
				"model.encoder.num_self_attention_v_channels": 4,
				"model.encoder.dropout": 0.3,

				"model.decoder.num_cross_attention_heads": 2,
				"model.decoder.num_cross_attention_qk_channels": 4,
				"model.decoder.num_cross_attention_v_channels": 4,
				"model.decoder.cross_attention_residual": True,
				"model.decoder.dropout": 0.3,
			}
		)


if __name__ == "__main__":
	SegmentationMapperCLI(LitSegmentationMapper, description="Segmentation map generator", run=True)
