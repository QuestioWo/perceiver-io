from pytorch_lightning.cli import LightningArgumentParser

from perceiver.model.segmentation.segmentation import LitSegmentationMapper
from perceiver.scripts.cli import CLI

class DefaultWrapperCLI(CLI):
	def __init__(self, model_class, default_parameters, run=True, **kwargs):
		self._default_parameters = default_parameters

		super().__init__(model_class, run, **kwargs)

	def get_default_parameters(self) :
		return self._default_parameters


class SegmentationMapperCLI(DefaultWrapperCLI):
	def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
		super().add_arguments_to_parser(parser)
		parser.link_arguments("data.num_classes", "model.decoder.num_classes", apply_on="instantiate")
		parser.set_defaults(
			self.get_default_parameters()
		)


if __name__ == "__main__":



	current_default_parameters = {
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
	
	SegmentationMapperCLI(LitSegmentationMapper, default_parameters=current_default_parameters, description="Segmentation map generator", run=True)
