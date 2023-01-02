from pytorch_lightning.cli import LightningArgumentParser

from perceiver.model.segmentation.segmentation import LitSegmentationMapper
from perceiver.scripts.cli import CLI


class SegmentationMapperCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        # parser.link_arguments("data.image_shape", "model.encoder.image_shape", apply_on="instantiate")
        parser.link_arguments("data.num_classes", "model.decoder.num_classes", apply_on="instantiate")
        parser.set_defaults(
            {
                "model.num_latents": 384,
                "model.num_latent_channels": 2048,
                "model.encoder.num_frequency_bands": 128,
                "model.encoder.num_cross_attention_layers": 4,
                "model.encoder.num_cross_attention_heads": 4,
				"model.encoder.num_cross_attention_qk_channels": 4,
                "model.encoder.num_self_attention_heads": 8,
                "model.encoder.num_self_attention_layers_per_block": 8,
                "model.encoder.num_self_attention_blocks": 8,
                "model.encoder.dropout": 0.2,
                "model.decoder.num_cross_attention_heads": 8,
				"model.decoder.num_cross_attention_qk_channels": 8,
                "model.decoder.dropout": 0.2,
            }

			# {
            #     "model.num_latents": 1024,
            #     "model.num_latent_channels": 1024,
            #     "model.encoder.num_frequency_bands": 64,
            #     "model.encoder.num_cross_attention_layers": 1,
            #     "model.encoder.num_cross_attention_heads": 1,
            #     "model.encoder.num_self_attention_heads": 8,
            #     "model.encoder.num_self_attention_layers_per_block": 6,
            #     "model.encoder.num_self_attention_blocks": 8,
            #     "model.encoder.dropout": 0.1,
            #     "model.decoder.num_cross_attention_heads": 1,
            #     "model.decoder.dropout": 0.1,
            # }
        )


if __name__ == "__main__":
    SegmentationMapperCLI(LitSegmentationMapper, description="Segmentation map generator", run=True)
