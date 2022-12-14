import os

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch.nn as nn
import torchmetrics as tm
import torch

from einops import rearrange
from transformers import PerceiverConfig as _, PerceiverForImageClassificationFourier

from perceiver.model.core import (
	EncoderConfig,
	InputAdapter,
	PerceiverConfig,
	PerceiverDecoder,
	PerceiverEncoder,
	PerceiverIO,
)
from perceiver.model.core.lightning import LitModel
from perceiver.model.core.config import DecoderConfig, EncoderConfig, PerceiverConfig
from perceiver.model.image.common import FourierPositionEncoding
from perceiver.data.segmentation.miccai import IMAGE_SIZE, NUM_CLASSES
from perceiver.model.core.modules import OutputAdapter

@dataclass
class SegmentationDecoderConfig(DecoderConfig):
	num_output_queries: int = (IMAGE_SIZE[1] * IMAGE_SIZE[2])
	num_output_query_channels: int = None
	num_classes: int = 16


@dataclass
class SegmentationEncoderConfig(EncoderConfig):
	image_shape: Tuple[int, int, int] = (IMAGE_SIZE[1], IMAGE_SIZE[2], 1) # TODO: make 3D
	num_frequency_bands: int = 64


class LitMapper(LitModel):
	def __init__(self, *args: Any, **kwargs: Any):
		super().__init__(*args, **kwargs)
		self.loss = nn.CrossEntropyLoss() # TODO: use a dice loss instead
		self.loss_tm = tm.Dice()
		self.acc = tm.classification.accuracy.Accuracy(task="multiclass", num_classes=NUM_CLASSES, mdmc_reduce="global")

	def step(self, batch):
		logits, y = self(batch)

		# TODO: remove when making 3d
		y: torch.Tensor = y[:,:,:,67]
		y = y[:,:,:,None]

		logits = torch.reshape(logits, [*y.shape, NUM_CLASSES])
		logits = torch.einsum("b w h d c -> b c w h d", logits)
		loss = self.loss(logits, y.long())
		y_pred = logits.argmax(dim=1).int()
		loss_dice = self.loss_tm(y_pred, y)
		acc = self.acc(y_pred, y)
		return loss, acc, loss_dice

	def training_step(self, batch, batch_idx):
		loss, acc, loss_dice = self.step(batch)
		self.log("train_loss", loss)
		self.log("train_acc", acc, prog_bar=True)
		self.log("train_loss_dice", loss_dice, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, acc, loss_dice = self.step(batch)
		self.log("val_loss", loss, prog_bar=True, sync_dist=True)
		self.log("val_acc", acc, prog_bar=True, sync_dist=True)
		self.log("val_loss_dice", loss_dice, prog_bar=True, sync_dist=True)

	def test_step(self, batch, batch_idx):
		loss, acc, loss_dice = self.step(batch)
		self.log("test_loss", loss, sync_dist=True)
		self.log("test_acc", acc, sync_dist=True)
		self.log("test_loss_dice", loss_dice, sync_dist=True)


class SegmentationInputAdapter(InputAdapter):
	def __init__(self, image_shape: Tuple[int, ...], num_frequency_bands: int):
		*spatial_shape, scan_depth = image_shape
		position_encoding = FourierPositionEncoding(spatial_shape, num_frequency_bands)

		super().__init__(num_input_channels=scan_depth + position_encoding.num_position_encoding_channels())
		self.image_shape = image_shape
		self.position_encoding = position_encoding

	def forward(self, x):
		# TODO: remove when making 3d
		x = x[:,:,:,67]
		x = x[:,:,:,None]
		
		b, *d = x.shape

		if tuple(d) != self.image_shape:
			raise ValueError(f"Input image shape {tuple(d)} different from required shape {self.image_shape}")

		x_enc = self.position_encoding(b)
		x = rearrange(x, "b ... c -> b (...) c")
		x = torch.cat([x, x_enc], dim=-1)
		return x


class SegmentationOutputAdapter(OutputAdapter):
	def __init__(
		self,
		num_classes: int,
		num_output_queries: int = 1,
		num_output_query_channels: Optional[int] = None,
		init_scale: float = 0.02,
	):

		if num_output_query_channels is None:
			num_output_query_channels = num_classes

		super().__init__(output_query=torch.empty(num_output_queries, num_output_query_channels), init_scale=init_scale)
		self.linear = nn.Linear(num_output_query_channels, num_classes)

	def forward(self, x):
		return self.linear(x).squeeze(dim=1)


class ScanSegmenter(PerceiverIO):
	def __init__(self, config: PerceiverConfig[SegmentationEncoderConfig, SegmentationDecoderConfig]):
		input_adapter = SegmentationInputAdapter(
			image_shape=config.encoder.image_shape, num_frequency_bands=config.encoder.num_frequency_bands
		)

		encoder_kwargs = config.encoder.base_kwargs()
		if encoder_kwargs["num_cross_attention_qk_channels"] is None:
			encoder_kwargs["num_cross_attention_qk_channels"] = input_adapter.num_input_channels

		encoder = PerceiverEncoder(
			input_adapter=input_adapter,
			num_latents=config.num_latents,
			num_latent_channels=config.num_latent_channels,
			activation_checkpointing=config.activation_checkpointing,
			activation_offloading=config.activation_offloading,
			**encoder_kwargs,
		)
		output_adapter = SegmentationOutputAdapter(
			num_classes=config.decoder.num_classes,
			num_output_queries=config.decoder.num_output_queries,
			num_output_query_channels=config.decoder.num_output_query_channels,
			init_scale=config.decoder.init_scale,
		)
		decoder = PerceiverDecoder(
			output_adapter=output_adapter,
			num_latent_channels=config.num_latent_channels,
			activation_checkpointing=config.activation_checkpointing,
			activation_offloading=config.activation_offloading,
			**config.decoder.base_kwargs(),
		)
		super().__init__(encoder, decoder)

		if config.params is None:
			pass
		elif os.path.isfile(config.params):
			self.load_state_dict(torch.load(config.params))
		else:
			raise ValueError("Cannot find pretrained model")
		#     # import model params from Hugging Face Perceiver
		#     model = PerceiverForImageClassificationFourier.from_pretrained(config.params)
		#     copy_encoder_params(model, self.encoder)
		#     copy_decoder_params(model, self.decoder)


class LitSegmentationMapper(LitMapper):
	def __init__(self, encoder: SegmentationEncoderConfig, decoder: SegmentationDecoderConfig, *args: Any, **kwargs: Any):
		super().__init__(encoder, decoder, *args, **kwargs)
		self.model = ScanSegmenter(
			PerceiverConfig(
				encoder=encoder,
				decoder=decoder,
				num_latents=self.hparams.num_latents,
				num_latent_channels=self.hparams.num_latent_channels,
				activation_checkpointing=self.hparams.activation_checkpointing,
				activation_offloading=self.hparams.activation_offloading,
				params=self.hparams.params,
			)
		)

	def forward(self, batch):
		y, x = batch["label"], batch["image"]
		return self.model(x), y