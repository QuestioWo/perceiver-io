import os

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch.nn as nn
import torchmetrics as tm
import torch

from einops import rearrange

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

# SLICE_INDEX_FROM, SLICE_INDEX_TO = (64, 124) # For 220,256,256 NOTE: cannot run
# SLICE_INDEX_FROM, SLICE_INDEX_TO = (48, 52) # For 165,192,192 NOTE: cannot run
# SLICE_INDEX_FROM, SLICE_INDEX_TO = (37, 38) # For 110,128,128 NOTE: can run
SLICE_INDEX_FROM, SLICE_INDEX_TO = (37, 42) # For 110,128,128 NOTE: can run

SLABS_START = 30
# SLABS_START = 0
SLABS_DEPTH = 10
# SLABS_DEPTH = IMAGE_SIZE[0]
SLABS_SIZE = SLICE_INDEX_TO - SLICE_INDEX_FROM


@dataclass
class SegmentationDecoderConfig(DecoderConfig):
	num_output_queries: int = (IMAGE_SIZE[1] * IMAGE_SIZE[2] * SLABS_SIZE)
	num_output_query_channels: int = None
	num_classes: int = NUM_CLASSES


@dataclass
class SegmentationEncoderConfig(EncoderConfig):
	image_shape: Tuple[int, int, int] = (IMAGE_SIZE[1], IMAGE_SIZE[2], SLABS_SIZE)
	num_frequency_bands: int = 64

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class LitMapper(LitModel):
	def __init__(self, *args: Any, **kwargs: Any):
		super().__init__(*args, **kwargs)
		self.ce_loss = nn.CrossEntropyLoss()
		self.dice_loss = DiceLoss(NUM_CLASSES)
		# self.loss = SegmentationClassificationLoss()
		self.dice = tm.Dice()
		self.acc = tm.classification.accuracy.Accuracy(task="multiclass", num_classes=NUM_CLASSES, mdmc_reduce="global")

	def step(self, batch):
		logits, y = self(batch)

		y = y[:,:,:,SLABS_START:SLABS_START+SLABS_DEPTH]
		
		ce_loss = self.ce_loss(logits, y.long())
		dice_loss = self.dice_loss(logits, y.long(), softmax=True)
		loss = 0.4 * ce_loss + 0.6 * dice_loss

		y_pred = logits.argmax(dim=1).int()
		dice_acc = self.dice(y_pred, y)
		raw_acc = self.acc(y_pred, y)

		return loss, raw_acc, dice_acc

	def training_step(self, batch, batch_idx):
		loss, acc, dice = self.step(batch)
		self.log("train_loss", loss)
		self.log("train_acc", acc, prog_bar=True)
		self.log("train_dice", dice, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, acc, dice = self.step(batch)
		self.log("val_loss", loss, prog_bar=True, sync_dist=True)
		self.log("val_acc", acc, prog_bar=True, sync_dist=True)
		self.log("val_dice", dice, prog_bar=True, sync_dist=True)

	def test_step(self, batch, batch_idx):
		loss, acc, dice = self.step(batch)
		self.log("test_loss", loss, sync_dist=True)
		self.log("test_acc", acc, sync_dist=True)
		self.log("test_dice", dice, sync_dist=True)


class SegmentationInputAdapter(InputAdapter):
	def __init__(self, image_shape: Tuple[int, ...], num_frequency_bands: int):
		*spatial_shape, scan_depth = image_shape
		position_encoding = FourierPositionEncoding(spatial_shape, num_frequency_bands)

		super().__init__(num_input_channels=scan_depth + position_encoding.num_position_encoding_channels())
		self.image_shape = image_shape
		self.position_encoding = position_encoding

	def forward(self, x):
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
		x = batch["image"]

		b, *_ = x.shape
		
		full_results = torch.zeros((b, NUM_CLASSES, IMAGE_SIZE[1], IMAGE_SIZE[2], SLABS_DEPTH), device=x.device)

		for i in range(SLABS_DEPTH // SLABS_SIZE) :
			offset = i * SLABS_SIZE

			current_slab = x[:,:,:,SLABS_START+offset:SLABS_START+(offset + SLABS_SIZE)]

			logits = self.model(current_slab)

			logits = torch.reshape(logits, [b, *x.shape[1:-1], SLABS_SIZE, NUM_CLASSES])
			logits = torch.einsum("b w h d c -> b c w h d", logits)

			full_results[:,:,:,:,offset:(offset + SLABS_SIZE)] = logits

		return full_results, batch["label"]