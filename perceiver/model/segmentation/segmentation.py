import os

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch.nn as nn
import torchmetrics as tm
import torch
import torch.utils.checkpoint

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
from perceiver.data.segmentation.miccai import CT_ONLY, IMAGE_SIZE, NUM_CLASSES
from perceiver.model.core.modules import OutputAdapter

@dataclass
class SegmentationDecoderConfig(DecoderConfig):
	num_output_query_channels: int = None
	num_classes: int = NUM_CLASSES


@dataclass
class SegmentationEncoderConfig(EncoderConfig):
	num_frequency_bands: int = NUM_CLASSES

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
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / (self.n_classes)

class LitMapper(LitModel):
	def __init__(self, *args: Any, **kwargs: Any):
		super().__init__(*args, **kwargs)
		self.ce_loss = nn.modules.loss.CrossEntropyLoss()
		self.dice_loss = DiceLoss(NUM_CLASSES)
		self.dice = tm.Dice(average='macro', num_classes=NUM_CLASSES)

		self.ct_only = CT_ONLY
		with open("/dev/shm/ct_only", "r") as f :
			self.ct_only = int(f.read())

	def step(self, batch):
		logits, y = self(batch)

		ce_loss = self.ce_loss(logits, y.long())
		dice_loss = self.dice_loss(logits, y.long(), softmax=True)
		loss: torch.Tensor = 0.9 * ce_loss + 0.1 * dice_loss

		current_dev = torch.device("cuda", 0)
		y_pred = logits.argmax(dim=1).int().to(current_dev)

		y = y.to(current_dev)
		self.dice = self.dice.to(current_dev)

		dice_acc = self.dice(y_pred, y)

		return loss, dice_acc

	def training_step(self, batch, batch_idx):
		loss, dice = self.step(batch)
		self.log("train_loss", loss)
		# self.log("train_acc", acc, prog_bar=True)
		self.log("train_dice", dice, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, dice = self.step(batch)
		self.log("val_loss", loss, prog_bar=True, sync_dist=True)
		# self.log("val_acc", acc, prog_bar=True, sync_dist=True)
		self.log("val_dice", dice, prog_bar=True, sync_dist=True)

	def test_step(self, batch, batch_idx):
		loss, dice = self.step(batch)
		self.log("test_loss", loss, sync_dist=True)
		# self.log("test_acc", acc, sync_dist=True)
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


class LitSegmentationMapper(LitMapper):
	def __init__(self, encoder: SegmentationEncoderConfig, decoder: SegmentationDecoderConfig, *args: Any, **kwargs: Any):
		super().__init__(encoder, decoder, *args, **kwargs)
		self.recursive_slices = self.hparams.recursive_slices
		self.overlap_slices = self.hparams.overlap_slices

		self.slabs_size = self.hparams.slabs_size

		self.slabs_start = self.hparams.slabs_start
		self.slabs_depth = self.hparams.slabs_depth

		encoder.image_shape = (IMAGE_SIZE[1], IMAGE_SIZE[2], self.slabs_size+self.recursive_slices)
		decoder.num_output_queries = (IMAGE_SIZE[1] * IMAGE_SIZE[2] * self.slabs_size)

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

		if self.overlap_slices > 0 :
			assert int(self.slabs_depth / self.overlap_slices) == (self.slabs_depth / self.overlap_slices), "depth of scan must be divisible by slab overlap"
		assert int(self.slabs_depth / self.slabs_size) == (self.slabs_depth / self.slabs_size), "depth of scan must be divisible by slab size"

		# -1 as first slab is double the height due to having nothing to overlap with
		# the second part of the last slab will have no overlap on the bottom half
		self.slab_iterations = (self.slabs_depth // (self.slabs_size - self.overlap_slices)) - (1 if (self.slabs_size // 2 == self.overlap_slices) else 0)

	def forward(self, batch):
		x: torch.Tensor = batch["image"]

		x.requires_grad = self.model.training

		b, *_ = x.shape
		
		full_results: torch.Tensor = torch.zeros((b, NUM_CLASSES, IMAGE_SIZE[1], IMAGE_SIZE[2], self.slabs_depth), device=x.device)
		prev_recursion_predictions: torch.Tensor = torch.zeros((b, IMAGE_SIZE[1], IMAGE_SIZE[2], self.recursive_slices), device=x.device, dtype=x.dtype)

		for i in range(self.slab_iterations) :
			offset = i * self.slabs_size

			slab_overlap_const = self.overlap_slices*i

			# No need to add slab start as full_results will always be slabs_depth sized 
			start_location = (offset-slab_overlap_const)
			end_location = (self.slabs_size+offset-slab_overlap_const)

			# print("start :=", start_location)
			# print("end :=", end_location)

			current_inputs: torch.Tensor = torch.zeros((b, IMAGE_SIZE[1], IMAGE_SIZE[2], self.slabs_size+self.recursive_slices), device=x.device)
			
			current_inputs[:,:,:,:self.recursive_slices] = prev_recursion_predictions
			current_inputs[:,:,:,self.recursive_slices:(self.slabs_size+self.recursive_slices)] = x[:,:,:,start_location:end_location]

			logits: torch.Tensor = None
			if current_inputs.requires_grad :
				logits = torch.utils.checkpoint.checkpoint(self.model, current_inputs)
			else :
				logits = self.model(current_inputs)

			logits = torch.reshape(logits, [b, *x.shape[1:-1], self.slabs_size, NUM_CLASSES])
			logits = torch.einsum("b w h d c -> b c w h d", logits)

			# always just add logits as otherwise previous logits will be overwritten.
			# addition means that no averaging has to take place and the maxes/predictions
			# will be correct and efficient
			full_results[:,:,:,:,start_location:end_location] += logits
			if self.recursive_slices > 0 :
				prev_recursion_predictions[:,:,:,:] = torch.argmax(logits[:,:,:,:,-self.recursive_slices:], dim=1)

		y = batch["label"] 
		if y != None :
			y = y[:,:,:,self.slabs_start:self.slabs_start+self.slabs_depth]

		return full_results, y
