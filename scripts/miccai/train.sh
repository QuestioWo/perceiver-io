python -m perceiver.scripts.segmentation.mapper fit \
  --optimizer=AdamW \
  --optimizer.lr=1e-4 \
  --trainer.max_epochs=5000 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=miccai_seg \
  --data=MICCAIDataModule \
  --data.batch_size=1 \
  --trainer.log_every_n_steps=2 \
  --data.num_workers=8 \
  --trainer.check_val_every_n_epoch=1 \
  --trainer.accelerator=gpu \
  --trainer.devices=1
#   --trainer.fast_dev_run=True \
#   --trainer.resume_from_checkpoint=logs/miccai_seg/version_5/checkpoints/last.ckpt