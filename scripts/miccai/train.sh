mkdir -p logs/miccai_seg && python -m perceiver.scripts.segmentation.mapper fit \
  --optimizer=AdamW \
  --optimizer.lr=5e-4 \
  --optimizer.eps=6.2e-5 \
  --trainer.max_epochs=350 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=miccai_seg \
  --data=MICCAIDataModule \
  --data.batch_size=1 \
  --data.dataset_dir=/mnt/d/amos22 \
  --trainer.log_every_n_steps=2 \
  --data.num_workers=8 \
  --trainer.check_val_every_n_epoch=1 \
  --trainer.accelerator=gpu \
  --trainer.resume_from_checkpoint=logs/miccai_seg/most_recent_baseline_multi_modality/checkpoints/last.ckpt \
  --trainer.devices=1
#   --trainer.fast_dev_run=True \
