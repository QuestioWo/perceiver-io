mkdir -p logs_recursion

python -m perceiver.scripts.segmentation.mapper \
	fit \
	--optimizer=AdamW \
	--optimizer.lr=5e-4 \
	--model.recursive_slices=2 \
	--trainer.max_epochs=350 \
	--trainer.precision=32 \
	--trainer.logger=TensorBoardLogger \
	--trainer.logger.save_dir=logs_recursion \
	--trainer.logger.name=miccai_seg_recursion \
	--data=MICCAIDataModule \
	--data.batch_size=1 \
	--data.num_workers=8 \
	--data.dataset_dir=/mnt/d/amos22 \
	--trainer.log_every_n_steps=1 \
	--trainer.check_val_every_n_epoch=1 \
	--trainer.accelerator=gpu \
	--trainer.devices='-1' 2>&1 | tee log_recursion.txt

python -m perceiver.scripts.segmentation.mapper \
	fit \
	--optimizer=AdamW \
	--optimizer.lr=5e-4 \
	--model.recursive_slices=0 \
	--trainer.max_epochs=350 \
	--trainer.precision=32 \
	--trainer.logger=TensorBoardLogger \
	--trainer.logger.save_dir=logs_recursion \
	--trainer.logger.name=miccai_seg_recursion \
	--data=MICCAIDataModule \
	--data.batch_size=1 \
	--data.num_workers=8 \
	--data.dataset_dir=/mnt/d/amos22 \
	--trainer.log_every_n_steps=1 \
	--trainer.check_val_every_n_epoch=1 \
	--trainer.accelerator=gpu \
	--trainer.devices='-1' 2>&1 | tee -a log_recursion.txt