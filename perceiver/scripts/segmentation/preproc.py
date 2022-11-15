import multiprocessing as mp
from typing import Optional

import jsonargparse

from perceiver.data.segmentation import (
	MICCAIDataModule
)


DATAMODULE_CLASSES = {
    "miccai": MICCAIDataModule
}


def main(args):
    module = DATAMODULE_CLASSES[args.dataset](**args)
    module.prepare_data()
    module.setup()


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description="Preprocess dataset for training")
    parser.add_argument("dataset", default="miccai", choices=list(DATAMODULE_CLASSES.keys()))
    parser.add_argument("--dataset_dir", default="AMOS22", type=str)
    parser.add_argument("--normalize", default=True, type=bool)
    parser.add_argument("--channels_last", default=True, type=bool)
    parser.add_argument("--random_crop", type=Optional[int])
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=mp.cpu_count(), type=int)
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--shuffle", default=True, type=bool)
    main(parser.parse_args())
