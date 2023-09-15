import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
from functools import partial

import math
import os
from argparse import ArgumentParser

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, IterableDataset
import numpy as np

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load

model_name = "pythia-2.8b"
name = "pythia-2.8b"
out_dir = Path("out") / name
save_interval = 3
eval_interval = 500
eval_iters = 100
log_interval = 5

# Hyperparameters
learning_rate = 6e-4
batch_size = 20
micro_batch_size = 1
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
max_iters = 101  # num_epochs * (epoch_size // micro_batch_size) // devices  60000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5


# Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
data_config = [
    ("arxiv", 2.5),
    ("book", 4.5),
    ("c4", 15.0),
    ("cc", 67.0),
    ("github", 4.5),
    ("stackexchange", 2.0),
    ("wikipedia", 4.5),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", name, flush_logs_every_n_steps=log_interval)


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345
) -> DataLoader:
    datasets = []
    # for prefix, _ in data_config:
    #     filenames = glob.glob(str(data_dir / f"{prefix}*"))
    #     dataset = PackedDataset(
    #         filenames,
    #         n_chunks=4,
    #         block_size=block_size,
    #         shuffle=shuffle,
    #         seed=seed,
    #         num_processes=fabric.world_size,
    #         process_rank=fabric.global_rank,
    #     )
    #     datasets.append(dataset)

    # if not datasets:
    #     raise RuntimeError(
    #         f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
    #     )

    # weights = [weight for _, weight in data_config]
    # sum_weights = sum(weights)
    # weights = [el / sum_weights for el in weights]

    # combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    # return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    import os
    filenames = [data_dir / i for i in os.listdir(str(data_dir))]
    fabric.print(f"FOUND DATAFILES {data_dir=}: {filenames[:5]=}")

    dataset = PackedDataset(
            filenames,
            n_chunks=4,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    # torch.set_float32_matmul_precision("high")

    config = Config.from_name(model_name)
    model = GPT(config)

    parser = ArgumentParser()
    parser.add_argument('--train_data_dir', default=None, type=str, required=True)
    parser.add_argument('--val_data_dir', default=None, type=str)
    parser.add_argument('--devices', default=1, type=str)
    parser.add_argument('--strategy', default='auto', type=str)
    parser.add_argument('--num_nodes', default=0, type=int)
    args = parser.parse_args()

    print("input", args.__dict__)
    dicty = args.__dict__.copy()

    train_data_dir = Path(dicty["train_data_dir"]) if "train_data_dir" in dicty else None
    val_data_dir = Path(dicty["val_data_dir"]) if "val_data_dir" in dicty else None
    del dicty['train_data_dir']
    del dicty['val_data_dir']

    trainer = L.Trainer(
        **dicty,
        max_epochs=10,
        gradient_clip_val=1.0,
    )

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=trainer,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        seed=(1337 + trainer.global_rank),
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)