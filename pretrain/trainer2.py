import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import torch
from torch.utils.data import DataLoader

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, IterableDataset
import numpy as np

from lit_gpt.speed_monitor import estimate_flops, measure_flops, SpeedMonitorCallback
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load


# name = "redpajama"
# out_dir = Path("out") / name
save_interval = 1000
eval_interval = 1000
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 6e-4
batch_size = 125
micro_batch_size = 6
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
max_iters = 600000  # num_epochs * (epoch_size // micro_batch_size) // devices
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
logger = step_csv_logger("out", __file__, flush_logs_every_n_steps=log_interval)


def setup(
    devices: int = 4,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    precision: Optional[str] = None,
    resume: Union[bool, Path] = False,
    from_model: str = '',
    model_name = "Llama-2-7b-hf",
    out_dir: str = 'out/None',
    micro_batch_size = 1,
) -> None:

    precision = precision or get_default_supported_precision(training=True)
    out_dir = Path(out_dir)

    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    logger = step_csv_logger("out", model_name, cls=CSVLogger, flush_logs_every_n_steps=log_interval)
    speed_monitor = SpeedMonitorCallback(
        length_fn=lambda batch: batch[0].size(1), batch_size=micro_batch_size, window_size=50, time_unit="seconds"
    )
    model_checkpoint = ModelCheckpoint(dirpath=out_dir, every_n_train_steps=save_interval, save_last=True, verbose=True)
    trainer = L.Trainer(
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=logger,
        callbacks=[speed_monitor, model_checkpoint],
        max_steps=max_iters,
        # max_epochs=1,
        limit_val_batches=eval_iters,
        accumulate_grad_batches=gradient_accumulation_steps,
        log_every_n_steps=log_interval,
        val_check_interval=eval_interval,
    )

    if trainer.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    train_data = Dataset(str(train_data_dir / "train.bin"), config.block_size)
    val_data = Dataset(str(val_data_dir / "val.bin"), config.block_size)
    train_dataloader = DataLoader(train_data, batch_size=micro_batch_size, num_workers=20)
    val_dataloader = DataLoader(val_data, batch_size=micro_batch_size, num_workers=20)

    with trainer.init_module(empty_init=True):
        model = GPT(config)
        model.apply(model._init_weights)

        if not from_model:
            model.apply(model._init_weights)

        else:
            checkpoint_dir  = Path(from_model)
            checkpoint_path = checkpoint_dir / "lit_model.pth"
            with lazy_load(checkpoint_path) as checkpoint:
                model.load_state_dict(checkpoint)

    trainer.fit(model, train_dataloader)

    t0 = time.perf_counter()
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="last")
    trainer.print(f"Training time: {(time.perf_counter()-t0):.2f}s")

    if trainer.strategy.root_device.type == "cuda":
        trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
        trainer.print(torch.cuda.memory_summary())

def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = glob.glob(str(data_dir / f"{prefix}*"))
        dataset = PackedDataset(
            filenames,
            n_chunks=4,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


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

class Dataset(IterableDataset):
    def __init__(self, data_file: Path, block_size: int):
        super().__init__()
        self.data_file = data_file
        self.block_size = block_size

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            i = torch.randint(len(data) - self.block_size, (1,)).item()
            x = torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
            y = torch.from_numpy((data[i + 1 : i + 1 + self.block_size]).astype(np.int64))
            yield x, y

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
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
