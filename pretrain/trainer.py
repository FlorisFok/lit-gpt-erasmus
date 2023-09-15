import math
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import FSDPStrategy
from torch.utils.data import DataLoader, IterableDataset


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.model import GPT, Block
from lit_gpt.speed_monitor import SpeedMonitorCallback, estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, step_csv_logger

model_name = "pythia-6.9b"
name = "openwebtext"
out_dir = Path("out") / name
data_dir = Path("data") / name
save_interval = 1000
eval_interval = 1000
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 6e-4
batch_size = 125
micro_batch_size = 5
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
max_iters = 600000  # num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


class LightningGPTModule(L.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.module: Optional[torch.nn.Module] = None
        self.measured_flops: Optional[int] = None

    def configure_model(self) -> None:
        self.module = GPT(self.config)
        self.module.apply(self.module._init_weights)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.module.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
        )

    def on_fit_start(self) -> None:
        trainer = self.trainer
        with torch.device("meta"):
            meta_model = GPT(self.module.config)
            # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
            # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
            # consider setting `self.measured_flops = estimated_flops` instead
            estimated_flops = estimate_flops(meta_model) * micro_batch_size
            self.print(f"Estimated TFLOPs: {estimated_flops * trainer.world_size / 1e12:.2f}")
            x = torch.randint(0, 1, (micro_batch_size, meta_model.config.block_size))
            self.measured_flops = measure_flops(meta_model, x)
            self.print(f"Measured TFLOPs: {self.measured_flops * trainer.world_size / 1e12:.2f}")

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if not decay_lr:
            return
        # determine and set the learning rate for this iteration
        lr = get_lr(self.trainer.fit_loop.total_batch_idx)
        for optimizer in self.trainer.strategy.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input_ids = batch[:, 0 : self.config.block_size].contiguous()
        targets = batch[:, 1 : self.config.block_size + 1].contiguous()
        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        input_ids = batch[:, 0 : self.config.block_size].contiguous()
        targets = batch[:, 1 : self.config.block_size + 1].contiguous()
        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


def main(devices: int = 4, precision: Optional[str] = None) -> None:
    precision = precision or get_default_supported_precision(training=True)

    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            # the argument is not available in the Trainer strategy, but it's the default anyways
            # state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    logger = step_csv_logger("out", name, cls=CSVLogger, flush_logs_every_n_steps=log_interval)
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
        max_epochs=1,
        limit_val_batches=eval_iters,
        accumulate_grad_batches=gradient_accumulation_steps,
        log_every_n_steps=log_interval,
        val_check_interval=eval_interval,
    )

    L.seed_everything(1337, workers=True)  # same seed for every process to init model (FSDP)

    trainer.print(hparams)

    if trainer.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)
    trainer.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    model = LightningGPTModule(config)
    trainer.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=trainer,
        train_data_dir=data_dir,
        val_data_dir=None,
        seed=(1337 + trainer.global_rank),
    )

    t0 = time.perf_counter()
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="last")
    trainer.print(f"Training time: {(time.perf_counter()-t0):.2f}s")
    if trainer.strategy.root_device.type == "cuda":
        trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")



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
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)