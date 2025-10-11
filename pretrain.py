import copy
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import coolname
import hydra
import pydantic
import torch
import torch.distributed as dist
import tqdm
import yaml
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

import wandb
from models.ema import EMAHelper
from models.recursive_reasoning.trm import TinyRecursiveModelCarry
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import get_model_source_path, load_model_class


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: list[str]
    data_paths_test: list[str] = []
    # Evaluators
    evaluators: list[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    gradient_accumulation_steps: int = 1
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0  # when to start eval
    eval_save_outputs: list[str] = []

    ema: bool = False  # use Exponential-Moving-Average
    ema_rate: float = 0.999  # EMA-rate
    freeze_weights: bool = (
        False  # If True, freeze weights and only learn the embeddings
    )

    # Benchmarking
    benchmark_mode: bool = False  # Enable detailed timing statistics
    benchmark_steps: Optional[int] = None  # If set, stop after N steps


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: TinyRecursiveModelCarry | None

    step: int
    total_steps: int
    accumulation_step: int = 0  # Track gradient accumulation


def create_dataloader(
    config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs
):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths_test
            if len(config.data_paths_test) > 0 and split == "test"
            else config.data_paths,
            rank=rank,
            num_replicas=world_size,
            **kwargs,
        ),
        split=split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader, dataset.metadata


def count_parameters(model: nn.Module) -> dict[str, float]:
    """Count parameters in different parts of the model (in millions)."""
    inner = (
        model.model.inner
        if hasattr(model, "model")
        else model.inner
        if hasattr(model, "inner")
        else model
    )

    # Detailed parameter breakdown
    # 1. Input embeddings (used once to create input_embeddings)
    embed_tokens_params = (
        sum(p.numel() for p in inner.embed_tokens.parameters())
        if hasattr(inner, "embed_tokens")
        else 0
    )
    embed_pos_params = (
        sum(p.numel() for p in inner.embed_pos.parameters())
        if hasattr(inner, "embed_pos")
        else 0
    )
    puzzle_emb_params = (
        sum(p.numel() for p in inner.puzzle_emb.parameters())
        if hasattr(inner, "puzzle_emb")
        else 0
    )
    rotary_emb_params = (
        sum(p.numel() for p in inner.rotary_emb.parameters())
        if hasattr(inner, "rotary_emb")
        else 0
    )

    input_embed_params = (
        embed_tokens_params + embed_pos_params + puzzle_emb_params + rotary_emb_params
    )

    # 2. Recurrent reasoning module (net) - used y_cycles * (z_cycles + 1) times
    net_params = (
        sum(p.numel() for p in inner.net.parameters()) if hasattr(inner, "net") else 0
    )

    # 3. Output heads (used once per forward pass)
    lm_head_params = (
        sum(p.numel() for p in inner.lm_head.parameters())
        if hasattr(inner, "lm_head")
        else 0
    )
    q_head_params = (
        sum(p.numel() for p in inner.q_head.parameters())
        if hasattr(inner, "q_head")
        else 0
    )

    # 4. Initial states (y_init, z_init - used once)
    init_params = 0
    if hasattr(inner, "y_init"):
        init_params += inner.y_init.numel()
    if hasattr(inner, "z_init"):
        init_params += inner.z_init.numel()

    output_params = lm_head_params + q_head_params + init_params

    # Total unique parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Verify our breakdown matches
    accounted = input_embed_params + net_params + output_params
    rest_params = net_params + output_params

    # Get number of cycles for effective parameter count
    y_cycles = inner.config.y_cycles if hasattr(inner, "config") else 1
    z_cycles = inner.config.z_cycles if hasattr(inner, "config") else 1

    # Trace through a forward pass:
    # 1. Compute input_embeddings (uses input embeddings once)
    # 2. For each of y_cycles y-cycles:
    #    - z_cycles times: z = net(z, y + input_embeddings)  [net called z_cycles times]
    #    - 1 time: y = net(y, z)                              [net called 1 time]
    # 3. Compute output: lm_head(y), q_head(y)                [output heads used once]
    total_net_passes = y_cycles * (z_cycles + 1)

    # Effective parameters accounting for reuse
    effective_params = (
        input_embed_params * 1  # Input embeddings used once
        + net_params * total_net_passes  # Net used multiple times
        + output_params * 1  # Output heads/states used once
    )

    return {
        "input_embeddings_M": input_embed_params / 1e6,
        "net_M": net_params / 1e6,
        "output_M": output_params / 1e6,
        "total_M": total_params / 1e6,
        "effective_M": effective_params / 1e6,
        "net_passes": total_net_passes,
        "y_cycles": y_cycles,
        "z_cycles": z_cycles,
    }


def create_model(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
    device: torch.device,
):
    # Model batch_size is the per-device batch size (global_batch_size is per device in HF convention)
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

        # Log parameter counts
        if rank == 0:
            param_counts = count_parameters(model)
            print("\n" + "=" * 70)
            print("MODEL PARAMETER COUNTS")
            print("=" * 70)
            print(
                f"Input embeddings:     {param_counts['input_embeddings_M']:>8.2f}M  (used 1x per forward)"
            )
            print(
                f"Recurrent net:        {param_counts['net_M']:>8.2f}M  (used {param_counts['net_passes']:.0f}x per forward)"
            )
            print(
                f"Output heads/states:  {param_counts['output_M']:>8.2f}M  (used 1x per forward)"
            )
            print("-" * 70)
            print(f"Total unique params:  {param_counts['total_M']:>8.2f}M")
            print(f"Effective params:     {param_counts['effective_M']:>8.2f}M")
            print()
            print(
                f"Recursion structure: {param_counts['y_cycles']:.0f} y-cycles × ({param_counts['z_cycles']:.0f} z-cycles + 1 y-update)"
            )
            print(
                f"                   = {param_counts['net_passes']:.0f} passes through net per forward"
            )
            print("=" * 70 + "\n")

        # torch.compile not supported on MPS yet
        if "DISABLE_COMPILE" not in os.environ and device.type != "mps":
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config, device)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            torch.optim.AdamW(
                model.parameters(),
                lr=1e-8,  # Will be overridden by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            )
        ]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            torch.optim.AdamW(
                model.parameters(),
                lr=1e-8,  # Will be overridden by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            ),
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0] * sd[0][k].to(device)
        for i in range(1, len(nets)):
            comb_net += alpha[i] * sd[i][k].to(device)
        sd_alpha[k] = comb_net
    net.load_state_dict(sd_alpha)
    return net


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(
            0.0,
            (1 - min_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    )


def init_train_state(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
    device: torch.device,
):
    # Estimated total training steps (optimizer steps, not forward passes)
    # Effective batch size = global_batch_size * gradient_accumulation_steps
    effective_batch_size = config.global_batch_size * config.gradient_accumulation_steps
    total_steps = int(
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        / effective_batch_size
    )

    # Model
    model, optimizers, optimizer_lrs = create_model(
        config, train_metadata, rank=rank, world_size=world_size, device=device
    )

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(
        train_state.model.state_dict(),
        os.path.join(config.checkpoint_path, f"step_{train_state.step}"),
    )


def load_checkpoint(model: nn.Module, config: PretrainConfig, device: torch.device):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location=device)

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(
                    f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}"
                )
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True)
                    .expand(expected_shape)
                    .contiguous()
                )
        model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )


def create_evaluators(
    config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata
) -> list[Any]:
    data_paths = (
        config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    )
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path,
                eval_metadata=eval_metadata,
                **cfg.__pydantic_extra__,
            )  # type: ignore
            evaluators.append(cls)

    return evaluators


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
):
    # Increment accumulation step
    train_state.accumulation_step += 1
    is_accumulation_step = (
        train_state.accumulation_step % config.gradient_accumulation_steps != 0
    )

    # Only increment main step counter when accumulation is complete
    if not is_accumulation_step:
        train_state.step += 1
        if train_state.step > train_state.total_steps:  # At most train_total_steps
            return

    # To device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Init carry
    # For text training (no puzzle embeddings), reset every batch for fresh text windows
    # For puzzle training with ACT, carry persists to continue iterating on same puzzle
    if train_state.carry is None or config.arch.puzzle_emb_ndim == 0:
        with torch.device(device):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    # Scale loss by effective batch size (HuggingFace convention)
    # Effective batch = global_batch_size * gradient_accumulation_steps
    effective_batch_size = config.global_batch_size * config.gradient_accumulation_steps
    ((1 / effective_batch_size) * loss).backward()

    # Only step optimizer when accumulation is complete
    if not is_accumulation_step:
        # Allreduce
        if world_size > 1:
            for param in train_state.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)

        # Apply optimizer
        lr_this_step = None
        for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
            lr_this_step = compute_lr(base_lr, config, train_state)

            for param_group in optim.param_groups:
                param_group["lr"] = lr_this_step

            optim.step()
            optim.zero_grad()

    # Only reduce and return metrics when accumulation is complete
    if not is_accumulation_step:
        # Reduce metrics
        if len(metrics):
            assert not any(v.requires_grad for v in metrics.values())

            metric_keys = list(
                sorted(metrics.keys())
            )  # Sort keys to guarantee all processes use the same order.
            # Reduce and reconstruct
            metric_values = torch.stack([metrics[k] for k in metric_keys])
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                metric_values = metric_values.cpu().numpy()
                reduced_metrics = {
                    k: metric_values[i] for i, k in enumerate(metric_keys)
                }

                # Postprocess
                count = max(reduced_metrics.get("count", 0), 1)  # Avoid NaNs
                effective_batch_size = (
                    config.global_batch_size * config.gradient_accumulation_steps
                )
                loss_count = max(
                    reduced_metrics.get("loss_count", effective_batch_size), 1
                )
                reduced_metrics = {
                    f"train/{k}": v / (loss_count if k.endswith("loss") else count)
                    for k, v in reduced_metrics.items()
                    if k not in ["loss_count"]  # Don't log loss_count itself
                }

                reduced_metrics["train/lr"] = lr_this_step
                return reduced_metrics


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: list[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    device: torch.device,
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds: dict[str, list[torch.Tensor]] = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0

        for set_name, batch, _ in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            # To device
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.device(device):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(
                            v.cpu()
                        )  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())),
                    dtype=torch.float32,
                    device=device,
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds,
                os.path.join(
                    config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"
                ),
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics_array = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics_array[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")

        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(
                    f"Running evaluator {i + 1}/{len(evaluators)}: {evaluator.__class__.__name__}"
                )

            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(
                evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group
            )
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")

        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name),
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(
    hydra_config: DictConfig, rank: int, world_size: int
) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = (
                f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
            )
        if config.run_name is None:
            config.run_name = (
                f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
            )
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join(
                "checkpoints", config.project_name, config.run_name
            )

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Determine device - prefer CUDA, fallback to MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        if device.type != "cuda":
            raise RuntimeError("Distributed training only supported with CUDA")
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK
            and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = (
        config.eval_interval if config.eval_interval is not None else config.epochs
    )
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, (
        "Eval interval must be a divisor of total epochs."
    )

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,  # HF convention: this is the per-step batch size
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=RANK,
            world_size=WORLD_SIZE,
        )
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(
        config, train_metadata, rank=RANK, world_size=WORLD_SIZE, device=device
    )

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(x_disable_stats=True),
        )  # type: ignore
        wandb.log(
            {"num_params": sum(x.numel() for x in train_state.model.parameters())},
            step=0,
        )
        save_code_and_config(config)
    if config.ema:
        print("Setup EMA")
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Benchmarking setup
    benchmark_timings = []
    benchmark_start_step = None
    benchmark_start_time = None

    # Training Loop
    for _iter_id in range(total_iters):
        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            # Benchmark timing
            if config.benchmark_mode and RANK == 0:
                if benchmark_start_step is None:
                    benchmark_start_step = train_state.step
                    benchmark_start_time = time.perf_counter()
                step_start = time.perf_counter()

            metrics = train_batch(
                config,
                train_state,
                batch,
                batch_size=global_batch_size,  # From dataloader
                rank=RANK,
                world_size=WORLD_SIZE,
                device=device,
            )

            # Benchmark timing
            if config.benchmark_mode and RANK == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                step_time = time.perf_counter() - step_start
                benchmark_timings.append(step_time)

            if RANK == 0:
                if metrics is not None:
                    wandb.log(metrics, step=train_state.step)
                    progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                assert ema_helper is not None
                ema_helper.update(train_state.model)

            # Stop early if benchmarking
            if (
                config.benchmark_steps is not None
                and train_state.step >= config.benchmark_steps
            ):
                if RANK == 0 and config.benchmark_mode:
                    # Print benchmark stats
                    import numpy as np

                    timings = np.array(
                        benchmark_timings[10:]
                    )  # Skip first 10 for warmup
                    total_time = time.perf_counter() - benchmark_start_time
                    steps = len(timings)

                    benchmark_results = {
                        "total_steps": int(steps),
                        "total_time_seconds": float(total_time),
                        "steps_per_second": float(steps / total_time),
                        "avg_step_time_ms": float(timings.mean() * 1000),
                        "std_step_time_ms": float(timings.std() * 1000),
                        "min_step_time_ms": float(timings.min() * 1000),
                        "max_step_time_ms": float(timings.max() * 1000),
                        "median_step_time_ms": float(np.median(timings) * 1000),
                        "all_timings_ms": [float(t * 1000) for t in timings.tolist()],
                    }

                    # Save to JSON
                    benchmark_file = "benchmark_results.json"
                    with open(benchmark_file, "w") as f:
                        json.dump(benchmark_results, f, indent=2)

                    print("\n" + "=" * 60)
                    print("BENCHMARK RESULTS")
                    print("=" * 60)
                    print(f"Total steps: {steps}")
                    print(f"Total time: {total_time:.2f}s")
                    print(f"Steps/sec: {benchmark_results['steps_per_second']:.2f}")
                    print(
                        f"Avg step time: {benchmark_results['avg_step_time_ms']:.2f}ms"
                    )
                    print(
                        f"Std step time: {benchmark_results['std_step_time_ms']:.2f}ms"
                    )
                    print(
                        f"Min step time: {benchmark_results['min_step_time_ms']:.2f}ms"
                    )
                    print(
                        f"Max step time: {benchmark_results['max_step_time_ms']:.2f}ms"
                    )
                    print(
                        f"Median step time: {benchmark_results['median_step_time_ms']:.2f}ms"
                    )
                    print(f"\nResults saved to: {benchmark_file}")
                    print("=" * 60)
                if dist.is_initialized():
                    dist.destroy_process_group()
                wandb.finish()
                return

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                assert ema_helper is not None
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(
                config,
                train_state_eval,
                eval_loader,
                eval_metadata,
                evaluators,
                rank=RANK,
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP,
                device=device,
            )

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)

            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (
                config.checkpoint_every_eval or (_iter_id == total_iters - 1)
            ):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
