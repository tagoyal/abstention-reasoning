"""
Pipeline commands - re-exports all commands for convenient access.

Usage:
    from pipeline import commands
    commands.create_primitives(...)
    commands.generate(...)
    commands.train_sft(...)
"""

from .data import (
    OOD_DATASETS,
    create_ood_prompts,
    create_partitions,
    create_primitives,
    create_prompts,
    create_verification_data,
)
from .inference import generate, generate_until_target, evaluate, analyze
from .training import train_sft, train_rl, convert_checkpoint

__all__ = [
    # Data commands
    "create_primitives",
    "create_partitions",
    "create_prompts",
    "create_verification_data",
    "create_ood_prompts",
    "OOD_DATASETS",
    # Inference commands
    "generate",
    "generate_until_target",
    "evaluate",
    "analyze",
    # Training commands
    "train_sft",
    "train_rl",
    "convert_checkpoint",
]
