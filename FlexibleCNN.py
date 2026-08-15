from dataclasses import dataclass
import math
import random

import torch
import torch.nn as nn


CHANNEL_PATTERNS = {
    "uniform",
    "increasing",
    "decreasing",
    "hourglass",
    "inverse_hourglass"
}
GROWTH_PATTERNS = {"linear", "early", "late"}

def generate_channels(
    depth: int,
    pattern: str,
    seed: int,
    growth_pattern: str = "linear",
    noise_ratio: float = 0.0,
    min_channels: int = 32,
    max_channels: int = 128
) -> list[int]:
    """Generate CNN channel widths while preserving the requested geometry.

    ``growth_pattern`` controls how quickly a curve changes: ``early`` changes
    quickly near its start, ``late`` changes quickly near its end, and
    ``linear`` changes at a constant rate.  Noise is applied only to internal
    layers, then each shape is corrected so noise cannot alter its geometry.
    """
    if depth <= 0:
        raise ValueError("depth must be a positive integer.")
    if pattern not in CHANNEL_PATTERNS:
        raise ValueError(f"Unknown pattern: {pattern}. Expected one of {sorted(CHANNEL_PATTERNS)}.")
    if growth_pattern not in GROWTH_PATTERNS:
        raise ValueError(
            f"Unknown growth_pattern: {growth_pattern}. "
            f"Expected one of {sorted(GROWTH_PATTERNS)}."
        )
    if not 0.0 <= noise_ratio < 1.0:
        raise ValueError("noise_ratio must be in the range [0.0, 1.0).")
    if min_channels <= 0 or max_channels <= 0 or min_channels > max_channels:
        raise ValueError("Channel bounds must satisfy 0 < min_channels <= max_channels.")

    generator = random.Random(seed)

    def curve(t: float) -> float:
        if growth_pattern == "early":
            return math.sqrt(t)
        if growth_pattern == "late":
            return t**2
        return t

    def interpolate(t: float) -> float:
        return min_channels + (max_channels - min_channels) * curve(t)

    if pattern == "uniform":
        # A uniform model is an explicit control group, so it remains noiseless.
        channels = [round((min_channels + max_channels) / 2)] * depth
    elif depth == 1:
        channels = [min_channels]
    else:
        positions = [index / (depth - 1) for index in range(depth)]
        if pattern == "increasing":
            channels = [round(interpolate(t)) for t in positions]
        elif pattern == "decreasing":
            channels = [round(interpolate(1.0 - t)) for t in positions]
        elif pattern == "hourglass":
            channels = [round(interpolate(abs(2 * t - 1))) for t in positions]
        else:  # inverse_hourglass
            channels = [round(interpolate(1.0 - abs(2 * t - 1))) for t in positions]

    for index in range(1, depth - 1):
        if pattern != "uniform" and noise_ratio:
            multiplier = generator.uniform(1.0 - noise_ratio, 1.0 + noise_ratio)
            channels[index] = round(channels[index] * multiplier)
        channels[index] = min(max(channels[index], min_channels), max_channels)

    if pattern == "increasing":
        channels = sorted(channels)
    elif pattern == "decreasing":
        channels = sorted(channels, reverse=True)
    elif pattern == "hourglass":
        midpoint = depth // 2
        if depth % 2:
            channels[midpoint] = min_channels
            channels[:midpoint] = sorted(channels[:midpoint], reverse=True)
            channels[midpoint + 1 :] = sorted(channels[midpoint + 1 :])
        else:
            channels[midpoint - 1 : midpoint + 1] = [min_channels, min_channels]
            channels[: midpoint - 1] = sorted(channels[: midpoint - 1], reverse=True)
            channels[midpoint + 1 :] = sorted(channels[midpoint + 1 :])
    elif pattern == "inverse_hourglass":
        midpoint = depth // 2
        if depth % 2:
            channels[midpoint] = max_channels
            channels[:midpoint] = sorted(channels[:midpoint])
            channels[midpoint + 1 :] = sorted(channels[midpoint + 1 :], reverse=True)
        else:
            channels[midpoint - 1 : midpoint + 1] = [max_channels, max_channels]
            channels[: midpoint - 1] = sorted(channels[: midpoint - 1])
            channels[midpoint + 1 :] = sorted(channels[midpoint + 1 :], reverse=True)

    return channels

def generate_pools(
    depth:int,
    pool_count:int,
    seed:int
) -> list[int]:
    if depth <= 0:
        raise ValueError("depth must be a positive integer.")
    if pool_count > depth:
        raise ValueError("pooling count must be smaller than depth.")
    if pool_count > 5:
        raise ValueError("pooling count must be smaller than 6 in cifar-10 dataset.")

    rng = random.Random(seed)

    return sorted(rng.sample(list(range(1, depth + 1)), pool_count))


def count_parameters(channels: list[int], input_channels: int = 3, num_classes: int = 10) -> int:
    """Return the exact trainable-parameter count of ``FlexibleCNN``.

    Each Conv2d and the final Linear layer use PyTorch's default bias=True,
    so their bias parameters are included in the calculation.
    """
    if not channels:
        raise ValueError("channels must contain at least one layer.")

    total = 0
    previous_channels = input_channels
    for output_channels in channels:
        total += output_channels * (previous_channels * 3 * 3 + 1)
        previous_channels = output_channels
    total += previous_channels * num_classes + num_classes
    return total


@dataclass
class ModelConfig:
    """CNN 한 개의 구조를 재현 가능하게 기록하는 설정값."""

    depth: int
    channels: list[int]
    pools: list[int]
    pattern: str
    growth_pattern: str
    noise_ratio: float
    min_channels: int
    max_channels: int
    seed: int
    parameter_count: int

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("depth는 1 이상의 정수여야 합니다.")
        if len(self.channels) != self.depth:
            raise ValueError("channels의 길이는 depth와 같아야 합니다.")
        if any(channel <= 0 for channel in self.channels):
            raise ValueError("모든 channel 값은 1 이상이어야 합니다.")
        if len(self.pools) > 5:
            raise ValueError("CIFAR-10 input supports at most five pooling operations.")
        if len(set(self.pools)) != len(self.pools):
            raise ValueError("pools에는 같은 위치를 중복해서 넣을 수 없습니다.")
        if self.pools != sorted(self.pools):
            raise ValueError("pools는 오름차순으로 정렬된 Conv 블록 위치여야 합니다.")
        if any(pool < 1 or pool > self.depth for pool in self.pools):
            raise ValueError("pool 위치는 1부터 depth 사이여야 합니다.")

class FlexibleCNN(nn.Module):
    """ModelConfig에 따라 생성되는 CIFAR-10용 CNN."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_channels = 3

        for block_index, out_channels in enumerate(config.channels, start=1):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if block_index in config.pools:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_channels, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def random_config(
    seed: int,
    depth: int,
    pattern: str,
    growth_pattern: str,
    noise_ratio: float,
    pool_count: int,
    min_channel_lower: int = 8,
    min_channel_upper: int = 96,
    max_channel_limit: int = 256,
    min_parameters: int | None = 5_000,
    max_parameters: int | None = 2_000_000,
    max_attempts: int = 1_000,
) -> ModelConfig:
    """Create one reproducible CNN configuration.

    The same config seed always produces the same channel bounds, noisy channel
    sequence, and pooling positions. ``min_channels`` and ``max_channels`` are
    sampled per model so the dataset includes both small and large CNNs.
    Candidates outside the requested parameter-count range are discarded.
    """
    if min_channel_lower <= 0 or min_channel_lower > min_channel_upper:
        raise ValueError("Invalid minimum-channel sampling range.")
    if min_channel_upper > max_channel_limit:
        raise ValueError("min_channel_upper cannot exceed max_channel_limit.")
    if min_parameters is not None and min_parameters < 0:
        raise ValueError("min_parameters must be non-negative.")
    if max_parameters is not None and max_parameters < 0:
        raise ValueError("max_parameters must be non-negative.")
    if (
        min_parameters is not None
        and max_parameters is not None
        and min_parameters > max_parameters
    ):
        raise ValueError("min_parameters cannot exceed max_parameters.")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")

    master_rng = random.Random(seed)
    for _ in range(max_attempts):
        min_channels = master_rng.randint(min_channel_lower, min_channel_upper)

        # Uniform is a true control group: every layer has one sampled width.
        max_channels = (
            min_channels
            if pattern == "uniform"
            else master_rng.randint(min_channels, max_channel_limit)
        )
        channels = generate_channels(
            depth=depth,
            pattern=pattern,
            seed=master_rng.randrange(2**63),
            growth_pattern=growth_pattern,
            noise_ratio=noise_ratio,
            min_channels=min_channels,
            max_channels=max_channels,
        )
        parameter_count = count_parameters(channels)
        if min_parameters is not None and parameter_count < min_parameters:
            continue
        if max_parameters is not None and parameter_count > max_parameters:
            continue

        pools = generate_pools(
            depth=depth,
            pool_count=pool_count,
            seed=master_rng.randrange(2**63),
        )
        return ModelConfig(
            depth=depth,
            channels=channels,
            pools=pools,
            pattern=pattern,
            growth_pattern=growth_pattern,
            noise_ratio=noise_ratio,
            min_channels=min_channels,
            max_channels=max_channels,
            seed=seed,
            parameter_count=parameter_count,
        )

    raise ValueError(
        f"No configuration satisfied the parameter range after {max_attempts} attempts. "
        "Widen the range or adjust the channel bounds."
    )



if __name__ == "__main__":
    pass
