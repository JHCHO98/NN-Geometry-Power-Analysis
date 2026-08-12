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
    "inverse_hourglass",
}
GROWTH_PATTERNS = {"linear", "early", "late"}


def generate_channels(
    depth: int,
    pattern: str,
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

    generator = random.Random(42)

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


@dataclass
class ModelConfig:
    """CNN 한 개의 구조를 재현 가능하게 기록하는 설정값."""

    depth: int
    channels: list[int]
    pools: list[int]
    pattern: str

    def __new__(cls, depth: int, channels: list[int], pools: list[int], pattern: str):
        # CIFAR-10 입력은 32 x 32이니까 2배 MaxPool을 5번 적용하면 1 x 1이 되므로, 6번째 pooling은 feature map 크기를 0으로 만들어 모델 생성을 막아야 한다.
        if len(pools) >= 6:
            raise ValueError(
                "CIFAR-10 입력(32x32)에서는 pooling을 최대 5번만 사용할 수 있습니다."
            )
        return super().__new__(cls)

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("depth는 1 이상의 정수여야 합니다.")
        if len(self.channels) != self.depth:
            raise ValueError("channels의 길이는 depth와 같아야 합니다.")
        if any(channel <= 0 for channel in self.channels):
            raise ValueError("모든 channel 값은 1 이상이어야 합니다.")
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
