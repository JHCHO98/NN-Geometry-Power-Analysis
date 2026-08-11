from dataclasses import dataclass

import torch
import torch.nn as nn


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
