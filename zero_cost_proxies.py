"""Reproducible Zero-Cost proxy calculations for FlexibleCNN architectures."""

from __future__ import annotations

import torch
import torch.nn as nn

from FlexibleCNN import FlexibleCNN, ModelConfig


def compute_zero_cost_proxies(config: ModelConfig, device: str = "cpu") -> dict[str, float]:
    """Return SynFlow, GradNorm, and Jacobian-covariance proxies.

    All random inputs and parameter initializations are seeded from ``config.seed``.
    This makes a score reproducible for a given candidate even when calculation is
    resumed later.  JacobCov uses correlations between input-Jacobian vectors, not
    correlations between the ten output logits; the latter is nearly constant for
    this model family and is not a Jacobian proxy.
    """
    device_obj = torch.device(device)
    seed = int(config.seed) % (2**63 - 1)
    cuda_devices = [device_obj.index or 0] if device_obj.type == "cuda" else []

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device_obj.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # SynFlow: preserve only positive parameter flow and score |w * dw|.
        synflow_model = FlexibleCNN(config).to(device_obj).eval()
        with torch.no_grad():
            for parameter in synflow_model.parameters():
                parameter.abs_()
        synflow_output = synflow_model(torch.ones(1, 3, 32, 32, device=device_obj))
        synflow_output.sum().backward()
        synflow_score = sum(
            (parameter * parameter.grad).abs().sum().item()
            for parameter in synflow_model.parameters()
            if parameter.grad is not None
        )
        del synflow_model

        # GradNorm: gradient magnitude under a fixed synthetic classification batch.
        model = FlexibleCNN(config).to(device_obj).eval()
        criterion = nn.CrossEntropyLoss()
        inputs = torch.randn(16, 3, 32, 32, device=device_obj)
        targets = torch.randint(0, 10, (16,), device=device_obj)
        criterion(model(inputs), targets).backward()
        grad_norm_score = sum(
            parameter.grad.norm(2).item() ** 2
            for parameter in model.parameters()
            if parameter.grad is not None
        ) ** 0.5
        del model

        # JacobCov: covariance spectrum of per-example input Jacobians.
        jacob_model = FlexibleCNN(config).to(device_obj).eval()
        jacob_inputs = torch.randn(16, 3, 32, 32, device=device_obj, requires_grad=True)
        jacob_outputs = jacob_model(jacob_inputs)
        jacobian = torch.autograd.grad(jacob_outputs.sum(), jacob_inputs)[0].flatten(1)
        jacobian = jacobian - jacobian.mean(dim=1, keepdim=True)
        jacobian = jacobian / jacobian.norm(dim=1, keepdim=True).clamp_min(1e-12)
        correlation = jacobian @ jacobian.T
        eigenvalues = torch.linalg.eigvalsh(correlation).clamp_min(1e-5)
        jacob_cov_score = -(torch.log(eigenvalues) + 1.0 / eigenvalues).sum().item()

    scores = {
        "synflow_score": float(synflow_score),
        "grad_norm_score": float(grad_norm_score),
        "jacob_cov_score": float(jacob_cov_score),
    }
    if not all(torch.isfinite(torch.tensor(value)) for value in scores.values()):
        raise FloatingPointError(f"Non-finite Zero-Cost score for seed {config.seed}: {scores}")
    return scores
