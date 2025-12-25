"""
Generic feed-forward scorer in PyTorch.

- Operates on fixed-length feature vectors (no temporal component)
- Outputs a single score in [0, 1] per sample
- Supports arbitrary depth/width MLPs with configurable activation and dropout

Author: Yubin Kim (ykim3041@gatech.edu)
Python: 3.9+
PyTorch: 1.12+
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
ActivationSpec = Union[str, Callable[[], nn.Module], nn.Module]


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _build_activation(spec: ActivationSpec) -> nn.Module:
    """Resolve an activation spec into an ``nn.Module`` instance."""
    if isinstance(spec, nn.Module):
        return copy.deepcopy(spec)
    if isinstance(spec, str):
        try:
            act_cls = _ACTIVATIONS[spec.lower()]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported activation '{spec}'. Available: {sorted(_ACTIVATIONS)}"
            ) from exc
        return act_cls()
    act = spec()
    if not isinstance(act, nn.Module):
        raise TypeError("Activation callable must return an nn.Module instance")
    return act


class GenericMLP(nn.Module):
    """A configurable feature-to-score MLP.

    Parameters
    ----------
    input_size : int
        Dimensionality of the input feature vector.
    hidden_sizes : Sequence[int], optional
        Hidden layer sizes, by default (128, 64).
    dropout : Union[float, Sequence[float]], optional
        Dropout probability applied after each hidden layer. Provide a single float to
        reuse across layers or a sequence matching ``hidden_sizes`` for per-layer values.
        Defaults to 0.0.
    activation : ActivationSpec, optional
        Activation to apply after each hidden layer, by default "relu".
    batch_norm : bool, optional
        Insert BatchNorm1d layers before activations, by default False.
    final_dropout : float, optional
        Dropout applied right before the final linear head, by default 0.0.

    Notes
    -----
    * Call ``forward(x, return_logits=True)`` during training and pair with
      ``BCEWithLogitsLoss`` for numerical stability.
    * Set ``batch_norm=True`` for tabular datasets where feature distributions vary.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (128, 64),
        dropout: Union[float, Sequence[float]] = 0.0,
        activation: ActivationSpec = "relu",
        batch_norm: bool = False,
        final_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not hidden_sizes:
            raise ValueError("Provide at least one hidden layer size for GenericMLP")

        if isinstance(dropout, Sequence) and not isinstance(dropout, (str, bytes)):
            if len(dropout) not in (1, len(hidden_sizes)):
                raise ValueError(
                    "When specifying per-layer dropout, provide 1 value or len(hidden_sizes) values."
                )
            dropouts = list(dropout)
            if len(dropouts) == 1:
                dropouts = dropouts * len(hidden_sizes)
        else:
            dropouts = [float(dropout)] * len(hidden_sizes)

        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.dropout = [float(p) for p in dropouts]
        self.batch_norm = batch_norm
        self.final_dropout = float(final_dropout)

        layers: list[nn.Module] = []
        in_dim = input_size

        for out_dim, p in zip(self.hidden_sizes, self.dropout):
            p = float(p)
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(_build_activation(activation))
            if p > 0.0:
                layers.append(nn.Dropout(p))
            in_dim = out_dim

        if self.final_dropout > 0.0:
            layers.append(nn.Dropout(self.final_dropout))
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, return_logits: bool = False) -> Tensor:
        """Forward pass for a batch of feature vectors -> one score per sample."""
        logits = self.net(x).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


@dataclass
class Batch:
    x: Tensor  # (batch, feat)
    y: Tensor  # (batch,) float targets in {0,1} or [0,1]


def collate_dense_batch(batch: Sequence[Tuple[Tensor, float]]) -> Batch:
    """Collate function for dense feature tensors.

    Expects each item as (feature_vec, target) where ``feature_vec`` is shape (feat,).
    Returns stacked tensor (batch, feat) and targets.
    """
    features, targets = zip(*batch)
    x = torch.stack([torch.as_tensor(f, dtype=torch.float32) for f in features], dim=0)
    y = torch.as_tensor(targets, dtype=torch.float32).view(-1)
    return Batch(x=x, y=y)


if __name__ == "__main__":
    # Synthetic demo showcasing usage; replace with your real data pipeline.
    torch.manual_seed(0)

    input_size = 32
    model = GenericMLP(
        input_size=input_size,
        hidden_sizes=(128, 64),
        dropout=(0.2, 0.1),
        activation="relu",
        batch_norm=True,
        final_dropout=0.1,
    )

    # Fake batch of feature vectors + binary targets.
    batch_size = 16
    features = torch.randn(batch_size, input_size)
    targets = torch.rand(batch_size)

    # Training-time forward (prefer logits + BCEWithLogitsLoss).
    logits = model(features, return_logits=True)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    print("Train pass ok | loss=", float(loss))

    # Batch inference -> probabilities in [0,1].
    probs = model(features)
    print("Batch inference | probs shape:", tuple(probs.shape))

    # Using the collate helper with a list of samples.
    samples = [(torch.randn(input_size), float(torch.rand(()))) for _ in range(8)]
    batch = collate_dense_batch(samples)
    probs = model(batch.x)
    print("Collate helper | batch probs shape:", tuple(probs.shape))
