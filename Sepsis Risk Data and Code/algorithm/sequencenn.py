"""
Generic RNN/LSTM/GRU scorer in PyTorch

- Trains on batches of variable-length sequences (e.g., ~1000 sequences)
- Outputs a single score in [0, 1] per sequence
- Supports classic RNN, GRU, and LSTM via one class
- Streaming/online inference: feed one timestep at a time and get a score each time

Author: Yubin Kim (ykim3041@gatech.edu)
Python: 3.9+
PyTorch: 1.12+
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

Tensor = torch.Tensor


class GenericRNN(nn.Module):
    """A generic sequence-to-score model using RNN/GRU/LSTM backbones.

    Parameters
    ----------
    input_size : int
        Dimensionality of each timestep's input feature vector.
    hidden_size : int, optional
        Hidden size of the recurrent backbone, by default 128.
    rnn_type : {"rnn", "gru", "lstm"}, optional
        Which recurrent cell to use, by default "lstm".
    num_layers : int, optional
        Number of recurrent layers, by default 1.
    bidirectional : bool, optional
        Use a bidirectional backbone, by default False.
    dropout : float, optional
        Dropout between recurrent layers (if num_layers>1) and after MLP layers, by default 0.0.
    fc_hidden_sizes : List[int], optional
        Optional MLP hidden sizes after the recurrent backbone, by default None (linear to 1).
    batch_first : bool, optional
        Use (batch, seq, feat) layout, by default True.

    Notes
    -----
    * During training, pass variable sequence lengths via `lengths` to ignore padding.
    * For inference, you can:
        1) Batch infer entire sequences (pass `lengths` if padded), or
        2) Do streaming: call `init_state` once, then repeatedly call `step(x_t, state)`
           with one timestep at a time to get a score each measurement.
    * For stability, prefer `BCEWithLogitsLoss` with `return_logits=True`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        fc_hidden_sizes: Optional[List[int]] = None,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        rnn_type = rnn_type.lower()
        if rnn_type not in {"rnn", "gru", "lstm"}:
            raise ValueError("rnn_type must be one of {'rnn','gru','lstm'}")

        RNNClass = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first

        self.rnn = RNNClass(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=batch_first,
        )

        # Build MLP head -> scalar
        dims = [hidden_size * self.num_directions] + (fc_hidden_sizes or []) + [1]
        mlp: List[nn.Module] = []
        for i in range(len(dims) - 2):
            mlp.append(nn.Linear(dims[i], dims[i + 1]))
            mlp.append(nn.ReLU())
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
        mlp.append(nn.Linear(dims[-2], dims[-1]))
        self.head = nn.Sequential(*mlp)

    # -----------------------------
    # Utilities: state, device, etc
    # -----------------------------
    def init_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Create a zero-initialized hidden state for streaming or custom control.

        Returns
        -------
        h or (h, c):
            Shape: (num_layers*num_directions, batch, hidden_size)
        """
        h = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device
        )
        if self.rnn_type == "lstm":
            c = torch.zeros_like(h)
            return (h, c)
        return h

    # -----------------------------
    # Forward (batched sequences)
    # -----------------------------
    def forward(
        self,
        x: Tensor,
        lengths: Optional[Union[Tensor, List[int]]] = None,
        state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        return_logits: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Forward pass for a batch of sequences -> one score per sequence.

        Parameters
        ----------
        x : Tensor
            Shape (batch, seq, feat) when `batch_first=True` (default). If you pad
            sequences to the same length, pass the true `lengths` so padding is ignored.
        lengths : 1D Tensor or list[int], optional
            True lengths for each sequence (required if `x` is padded). If omitted, all
            timesteps are treated as valid.
        state : hidden state, optional
            Initial hidden/cell state. Rarely needed for full-sequence inference.
        return_logits : bool, optional
            If True, return raw logits (pre-sigmoid). Use with BCEWithLogitsLoss.

        Returns
        -------
        (score_or_logits, state)
            score_or_logits has shape (batch,). If `return_logits=False`, values are in [0, 1].
        """
        if lengths is not None:
            if not torch.is_tensor(lengths):
                lengths = torch.as_tensor(lengths, device=x.device)
            # pack for efficiency; we can use the final hidden state safely with packing
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False
            )
            _, state = self.rnn(packed, state)
        else:
            _, state = self.rnn(x, state)

        # Get last-layer hidden for each direction and concatenate
        if self.rnn_type == "lstm":
            h_n = state[0]  # (layers*dirs, batch, hidden)
        else:
            h_n = state
        # reshape to (layers, dirs, batch, hidden) and select last layer
        h_last = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_size)[-1]
        # -> (dirs, batch, hidden) -> (batch, dirs*hidden)
        h_last = h_last.transpose(0, 1).contiguous().view(x.size(0), -1)

        logits = self.head(h_last).squeeze(-1)  # (batch,)
        if return_logits:
            return logits, state
        else:
            return torch.sigmoid(logits), state

    # -----------------------------
    # Streaming (one timestep at a time)
    # -----------------------------
    @torch.no_grad()
    def step(
        self,
        x_t: Tensor,
        state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        return_logits: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Consume **one timestep** (or a batch of single timesteps) and return a score.

        Parameters
        ----------
        x_t : Tensor
            Shape (feat,) for a single sequence or (batch, feat) for a batch.
        state : hidden state, optional
            Carry the state returned from the previous call to `step`.
        return_logits : bool, optional
            If True, return raw logits (pre-sigmoid).

        Returns
        -------
        (score_or_logits, new_state)
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)  # (1, feat)
        x_t = x_t.unsqueeze(1)  # (batch, 1, feat) for batch_first

        out, new_state = self.rnn(x_t, state)
        # `out` shape: (batch, 1, hidden*dirs) for GRU/RNN/LSTM alike
        last = out[:, -1, :]  # (batch, hidden*dirs)
        logits = self.head(last).squeeze(-1)  # (batch,)
        if return_logits:
            return logits, new_state
        else:
            return torch.sigmoid(logits), new_state


# -----------------------------------------------------------
# Dataloader helper: pad variable-length batches for training
# -----------------------------------------------------------
@dataclass
class Batch:
    x: Tensor  # (batch, max_len, feat)
    lengths: Tensor  # (batch,)
    y: Tensor  # (batch,) float targets in {0,1} or [0,1]


def pad_collate(batch: Sequence[Tuple[Tensor, float]]) -> Batch:
    """Collate function for DataLoader.

    Expects each item as (seq, target) where `seq` is (len_i, feat).
    Returns padded tensor (batch, max_len, feat), lengths, and targets.
    """
    seqs, targets = zip(*batch)
    seqs = [torch.as_tensor(s, dtype=torch.float32) for s in seqs]
    lengths = torch.as_tensor([s.size(0) for s in seqs], dtype=torch.long)
    x = pad_sequence(seqs, batch_first=True)  # (batch, max_len, feat)
    y = torch.as_tensor(targets, dtype=torch.float32).view(-1)
    return Batch(x=x, lengths=lengths, y=y)


# -----------------------------------------------------------
# Minimal training/inference examples (replace with your data)
# -----------------------------------------------------------
if __name__ == "__main__":
    # Synthetic demo just to show shapes; replace with real data pipeline.
    torch.manual_seed(0)

    # Suppose each timestep has 16 features
    input_size = 16
    model = GenericRNN(input_size=input_size, hidden_size=64, rnn_type="lstm")

    # Fake batch of variable-length sequences (e.g., 1000 sequences)
    batch_size = 8
    lengths = torch.randint(low=1, high=50, size=(batch_size,))
    seqs = [torch.randn(l.item(), input_size) for l in lengths]
    targets = torch.rand(batch_size)  # scores in [0,1] (binary or continuous)

    batch = pad_collate(list(zip(seqs, targets)))

    # Training-time forward (prefer logits + BCEWithLogitsLoss)
    logits, _ = model(batch.x, lengths=batch.lengths, return_logits=True)
    loss = F.binary_cross_entropy_with_logits(logits, batch.y)
    loss.backward()
    print("Train pass ok | loss=", float(loss))

    # Batch inference on full sequences -> probabilities in [0,1]
    probs, _ = model(batch.x, lengths=batch.lengths, return_logits=False)
    print("Batch inference | probs shape:", tuple(probs.shape))

    # Streaming/online inference (one sequence, one timestep at a time)
    model.eval()
    state = model.init_state(batch_size=1)
    # Consume first 5 timesteps of a new sequence (len can be 1 as well)
    for t in range(5):
        x_t = torch.randn(input_size)
        score, state = model.step(x_t, state, return_logits=False)
        print(f"t={t} score={float(score):.4f}")

    # Reset state to start a fresh sequence
    state = model.init_state(batch_size=1)
    score, state = model.step(torch.randn(input_size), state)
    print("Single-timestep prediction:", float(score))
