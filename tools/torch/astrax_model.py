"""PyTorch implementation of the Astrax holistic dialogue model.

This mirrors the C++ architecture exactly so that weights can be exchanged
between the CPU (C++) trainer and the GPU (PyTorch) trainer:

    subword ids -> pooled embedding  (input features)
    Osten condition vector
    -> shared hidden projection
    -> per-slot hidden (slot embedding + round embedding added)
    -> dense byte-class logits (256 bytes + 1 stop)
    -> iterative whole-sequence refinement over several rounds

It is deliberately NOT a Transformer and NOT next-token autoregressive: every
slot is refined in parallel over rounds, and no slot performs left-to-right
generation.
"""

from __future__ import annotations

import torch
from torch import nn


class HolisticDialogueModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        max_response_codepoints: int = 64,
        condition_dim: int = 128,
        hidden_dim: int = 192,
        slot_dim: int = 64,
        round_dim: int = 32,
        agg_dim: int = 128,
        rounds: int = 3,
        class_count: int = 257,
        vocab_size: int = 30000,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.max_response_codepoints = max_response_codepoints
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.slot_dim = slot_dim
        self.round_dim = round_dim
        self.agg_dim = agg_dim
        self.rounds = rounds
        self.class_count = class_count

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embeddings.weight, std=0.05)

        self.in_weights = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.cond_weights = nn.Parameter(torch.empty(hidden_dim, condition_dim))
        self.slot_weights = nn.Parameter(torch.empty(hidden_dim, slot_dim))
        self.round_weights = nn.Parameter(torch.empty(hidden_dim, round_dim))
        self.agg_weights = nn.Parameter(torch.empty(hidden_dim, agg_dim))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.out_weights = nn.Parameter(torch.empty(class_count, hidden_dim))
        self.out_bias = nn.Parameter(torch.zeros(class_count))

        self.slot_embedding = nn.Parameter(
            torch.empty(max_response_codepoints, slot_dim))
        self.round_embedding = nn.Parameter(torch.empty(rounds, round_dim))

        for parameter in (
            self.in_weights,
            self.cond_weights,
            self.slot_weights,
            self.round_weights,
            self.agg_weights,
            self.out_weights,
        ):
            nn.init.normal_(parameter, std=0.05)
        nn.init.normal_(self.slot_embedding, std=0.1)
        nn.init.normal_(self.round_embedding, std=0.1)

    def encode_features(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ids: (batch, seq), mask: (batch, seq) 1 for real tokens.
        embedded = self.embeddings(ids) * mask.unsqueeze(-1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = embedded.sum(dim=1) / counts  # (batch, embed_dim)
        # Project/pad the pooled vector to input_dim by modular accumulation,
        # matching the C++ encode_text behavior.
        features = torch.zeros(
            ids.shape[0], self.input_dim, device=ids.device, dtype=pooled.dtype)
        for index in range(pooled.shape[1]):
            features[:, index % self.input_dim] += pooled[:, index]
        norm = features.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return features / norm.clamp(min=1.0)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # features: (batch, input_dim), condition: (batch, condition_dim)
        batch = features.shape[0]
        base = (
            features @ self.in_weights.t()
            + condition @ self.cond_weights.t()
            + self.hidden_bias
        )  # (batch, hidden)

        slot_ids = torch.arange(self.max_response_codepoints, device=features.device)
        aggregate = torch.zeros(batch, self.agg_dim, device=features.device)
        logits = None
        for round_index in range(self.rounds):
            slot_bias = (
                self.slot_embedding[slot_ids] @ self.slot_weights.t()
            )  # (slots, hidden)
            round_bias = (
                self.round_embedding[round_index] @ self.round_weights.t()
            )  # (hidden,)
            hidden = torch.tanh(
                base.unsqueeze(1) + slot_bias.unsqueeze(0) + round_bias
            )  # (batch, slots, hidden)
            logits = (
                hidden @ self.out_weights.t() + self.out_bias
            )  # (batch, slots, class_count)
            # Aggregate slot distributions into the fixed-size channel.
            probabilities = torch.softmax(logits, dim=-1)
            flat = probabilities.reshape(batch, -1)
            aggregate = torch.zeros_like(aggregate)
            index = torch.arange(flat.shape[1], device=features.device) % self.agg_dim
            aggregate.scatter_add_(1, index.unsqueeze(0).expand(batch, -1), flat)
        return logits


def byte_loss(logits: torch.Tensor, targets: torch.Tensor,
              active: torch.Tensor) -> torch.Tensor:
    # logits: (batch, slots, class_count), targets/active: (batch, slots)
    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * active
    denominator = active.sum().clamp(min=1)
    return -gathered.sum() / denominator
