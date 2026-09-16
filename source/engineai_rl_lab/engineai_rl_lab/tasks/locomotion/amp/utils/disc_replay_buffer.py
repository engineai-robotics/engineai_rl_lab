"""Replay buffer holding past policy discriminator observations.

Mirrors MimicKit's ``ExperienceBuffer(buffer_length=disc_buffer_size, batch_size=1)``,
simplified to a single flat ring buffer over arbitrary sample shapes.
"""

from __future__ import annotations

import torch


class DiscReplayBuffer:
    """Ring buffer of discriminator observations sampled from earlier policies.

    The storage is allocated lazily on the first :meth:`push` so the same buffer works for
    the 3D ``[history, frame]`` observations and the 2D flattened-with-condition layout.
    """

    def __init__(self, capacity: int, device: str = "cpu") -> None:
        if capacity <= 0:
            raise ValueError(f"The replay buffer capacity must be positive, got {capacity}.")
        self.capacity = int(capacity)
        self.device = device
        self._buffer: torch.Tensor | None = None
        self._head = 0
        self._total_samples = 0

    @property
    def sample_count(self) -> int:
        """Number of valid samples currently stored."""
        return min(self._total_samples, self.capacity)

    @property
    def total_samples(self) -> int:
        """Number of samples ever pushed into the buffer."""
        return self._total_samples

    def is_full(self) -> bool:
        """Whether the buffer has seen at least ``capacity`` samples."""
        return self._total_samples >= self.capacity

    def push(self, data: torch.Tensor) -> None:
        """Ring-write ``data`` (shape ``[n, ...]``), wrapping around the buffer end."""
        if data.ndim < 2:
            raise ValueError(f"Expected samples with shape [n, ...], got {tuple(data.shape)}.")
        num_samples = data.shape[0]
        if num_samples == 0:
            return
        if num_samples > self.capacity:
            raise ValueError(
                f"Cannot push {num_samples} samples into a replay buffer of capacity {self.capacity}."
            )

        if self._buffer is None:
            self._buffer = torch.zeros(
                (self.capacity, *data.shape[1:]), dtype=data.dtype, device=self.device
            )
        elif tuple(self._buffer.shape[1:]) != tuple(data.shape[1:]):
            raise ValueError(
                f"Sample shape {tuple(data.shape[1:])} does not match the buffer's "
                f"{tuple(self._buffer.shape[1:])}."
            )

        data = data.detach().to(device=self.device, dtype=self._buffer.dtype)
        head_samples = min(num_samples, self.capacity - self._head)
        self._buffer[self._head : self._head + head_samples] = data[:head_samples]
        remainder = num_samples - head_samples
        if remainder > 0:
            self._buffer[:remainder] = data[head_samples:]

        self._head = (self._head + num_samples) % self.capacity
        self._total_samples += num_samples

    def sample(self, num_samples: int) -> torch.Tensor:
        """Draw ``num_samples`` stored samples uniformly with replacement."""
        if self._buffer is None or self.sample_count == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = torch.randint(0, self.sample_count, (num_samples,), device=self.device)
        return self._buffer[indices]
