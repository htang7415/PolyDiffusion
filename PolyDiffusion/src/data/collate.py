"""Batch collation helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch


def collate_token_batch(
    sequences: Iterable[Sequence[int]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pad a batch of token id sequences into a tensor.

    Args:
        sequences: Iterable of token id sequences (BOS/EOS already handled upstream).
        pad_token_id: Token id used for padding.

    Returns:
        Dictionary containing:
            ``tokens`` (LongTensor): shape (batch, max_len)
            ``mask`` (BoolTensor): shape (batch, max_len) marking valid tokens
            ``lengths`` (LongTensor): lengths prior to padding
    """
    seq_list: List[Sequence[int]] = [tuple(seq) for seq in sequences]
    if not seq_list:
        raise ValueError("collate_token_batch received an empty batch.")

    max_len = max(len(seq) for seq in seq_list)
    batch_size = len(seq_list)

    tokens = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for idx, seq in enumerate(seq_list):
        seq_len = len(seq)
        lengths[idx] = seq_len
        if seq_len == 0:
            continue
        tokens[idx, :seq_len] = torch.tensor(seq, dtype=torch.long)
        mask[idx, :seq_len] = True

    return {"tokens": tokens, "mask": mask, "lengths": lengths}
