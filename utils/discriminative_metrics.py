"""Discriminative score metric utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam


class Discriminator(nn.Module):
    """Simple GRU based discriminator used for the discriminative score."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, sequences: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor, None]:
        lengths_cpu = lengths.detach().cpu()
        packed = pack_padded_sequence(
            sequences, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, last_hidden = self.gru(packed)
        last_hidden = last_hidden[-1]
        logits = self.output(last_hidden)
        probabilities = torch.sigmoid(logits)
        return logits, probabilities, None


def extract_time(data: Sequence[np.ndarray]) -> Tuple[np.ndarray, int]:
    lengths = []
    max_seq_len = 0
    for sequence in data:
        array = np.asarray(sequence)
        if array.ndim == 1:
            array = np.expand_dims(array, -1)
        if array.size == 0:
            length = 0
        else:
            row_sums = np.abs(array).sum(axis=-1)
            non_padding = np.where(row_sums > 0)[0]
            length = int(non_padding[-1] + 1) if non_padding.size > 0 else array.shape[0]
        lengths.append(max(1, length))
        max_seq_len = max(max_seq_len, array.shape[0])
    return np.asarray(lengths, dtype=np.int64), max_seq_len


def _ensure_numpy_array(data: Iterable, max_seq_len: int) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim == 3 and array.dtype != object and array.shape[1] == max_seq_len:
        return array.astype(np.float32)

    padded = []
    for sequence in data:
        seq_array = np.asarray(sequence, dtype=np.float32)
        if seq_array.ndim == 1:
            seq_array = np.expand_dims(seq_array, -1)
        if seq_array.shape[0] > max_seq_len:
            raise ValueError("Sequence length exceeds maximum sequence length.")
        pad_len = max_seq_len - seq_array.shape[0]
        if pad_len > 0:
            padding = np.zeros((pad_len, seq_array.shape[-1]), dtype=np.float32)
            seq_array = np.concatenate([seq_array, padding], axis=0)
        padded.append(seq_array)
    return np.stack(padded, axis=0)


@dataclass
class TrainTestSplit:
    train_x: np.ndarray
    train_x_hat: np.ndarray
    test_x: np.ndarray
    test_x_hat: np.ndarray
    train_t: np.ndarray
    train_t_hat: np.ndarray
    test_t: np.ndarray
    test_t_hat: np.ndarray


def train_test_divide(
    ori_data: np.ndarray,
    generated_data: np.ndarray,
    ori_time: np.ndarray,
    generated_time: np.ndarray,
) -> TrainTestSplit:
    num_samples = ori_data.shape[0]
    if num_samples != generated_data.shape[0]:
        raise ValueError("Original and generated data must have the same number of samples.")
    if num_samples < 2:
        raise ValueError("Need at least two sequences to compute the discriminative score.")
    split = int(0.8 * num_samples)
    split = min(max(split, 1), num_samples - 1)

    return TrainTestSplit(
        train_x=ori_data[:split],
        train_x_hat=generated_data[:split],
        test_x=ori_data[split:],
        test_x_hat=generated_data[split:],
        train_t=ori_time[:split],
        train_t_hat=generated_time[:split],
        test_t=ori_time[split:],
        test_t_hat=generated_time[split:],
    )


def batch_generator(
    data: np.ndarray, time: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    num_samples = data.shape[0]
    if num_samples == 0:
        raise ValueError("Cannot generate a batch from an empty dataset.")

    indices = np.random.permutation(num_samples)[:batch_size]
    return data[indices], time[indices]


def discriminative_score_metrics(
    ori_data: Sequence[Sequence[float]],
    generated_data: Sequence[Sequence[float]],
    iterations: int = 2000,
    batch_size: int = 128,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ori_sequences = [np.asarray(sequence) for sequence in ori_data]
    generated_sequences = [np.asarray(sequence) for sequence in generated_data]

    if len(ori_sequences) == 0 or len(generated_sequences) == 0:
        raise ValueError("Original and generated data must be non-empty.")
    if len(ori_sequences) != len(generated_sequences):
        raise ValueError("Original and generated data must contain the same number of sequences.")

    ori_time, ori_max_seq_len = extract_time(ori_sequences)
    generated_time, generated_max_seq_len = extract_time(generated_sequences)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)
    if max_seq_len == 0:
        raise ValueError("Sequences must contain at least one timestep.")

    ori_array = _ensure_numpy_array(ori_sequences, max_seq_len)
    generated_array = _ensure_numpy_array(generated_sequences, max_seq_len)

    if ori_array.shape[-1] == 1:
        ori_array = np.concatenate([ori_array, ori_array], axis=-1)
        generated_array = np.concatenate([generated_array, generated_array], axis=-1)

    _, _, feature_dim = ori_array.shape
    hidden_dim = max(1, feature_dim // 2)

    discriminator_model = Discriminator(feature_dim, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(discriminator_model.parameters())

    split = train_test_divide(ori_array, generated_array, ori_time, generated_time)

    discriminator_model.train()
    for _ in range(iterations):
        batch_x, batch_t = batch_generator(split.train_x, split.train_t, batch_size)
        batch_x_hat, batch_t_hat = batch_generator(
            split.train_x_hat, split.train_t_hat, batch_size
        )

        real_sequences = torch.from_numpy(batch_x).to(device)
        real_lengths = torch.from_numpy(batch_t).to(device)
        fake_sequences = torch.from_numpy(batch_x_hat).to(device)
        fake_lengths = torch.from_numpy(batch_t_hat).to(device)

        logits_real, _, _ = discriminator_model(real_sequences, real_lengths)
        logits_fake, _, _ = discriminator_model(fake_sequences, fake_lengths)

        loss_real = criterion(logits_real, torch.ones_like(logits_real))
        loss_fake = criterion(logits_fake, torch.zeros_like(logits_fake))
        loss = loss_real + loss_fake

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    discriminator_model.eval()
    with torch.no_grad():
        test_x = torch.from_numpy(split.test_x).to(device)
        test_t = torch.from_numpy(split.test_t).to(device)
        test_x_hat = torch.from_numpy(split.test_x_hat).to(device)
        test_t_hat = torch.from_numpy(split.test_t_hat).to(device)

        logits_real, _, _ = discriminator_model(test_x, test_t)
        logits_fake, _, _ = discriminator_model(test_x_hat, test_t_hat)

    probabilities = torch.sigmoid(torch.cat([logits_real, logits_fake], dim=0))
    y_pred = probabilities.cpu().numpy().squeeze()
    labels = np.concatenate(
        [np.ones(len(logits_real)), np.zeros(len(logits_fake))], axis=0
    )

    accuracy = accuracy_score(labels, y_pred > 0.5)
    return float(np.abs(0.5 - accuracy))
