"""Train the VAE as Keras's ``fit`` did and read out its latent means."""

import math
import time
from pathlib import Path

import polars as pl
import torch
from safetensors.torch import save_file

from vae.model import loss_terms

VALIDATION_SPLIT = 0.1


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mean_loss_terms(model, sequences, targets, batches, optimizer=None):
    totals = {}
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches:
            batch_sequences = sequences[batch]
            terms = loss_terms(model(batch_sequences), batch_sequences, targets[batch])
            if optimizer is not None:
                optimizer.zero_grad()
                terms["loss"].backward()
                optimizer.step()
            for name, value in terms.items():
                totals[name] = totals.get(name, 0) + value.detach() * len(batch)
    return {name: float(total) / len(sequences) for name, total in totals.items()}


def fit(model, sequences, targets, epochs, checkpoints, batch_size=32):
    """Train with Adam and save the weights after every epoch.

    This mirrors Keras ``fit(..., validation_split=0.1)``. The last 10% of
    rows, unshuffled, are held out for validation. The training rows are
    reshuffled each epoch, and the last partial batch is kept. Adam uses
    learning rate 1e-4 and Keras's epsilon of 1e-7. Training runs on the GPU
    when one is visible.

    Parameters
    ----------
    model : vae.model.VAE
        Model to train in place.
    sequences : numpy.ndarray
        One-hot sequences of shape ``(n, 4, SEQUENCE_LENGTH)``.
    targets : numpy.ndarray
        One-hot class labels of shape ``(n, n_classes)``.
    epochs : int
        Number of passes over the training rows.
    checkpoints : str or pathlib.Path
        Directory for the weights. Epoch ``e`` goes to
        ``f"{e:02d}.safetensors"``, counting from 1. It is created if needed.
    batch_size : int, optional
        Rows per optimizer step.

    Returns
    -------
    polars.DataFrame
        One row per epoch with Keras's history columns: ``epoch``, the
        training ``loss``, ``kl_div_loss``, ``decoder_loss`` and
        ``classifier_loss``, and the same four with a ``val_`` prefix. Each
        is a mean over rows.

    Examples
    --------
    >>> import tempfile
    >>> import numpy as np
    >>> from vae.model import SEQUENCE_LENGTH, VAE
    >>> sequences = np.zeros((20, 4, SEQUENCE_LENGTH), dtype=np.float32)
    >>> targets = np.eye(3)[np.arange(20) % 3]
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     history = fit(VAE(n_classes=3), sequences, targets, 1, tmp, batch_size=8)
    ...     sorted(path.name for path in Path(tmp).iterdir())  # doctest: +ELLIPSIS
    epoch 1/1 ...
    ['01.safetensors']
    >>> history.columns
    ... # doctest: +NORMALIZE_WHITESPACE
    ['epoch', 'loss', 'kl_div_loss', 'decoder_loss', 'classifier_loss',
     'val_loss', 'val_kl_div_loss', 'val_decoder_loss', 'val_classifier_loss']
    """
    device = _device()
    model.to(device)
    sequences = torch.as_tensor(sequences, device=device)
    targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
    n_train = math.floor(len(sequences) * (1.0 - VALIDATION_SPLIT))
    validation_batches = torch.arange(len(sequences) - n_train).split(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)
    checkpoints = Path(checkpoints)
    checkpoints.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        shuffled = torch.randperm(n_train, device=device).split(batch_size)
        train = _mean_loss_terms(
            model, sequences[:n_train], targets[:n_train], shuffled, optimizer
        )
        model.eval()
        validation = _mean_loss_terms(
            model,
            sequences[n_train:],
            targets[n_train:],
            validation_batches,
        )
        save_file(model.state_dict(), checkpoints / f"{epoch:02d}.safetensors")
        history.append(
            {"epoch": epoch}
            | train
            | {f"val_{name}": value for name, value in validation.items()}
        )
        print(
            f"epoch {epoch}/{epochs}",
            *(
                f"{name}={value:.4f}"
                for name, value in history[-1].items()
                if name != "epoch"
            ),
            f"time={time.perf_counter() - start:.0f}s",
            flush=True,
        )
    return pl.DataFrame(history)


@torch.inference_mode()
def latent_means_of(model, sequences, batch_size=1024):
    """Encode sequences to the means of their latent vectors.

    Parameters
    ----------
    model : vae.model.VAE
        Trained model. It runs on the GPU when one is visible.
    sequences : numpy.ndarray
        One-hot sequences of shape ``(n, 4, SEQUENCE_LENGTH)``.
    batch_size : int, optional
        Rows per forward pass.

    Returns
    -------
    numpy.ndarray
        Float32 array of ``z_mu``, of shape ``(n, LATENT_DIM)``.

    Examples
    --------
    >>> import numpy as np
    >>> from vae.model import SEQUENCE_LENGTH, VAE
    >>> sequences = np.zeros((5, 4, SEQUENCE_LENGTH), dtype=np.float32)
    >>> latent_means_of(VAE(n_classes=3), sequences, batch_size=2).shape
    (5, 256)
    """
    device = _device()
    model.to(device).eval()
    means = [
        model.encode(
            torch.as_tensor(sequences[start : start + batch_size], device=device)
        )[0]
        for start in range(0, len(sequences), batch_size)
    ]
    return torch.cat(means).cpu().numpy()
