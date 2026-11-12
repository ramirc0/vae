"""The b-cCRE VAE classifier from Pratt et al. 2024, as in Greg Andrews's notebooks."""

import torch
import torch.nn.functional as F
from torch import nn

SEQUENCE_LENGTH = 512
LATENT_DIM = 256

_POOLED_CHANNELS = 128
_POOLED_LENGTH = SEQUENCE_LENGTH // 16


class _SameConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(
            in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        # An even kernel leaves one extra position. Keras "same" drops the last one.
        return super().forward(x)[..., : x.shape[-1]]


class VAE(nn.Module):
    """VAE with a classifier head on its sampled latent vector.

    Layers mirror the Keras model from the paper, channels first. The
    encoder's two dense layers and the decoder's three have no activation,
    as in the original. The latent vector is sampled in both training and
    evaluation mode.

    Parameters
    ----------
    n_classes : int
        Number of classes the head predicts. The neuron/glia and fetal/adult
        models use 3. The Corces cell-type model uses 7.

    Examples
    --------
    >>> model = VAE(n_classes=3)
    >>> sum(parameter.numel() for parameter in model.parameters())
    12465423
    >>> outputs = model(torch.zeros(2, 4, SEQUENCE_LENGTH))
    >>> {name: tuple(output.shape) for name, output in outputs.items()}
    ... # doctest: +NORMALIZE_WHITESPACE
    {'z_mu': (2, 256), 'z_log_var': (2, 256), 'latent_features': (2, 256),
     'kl_div': (2, 256), 'decoder': (2, 4, 512), 'classifier': (2, 3)}
    """

    def __init__(self, n_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 256, 32, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.Conv1d(256, _POOLED_CHANNELS, 8, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(_POOLED_CHANNELS * _POOLED_LENGTH, 1028),
            nn.Linear(1028, 512),
        )
        self.z_mu = nn.Linear(512, LATENT_DIM)
        self.z_log_var = nn.Linear(512, LATENT_DIM)
        self.classifier = nn.Linear(LATENT_DIM, n_classes)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.Linear(512, 1028),
            nn.Linear(1028, _POOLED_CHANNELS * _POOLED_LENGTH),
            nn.Unflatten(1, (_POOLED_CHANNELS, _POOLED_LENGTH)),
            nn.Upsample(scale_factor=2),
            _SameConvTranspose1d(_POOLED_CHANNELS, 128, 8),
            nn.Upsample(scale_factor=8),
            _SameConvTranspose1d(128, SEQUENCE_LENGTH, 32),
            _SameConvTranspose1d(SEQUENCE_LENGTH, 4, 32),
            nn.ReLU(),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, sequences):
        """Map sequences to the mean and log variance of their latent vectors.

        Parameters
        ----------
        sequences : torch.Tensor
            One-hot sequences of shape ``(n, 4, SEQUENCE_LENGTH)``.

        Returns
        -------
        z_mu, z_log_var : torch.Tensor
            Each of shape ``(n, LATENT_DIM)``.

        Examples
        --------
        >>> z_mu, z_log_var = VAE(n_classes=3).encode(torch.zeros(2, 4, SEQUENCE_LENGTH))
        >>> z_mu.shape, z_log_var.shape
        (torch.Size([2, 256]), torch.Size([2, 256]))
        """
        hidden = self.encoder(sequences)
        return self.z_mu(hidden), self.z_log_var(hidden)

    def forward(self, sequences):
        """Encode, sample, reconstruct and classify sequences.

        Parameters
        ----------
        sequences : torch.Tensor
            One-hot sequences of shape ``(n, 4, SEQUENCE_LENGTH)``.

        Returns
        -------
        dict of str to torch.Tensor
            The Keras model's six outputs. ``z_mu`` and ``z_log_var``
            describe the latent distribution. ``latent_features`` is the
            sampled ``z``. ``kl_div`` is the KL divergence from the standard
            normal per latent dimension. ``decoder`` is the reconstruction,
            of shape ``(n, 4, SEQUENCE_LENGTH)``. ``classifier`` holds the
            class logits.

        Examples
        --------
        >>> outputs = VAE(n_classes=7)(torch.zeros(2, 4, SEQUENCE_LENGTH))
        >>> outputs["classifier"].shape
        torch.Size([2, 7])
        """
        z_mu, z_log_var = self.encode(sequences)
        z = z_mu + torch.exp(0.5 * z_log_var) * torch.randn_like(z_log_var)
        return {
            "z_mu": z_mu,
            "z_log_var": z_log_var,
            "latent_features": z,
            "kl_div": -0.5 * (1 + z_log_var - z_mu**2 - torch.exp(z_log_var)),
            "decoder": self.decoder(z),
            "classifier": self.classifier(z),
        }


def loss_terms(outputs, sequences, targets):
    """Compute the training loss and its three terms, weighted 1:1:1.

    The KL term is the mean absolute KL divergence, which is how Keras's mean
    absolute error against zeros trained it. The reconstruction term is the
    mean squared error. The classifier term is softmax cross-entropy on the
    logits.

    Parameters
    ----------
    outputs : dict of str to torch.Tensor
        Output of `VAE.forward`.
    sequences : torch.Tensor
        The one-hot sequences the model reconstructs.
    targets : torch.Tensor
        Float one-hot class labels of shape ``(n, n_classes)``.

    Returns
    -------
    dict of str to torch.Tensor
        Scalars named as in Keras's history: ``loss`` (the sum),
        ``kl_div_loss``, ``decoder_loss`` and ``classifier_loss``.

    Examples
    --------
    >>> sequences = torch.zeros(2, 4, SEQUENCE_LENGTH)
    >>> outputs = VAE(n_classes=3)(sequences)
    >>> terms = loss_terms(outputs, sequences, torch.eye(3)[:2])
    >>> list(terms)
    ['loss', 'kl_div_loss', 'decoder_loss', 'classifier_loss']
    >>> bool(terms["loss"] == sum(list(terms.values())[1:]))
    True
    """
    kl_div = outputs["kl_div"].abs().mean()
    decoder = F.mse_loss(outputs["decoder"], sequences)
    classifier = F.cross_entropy(outputs["classifier"], targets)
    return {
        "loss": kl_div + decoder + classifier,
        "kl_div_loss": kl_div,
        "decoder_loss": decoder,
        "classifier_loss": classifier,
    }
