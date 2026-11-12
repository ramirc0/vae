"""The b-cCRE VAE classifier from Pratt et al. 2024, as in Greg Andrews's notebooks."""

import keras
import tensorflow as tf
from keras import layers

SEQUENCE_LENGTH = 512
LATENT_DIM = 256

LOSSES = {
    "kl_div": "mean_absolute_error",
    "classifier": "categorical_crossentropy",
    "decoder": "mean_squared_error",
}


def _sample_latent(distribution):
    mean, log_var = distribution
    return mean + tf.exp(0.5 * log_var) * tf.random.normal(tf.shape(log_var))


def _kl_divergence(distribution):
    mean, log_var = distribution
    return -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))


def build_vae(n_classes, activation):
    """Build the VAE with a classifier head on its sampled latent vector.

    The model returns a dict of six outputs: ``z_mu``, ``z_log_var``,
    ``latent_features`` (the sampled ``z``), ``kl_div``, ``decoder`` (the
    reconstructed one-hot sequence) and ``classifier``.

    Parameters
    ----------
    n_classes : int
        Number of classes the head predicts. The neuron/glia and fetal/adult
        models use 3. The Corces cell-type model uses 7.
    activation : str
        Activation of the classifier head. The 3-class models use
        ``"sigmoid"``. The 7-class model uses ``"softmax"``.

    Returns
    -------
    keras.Model
        Uncompiled model that takes one-hot sequences of shape
        ``(SEQUENCE_LENGTH, 4)``.

    Examples
    --------
    >>> model = build_vae(n_classes=3, activation="sigmoid")
    >>> {name: tuple(output.shape) for name, output in model.output.items()}
    ... # doctest: +NORMALIZE_WHITESPACE
    {'z_mu': (None, 256), 'z_log_var': (None, 256), 'latent_features': (None, 256),
     'kl_div': (None, 256), 'decoder': (None, 512, 4), 'classifier': (None, 3)}
    """
    sequence = keras.Input(shape=(SEQUENCE_LENGTH, 4))
    x = layers.Conv1D(256, 32, activation="relu", padding="same", name="conv")(sequence)
    x = layers.MaxPooling1D(8)(x)
    x = layers.Conv1D(128, 8, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    pooled_length, pooled_channels = x.shape[1:]
    x = layers.Flatten()(x)
    x = layers.Dense(1028)(x)
    x = layers.Dense(512)(x)

    z_mu = layers.Dense(LATENT_DIM, name="z_mu")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = layers.Lambda(_sample_latent, name="latent_features")([z_mu, z_log_var])
    kl_div = layers.Lambda(_kl_divergence, name="kl_div")([z_mu, z_log_var])
    classifier = layers.Dense(n_classes, activation=activation, name="classifier")(z)

    y = layers.Dense(512)(z)
    y = layers.Dense(1028)(y)
    y = layers.Dense(pooled_length * pooled_channels)(y)
    y = layers.Reshape((pooled_length, pooled_channels))(y)
    y = layers.UpSampling1D(2)(y)
    y = layers.Conv1DTranspose(128, 8, padding="same")(y)
    y = layers.UpSampling1D(8)(y)
    y = layers.Conv1DTranspose(SEQUENCE_LENGTH, 32, padding="same")(y)
    decoder = layers.Conv1DTranspose(
        4, 32, activation="relu", padding="same", name="decoder"
    )(y)

    outputs = {
        "z_mu": z_mu,
        "z_log_var": z_log_var,
        "latent_features": z,
        "kl_div": kl_div,
        "decoder": decoder,
        "classifier": classifier,
    }
    return keras.Model(sequence, outputs, name="vae")


def compile_vae(model):
    """Compile the VAE with the notebooks' losses, weights and optimizer.

    The KL output trains toward zeros with mean absolute error. The
    reconstruction uses mean squared error and the classifier categorical
    cross-entropy. All three losses carry weight 1. The optimizer is Adam
    with learning rate 1e-4.

    Parameters
    ----------
    model : keras.Model
        Model from `build_vae`. It is compiled in place.

    Returns
    -------
    keras.Model
        The same model, compiled.

    Examples
    --------
    >>> model = compile_vae(build_vae(n_classes=7, activation="softmax"))
    >>> model.compiled
    True
    """
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=LOSSES)
    return model
