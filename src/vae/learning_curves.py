"""Plot training and validation metrics by epoch."""

import matplotlib.pyplot as plt
import poikilos as pk


def plot_learning_curves(history, path):
    """Plot each metric by epoch in its own panel and save the figure.

    Every column with a ``val_`` twin is a metric. Its panel draws the
    training value as a solid line and the validation value as a dashed
    one. Panels keep their own y scale. A grey line marks the epoch with
    the lowest validation value of the first metric. Apply a poikilos theme
    with ``pk.use()`` first.

    Parameters
    ----------
    history : polars.DataFrame
        One row per epoch, as `vae.training.fit` returns. It needs an
        ``epoch`` column.
    path : str or pathlib.Path
        Output path without extension. SVG and PNG are written.

    Returns
    -------
    matplotlib.figure.Figure
        The saved figure.

    Examples
    --------
    >>> import polars as pl, tempfile
    >>> history = pl.DataFrame(
    ...     {"epoch": [1, 2], "loss": [2.0, 1.0], "val_loss": [2.5, 2.0], "kl": [0.1, 0.2], "val_kl": [0.1, 0.3]}
    ... )
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     fig = plot_learning_curves(history, f"{tmp}/curves")
    >>> [ax.get_title() for ax in fig.axes]
    ['loss', 'kl']
    """
    metrics = [name for name in history.columns if f"val_{name}" in history.columns]
    best_epoch = history["epoch"][history[f"val_{metrics[0]}"].arg_min()]
    colors = pk.PALETTES["zed"].colors
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(2.6 * len(metrics), 2.6), squeeze=False
    )
    for ax, metric in zip(axes[0], metrics):
        ax.axvline(best_epoch, color=colors["silver"], linewidth=0.5, zorder=0)
        ax.plot(history["epoch"], history[metric], color=colors["black"], label="train")
        ax.plot(
            history["epoch"],
            history[f"val_{metric}"],
            color=colors["red"],
            linestyle="--",
            label="validation",
        )
        ax.set(title=metric, xlabel="Epoch")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside right upper")
    pk.fit_legend(fig)
    for ax in axes[0]:
        pk.despine(ax, offset=7.5)
    pk.save_figure(fig, path)
    return fig
