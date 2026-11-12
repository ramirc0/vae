"""Project VAE latent means to 2D and plot them by class."""

from importlib.util import find_spec

import matplotlib.pyplot as plt
import poikilos as pk
from sklearn.decomposition import PCA


def _umap():
    if find_spec("cuml"):
        import cupy

        if cupy.cuda.is_available():
            from cuml.manifold import UMAP

            return UMAP()
    from umap import UMAP

    return UMAP()


def umap_of(latent_means, n_components=64):
    """Embed latent means in 2D with PCA, then UMAP.

    UMAP runs on the GPU with cuML when the ``gpu`` extra is installed and a
    GPU is visible. Otherwise it runs on the CPU with umap-learn. The two
    give different embeddings for the same input.

    Parameters
    ----------
    latent_means : numpy.ndarray
        Array of shape ``(n_regions, latent_dim)``, the model's ``z_mu``.
    n_components : int, optional
        Number of principal components passed to UMAP.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_regions, 2)``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> umap_of(rng.normal(size=(100, 16)), n_components=8).shape
    (100, 2)
    """
    components = PCA(n_components=n_components).fit_transform(latent_means)
    return _umap().fit_transform(components)


def plot_umap(embedding, labels, colors, path, axes=True):
    """Scatter a 2D embedding, one color per class, and save it.

    A region in several classes is drawn once per class, in the order of
    `colors`, so later classes cover earlier ones.

    Parameters
    ----------
    embedding : numpy.ndarray
        Array of shape ``(n_regions, 2)`` from `umap_of`.
    labels : polars.DataFrame
        One 0/1 column per class, one row per region.
    colors : dict of str to str
        Hex color per class column. Classes left out are not drawn.
    path : str or pathlib.Path
        Output path without extension. SVG and PNG are written.
    axes : bool, optional
        Draw axes and a legend. Set False for the bare point cloud.

    Returns
    -------
    matplotlib.figure.Figure
        The saved figure.

    Examples
    --------
    >>> import numpy as np, polars as pl, tempfile
    >>> labels = pl.DataFrame({"a": [1, 0], "b": [0, 1]})
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     fig = plot_umap(np.eye(2), labels, {"a": "#FF0000", "b": "#0000FF"}, f"{tmp}/u")
    >>> len(fig.axes[0].collections)
    2
    """
    with pk.context():
        fig, ax = plt.subplots()
        for name, color in colors.items():
            members = labels[name].to_numpy() == 1
            ax.scatter(
                *embedding[members].T,
                s=0.1,
                c=color,
                alpha=0.5,
                linewidths=0,
                label=name,
                rasterized=True,
            )
        ax.set_box_aspect(1)
        if axes:
            ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
            fig.legend(loc="outside right upper", markerscale=10)
            pk.despine(ax, offset=7.5)
            pk.fit_legend(fig)
        else:
            ax.set_axis_off()
        pk.save_figure(fig, path)
    return fig
