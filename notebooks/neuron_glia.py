import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from keras.callbacks import ModelCheckpoint

    from vae.latent import plot_umap, umap_of
    from vae.model import build_vae, compile_vae
    from vae.regions import encode_regions, read_labels

    return (
        ModelCheckpoint,
        build_vae,
        compile_vae,
        encode_regions,
        mo,
        np,
        pl,
        plot_umap,
        read_labels,
        umap_of,
    )


@app.cell
def _(mo):
    mo.md("""
    # Neuron vs glia b-cCREs

    VAE with a 3-class sigmoid head: neuron-specific, glia-specific and
    shared b-cCREs. Port of `Neuron-Glia.ipynb` from Pratt et al. 2024.
    """)
    return


@app.cell
def _(mo):
    CLASSES = ["neuron", "glia", "shared"]
    # zlab psychscreen-neun palette: NeuN+ only, NeuN- only, NeuN+ and NeuN-
    COLORS = {"neuron": "#6D6E71", "glia": "#8856A7", "shared": "#3E85C6"}
    CHECKPOINTS = mo.notebook_dir().parent / "checkpoints" / "neuron-glia"
    RESULTS = mo.notebook_dir().parent / "results" / "neuron-glia"
    RESULTS.mkdir(parents=True, exist_ok=True)
    return CHECKPOINTS, CLASSES, COLORS, RESULTS


@app.cell
def _(read_labels):
    labels = read_labels("VAE-Neuron-Glia.txt")
    labels
    return (labels,)


@app.cell
def _(CLASSES, labels, pl):
    labels.filter(pl.col("n") == 1).select(CLASSES).sum()
    return


@app.cell
def _(CLASSES, build_vae, compile_vae):
    vae_model = compile_vae(build_vae(n_classes=len(CLASSES), activation="sigmoid"))
    vae_model.summary()
    return (vae_model,)


@app.cell
def _(encode_regions, labels):
    X = encode_regions(labels)
    X.shape
    return (X,)


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train 20 epochs")
    train_button
    return (train_button,)


@app.cell
def _(CHECKPOINTS, CLASSES, ModelCheckpoint, X, labels, np, train_button, vae_model):
    if train_button.value:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        vae_model.fit(
            X,
            {
                "decoder": X,
                "kl_div": np.zeros((len(X), 1)),
                "classifier": labels.select(CLASSES).to_numpy(),
            },
            epochs=20,
            validation_split=0.1,
            callbacks=[
                ModelCheckpoint(
                    str(CHECKPOINTS / "{epoch:02d}.weights.h5"), save_weights_only=True
                )
            ],
        )
    weights = CHECKPOINTS / "14.weights.h5"
    return (weights,)


@app.cell
def _(X, mo, vae_model, weights):
    mo.stop(not weights.exists(), mo.md(f"No checkpoint at `{weights}`. Train first."))
    vae_model.load_weights(weights)
    latent_means = vae_model.predict(X)["z_mu"]
    return (latent_means,)


@app.cell
def _(latent_means, umap_of):
    embedding = umap_of(latent_means)
    return (embedding,)


@app.cell
def _(COLORS, RESULTS, embedding, labels, plot_umap):
    plot_umap(embedding, labels, COLORS, RESULTS / "umap")
    return


@app.cell
def _(COLORS, RESULTS, embedding, labels, plot_umap):
    plot_umap(embedding, labels, COLORS, RESULTS / "umap-bare", axes=False)
    return


if __name__ == "__main__":
    app.run()
