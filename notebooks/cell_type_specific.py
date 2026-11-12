import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import poikilos as pk
    import polars as pl
    from keras.callbacks import ModelCheckpoint

    from vae.latent import plot_umap, umap_of
    from vae.model import build_vae, compile_vae
    from vae.regions import encode_regions, read_labels

    pk.use()
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
    # Cell type-specific b-cCREs

    VAE with a 7-class softmax head over the Corces et al. 2020 brain cell
    types. It trains only on b-cCREs active in exactly one cell type. Port of
    `Cell-Type-Specific.ipynb` from Pratt et al. 2024. The UMAP leaves out
    nigral neurons, as the original does.
    """)
    return


@app.cell
def _(mo):
    CLASSES = [
        "Microglia",
        "Oligodendrocytes",
        "ExcitatoryNeurons",
        "NigralNeurons",
        "OPCs",
        "Astrocytes",
        "InhibitoryNeurons",
    ]
    # zlab single-cell-brain palette: Mic, Oli, Ex, OPC, Ast, In
    COLORS = {
        "Microglia": "#C7E9C0",
        "Oligodendrocytes": "#FF7F00",
        "ExcitatoryNeurons": "#E31A1C",
        "OPCs": "#FFD92F",
        "Astrocytes": "#33A02C",
        "InhibitoryNeurons": "#1F78B4",
    }
    CHECKPOINTS = mo.notebook_dir().parent / "checkpoints" / "cell-type-specific"
    RESULTS = mo.notebook_dir().parent / "results" / "cell-type-specific"
    RESULTS.mkdir(parents=True, exist_ok=True)
    return CHECKPOINTS, CLASSES, COLORS, RESULTS


@app.cell
def _(pl, read_labels):
    labels = read_labels("Corces.txt").filter(pl.col("nCells") == 1)
    labels
    return (labels,)


@app.cell
def _(CLASSES, labels):
    labels.select(CLASSES).sum()
    return


@app.cell
def _(CLASSES, build_vae, compile_vae):
    vae_model = compile_vae(build_vae(n_classes=len(CLASSES), activation="softmax"))
    vae_model.summary()
    return (vae_model,)


@app.cell
def _(encode_regions, labels):
    X = encode_regions(labels)
    X.shape
    return (X,)


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train 10 epochs")
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
            epochs=10,
            validation_split=0.1,
            callbacks=[
                ModelCheckpoint(
                    str(CHECKPOINTS / "{epoch:02d}.weights.h5"), save_weights_only=True
                )
            ],
        )
    weights = CHECKPOINTS / "10.weights.h5"
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
