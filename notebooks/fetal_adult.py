import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import poikilos as pk
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
        plot_umap,
        read_labels,
        umap_of,
    )


@app.cell
def _(mo):
    mo.md("""
    # Fetal vs adult b-cCREs

    VAE with a 3-class sigmoid head: fetal-specific, adult-specific and
    shared b-cCREs. Port of `Fetal-Adult.ipynb` from Pratt et al. 2024.
    """)
    return


@app.cell
def _(mo):
    CLASSES = ["fetal", "adult", "shared"]
    # zlab psychscreen-age palette: Fetal-specific, Adult-specific, Fetal and adult
    COLORS = {"fetal": "#92D050", "adult": "#FF0000", "shared": "#3D85C6"}
    CHECKPOINTS = mo.notebook_dir().parent / "checkpoints" / "fetal-adult"
    RESULTS = mo.notebook_dir().parent / "results" / "fetal-adult"
    RESULTS.mkdir(parents=True, exist_ok=True)
    return CHECKPOINTS, CLASSES, COLORS, RESULTS


@app.cell
def _(read_labels):
    labels = read_labels("VAE-Fetal-Adult.txt")
    labels
    return (labels,)


@app.cell
def _(CLASSES, labels):
    labels.select(CLASSES).sum()
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
