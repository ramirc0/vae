import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import poikilos as pk
    from safetensors.torch import load_file

    from vae.latent import plot_umap, umap_of
    from vae.model import VAE
    from vae.regions import encode_regions, read_labels
    from vae.training import fit, latent_means_of

    pk.use()
    return (
        VAE,
        encode_regions,
        fit,
        latent_means_of,
        load_file,
        mo,
        plot_umap,
        read_labels,
        umap_of,
    )


@app.cell
def _(mo):
    mo.md("""
    # Fetal vs adult b-cCREs

    VAE with a 3-class head: fetal-specific, adult-specific and
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
def _(CLASSES, VAE):
    vae_model = VAE(n_classes=len(CLASSES))
    vae_model
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
def _(CHECKPOINTS, CLASSES, RESULTS, X, fit, labels, mo, train_button, vae_model):
    if train_button.value or "train" in mo.cli_args():
        history = fit(
            vae_model,
            X,
            labels.select(CLASSES).to_numpy(),
            epochs=20,
            checkpoints=CHECKPOINTS,
        )
        history.write_csv(RESULTS / "history.csv")
    weights = CHECKPOINTS / "14.safetensors"
    return (weights,)


@app.cell
def _(X, latent_means_of, load_file, mo, vae_model, weights):
    mo.stop(not weights.exists(), mo.md(f"No checkpoint at `{weights}`. Train first."))
    vae_model.load_state_dict(load_file(weights))
    latent_means = latent_means_of(vae_model, X)
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
