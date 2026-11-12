import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import poikilos as pk
    import polars as pl
    from safetensors.torch import load_file

    from vae.latent import plot_umap, umap_of
    from vae.learning_curves import plot_learning_curves
    from vae.model import VAE
    from vae.regions import encode_regions, read_labels
    from vae.training import best_epoch, first_validation_row, fit, latent_means_of

    pk.use()
    return (
        VAE,
        best_epoch,
        encode_regions,
        first_validation_row,
        fit,
        latent_means_of,
        load_file,
        mo,
        pl,
        plot_learning_curves,
        plot_umap,
        read_labels,
        umap_of,
    )


@app.cell
def _(mo):
    mo.md("""
    # Fetal vs adult b-cCREs

    VAE with a 3-class head: fetal-specific, adult-specific and
    shared b-cCREs. Based on `Fetal-Adult.ipynb` from Pratt et al. 2024.
    The UMAP shows the held-out rows.
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
    train_button = mo.ui.run_button(label="Train 30 epochs")
    train_button
    return (train_button,)


@app.cell
def _(
    CHECKPOINTS,
    CLASSES,
    RESULTS,
    X,
    best_epoch,
    fit,
    labels,
    mo,
    pl,
    train_button,
    vae_model,
):
    if train_button.value or "train" in mo.cli_args():
        history = fit(
            vae_model,
            X,
            labels.select(CLASSES).to_numpy(),
            epochs=30,
            checkpoints=CHECKPOINTS,
        )
        history.write_csv(RESULTS / "history.csv")
    else:
        history = pl.read_csv(RESULTS / "history.csv")
    loaded_epoch = best_epoch(history)
    weights = CHECKPOINTS / f"{loaded_epoch:02d}.safetensors"
    return history, loaded_epoch, weights


@app.cell
def _(RESULTS, history, loaded_epoch, plot_learning_curves):
    plot_learning_curves(history, loaded_epoch, RESULTS / "learning-curves")
    return


@app.cell
def _(X, first_validation_row, latent_means_of, load_file, mo, vae_model, weights):
    mo.stop(not weights.exists(), mo.md(f"No checkpoint at `{weights}`. Train first."))
    vae_model.load_state_dict(load_file(weights))
    held_out = slice(first_validation_row(len(X)), None)
    latent_means = latent_means_of(vae_model, X[held_out])
    return held_out, latent_means


@app.cell
def _(latent_means, umap_of):
    embedding = umap_of(latent_means)
    return (embedding,)


@app.cell
def _(COLORS, RESULTS, embedding, held_out, labels, plot_umap):
    plot_umap(embedding, labels[held_out], COLORS, RESULTS / "umap")
    return


@app.cell
def _(COLORS, RESULTS, embedding, held_out, labels, plot_umap):
    plot_umap(embedding, labels[held_out], COLORS, RESULTS / "umap-bare", axes=False)
    return


if __name__ == "__main__":
    app.run()
