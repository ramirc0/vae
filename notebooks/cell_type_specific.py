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
        pl,
        plot_umap,
        read_labels,
        umap_of,
    )


@app.cell
def _(mo):
    mo.md("""
    # Cell type-specific b-cCREs

    VAE with a 7-class head over the Corces et al. 2020 brain cell
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
    train_button = mo.ui.run_button(label="Train 10 epochs")
    train_button
    return (train_button,)


@app.cell
def _(CHECKPOINTS, CLASSES, RESULTS, X, fit, labels, mo, train_button, vae_model):
    if train_button.value or "train" in mo.cli_args():
        history = fit(
            vae_model,
            X,
            labels.select(CLASSES).to_numpy(),
            epochs=10,
            checkpoints=CHECKPOINTS,
        )
        history.write_csv(RESULTS / "history.csv")
    weights = CHECKPOINTS / "10.safetensors"
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
