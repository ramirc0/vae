# vae

Keras 3 port of the variational autoencoder (VAE) classifiers of brain cCREs (b-cCREs) from Pratt et al. 2024.

![VAE architecture](results/architecture.svg)

## Description

Pratt et al. trained VAEs to show how well DNA sequence alone separates subsets of b-cCREs. Each model encodes 512 bp of one-hot DNA around a b-cCRE center into a 256-dimensional latent space. A classifier head on the sampled latent vector trains jointly with the reconstruction and KL losses. A UMAP of the latent means then shows how the classes separate. There are three models:

| Notebook | Classes | Head |
| --- | --- | --- |
| `notebooks/neuron_glia.py` | neuron, glia, shared | 3, sigmoid |
| `notebooks/fetal_adult.py` | fetal, adult, shared | 3, sigmoid |
| `notebooks/cell_type_specific.py` | 7 Corces et al. 2020 brain cell types | 7, softmax |

The model and notebooks port Greg Andrews's TensorFlow 2.8 notebooks from the paper's [Zenodo record](https://zenodo.org/records/10605867) to Keras 3 and marimo. The shared code is in `src/vae/`. UMAP runs on the GPU with cuML when one is available.

## Getting Started

### Dependencies

* Linux with an NVIDIA GPU. Training and UMAP fall back to the CPU without one.
* [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).
* Access to the Weng lab servers. The notebooks read labels from `/data/projects/psychSCREEN/Zenodo/code/VAE` and the genome from `/data/common/genome/hg38.fa`.

### Installing

```
git clone https://github.com/ramirc0/vae.git
cd vae
uv sync --extra gpu
```

### Executing program

* Open a notebook in marimo:
```
just edit notebooks/neuron_glia.py
```
* Click **Train** to fit the model. Weights for each epoch go to `checkpoints/<model>/`.
* The notebook loads the epoch the original notebook used and writes the UMAP to `results/<model>/`.
* Redraw the architecture diagram:
```
uv run python scripts/architecture.py results/architecture.svg
```

## Help

Launch notebooks through `just`. It puts the CUDA libraries that uv installed ahead of Ubuntu's CUDA 12.0 libraries in `/usr/lib`. Without it, TensorFlow finds no GPUs and cuML fails to import. Check what TensorFlow sees with:
```
just --command uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Authors

Christian Ramirez, christian.ramirez1@umassmed.edu

The original model and notebooks are by Gregory Andrews.

## Acknowledgments

Pratt HE, Andrews G, Shedd N, et al. Using a comprehensive atlas and predictive models to reveal the complexity and evolution of brain-active regulatory elements. *Sci Adv* 10, eadj4452 (2024). https://doi.org/10.1126/sciadv.adj4452
