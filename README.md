# vae

Variational autoencoder (VAE) classifiers of brain cCREs (b-cCREs) in PyTorch, based on the models of Pratt et al. 2024.

![VAE architecture](results/architecture.svg)

## Description

Pratt et al. trained VAEs to show how well DNA sequence alone separates subsets of b-cCREs. Each model encodes 512 bp of one-hot DNA around a b-cCRE center into a 256-dimensional latent space. A classifier head on the latent vector trains jointly with the reconstruction and KL losses. A UMAP of the latent means of held-out b-cCREs then shows how the classes separate. There are three models:

| Notebook | Classes |
| --- | --- |
| `notebooks/neuron_glia.py` | neuron, glia, shared |
| `notebooks/fetal_adult.py` | fetal, adult, shared |
| `notebooks/cell_type_specific.py` | 7 Corces et al. 2020 brain cell types |

The model and notebooks start from Greg Andrews's TensorFlow 2.8 notebooks in the paper's [Zenodo record](https://zenodo.org/records/10605867). The model keeps the original layers and changes how it trains:

* A ReLU follows every dense layer and every inner transposed convolution. In the original, these layers are linear.
* Dropout of 0.5 precedes the first dense layer.
* The decoder returns per-base logits. The loss is the negative ELBO per base (reconstruction cross-entropy plus KL) plus the classifier's softmax cross-entropy. The original summed MSE, the mean absolute KL and the classifier loss.
* AdamW trains it with learning rate 3e-4, weight decay 0.1, batches of 128 and reverse-complement augmentation, for 30 epochs.
* The notebook loads the epoch with the lowest validation classifier loss and draws the UMAP from the held-out rows only.

The original setup overfits after about epoch 4. At the epochs it loads, the UMAP of all rows mostly separates memorized training rows. The validation split holds out the last 10% of rows, as Keras does: the tail of chr8, then chr9, chrX and chrY. On those rows, balanced accuracy is:

| Model | Original setup, best epoch | This model | 6-mer logistic regression |
| --- | --- | --- | --- |
| neuron, glia, shared | 0.62 | 0.71 | 0.64 |
| fetal, adult, shared | 0.65 | 0.70 | 0.62 |
| 7 cell types | 0.29 | 0.43 | 0.39 |

The decoder recovers little of the sequence, about 1.96 bits per base out of 2. DNA barely compresses, so the latent holds mostly the class and GC content.

The shared code is in `src/vae/`. `model.py` holds the model and its loss. `training.py` holds the training loop and picks the epoch to load. `latent.py` draws the UMAP, on the GPU with cuML when one is available. `learning_curves.py` plots the metrics by epoch.

## Getting Started

### Dependencies

* Linux with an NVIDIA GPU and a driver for CUDA 13.0 or later. Training and UMAP fall back to the CPU without one.
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
* Click **Train** to fit the model. Weights for each epoch go to `checkpoints/<model>/` as safetensors files. The loss history goes to `results/<model>/history.csv`, and its learning curves to `results/<model>/learning-curves.svg`, one panel per metric.
* Or train from the command line:
```
just run notebooks/neuron_glia.py --train
```
* The notebook loads the epoch with the lowest validation classifier loss and writes the UMAP of held-out rows to `results/<model>/`. The repository includes those three checkpoints, so the notebooks run without training. Training overwrites them, and it can pick a different epoch. Update the checkpoint exceptions in `.gitignore` when it does.
* Redraw the architecture diagram:
```
uv run --extra gpu python scripts/architecture.py results/architecture.svg
```

## Help

Pass `--extra gpu` to every `uv run` and `uv sync`. Without it, uv removes cuML from the environment. On a shared host, pick idle GPUs with `CUDA_VISIBLE_DEVICES`. Check what PyTorch sees with:
```
uv run --extra gpu python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Authors

Christian Ramirez, christian.ramirez1@umassmed.edu

The original model and notebooks are by Gregory Andrews.

## Acknowledgments

Pratt HE, Andrews G, Shedd N, et al. Using a comprehensive atlas and predictive models to reveal the complexity and evolution of brain-active regulatory elements. *Sci Adv* 10, eadj4452 (2024). https://doi.org/10.1126/sciadv.adj4452
