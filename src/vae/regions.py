"""Read labelled b-cCREs and one-hot encode the sequence around their centers."""

from pathlib import Path

import numpy as np
import polars as pl
from pyfaidx import Fasta

from vae.model import SEQUENCE_LENGTH

LABELS_DIR = Path("/data/projects/psychSCREEN/Zenodo/code/VAE")
HG38 = Path("/data/common/genome/hg38.fa")

_ONE_HOT = np.zeros((256, 4), dtype=np.float32)
_ONE_HOT[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.eye(4)
_ONE_HOT[ord("N")] = 0.25


def read_labels(filename):
    """Read a b-cCRE labels file from the paper's Zenodo VAE folder.

    Parameters
    ----------
    filename : str
        Name of a file in `LABELS_DIR`, such as ``"VAE-Neuron-Glia.txt"``.
        Each row is a b-cCRE: ``chrom``, ``start``, ``stop`` and ``rDHS``,
        then one 0/1 column per class.

    Returns
    -------
    polars.DataFrame
        One row per b-cCRE.

    Examples
    --------
    >>> read_labels("VAE-Fetal-Adult.txt").columns
    ['chrom', 'start', 'stop', 'rDHS', 'fetal', 'adult', 'shared']
    """
    return pl.read_csv(LABELS_DIR / filename, separator="\t")


def one_hot(sequence):
    """One-hot encode a DNA sequence in A, C, G, T order.

    ``N`` becomes 0.25 in every channel. Other characters become zeros.

    Parameters
    ----------
    sequence : str
        Uppercase DNA sequence.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(len(sequence), 4)``.

    Examples
    --------
    >>> one_hot("ACGTN")
    array([[1.  , 0.  , 0.  , 0.  ],
           [0.  , 1.  , 0.  , 0.  ],
           [0.  , 0.  , 1.  , 0.  ],
           [0.  , 0.  , 0.  , 1.  ],
           [0.25, 0.25, 0.25, 0.25]], dtype=float32)
    """
    return _ONE_HOT[np.frombuffer(sequence.encode(), dtype=np.uint8)]


def encode_regions(regions, genome=HG38):
    """One-hot encode `SEQUENCE_LENGTH` bp centered on each region.

    Parameters
    ----------
    regions : polars.DataFrame
        Regions with ``chrom``, ``start`` and ``stop`` columns.
    genome : str or pathlib.Path, optional
        Indexed FASTA file. Defaults to hg38.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(len(regions), SEQUENCE_LENGTH, 4)``.

    Examples
    --------
    >>> regions = pl.DataFrame({"chrom": ["chr1"], "start": [181289], "stop": [181639]})
    >>> encode_regions(regions).shape
    (1, 512, 4)
    """
    fasta = Fasta(genome, sequence_always_upper=True, as_raw=True)
    half = SEQUENCE_LENGTH // 2
    centers = (regions["start"] + regions["stop"]) // 2
    return np.stack(
        [
            one_hot(fasta[chrom][center - half : center + half])
            for chrom, center in zip(regions["chrom"], centers)
        ]
    )
