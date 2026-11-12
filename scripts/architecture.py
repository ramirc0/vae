"""Draw the Pratt et al. 2024 b-cCRE VAE as an SVG.

Run ``python scripts/architecture.py out.svg``. Values follow
``vae.model.VAE``. Shapes are channels × length.
"""

import sys

FONT = "Helvetica, Nimbus Sans, Arial, sans-serif"
INK = "#333333"
LABEL = "#555555"
GRAY = ("#EBEBEB", "#555555")
PINK = ("#E8AED6", "#B576A5")
GREEN = ("#E4F59F", "#A9BA6C")
YELLOW = ("#FFE58F", "#D8B44F")
ORANGE = ("#FFD3A6", "#DFA162")

ROW = 52
STEP = ROW + 40
LEFT, RIGHT = 340, 1220
SAMPLE = 495
WIDTH = 480

ENCODER = [
    ("Conv1d(4→256, k=32), ReLU", "256 × 512"),
    ("MaxPool1d(8)", "256 × 64"),
    ("Conv1d(256→128, k=8), ReLU", "128 × 64"),
    ("MaxPool1d(2)", "128 × 32"),
    ("Flatten", "4,096"),
    ("Linear(4,096→1,028)", "1,028"),
    ("Linear(1,028→512)", "512"),
]
DECODER = [
    ("Linear(256→512)", "512"),
    ("Linear(512→1,028)", "1,028"),
    ("Linear(1,028→4,096)", "4,096"),
    ("Unflatten", "128 × 32"),
    ("Upsample(×2)", "128 × 64"),
    ("ConvTranspose1d(128→128, k=8)", "128 × 64"),
    ("Upsample(×8)", "128 × 512"),
    ("ConvTranspose1d(128→512, k=32)", "512 × 512"),
    ("ConvTranspose1d(512→4, k=32), ReLU", "4 × 512"),
]


def text(x, y, lines, size=20, anchor="middle", fill=INK):
    """Return centered text, one ``tspan`` per line, around baseline ``y``."""
    dy = size * 1.25
    y0 = y - dy * (len(lines) - 1) / 2 + size * 0.35
    spans = "".join(
        f'<tspan x="{x}" y="{y0 + i * dy:.1f}">{line}</tspan>'
        for i, line in enumerate(lines)
    )
    return (
        f'<text font-family="{FONT}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}">{spans}</text>'
    )


def box(x, y, w, h, lines, fill="#FFFFFF"):
    """Return a rounded box with centered text."""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
        f'fill="{fill}" stroke="{INK}" stroke-width="2"/>'
        + text(x + w / 2, y + h / 2, lines)
    )


def panel(x, y, w, h, colors):
    """Return a colored group background."""
    fill, stroke = colors
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
    )


def path(points, arrow=True):
    """Return a polyline through ``points``, ending in an arrowhead."""
    d = " ".join(f"{x},{y}" for x, y in points)
    head = ' marker-end="url(#head)"' if arrow else ""
    return f'<polyline points="{d}" fill="none" stroke="{INK}" stroke-width="2"{head}/>'


def shape(x, y, value):
    """Return a tensor-shape label placed right of an arrow at ``x``."""
    return text(x + 10, y, [value], size=18, anchor="start", fill=LABEL)


def title(x, y, value, anchor):
    """Return a panel title."""
    return text(x, y, [value], size=22, anchor=anchor, fill=LABEL)


def encoder():
    """Return the input and the encoder, top to bottom. Ends at y=764."""
    x = LEFT - WIDTH / 2
    parts = [
        panel(x - 20, 140, WIDTH + 40, 644, GREEN),
        title(620, 170, "Encoder", "start"),
        box(x, 10, WIDTH, 80, ["Input", "One-hot DNA, 512 bp"], GRAY[0]),
        path([(LEFT, 90), (LEFT, 160)]),
        shape(LEFT, 125, "4 × 512"),
    ]
    for i, (label, out) in enumerate(ENCODER):
        y = 160 + i * STEP
        parts.append(box(x, y, WIDTH, ROW, [label]))
        if i < len(ENCODER) - 1:
            parts += [
                path([(LEFT, y + ROW), (LEFT, y + STEP)]),
                shape(LEFT, y + 72, out),
            ]
    return parts


def latent():
    """Return the latent block, the classifier and KL heads, and their outputs."""
    mu, log_var, kl, z = 215, 465, 185, SAMPLE
    return [
        panel(20, 815, 640, 225, PINK),
        title(680, 845, "Latent", "start"),
        path([(LEFT, 764), (LEFT, 800)], arrow=False),
        path([(mu, 800), (log_var, 800)], arrow=False),
        path([(mu, 800), (mu, 830)]),
        path([(log_var, 800), (log_var, 830)]),
        box(100, 830, 230, 64, ["μ", "Linear(512→256)"]),
        box(350, 830, 230, 64, ["log σ²", "Linear(512→256)"]),
        path([(mu, 894), (mu, 920)], arrow=False),
        path([(log_var, 894), (log_var, 920)], arrow=False),
        path([(kl, 920), (z, 920)], arrow=False),
        path([(kl, 920), (kl, 960)]),
        path([(z, 920), (z, 960)]),
        box(40, 960, 290, 64, ["KL divergence", "−½(1 + log σ² − μ² − σ²)"]),
        box(350, 960, 290, 64, ["Sample z", "μ + exp(½ log σ²) · ε"]),
        path([(kl, 1024), (kl, 1230)]),
        shape(kl, 1127, "256"),
        box(40, 1230, 290, 80, ["Output", "KL per dimension"], GRAY[0]),
        text(kl, 1335, ["MAE against 0"], size=16, fill=LABEL),
        path([(z, 1024), (z, 1060)], arrow=False),
        panel(330, 1085, 330, 94, YELLOW),
        path([(z, 1060), (z, 1100)]),
        box(350, 1100, 290, 64, ["Classifier", "Linear(256→C)"]),
        path([(z, 1164), (z, 1230)]),
        shape(z, 1197, "C"),
        box(350, 1230, 290, 80, ["Output", "Class logits"], GRAY[0]),
        text(z, 1335, ["Softmax cross-entropy"], size=16, fill=LABEL),
    ]


def decoder():
    """Return the decoder, bottom to top, and the reconstruction output."""
    x = RIGHT - WIDTH / 2
    parts = [
        panel(x - 20, 140, WIDTH + 40, 828, ORANGE),
        title(940, 170, "Decoder", "end"),
        path([(SAMPLE, 1060), (RIGHT, 1060), (RIGHT, 948)]),
        shape(RIGHT, 1000, "256"),
        box(x, 10, WIDTH, 80, ["Output", "Reconstructed one-hot DNA"], GRAY[0]),
        text(940, 50, ["MSE against input"], size=16, anchor="end", fill=LABEL),
    ]
    for k, (label, out) in enumerate(DECODER):
        y = 896 - k * STEP
        parts += [
            box(x, y, WIDTH, ROW, [label]),
            path([(RIGHT, y), (RIGHT, y - 40 if k < len(DECODER) - 1 else 90)]),
            shape(RIGHT, y - 20 if k < len(DECODER) - 1 else 125, out),
        ]
    return parts


def notes():
    """Return the notes on class counts, training and activations."""
    lines = [
        "C = 3: neuron/glia, fetal/adult",
        "C = 7: Corces cell types",
        "Loss weights 1 : 1 : 1. Adam, learning rate 10⁻⁴.",
        "Layers without an activation are linear.",
    ]
    return [text(980, 1200, lines, size=16, anchor="start", fill=LABEL)]


def figure():
    """Return the full SVG document as a string."""
    marker = (
        '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>'
    )
    body = "\n".join(encoder() + latent() + decoder() + notes())
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1370" '
        'viewBox="0 0 1500 1370">\n'
        f'<rect width="1500" height="1370" fill="#FFFFFF"/>\n{marker}\n{body}\n</svg>\n'
    )


if __name__ == "__main__":
    with open(sys.argv[1], "w") as handle:
        handle.write(figure())
