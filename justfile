# Ubuntu's CUDA 12.0 libraries in /usr/lib shadow the pip CUDA 12.9 wheels,
# which hides the GPUs from TensorFlow and breaks cuML.
export LD_LIBRARY_PATH := `uv sync --extra gpu --quiet && ls -d "$PWD"/.venv/lib/python3*/site-packages/nvidia/*/lib | paste -sd: -`

# Open a notebook in the marimo editor, e.g. `just edit neuron_glia`.
edit notebook:
    uv run --extra gpu marimo edit notebooks/{{ notebook }}.py

# Run a notebook top to bottom as a script.
run notebook:
    uv run --extra gpu python notebooks/{{ notebook }}.py
