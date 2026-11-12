# Open a notebook in the marimo editor, e.g. `just edit notebooks/neuron_glia.py`.
edit notebook:
    uv run --extra gpu marimo edit {{ notebook }}

# Run a notebook top to bottom as a script. Add `--train` to fit the model first.
run notebook *args:
    uv run --extra gpu python {{ notebook }} {{ args }}
