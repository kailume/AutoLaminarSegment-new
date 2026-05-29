# Project Structure And Codex Instructions

## Project Structure

- `run_pipeline.py`: algorithm startup entry point for the normal pipeline.
- `run_auto_pipeline.py`: algorithm startup entry point for the automatic pipeline.
- `segment_boundaries.py`, `draw_boundaries.py`, `segment_large_image_fast.py`, `resultanalysis.py`, and files under `src/`: implementation files for the actual algorithm details.
- `input/`: runtime input data.
- `output/`: runtime output data.
- `tmp/`: temporary runtime artifacts.
- `venv-cellpose/`: active Cellpose Python environment.
- `.venv-cellpose/`: old/inactive Cellpose environment; do not use unless explicitly requested.

## Pipeline Entry Points

- `run_pipeline.py` and `run_auto_pipeline.py` are the algorithm startup entry points.
- `run_auto_pipeline.py` includes the preceding automatic segmentation steps for GM, WM, and grayMask before running the rest of the pipeline.
- When modifying pipeline behavior, keep the startup entry points aligned with the underlying algorithm files.

## Parameter Modification Rules

When adding or changing parameters in detailed algorithm logic:

1. Add a parameter modification entry near the beginning of the corresponding startup entry point:
   - `run_auto_pipeline.py`
   - `run_pipeline.py`
2. Add the matching parameter modification entry near the beginning of the corresponding actual algorithm file.
3. Keep parameter names and defaults consistent across the CLI, startup entry point, and implementation file.

Parameter precedence must always be:

```text
CLI > startup entry point (run_auto_pipeline.py / run_pipeline.py) > actual algorithm internal parameter
```

In other words:

- CLI arguments have the highest priority.
- Values configured in `run_auto_pipeline.py` or `run_pipeline.py` override internal algorithm defaults.
- Internal parameters inside the actual algorithm files are the fallback defaults.
