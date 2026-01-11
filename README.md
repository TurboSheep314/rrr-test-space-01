# Bubble Move Meter

A Streamlit application that visualizes ZIP-level composite scores on a map.
The composite score is computed from Overall Score and the two highest-variance
features using relative variance (CV), with interactive sliders for weighting.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py