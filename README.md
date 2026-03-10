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


in the future: piut the sheet ID in a side bar so people can add their sheets in there
SHEET_ID = st.sidebar.text_input(
    "Google Sheet ID",
    value="1ABCDEF1234567890"
)

df = load_scores(SHEET_ID)



Education
Healthcare + Fitness
Commute/Transit score
Accessibility
Culture/Entertainment 

All scored out of 100- so total score could be 500