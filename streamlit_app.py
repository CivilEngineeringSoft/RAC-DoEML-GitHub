"""RAC DoE-ML Streamlit app.   Run locally:  streamlit run streamlit_app.py"""

import sys
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

st.set_page_config(page_title="RAC DoE-ML", page_icon=":material/science:", layout="wide")

pages = [
    st.Page(APP / "views" / "home.py", title="Home", icon=":material/home:", default=True),
    st.Page(APP / "views" / "predict.py", title="Predict 28-day strength", icon=":material/online_prediction:"),
    st.Page(APP / "views" / "optimise.py", title="Optimal mix (DoE)", icon=":material/grid_on:"),
]
st.navigation(pages).run()
