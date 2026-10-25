import streamlit as st

from doeml.data import conversion_table

st.title("RAC DoE-ML")
st.subheader("28-day compressive strength of recycled aggregate concrete: ML prediction and "
             "optimal mix design with a design of experiments")

st.markdown(
    "This application accompanies the article *Experimental Analysis and Taguchi Optimization of the "
    "Residual Mechanical Behavior of Recycled Aggregate Concretes at Different Temperature Levels* "
    "(Cirhuza Badesire, Di Cesare, Vu, Bishweka). It uses the three machine-learning models of the "
    "paper (XGBoost, CatBoost, HistGB), trained on 594 RAC mixes from 44 studies.")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("#### :material/online_prediction: Predict the 28-day strength")
        st.markdown(
            "- Enter the composition of a mix (or upload several mixes from the Excel template)\n"
            "- Choose the specimen and whether a **high** (≥ 45 MPa) or **low** (< 45 MPa) strength "
            "is expected\n"
            "- The strength (specimen shape) coefficient is taken from the conversion table, with the "
            "strength grade deduced from the predicted strength\n"
            "- Strength on the specimen and equivalent strength on a 160 × 320 mm cylinder")
        st.page_link("views/predict.py", label="Open", icon=":material/arrow_forward:")
with c2:
    with st.container(border=True):
        st.markdown("#### :material/grid_on: Find the optimal mix with a DoE")
        st.markdown(
            "- Define one or several base compositions and the properties of the aggregates\n"
            "- Choose the factors and their levels (RCA and RFA replacement ratios, admixture, W/B, ...)\n"
            "- Choose the design: **full factorial** (as in the paper), **Taguchi** orthogonal array, "
            "**Box–Behnken**, **central composite** or **Latin hypercube**\n"
            "- Optimal mix (maximum strength, maximum recycled content for a target strength, or "
            "multi-criteria strength–cost–CO₂ with Grey Relational Analysis), main effects, ANOVA, downloads")
        st.page_link("views/optimise.py", label="Open", icon=":material/arrow_forward:")

with st.expander("Strength conversion factors used by the app (σ_cyl / σ_x)"):
    st.dataframe(conversion_table(), hide_index=True)
    st.caption("The factor depends on the specimen and on the strength grade. Unless a grade is imposed, the "
               "app starts from C20-40 (low strength) or C50 (high strength), predicts the strength, converts "
               "it to the 160 × 320 mm cylinder, updates the grade (limits 45, 55, 65 and 75 MPa) and repeats "
               "until the grade no longer changes.")

st.markdown("#### How to cite")
st.code("Cirhuza Badesire P., Di Cesare N., Vu X.H., Bishweka C. Experimental Analysis and Taguchi "
        "Optimization of the Residual Mechanical Behavior of Recycled Aggregate Concretes at Different "
        "Temperature Levels (2026, in preparation).", language=None)
st.caption("© 2026 Université Claude Bernard Lyon 1, LMC² (EA 7427) · ULPGL Goma. Predictions are given for "
           "research and pre-design purposes; they do not replace laboratory testing.")
