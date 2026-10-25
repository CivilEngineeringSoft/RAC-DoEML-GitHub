import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from doeml import models as M
from doeml import ui
from doeml.data import (GRADES, SPECIMENS, build_mix, conversion_table, mixes_from_table,
                        mixes_template, read_mixes, water_content)


@st.cache_resource(show_spinner="Loading the models...")
def get_models():
    return M.load()


@st.cache_data
def get_template():
    return mixes_template()


bundle, origin = get_models()
models, ranges = bundle["models"], bundle["ranges"]

st.title("Predict the 28-day compressive strength")
st.caption(f"XGBoost, CatBoost and HistGB trained on {bundle['n']} RAC mixes (80 % / 20 % split) with the "
           "optimal hyperparameters of the paper (Table 5)."
           + (" Models retrained at start-up for the installed library versions." if origin == "retrained" else ""))

SECTIONS = [":material/edit: One mix", ":material/table: Several mixes (Excel)",
            ":material/insights: Model performance"]
section = st.segmented_control("Section", SECTIONS, default=SECTIONS[0], required=True,
                               key="pr_section", label_visibility="collapsed")


def class_check(df: pd.DataFrame, high: bool):
    bad = df[(df["fc specimen (MPa)"] < 45) if high else (df["fc specimen (MPa)"] >= 45)]
    if len(bad):
        st.warning(
            f"The expected strength class (**{'high, ≥ 45 MPa' if high else 'low, < 45 MPa'}**) is not "
            f"consistent with the prediction of {', '.join(bad['Model'])}. The strength classification is an "
            "input of the models: check the class you expect for this mix, or compare with the other class.")


if section == SECTIONS[0]:
    c1, c2, c3 = st.columns([1.2, 1, 1.4])
    spec = c1.selectbox("Specimen", list(SPECIMENS), index=2, format_func=lambda k: SPECIMENS[k]["label"],
                        key="pr_spec")
    high = c2.radio("Expected compressive strength", ["High (≥ 45 MPa)", "Low (< 45 MPa)"],
                    key="pr_class").startswith("High")
    gsel = c3.selectbox("Strength grade (for the conversion factor)", ["Automatic"] + GRADES, key="pr_grade",
                        help="Automatic: deduced from the predicted strength (iterative).")
    names = st.multiselect("Model(s)", M.MODEL_NAMES, default=M.MODEL_NAMES, key="pr_models")

    with st.form("pr_form"):
        st.markdown("**Binder and water**")
        b1, b2, b3, b4 = st.columns(4)
        cement = b1.number_input("Cement (kg/m³)", 0.0, 800.0, 400.0, 5.0)
        fly = b2.number_input("Fly ash (kg/m³)", 0.0, 400.0, 0.0, 5.0)
        wb = b3.number_input("Water-to-binder ratio W/B", 0.2, 1.0, 0.50, 0.01)
        adm = b4.number_input("Admixture (% of cement)", 0.0, 6.0, 1.5, 0.1)
        st.markdown("**Coarse aggregates**")
        n1, n2, n3, n4 = st.columns(4)
        nca = n1.number_input("NCA content (kg/m³)", 0.0, 2000.0, 1027.58, 10.0)
        nsize = n2.number_input("NCA max. size (mm)", 0.0, 40.0, 10.0, 1.0)
        nabs = n3.number_input("NCA absorption (%)", 0.0, 10.0, 0.75, 0.05)
        nssd = n4.number_input("NCA SSD density (kg/m³)", 0.0, 3000.0, 2660.0, 10.0)
        r1, r2, r3, r4 = st.columns(4)
        rca = r1.number_input("RCA content (kg/m³)", 0.0, 2000.0, 0.0, 10.0)
        rsize = r2.number_input("RCA max. size (mm)", 0.0, 40.0, 10.0, 1.0)
        rabs = r3.number_input("RCA absorption (%)", 0.0, 20.0, 6.6, 0.1)
        rssd = r4.number_input("RCA SSD density (kg/m³)", 0.0, 3000.0, 2400.0, 10.0)
        st.markdown("**Fine aggregates**")
        f1, f2, f3, _ = st.columns(4)
        fna = f1.number_input("Natural fine aggregate (kg/m³)", 0.0, 1500.0, 786.24, 10.0)
        rfa = f2.number_input("RFA content (kg/m³)", 0.0, 1500.0, 0.0, 10.0)
        rfabs = f3.number_input("RFA absorption (%)", 0.0, 20.0, 13.3, 0.1)
        ok = st.form_submit_button("Predict", type="primary", icon=":material/online_prediction:")

    if ok:
        if not names:
            st.warning("Select at least one model.")
            st.stop()
        mix = build_mix(specimen=spec, high_strength=high, cement=cement, fly_ash=fly, wb=wb, admixture=adm,
                        fine_total=fna + rfa, coarse_total=nca + rca,
                        ca_ratio=100 * rca / (nca + rca) if nca + rca > 0 else 0,
                        fa_ratio=100 * rfa / (fna + rfa) if fna + rfa > 0 else 0,
                        nca_size=nsize, nca_abs=nabs, nca_ssd=nssd, rca_size=rsize, rca_abs=rabs, rca_ssd=rssd,
                        rfa_abs=rfabs)
        res = M.predict(models, [mix], names, [spec], [None if gsel == "Automatic" else gsel])
        st.markdown("##### Predicted 28-day compressive strength")
        cols = st.columns(len(names) + (1 if len(names) > 1 else 0))
        for c, (_, r) in zip(cols, res.iterrows()):
            c.metric(r["Model"], f"{r['fc specimen (MPa)']:.1f} MPa",
                     help=f"Equivalent 160 × 320 mm cylinder: {r['fc cylinder 160x320 (MPa)']:.1f} MPa")
        if len(names) > 1:
            cols[-1].metric("Mean", f"{res['fc specimen (MPa)'].mean():.1f} MPa",
                            help=f"Range {res['fc specimen (MPa)'].min():.1f} – {res['fc specimen (MPa)'].max():.1f} MPa")
        st.dataframe(res.drop(columns=["mix"]).set_index("Model").style.format(
            {"fc specimen (MPa)": "{:.1f}", "fc cylinder 160x320 (MPa)": "{:.1f}", "Strength coefficient": "{:.2f}"}))
        st.caption(f"{SPECIMENS[spec]['label']} · coarse aggregate replacement {mix['Coarse aggregate replacement ratio (%)']:.1f} % · "
                   f"fine aggregate replacement {mix['Fine aggregate replacement ratio (%)']:.1f} % · "
                   f"water {water_content(mix):.1f} kg/m³")
        class_check(res, high)
        oor = M.out_of_range(mix, ranges)
        if oor:
            st.warning("Outside the range of the training data (extrapolation): " + ", ".join(oor))
        out = pd.concat([pd.DataFrame([mix]).T.rename(columns={0: "Value"}),
                         res.set_index("Model")[["fc specimen (MPa)", "Grade", "Strength coefficient",
                                                  "fc cylinder 160x320 (MPa)"]].T.astype(str)], axis=0)
        st.download_button("Download the result (Excel)", ui.excel_bytes({"Prediction": out}),
                           "RAC_28d_prediction.xlsx", ui.XLSX, icon=":material/download:", on_click="ignore")

elif section == SECTIONS[1]:
    st.download_button("Download the Excel template", get_template(), "RAC_mixes_template.xlsx", ui.XLSX,
                       icon=":material/download:", on_click="ignore")
    names = st.multiselect("Model(s)", M.MODEL_NAMES, default=M.MODEL_NAMES, key="pr_models_b")
    up = st.file_uploader("Upload the mixes (Excel or CSV, one row per mix)", type=["xlsx", "xls", "csv"])
    if up is not None and names:
        try:
            rows = mixes_from_table(read_mixes(up))
        except Exception as e:                                   # noqa: BLE001
            st.error(f"The file could not be read: {e}")
            st.stop()
        res = M.predict(models, [r["mix"] for r in rows], names, [r["specimen"] for r in rows],
                        [r["grade"] for r in rows])
        wide = res.pivot(index="mix", columns="Model", values="fc specimen (MPa)").round(2)
        wide.columns = [f"fc {c} (MPa)" for c in wide.columns]
        if len(names) > 1:
            wide["fc mean (MPa)"] = wide.mean(axis=1).round(2)
        grade = res.pivot(index="mix", columns="Model", values="Grade")
        grade.columns = [f"Grade {c}" for c in grade.columns]
        info = pd.DataFrame({"Mix ID": [r["id"] for r in rows],
                             "Specimen": [SPECIMENS[r["specimen"]]["label"] for r in rows],
                             "Expected": ["High" if r["high"] else "Low" for r in rows],
                             "Outside training range": [", ".join(M.out_of_range(r["mix"], ranges)) for r in rows]})
        table = pd.concat([info, wide, grade], axis=1)
        st.dataframe(table, hide_index=True)
        full = pd.concat([table, pd.DataFrame([r["mix"] for r in rows])], axis=1)
        st.download_button("Download the predictions (Excel)", ui.excel_bytes({"Predictions": full}),
                           "RAC_28d_predictions.xlsx", ui.XLSX, icon=":material/download:", on_click="ignore")

else:
    st.markdown("##### Performance of the models (MSE in MPa², errors in MPa)")
    st.dataframe(bundle["performance"].style.format("{:.3f}"))
    st.caption(f"{bundle['n_train']} training and {bundle['n_test']} testing mixes (random seed 42).")
    with st.expander("Strength conversion factors (σ_cyl / σ_x)"):
        st.dataframe(conversion_table(), hide_index=True)
