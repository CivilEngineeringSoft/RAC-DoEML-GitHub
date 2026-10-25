"""pip install -r requirements.txt pytest && pytest tests"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from doeml import doe as D                                         # noqa: E402
from doeml import models as M                                      # noqa: E402
from doeml import optimise as O                                    # noqa: E402
from doeml.data import (FEATURES, TARGET, build_mix, grade_from_cylinder_strength,  # noqa: E402
                        load_data, mixes_from_table, mixes_template, prepare, specimen_code,
                        strength_coefficient)


@pytest.fixture(scope="module")
def bundle():
    return M.load()[0]


def test_preprocessing_reproduces_table2():
    X = prepare(load_data())
    assert X.shape == (594, 20) and list(X.columns) == FEATURES
    assert round(X["Size RCA (mm)"].mean(), 2) == 15.21
    assert round(X["Specimen"].mean(), 2) == 0.25
    assert np.percentile(X["Specimen"], [25, 50, 75]).round(2).tolist() == [0.2, 0.3, 0.3]


@pytest.mark.parametrize("label,code", [("100x200mm", 0.1), ("150x300 mm", 0.2), ("Cube 100 mm", 0.3),
                                        ("150x150x150 mm", 0.4), ("Cylinder 75 x 150 mm", 0.5)])
def test_specimen_coding(label, code):
    assert specimen_code(label) == code


def test_conversion_factors_and_grades():
    assert strength_coefficient("100x100x100", "C20-40") == 0.75
    assert strength_coefficient("150x150x150", "C80") == 0.88
    assert strength_coefficient("150x300", "C60") == 0.99
    assert [grade_from_cylinder_strength(v) for v in (30, 50, 60, 70, 90)] == GRADES_EXPECTED


GRADES_EXPECTED = ["C20-40", "C50", "C60", "C70", "C80"]


def test_models_close_to_table6(bundle):
    perf = bundle["performance"]
    for name, r2 in {"XGBoost": 0.83, "CatBoost": 0.86, "HistGB": 0.84}.items():
        assert abs(perf.loc[("Testing", "R2"), name] - r2) < 0.03
        assert perf.loc[("Training", "R2"), name] > 0.98


def test_table8_mix(bundle):
    mix = build_mix(specimen="100x100x100", high_strength=True, cement=400, fly_ash=0, wb=0.50, admixture=1.5,
                    fine_total=786.24, coarse_total=1027.58, ca_ratio=0, fa_ratio=0, nca_size=10, nca_abs=0.75,
                    nca_ssd=2660, rca_size=10, rca_abs=6.6, rca_ssd=2400, rfa_abs=13.3)
    r = M.predict(bundle["models"], [mix], ["XGBoost"], ["100x100x100"])
    assert abs(r["fc specimen (MPa)"].iloc[0] - 51.7) < 1.5          # paper: 51.7 MPa


def test_orthogonal_arrays():
    for name, (a, cl) in D.orthogonal_arrays().items():
        for i, j in itertools.combinations(range(a.shape[1]), 2):
            counts = pd.Series(list(zip(a[:, i], a[:, j]))).value_counts()
            assert counts.nunique() == 1 and len(counts) == cl[i] * cl[j], name


def test_designs_and_optimum(bundle):
    F = [D.Factor(n, [float(x) for x in O.FACTORS[n][1].split(",")]) for n in list(O.FACTORS)[:4]]
    base = D.Factor(O.BASE, list(O.DEFAULT_BASES["Name"]), numeric=False)
    dsg, info = O.make_design(F, "Full factorial", base)
    assert len(dsg) == 108                                            # as in the paper
    res = O.evaluate(dsg, O.DEFAULT_BASES, O.DEFAULT_MATERIALS, "100x100x100", True, bundle["models"], M.MODEL_NAMES)
    i, _, _ = O.choose_optimum(res, O.OBJECTIVES[0], "CatBoost")
    best = res.loc[i]
    assert best[O.BASE] == "Dreux-Gorisse" and best["CA replacement ratio (%)"] == 0 \
        and best["Admixture content (%)"] == 1.5                       # Table 8
    tab = D.anova(res, list(dsg.columns), "fc XGBoost (MPa)")
    assert tab.index[0] == "CA replacement ratio (%)"                 # Table 7, XGBoost
    for design in ["Taguchi orthogonal array", "Box-Behnken", "Central composite (face-centred)", "Latin hypercube"]:
        d, _ = O.make_design(F, design, base, n_runs=20)
        r = O.evaluate(d, O.DEFAULT_BASES, O.DEFAULT_MATERIALS, "100x100x100", True, bundle["models"], ["HistGB"])
        for obj in O.OBJECTIVES:
            O.choose_optimum(r, obj, "HistGB", 45)
    assert O.make_design(F, "Taguchi orthogonal array", base)[1] == "Taguchi L18 (2^1 3^7)"


def test_template_roundtrip(bundle):
    rows = mixes_from_table(pd.read_excel(pd.io.common.BytesIO(mixes_template())))
    assert len(rows) == 2 and rows[1]["mix"]["Coarse aggregate replacement ratio (%)"] == pytest.approx(20, abs=0.1)
    M.predict(bundle["models"], [r["mix"] for r in rows], M.MODEL_NAMES, [r["specimen"] for r in rows])


def test_pages():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=300)
    at.run()
    at.switch_page("views/predict.py").run()
    at.get("form_submit_button")[0].click().run()
    assert not at.exception and any(m.label == "XGBoost" for m in at.metric)
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=300)
    at.run()
    at.switch_page("views/optimise.py").run()
    at.button[0].click().run()
    assert not at.exception and any("Optimal mix" in s.value for s in at.success)
    for s in [":material/bar_chart: Main effects", ":material/functions: ANOVA", ":material/download: Download"]:
        at.session_state["op_section"] = s
        at.run()
        assert not at.exception
