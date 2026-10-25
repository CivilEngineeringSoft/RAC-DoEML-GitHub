"""Generate a DoE of mixes, predict them with the ML models and find the optimum."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import doe as D
from .data import build_mix
from .models import predict

# factor catalogue: display name -> (argument of build_mix, default levels, unit)
FACTORS = {
    "CA replacement ratio (%)": ("ca_ratio", "0, 20, 60"),
    "FA replacement ratio (%)": ("fa_ratio", "0, 20, 30"),
    "Admixture content (%)": ("admixture", "0, 1, 1.5"),
    "W/B": ("wb", "0.50, 0.53"),
    "Cement content (kg/m3)": ("cement", "350, 400, 450"),
    "Fly ash content (kg/m3)": ("fly_ash", "0, 50, 100"),
}
BASE = "Base composition"

DEFAULT_BASES = pd.DataFrame([
    {"Name": "Fuller", "Cement (kg/m3)": 400.0, "Fly ash (kg/m3)": 0.0, "W/B": 0.53, "Admixture (%)": 0.0,
     "Fine aggregate (kg/m3)": 753.94, "Coarse aggregate (kg/m3)": 914.6},
    {"Name": "Dreux-Gorisse", "Cement (kg/m3)": 400.0, "Fly ash (kg/m3)": 0.0, "W/B": 0.526, "Admixture (%)": 0.0,
     "Fine aggregate (kg/m3)": 786.24, "Coarse aggregate (kg/m3)": 1027.58},
])
DEFAULT_MATERIALS = {"nca_size": 10.0, "nca_abs": 0.75, "nca_ssd": 2660.0,
                     "rca_size": 10.0, "rca_abs": 6.6, "rca_ssd": 2400.0, "rfa_abs": 13.3}

OBJECTIVES = ["Maximum predicted strength (as in the paper)",
              "Maximum recycled aggregate content with a target strength",
              "Multi-criteria: strength, cost and CO2 (Grey Relational Analysis)"]


def make_design(factors: List[D.Factor], design: str, base_factor: Optional[D.Factor] = None,
                n_runs: int = 30, center_points: int = 1, array: Optional[str] = None,
                seed: int = 0) -> tuple:
    """Returns (design DataFrame, description)."""
    info = design
    if design == "Full factorial":
        allf = ([base_factor] if base_factor else []) + factors
        return D.full_factorial(allf), f"Full factorial ({' x '.join(str(f.n) for f in allf)})"
    if design == "Taguchi orthogonal array":
        allf = ([base_factor] if base_factor else []) + factors
        df, name = D.taguchi(allf, array)
        return df, f"Taguchi {name}"
    if design == "Latin hypercube":
        allf = ([base_factor] if base_factor else []) + factors
        return D.latin_hypercube(allf, n_runs, seed), f"Latin hypercube ({n_runs} runs)"
    if design == "Box-Behnken":
        df = D.box_behnken(factors, center_points)
    elif design.startswith("Central composite"):
        df = D.central_composite(factors, center_points)
    else:
        raise ValueError(design)
    if base_factor:
        df = D.cross(df, base_factor)
        info += f" x {base_factor.n} base compositions"
    return df, info


def evaluate(design: pd.DataFrame, bases: pd.DataFrame, materials: Dict[str, float], specimen: str,
             high: bool, models: Dict[str, object], names: List[str], grade: Optional[str] = None,
             prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build every mix of the design, predict its strength with each model and
    compute its cost and CO2. Returns one row per run."""
    bases = bases.set_index("Name")
    mixes, rows = [], []
    for i, r in design.reset_index(drop=True).iterrows():
        b = bases.loc[r[BASE]] if BASE in r else bases.iloc[0]
        args = dict(cement=float(b["Cement (kg/m3)"]), fly_ash=float(b["Fly ash (kg/m3)"]), wb=float(b["W/B"]),
                    admixture=float(b["Admixture (%)"]), fine_total=float(b["Fine aggregate (kg/m3)"]),
                    coarse_total=float(b["Coarse aggregate (kg/m3)"]), ca_ratio=0.0, fa_ratio=0.0)
        for f, (arg, _) in FACTORS.items():
            if f in r:
                args[arg] = float(r[f])
        mix = build_mix(specimen=specimen, high_strength=high, **materials, **args)
        mixes.append(mix)
        cost, co2 = D.cost_co2(mix, prices if prices is not None else D.PRICES)
        rows.append({"Run": i + 1, **{c: r[c] for c in design.columns},
                     "RCA (kg/m3)": mix["Quantity RCA (kg/m3)"], "NCA (kg/m3)": mix["Quantity NCA (kg/m3)"],
                     "RFA (kg/m3)": mix["RFA content (kg/m3)"], "FNA (kg/m3)": mix["Quantity FNA (kg/m3)"],
                     "Water (kg/m3)": D.masses(mix)["Water"],
                     "Recycled aggregate (%)": 100 * (mix["Quantity RCA (kg/m3)"] + mix["RFA content (kg/m3)"]) / max(
                         1e-9, args["fine_total"] + args["coarse_total"]),
                     "Cost ($/m3)": cost, "CO2 (kg/m3)": co2})
    out = pd.DataFrame(rows)
    pred = predict(models, mixes, names, [specimen] * len(mixes), [grade] * len(mixes))
    for n in names:
        p = pred[pred["Model"] == n].set_index("mix")
        out[f"fc {n} (MPa)"] = p["fc specimen (MPa)"].to_numpy()
        out[f"Grade {n}"] = p["Grade"].to_numpy()
    out["fc mean (MPa)"] = out[[f"fc {n} (MPa)" for n in names]].mean(axis=1)
    out.attrs["mixes"] = mixes
    return out


def response_column(decision: str) -> str:
    return "fc mean (MPa)" if decision == "Mean of the selected models" else f"fc {decision} (MPa)"


def choose_optimum(res: pd.DataFrame, objective: str, decision: str, target: float = 45.0,
                   weights: Optional[Dict[str, float]] = None) -> tuple:
    """Returns (index of the optimal run, scored DataFrame, message)."""
    y = response_column(decision)
    df = res.copy()
    if objective == OBJECTIVES[0]:
        df["Score"] = df[y]
        msg = f"highest {y}"
    elif objective == OBJECTIVES[1]:
        ok = df[y] >= target
        if not ok.any():
            df["Score"] = df[y]
            return int(df["Score"].idxmax()), df, (f"no mix reaches {target:g} MPa: the strongest mix is shown")
        df["Score"] = np.where(ok, df["Recycled aggregate (%)"] + 1e-3 * df[y], -np.inf)
        msg = f"highest recycled content among the mixes with {y} ≥ {target:g} MPa"
    else:
        crit = {y: "max", "Cost ($/m3)": "min", "CO2 (kg/m3)": "min"}
        w = weights or {}
        df["Score"] = D.grey_relational_grade(df, crit, {y: w.get("strength", 1), "Cost ($/m3)": w.get("cost", 1),
                                                          "CO2 (kg/m3)": w.get("co2", 1)})
        df = df.rename(columns={"Score": "Grey relational grade"})
        df["Score"] = df["Grey relational grade"]
        msg = "highest grey relational grade (strength ↑, cost ↓, CO₂ ↓)"
    return int(df["Score"].idxmax()), df, msg
