"""Designs of experiments and analysis of their results.

Designs: full factorial, Taguchi orthogonal arrays (L4, L8, L9, L16, L16(4^5), L18,
L25, L27, L32), Box-Behnken, face-centred central composite, Latin hypercube and
random sampling. Analysis: main effects, Taguchi signal-to-noise ratio
(larger-the-better), ANOVA with factor contributions, and a multi-criteria Grey
Relational Analysis (strength, cost, CO2).
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ===================================================================== factors
class Factor:
    def __init__(self, name: str, levels: Sequence, numeric: bool = True):
        self.name = name
        self.levels = list(levels)
        self.numeric = numeric

    @property
    def n(self) -> int:
        return len(self.levels)

    def __repr__(self):
        return f"Factor({self.name!r}, {self.levels})"


# ===================================================================== full factorial
def full_factorial(factors: List[Factor]) -> pd.DataFrame:
    return pd.DataFrame(list(itertools.product(*[f.levels for f in factors])),
                        columns=[f.name for f in factors])


# ===================================================================== orthogonal arrays
def _gf_linear_array(p: int, n_basic: int) -> np.ndarray:
    """Orthogonal array L_{p^n}(p^((p^n-1)/(p-1))) over a prime field GF(p)."""
    runs = np.array(list(itertools.product(range(p), repeat=n_basic)))
    cols = []
    for coeffs in itertools.product(range(p), repeat=n_basic):
        nz = [c for c in coeffs if c]
        if not nz or nz[0] != 1:                 # first non-zero coefficient = 1
            continue
        cols.append(runs @ np.array(coeffs) % p)
    return np.array(cols).T


# GF(4) arithmetic for the L16(4^5) array
_GF4_MUL = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]])


def _gf4_array() -> np.ndarray:
    runs = list(itertools.product(range(4), repeat=2))
    cols = []
    for a, b in [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3)]:
        cols.append([_GF4_MUL[a, x] ^ _GF4_MUL[b, y] for x, y in runs])
    return np.array(cols).T


# Standard L18 (2^1 x 3^7) - Taguchi
_L18 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 2, 2, 2, 2, 2, 2],
    [0, 1, 0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2, 0, 0], [0, 1, 2, 2, 0, 0, 1, 1],
    [0, 2, 0, 1, 0, 2, 1, 2], [0, 2, 1, 2, 1, 0, 2, 0], [0, 2, 2, 0, 2, 1, 0, 1],
    [1, 0, 0, 2, 2, 1, 1, 0], [1, 0, 1, 0, 0, 2, 2, 1], [1, 0, 2, 1, 1, 0, 0, 2],
    [1, 1, 0, 1, 2, 0, 2, 1], [1, 1, 1, 2, 0, 1, 0, 2], [1, 1, 2, 0, 1, 2, 1, 0],
    [1, 2, 0, 2, 1, 2, 0, 1], [1, 2, 1, 0, 2, 0, 1, 2], [1, 2, 2, 1, 0, 1, 2, 0],
])


def orthogonal_arrays() -> Dict[str, tuple]:
    """name -> (array, list of column levels)."""
    arr = {
        "L4 (2^3)": _gf_linear_array(2, 2),
        "L8 (2^7)": _gf_linear_array(2, 3),
        "L9 (3^4)": _gf_linear_array(3, 2),
        "L16 (2^15)": _gf_linear_array(2, 4),
        "L16 (4^5)": _gf4_array(),
        "L18 (2^1 3^7)": _L18,
        "L25 (5^6)": _gf_linear_array(5, 2),
        "L27 (3^13)": _gf_linear_array(3, 3),
        "L32 (2^31)": _gf_linear_array(2, 5),
    }
    return {k: (a, [len(np.unique(a[:, j])) for j in range(a.shape[1])]) for k, a in arr.items()}


def _assign(col_levels: List[int], factor_levels: List[int]) -> Optional[List[int]]:
    """Assign each factor to a distinct column with at least as many levels
    (dummy-level technique when the column has more levels). Returns column indices."""
    order = sorted(range(len(factor_levels)), key=lambda i: -factor_levels[i])
    free = list(range(len(col_levels)))
    out = [None] * len(factor_levels)
    for i in order:
        cands = [c for c in free if col_levels[c] >= factor_levels[i]]
        if not cands:
            return None
        c = min(cands, key=lambda c: (col_levels[c] - factor_levels[i], c))   # tightest fit
        out[i] = c
        free.remove(c)
    return out


def taguchi(factors: List[Factor], array: Optional[str] = None) -> tuple:
    """Taguchi orthogonal-array design. Chooses the smallest array that fits the
    factors (or uses ``array``). Factors with fewer levels than their column use the
    dummy-level technique (level index modulo the number of levels).

    Returns (design DataFrame, array name)."""
    arrays = orthogonal_arrays()
    fl = [f.n for f in factors]
    candidates = []
    for name in ([array] if array else arrays):
        a, cl = arrays[name]
        cols = _assign(cl, fl)
        if cols is None:
            continue
        dummies = sum(cl[c] != n for c, n in zip(cols, fl))
        # prefer few runs, but penalise the dummy-level technique (unbalanced levels)
        candidates.append((a.shape[0] + 2 * dummies, a.shape[0], name, cols))
    if candidates:
        _, _, name, cols = min(candidates)
        a, _ = arrays[name]
        rows = [[f.levels[int(a[r, c]) % f.n] for f, c in zip(factors, cols)] for r in range(a.shape[0])]
        return pd.DataFrame(rows, columns=[f.name for f in factors]), name
    raise ValueError("No standard orthogonal array fits these factors (max. 5 levels per factor, "
                     "31 two-level, 13 three-level factors). Use a full factorial or a Latin hypercube.")


# ===================================================================== response-surface designs
def _coded_to_levels(coded: np.ndarray, factors: List[Factor]) -> pd.DataFrame:
    out = {}
    for j, f in enumerate(factors):
        lo, hi = min(f.levels), max(f.levels)
        mid = (lo + hi) / 2
        out[f.name] = mid + coded[:, j] * (hi - lo) / 2
    return pd.DataFrame(out)


def box_behnken(factors: List[Factor], center_points: int = 1) -> pd.DataFrame:
    """Box-Behnken design (>= 3 numeric factors): low / centre / high of each factor."""
    k = len(factors)
    if k < 3:
        raise ValueError("Box-Behnken needs at least 3 numeric factors.")
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        for a, b in itertools.product((-1, 1), repeat=2):
            r = np.zeros(k)
            r[i], r[j] = a, b
            rows.append(r)
    rows += [np.zeros(k)] * center_points
    return _coded_to_levels(np.array(rows), factors)


def central_composite(factors: List[Factor], center_points: int = 1) -> pd.DataFrame:
    """Face-centred central composite design (axial points on the faces, so every
    run stays within the factor ranges)."""
    k = len(factors)
    if k < 2:
        raise ValueError("A central composite design needs at least 2 numeric factors.")
    rows = [np.array(r, float) for r in itertools.product((-1, 1), repeat=k)]
    for i in range(k):
        for a in (-1, 1):
            r = np.zeros(k)
            r[i] = a
            rows.append(r)
    rows += [np.zeros(k)] * center_points
    return _coded_to_levels(np.array(rows), factors)


def latin_hypercube(factors: List[Factor], n: int, seed: int = 0) -> pd.DataFrame:
    """Latin hypercube sampling: numeric factors vary continuously between their
    minimum and maximum levels; categorical factors are sampled among their levels."""
    from scipy.stats import qmc
    u = qmc.LatinHypercube(d=len(factors), seed=seed).random(n)
    out = {}
    for j, f in enumerate(factors):
        if f.numeric:
            lo, hi = min(f.levels), max(f.levels)
            out[f.name] = lo + u[:, j] * (hi - lo)
        else:
            out[f.name] = [f.levels[min(int(x * f.n), f.n - 1)] for x in u[:, j]]
    return pd.DataFrame(out)


def random_design(factors: List[Factor], n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f.name: rng.choice(f.levels, size=n) for f in factors})


def cross(design: pd.DataFrame, factor: Factor) -> pd.DataFrame:
    """Repeat a design for every level of a categorical factor (e.g. base composition)."""
    parts = [design.assign(**{factor.name: lv}) for lv in factor.levels]
    return pd.concat(parts, ignore_index=True)[[factor.name] + list(design.columns)]


DESIGNS = {
    "Full factorial": "all combinations of the levels (the design used in the paper)",
    "Taguchi orthogonal array": "smallest standard array (L4 ... L32) that fits the factors",
    "Box-Behnken": "response-surface design, 3 levels (min, centre, max) of >= 3 numeric factors",
    "Central composite (face-centred)": "response-surface design, 2-level factorial + axial + centre points",
    "Latin hypercube": "space-filling random sampling between the minimum and maximum levels",
}


# ===================================================================== analysis
def main_effects(df: pd.DataFrame, factors: List[str], response: str) -> Dict[str, pd.Series]:
    return {f: df.groupby(f, sort=True)[response].mean() for f in factors}


def sn_larger_better(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    return -10 * np.log10(1.0 / np.maximum(y, 1e-9) ** 2)


def anova(df: pd.DataFrame, factors: List[str], response: str, interactions: bool = True) -> pd.DataFrame:
    """ANOVA (type II sums of squares) with the factors as categorical variables and,
    when the residual degrees of freedom allow it, all two-factor interactions.
    Returns Sum Sq, df, F, p-value and contribution (%), sorted by contribution."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    use = [f for f in factors if df[f].nunique() > 1]
    if not use:
        raise ValueError("No factor varies in the design.")
    alias = {f: f"F{i}" for i, f in enumerate(use)}
    import warnings
    d = df[use + [response]].rename(columns={**alias, response: "y"})
    terms = [f"C({a})" for a in alias.values()]
    main = "y ~ " + " + ".join(terms)
    formula = main
    inter = [f"{a}:{b}" for a, b in itertools.combinations(terms, 2)]
    if interactions and inter:
        n_par = 1 + sum(df[f].nunique() - 1 for f in use) + sum(
            (df[a].nunique() - 1) * (df[b].nunique() - 1) for a, b in itertools.combinations(use, 2))
        if n_par < len(d) - 1:
            formula += " + " + " + ".join(inter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=d).fit()
        # response-surface designs (Box-Behnken, CCD) cannot estimate every interaction
        # of categorical factors: fall back to the main effects
        if model.df_model < len(model.params) - 1 and formula != main:
            model = smf.ols(main, data=d).fit()
        tab = anova_lm(model, typ=2)
    inv = {v: k for k, v in alias.items()}

    def nice(term):
        parts = [inv[p.replace("C(", "").replace(")", "")] for p in term.split(":")]
        return " x ".join(parts)

    tab.index = [nice(t) if t != "Residual" else "Residual" for t in tab.index]
    tab = tab.rename(columns={"sum_sq": "Sum Sq", "df": "df", "F": "F-value", "PR(>F)": "p-value"})
    tab["Contribution (%)"] = 100 * tab["Sum Sq"] / tab["Sum Sq"].sum()
    res = tab.loc[["Residual"]] if "Residual" in tab.index else None
    tab = tab.drop(index="Residual", errors="ignore").sort_values("Contribution (%)", ascending=False)
    return pd.concat([tab, res]) if res is not None else tab


# ===================================================================== cost, CO2 and GRA
PRICES = pd.DataFrame({
    "Constituent": ["Cement", "Fly ash", "NFA", "NCA", "RCA", "RFA", "Water", "Superplasticizer"],
    "Price ($/kg)": [0.0689, 0.0, 0.0287, 0.0287, 0.0215, 0.0215, 0.000691, 1.45],
    "CO2 factor (kg CO2-eq/kg)": [0.889, 0.0, 0.0025, 0.0022, 0.0015, 0.0018, 0.0002, 0.001181],
})


def masses(mix: dict) -> Dict[str, float]:
    c, fa = mix["Cement Content (kg/m3)"], mix["Fly Ash Content (kg/m3)"]
    return {"Cement": c, "Fly ash": fa, "NFA": mix["Quantity FNA (kg/m3)"],
            "NCA": mix["Quantity NCA (kg/m3)"], "RCA": mix["Quantity RCA (kg/m3)"],
            "RFA": mix["RFA content (kg/m3)"], "Water": mix["Water-to-binder ratio (W/B)"] * (c + fa),
            "Superplasticizer": mix["Admixture Content (%)"] / 100 * c}


def cost_co2(mix: dict, prices: pd.DataFrame = PRICES) -> tuple:
    m = masses(mix)
    p = prices.set_index("Constituent")
    cost = sum(m[k] * float(p.loc[k, "Price ($/kg)"]) for k in m if k in p.index)
    co2 = sum(m[k] * float(p.loc[k, "CO2 factor (kg CO2-eq/kg)"]) for k in m if k in p.index)
    return cost, co2


def grey_relational_grade(df: pd.DataFrame, criteria: Dict[str, str], weights: Optional[Dict[str, float]] = None,
                          zeta: float = 0.5) -> pd.Series:
    """Grey relational grade of each row. ``criteria`` maps column -> 'max' | 'min'."""
    gc = pd.DataFrame(index=df.index)
    for c, sense in criteria.items():
        x = df[c].astype(float)
        rng = x.max() - x.min()
        eta = pd.Series(1.0, index=df.index) if rng < 1e-12 else (
            (x - x.min()) / rng if sense == "max" else (x.max() - x) / rng)
        delta = 1 - eta
        gc[c] = (zeta) / (delta + zeta)                # delta_min = 0, delta_max = 1
    w = pd.Series({c: (weights or {}).get(c, 1.0) for c in criteria}, dtype=float)
    w = w / w.sum()
    return (gc * w).sum(axis=1)
