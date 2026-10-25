"""The three regression models of the paper (Table 5) and their training / loading."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (max_error, mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split

from .data import (APP_DIR, COEF, FEATURES, GRADES, TARGET, grade_from_cylinder_strength,
                   initial_grade, load_data, prepare, strength_coefficient)

SEED = 42
TEST_SIZE = 0.2
MODEL_FILE = APP_DIR / "models" / "models.joblib"
META_FILE = APP_DIR / "models" / "models_meta.json"

# optimal hyperparameters (Table 5)
PAPER_PARAMS = {
    "XGBoost": {"n_estimators": 300, "learning_rate": 0.170, "max_depth": 3, "gamma": 0.9, "subsample": 0.940},
    "CatBoost": {"iterations": 1000, "learning_rate": 0.111, "depth": 4},
    "HistGB": {"max_iter": 283, "learning_rate": 0.104, "max_depth": 8},
}
MODEL_NAMES = list(PAPER_PARAMS)


def make(name: str):
    p = PAPER_PARAMS[name]
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0, **p)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=1, **p)
    return HistGradientBoostingRegressor(random_state=SEED, **p)


def metrics(y, p) -> Dict[str, float]:
    mse = mean_squared_error(y, p)
    return {"MSE": mse, "R2": r2_score(y, p), "Max error (MPa)": max_error(y, p),
            "MAPE": mean_absolute_percentage_error(y, p), "RMSE (MPa)": float(np.sqrt(mse)),
            "MAE (MPa)": mean_absolute_error(y, p)}


def library_versions() -> Dict[str, str]:
    import catboost, sklearn, xgboost
    return {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "catboost": catboost.__version__}


def train(progress=None) -> dict:
    df = load_data()
    X, y = prepare(df), df[TARGET].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    models, rows = {}, {}
    for name in MODEL_NAMES:
        if progress:
            progress(name)
        m = make(name).fit(Xtr, ytr)
        models[name] = m
        for part, (xx, yy) in {"Training": (Xtr, ytr), "Testing": (Xte, yte), "Total": (X, y)}.items():
            for k, v in metrics(yy, m.predict(xx)).items():
                rows.setdefault(name, {})[(part, k)] = v
    perf = pd.DataFrame(rows)
    perf.index = pd.MultiIndex.from_tuples(perf.index, names=["Dataset", "Metric"])
    ranges = {c: (float(X[c].min()), float(X[c].max())) for c in FEATURES}
    return {"models": models, "performance": perf, "ranges": ranges, "n": len(X),
            "n_train": len(Xtr), "n_test": len(Xte)}


def save(bundle: dict) -> None:
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE, compress=3)
    META_FILE.write_text(json.dumps({"versions": library_versions(), "seed": SEED, "test_size": TEST_SIZE}, indent=2))


def load(progress=None) -> Tuple[dict, str]:
    try:
        meta = json.loads(META_FILE.read_text())
        if meta.get("versions") == library_versions() and MODEL_FILE.exists():
            return joblib.load(MODEL_FILE), "file"
    except Exception:
        pass
    return train(progress), "retrained"


# --------------------------------------------------------------------------- prediction
def predict(models: Dict[str, object], mixes: List[dict], names: Iterable[str],
            specimens: List[str], grades: Optional[List[Optional[str]]] = None,
            max_iter: int = 6) -> pd.DataFrame:
    """Predict the 28-day compressive strength of mixes.

    ``mixes`` are raw input dicts (without the strength coefficient). The strength
    coefficient depends on the specimen and on the strength grade; when the grade is
    not imposed, it is found iteratively: start from C20-40 (low) or C50 (high),
    predict, convert to the equivalent 160x320 cylinder strength, update the grade,
    and repeat until the grade no longer changes.

    Returns one row per mix and model: prediction on the specimen, grade,
    coefficient and equivalent cylinder strength."""
    names = list(names)
    grades = grades or [None] * len(mixes)
    out = []
    for name in names:
        model = models[name]
        g = [gr if gr else initial_grade(m["Strength classification feature (Expected_strength)"] == 1)
             for m, gr in zip(mixes, grades)]
        fixed = [gr is not None for gr in grades]
        for it in range(max_iter):
            X = pd.DataFrame([{**m, COEF: strength_coefficient(s, gg)} for m, s, gg in zip(mixes, specimens, g)])[FEATURES]
            fc = model.predict(X)
            coef = X[COEF].to_numpy()
            new = [gg if fx else grade_from_cylinder_strength(f * c) for gg, fx, f, c in zip(g, fixed, fc, coef)]
            if new == g:
                break
            g = new
        for i, (f, c, gg) in enumerate(zip(fc, coef, g)):
            out.append({"mix": i, "Model": name, "fc specimen (MPa)": float(f), "Grade": gg,
                        "Strength coefficient": float(c), "fc cylinder 160x320 (MPa)": float(f * c),
                        "Grade iterations": it + 1})
    return pd.DataFrame(out)


def out_of_range(mix: dict, ranges: Dict[str, Tuple[float, float]]) -> List[str]:
    return [c for c, v in mix.items() if c in ranges and not (ranges[c][0] - 1e-9 <= v <= ranges[c][1] + 1e-9)]
