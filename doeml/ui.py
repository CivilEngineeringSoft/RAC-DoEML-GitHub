"""Small Streamlit helpers."""

from __future__ import annotations

import io
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MODEL_COLORS = {"XGBoost": "#1f77b4", "CatBoost": "#d62728", "HistGB": "#2ca02c", "Mean": "black"}


def fig_bytes(fig, fmt="png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300 if fmt == "png" else None, bbox_inches="tight")
    return buf.getvalue()


def show_fig(fig, key: str, name: str, width="stretch"):
    st.pyplot(fig, width=width)
    c1, c2, _ = st.columns([1, 1, 5])
    c1.download_button("PNG", fig_bytes(fig, "png"), f"{name}.png", "image/png", key=f"{key}_png",
                       icon=":material/download:", on_click="ignore")
    c2.download_button("PDF", fig_bytes(fig, "pdf"), f"{name}.pdf", "application/pdf", key=f"{key}_pdf",
                       icon=":material/download:", on_click="ignore")
    plt.close(fig)


def excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name[:31])
    return buf.getvalue()


def parse_levels(text: str, numeric: bool = True):
    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
    if not numeric:
        return parts
    vals = [float(p) for p in parts]
    if len(set(vals)) != len(vals):
        raise ValueError("duplicated levels")
    return vals
