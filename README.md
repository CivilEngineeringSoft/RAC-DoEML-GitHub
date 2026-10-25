# RAC DoE-ML — Streamlit app (private repository)

Web application accompanying the article *Experimental Analysis and Taguchi Optimization of the
Residual Mechanical Behavior of Recycled Aggregate Concretes at Different Temperature Levels*
(Cirhuza Badesire, Di Cesare, Vu, Bishweka).

- **Public app:** https://rac-doeml-app.streamlit.app (update this link once the app is deployed)
- This repository (code, dataset, trained models) is **private**. The app never displays or exports
  the dataset.

© 2026 Université Claude Bernard Lyon 1, LMC² (EA 7427). All rights reserved.

## What it does

### 1. Predict the 28-day compressive strength
- Three models of the paper, trained on 594 RAC mixes with the optimal hyperparameters of
  Table 5, an 80 % / 20 % split (random seed 42), and empty cells = 0:
  - **XGBoost**
  - **CatBoost**
  - **HistGB**
- Inputs:
  - the composition: cement, fly ash, W/B, admixture, natural and recycled coarse and fine
    aggregates with their properties;
  - the specimen;
  - the **expected strength**: high (≥ 45 MPa) or low (< 45 MPa).

  The replacement ratios are computed from the quantities.
- **Strength (specimen shape) coefficient:**
  - It is taken from the conversion table (σ_cyl/σ_x, by specimen and strength grade).
  - The grade is automatic: start from C20-40 (low) or C50 (high), predict, convert to the
    160 × 320 mm cylinder, update the grade (limits 45/55/65/75 MPa), and repeat until it is
    stable.
  - The grade can also be imposed.
- **Results:**
  - strength on the specimen and the equivalent 160 × 320 mm cylinder strength;
  - a warning when the prediction contradicts the expected strength class;
  - a warning when the inputs are outside the training data (extrapolation).
- One mix in a form, or several mixes from the Excel template. Results can be downloaded.

### 2. Find the optimal mix with a design of experiments
1. **Setup:**
   - specimen, strength class and models;
   - one or several **base compositions**: by default the Fuller and Dreux–Gorisse mixes of the
     paper; several bases become a factor of the design;
   - aggregate properties.
2. **Factors and levels:** CA and FA replacement ratios, admixture, W/B, cement, fly ash.
   - Defaults are those of the paper: 0/20/60 %, 0/20/30 %, 0/1/1.5 %, 0.50/0.53.
3. **Design:**
   - **full factorial**: 108 mixes, as in the paper;
   - **Taguchi** orthogonal arrays: L4, L8, L9, L16, L16(4⁵), L18, L25, L27, L32, chosen
     automatically or by the user, with the dummy-level technique;
   - **Box–Behnken**;
   - **face-centred central composite**;
   - **Latin hypercube**.
4. **Optimisation criterion:**
   - maximum predicted strength, as in the paper;
   - maximum recycled-aggregate content reaching a target strength;
   - **multi-criteria** strength–cost–CO₂ with Grey Relational Analysis, using the unit prices
     and emission factors of Table 4 (editable).
5. **Results:**
   - the optimal mix;
   - all mixes, with predictions, cost and CO₂;
   - predictions per run (as in Fig. 9) and main effects (as in Fig. 10), including the
     Taguchi signal-to-noise ratio;
   - ANOVA with contributions (as in Table 7);
   - an Excel download.

**Reproduction of the paper:**
- the full factorial gives 108 mixes;
- CatBoost's optimum is the Table 8 composition: Dreux–Gorisse base, 0 % RCA, 0 % RFA, 1.5 %
  admixture, W/B = 0.50;
- XGBoost predicts 51.5 MPa for it (paper: 51.7 MPa);
- the ANOVA ranks the CA replacement ratio first for XGBoost, as in Table 7.

## Run locally

```bash
git clone https://github.com/CivilEngineeringSoft/RAC-DoEML.git   # requires access
cd RAC-DoEML
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud (private repository, public app)

1. Create the **private** repository `CivilEngineeringSoft/RAC-DoEML` on GitHub. Upload the
   **contents** of this folder, so that `streamlit_app.py`, `requirements.txt` and
   `packages.txt` are at the top level of the repository.
2. Go to https://share.streamlit.io. Click **Create app → Deploy a public app from GitHub** and
   fill in:
   - **Repository:** `CivilEngineeringSoft/RAC-DoEML`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** `rac-doeml`
   - **Advanced settings:** Python 3.12
3. Click **Deploy**.
4. The app is private at first, because the repository is private. Open it and click **Share** →
   **"This app is public and searchable"**.

## Folder content

| Path | Content |
|---|---|
| `streamlit_app.py` | entry point |
| `views/` | pages: home, prediction, DoE optimisation |
| `doeml/data.py` | variables, specimen coding, conversion factors, grade, mix construction, template |
| `doeml/models.py` | the three models (Table 5), training/loading, prediction with grade iteration |
| `doeml/doe.py` | designs (full factorial, Taguchi, Box–Behnken, CCD, LHS), ANOVA, S/N, cost, CO₂, GRA |
| `doeml/optimise.py` | design → mixes → predictions → optimum |
| `data/` | dataset (594 mixes) — private |
| `models/` | trained models (retrained automatically if library versions differ) |
| `tests/` | `pytest tests` |
| `docs/` | text and figure to add to the paper |
