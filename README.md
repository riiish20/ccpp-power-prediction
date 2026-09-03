# Combined-Cycle Power Plant — Power Output Prediction

Predicting a power plant's net hourly electrical output from four ambient sensor
readings — and checking the model against turbine thermodynamics, not just accuracy.

**Tools:** Python · pandas · scikit-learn · matplotlib · seaborn

## Problem

A combined-cycle power plant's output changes with the ambient conditions around it.
Using 9,568 hours of real operating data (UCI dataset #294, 2006–2011, full load),
this project predicts **net hourly electrical output (PE, in MW)** from:

- **AT** — ambient temperature (°C)
- **V**  — exhaust vacuum (cm Hg)
- **AP** — ambient pressure (mbar)
- **RH** — relative humidity (%)

## Data & approach

- Loaded the data with `ucimlrepo`; no missing values, so no cleaning was needed.
- Explored feature–target scatter plots and a correlation matrix before modeling.
  Temperature (AT) is the strongest driver of power, with vacuum (V) second.
  - Engineered one physics-based feature — **inlet air density** — computed from AT, AP,
  and RH using the ideal gas law plus a water-vapor correction, since gas-turbine power
  is driven by inlet air mass flow.
- Fit a `LinearRegression` baseline, then read its residuals against each input.
  The residuals curve against temperature, so I added a squared temperature term
  (`AT²`), which improved the fit — a small physics-guided correction.
- Compared models with **5-fold cross-validation** (not a single split), so scores
  reflect performance across the whole dataset rather than one lucky test set:
  - `DummyRegressor` (predicts the mean) — the floor any real model must beat
  - `LinearRegression` (all four features)
  - `RandomForestRegressor`

## Results



| Model | RMSE (MW) | R² |
|---|---|---|
| Dummy (mean) | 17.067 | ~0.00 |
| Linear regression | 4.433 | .9285 |
| Random forest | 3.349 | .9620 |

- Adding the engineered **air-density** feature changed linear-model RMSE from 4.560 to 4.33
  and random-forest RMSE from 3.362 to 3.349 — it helped the simple model but barely moved the forest,
  which makes sense: trees already capture those interactions on their own.

- **Best model:** Random Forest, with cross-validated RMSE ≈ 3.349 MW and R² ≈ .9620.
- **Most important inputs (random forest):** AT first, then V — consistent with
  ambient temperature driving the gas turbine and exhaust vacuum driving the steam turbine.
- Adding `AT²` to the linear model changed its RMSE from `4.394` to `4.302`.

## The physics check

Holding the other inputs at their median and sweeping one at a time:

- **Temperature:** predicted power falls as air warms — matches
  gas-turbine behaviour, where colder, denser inlet air raises mass flow and output.
- **Exhaust vacuum:** predicted power shows a falling, smooth trend, consistent
  with the steam cycle.
- **Sanity check:** predictions stay within the real plant's range (425.6 – 495.5 MW) and
  never go negative — the model doesn't predict anything physically impossible.

## Limitations & next steps

- Trained only at **full load** — not valid for part-load operation.
- A random forest **can't extrapolate** beyond the temperatures it trained on, so it would be
  unreliable for conditions outside this dataset's range.
- Assumes the plant's physics stays fixed — equipment aging or fouling would shift real behaviour.
- **Next:** add prediction intervals for uncertainty, and test on held-out temperature
  extremes to measure how the model degrades outside its training range.

## How to run

```bash
pip install -r requirements.txt
```

Open `ccpp_power_prediction.ipynb` in Jupyter or Google Colab and run it top to bottom.

## Data source

Combined Cycle Power Plant dataset — UCI Machine Learning Repository (id 294):
https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
