"""Train and evaluate stress-level models.

Pipeline
  1. Load the cleaned dataset (374 real rows, only 132 unique).
  2. Grow it to 1000 rows with label-preserving synthetic augmentation
     (small jitter on numeric features, categories copied from a real parent row).
  3. Evaluate honestly: split by UNIQUE real person-profile, augment only the
     training side, and score on real, never-augmented, de-duplicated test rows.
  4. Compare plain, regularised (Ridge / Lasso / ElasticNet) and tree models.
  5. Save the best model, metrics and the 1000-row dataset for the app.

Run:  python train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
SRC = ROOT.parent / "Sleep-Health-ML-Project" / "Datasets" / "cleaned-dataset.csv"
TARGET_ROWS = 1000
AUGMENT = False   # synthetic rows gave no accuracy gain on real people, so the app uses real data only
SEED = 42

NUMERIC = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
           "Heart Rate", "Daily Steps", "BP High", "BP Low"]
CATEGORICAL = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]
FEATURES = CATEGORICAL[:1] + ["Age", "Occupation", "Sleep Duration", "Quality of Sleep",
                              "Physical Activity Level", "BMI Category", "Heart Rate",
                              "Daily Steps", "Sleep Disorder", "BP High", "BP Low"]
TARGET = "Stress Level"

# (noise std, decimals, step to round to)
JITTER = {
    "Age": (2.0, 0, 1), "Sleep Duration": (0.2, 1, None), "Quality of Sleep": (0.4, 0, 1),
    "Physical Activity Level": (5.0, 0, 1), "Heart Rate": (1.5, 0, 1),
    "Daily Steps": (400.0, 0, 100), "BP High": (2.0, 0, 1), "BP Low": (1.5, 0, 1),
}


def load_real() -> pd.DataFrame:
    df = pd.read_csv(SRC).drop(columns=["Person ID"])
    df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")
    return df[FEATURES + [TARGET]].reset_index(drop=True)


def augment(real: pd.DataFrame, n_total: int, rng: np.random.Generator) -> pd.DataFrame:
    """Return `real` plus synthetic rows so the result has n_total rows."""
    n_new = n_total - len(real)
    if n_new <= 0:
        return real.copy()
    parents = real.sample(n=n_new, replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    lo, hi = real[NUMERIC].min(), real[NUMERIC].max()
    for col, (std, dec, step) in JITTER.items():
        vals = parents[col].astype(float) + rng.normal(0, std, n_new)
        if step:
            vals = np.round(vals / step) * step
        vals = vals.clip(lo[col], hi[col]).round(dec)
        parents[col] = vals.astype(int) if dec == 0 else vals
    return pd.concat([real, parents], ignore_index=True)


def make_model(name: str) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    alphas = np.logspace(-3, 2, 30)
    en = ElasticNetCV(alphas=alphas, l1_ratio=[0.2, 0.5, 0.8], cv=5, max_iter=20000, random_state=SEED)
    est = {
        "Linear Regression": LinearRegression(),
        "Ridge (L2)": RidgeCV(alphas=alphas),
        "Lasso (L1)": LassoCV(alphas=alphas, cv=5, max_iter=20000, random_state=SEED),
        "ElasticNet (L1+L2)": ElasticNetCV(alphas=alphas, l1_ratio=[0.2, 0.5, 0.8], cv=5,
                                           max_iter=20000, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
        "Blend (ElasticNet + GB)": VotingRegressor([("en", en), ("gb", GradientBoostingRegressor(random_state=SEED))]),
    }[name]
    return Pipeline([("pre", pre), ("model", est)])


MODELS = ["Linear Regression", "Ridge (L2)", "Lasso (L1)", "ElasticNet (L1+L2)",
          "Random Forest", "Gradient Boosting", "Blend (ElasticNet + GB)"]


def score(y, p):
    return dict(r2=r2_score(y, p), rmse=float(np.sqrt(mean_squared_error(y, p))),
                mape=mean_absolute_percentage_error(y, p))


def evaluate(real: pd.DataFrame, n_splits: int = 20) -> pd.DataFrame:
    """Group-aware, leak-free evaluation (see module docstring)."""
    group = real.groupby(FEATURES + [TARGET]).ngroup().values
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=SEED)
    rows = []
    for k, (tr, te) in enumerate(gss.split(real, groups=group)):
        train = real.iloc[tr]
        test = real.iloc[te].drop_duplicates()          # unique, real, un-augmented
        rng = np.random.default_rng(SEED + k)
        scale = TARGET_ROWS / len(real)
        train_aug = augment(train, int(round(len(train) * scale)), rng)
        for name in MODELS:
            variants = [("real only", train)] + ([("augmented", train_aug)] if AUGMENT else [])
            for data_name, data in variants:
                m = make_model(name).fit(data[FEATURES], data[TARGET])
                s = score(test[TARGET], m.predict(test[FEATURES]))
                rows.append(dict(model=name, data=data_name, **s))
    res = pd.DataFrame(rows).groupby(["model", "data"], sort=False).agg(["mean", "std"])
    res.columns = ["_".join(c) for c in res.columns]
    return res.reset_index()


def main():
    real = load_real()
    print(f"real rows: {len(real)}  unique: {len(real.drop_duplicates())}")

    res = evaluate(real)
    print(res.round(3).to_string())

    aug = res[res["data"] == ("augmented" if AUGMENT else "real only")].sort_values("rmse_mean")
    best_name = aug.iloc[0]["model"]
    print("\nBest on real held-out data:", best_name)

    # ---- final dataset + model on all data -------------------------------
    rng = np.random.default_rng(SEED)
    full = augment(real, TARGET_ROWS, rng) if AUGMENT else real.copy()
    full.insert(0, "Source", ["Original"] * len(real) + ["Synthetic"] * (len(full) - len(real)))
    full.insert(0, "Person ID", range(1, len(full) + 1))
    full.to_csv(ROOT / "data" / "sleep_stress_data.csv", index=False)

    final = make_model(best_name).fit(full[FEATURES], full[TARGET])
    joblib.dump(final, ROOT / "models" / "stress_model.joblib")

    # regularisation story: coefficients of each linear model (standardised features)
    coefs = {}
    for name in MODELS[:4]:
        m = make_model(name).fit(full[FEATURES], full[TARGET])
        names = m.named_steps["pre"].get_feature_names_out()
        coefs[name] = dict(zip([n.split("__", 1)[1] for n in names],
                               np.round(m.named_steps["model"].coef_, 4).tolist()))
        est = m.named_steps["model"]
        coefs[name]["_alpha"] = float(getattr(est, "alpha_", 0.0))
    zeroed = [k for k, v in coefs["Lasso (L1)"].items() if k != "_alpha" and abs(v) < 1e-9]

    metrics = dict(
        best_model=best_name,
        n_original=int(len(real)), n_unique_original=int(len(real.drop_duplicates())),
        n_total=int(len(full)), n_synthetic=int(len(full) - len(real)),
        results=res.round(4).to_dict(orient="records"),
        coefficients=coefs, lasso_zeroed=zeroed,
        feature_ranges={c: [float(full[c].min()), float(full[c].max())] for c in NUMERIC},
        categories={c: sorted(full[c].unique().tolist()) for c in CATEGORICAL},
        stress_range=[int(full[TARGET].min()), int(full[TARGET].max())],
        occupation_stress=full.groupby("Occupation")[TARGET].mean().round(2).to_dict(),
    )
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("saved model, metrics and dataset; Lasso zeroed:", zeroed)


if __name__ == "__main__":
    main()
