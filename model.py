"""
model.py - data preparation and machine-learning logic
=======================================================
Everything that is NOT user-interface code lives here:

  1. generate_sample_data()   - creates a realistic historical sales dataset
  2. preprocess_data()        - cleans the raw data and adds date features
  3. make_monthly()           - turns daily records into one row per month (ML-ready)
  4. train_and_evaluate()     - trains Linear Regression + Random Forest, picks the best
  5. forecast_future()        - predicts revenue for future months
  6. backtest_forecast()      - checks how good the forecasting method really is

This file does not import Streamlit, so it can be tested on its own.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
REQUIRED_COLUMNS = ["Date", "Product", "Category", "Region", "UnitsSold",
                    "Price", "Revenue", "Discount", "MarketingSpend"]
NUMERIC_COLUMNS = ["UnitsSold", "Price", "Revenue", "Discount", "MarketingSpend"]

# Features used to predict Revenue (in the monthly table, "Month" is the month number 1-12)
FEATURES = ["UnitsSold", "Price", "Discount", "MarketingSpend", "Year", "Month", "Quarter"]
TARGET = "Revenue"
MODEL_NAMES = ["Linear Regression", "Random Forest"]
MIN_MONTHS = 12  # minimum months of history needed to train and forecast


# ----------------------------------------------------------------------
# 1. SAMPLE DATA GENERATOR
# ----------------------------------------------------------------------
def generate_sample_data(seed=42, start="2023-10-01", end="2026-09-30"):
    """
    Create 3 years of daily sales for 6 products (about 6,500 records).

    Built-in patterns the model can discover:
      * yearly growth (some products grow, some decline)
      * seasonality (Nov/Dec peak)
      * discounts increase units sold
      * marketing spend slightly increases units sold
    A few missing values, duplicates and extreme values are added on purpose
    so the cleaning step has something to do.
    """
    rng = np.random.default_rng(seed)

    # product: (category, base price, base units/day, yearly growth, base marketing/day)
    meta = pd.DataFrame.from_dict(
        {
            "Laptop":        ("Electronics", 850, 6, 0.10, 400),
            "Smartphone":    ("Electronics", 600, 10, 0.15, 450),
            "Office Chair":  ("Furniture", 220, 7, 0.05, 150),
            "Standing Desk": ("Furniture", 380, 4, -0.03, 150),
            "Running Shoes": ("Lifestyle", 90, 14, 0.18, 200),
            "Backpack":      ("Lifestyle", 45, 12, -0.06, 100),
        },
        orient="index",
        columns=["Category", "BasePrice", "BaseUnits", "Growth", "BaseMarketing"],
    )
    # Monthly seasonality (Jan..Dec) for each category
    season = {
        "Electronics": np.array([0.92, 0.88, 0.95, 1.0, 1.0, 0.95, 0.93, 1.0, 1.05, 1.08, 1.35, 1.45]),
        "Furniture":   np.array([0.95, 0.95, 1.05, 1.1, 1.1, 1.05, 1.0, 0.95, 0.95, 0.95, 1.05, 1.0]),
        "Lifestyle":   np.array([0.9, 0.9, 1.0, 1.1, 1.15, 1.2, 1.1, 1.05, 1.0, 0.95, 1.2, 1.3]),
    }

    # One record per product per day
    dates = pd.date_range(start, end, freq="D")
    grid = pd.MultiIndex.from_product([dates, list(meta.index)], names=["Date", "Product"]).to_frame(index=False)
    grid = grid.join(meta, on="Product")
    n = len(grid)

    years = ((grid["Date"] - pd.Timestamp(start)).dt.days / 365.25).values
    month = grid["Date"].dt.month.values

    seasonal = np.ones(n)
    for cat, factors in season.items():
        mask = (grid["Category"] == cat).values
        seasonal[mask] = factors[month[mask] - 1]
    weekday = np.array([0.95, 0.95, 0.95, 0.95, 1.0, 1.15, 1.1])[grid["Date"].dt.dayofweek.values]

    # Region (random per record) and its effect on demand
    regions = np.array(["North", "South", "East", "West"])
    region = rng.choice(regions, n, p=[0.30, 0.20, 0.25, 0.25])
    region_factor = pd.Series(region).map({"North": 1.0, "South": 0.9, "East": 1.05, "West": 1.1}).values

    # Discount: bigger and more common during the Nov/Dec sales season
    levels = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    normal = rng.choice(levels, n, p=[0.50, 0.15, 0.15, 0.10, 0.07, 0.03])
    sale = rng.choice(levels, n, p=[0.15, 0.10, 0.15, 0.20, 0.25, 0.15])
    discount = np.where(np.isin(month, [11, 12]), sale, normal)

    # Marketing spend (follows seasonality + noise) and its small effect on demand
    noise = rng.lognormal(0, 0.3, n)
    marketing = grid["BaseMarketing"].values * (1 + 0.08 * years) * (1 + 0.5 * (seasonal - 1)) * noise
    lift = np.clip(1 + 0.3 * (noise - 1), 0.6, 1.6)

    # Units sold = expected demand + random variation
    expected = (grid["BaseUnits"].values * (1 + grid["Growth"].values) ** years * seasonal
                * weekday * region_factor * (1 + 2.2 * discount) * lift)
    units = rng.poisson(expected).astype(float)

    # A few extreme values (e.g. data-entry mistakes) to demonstrate outlier handling
    spikes = rng.choice(n, 6, replace=False)
    units[spikes] *= 6

    price = grid["BasePrice"].values * (1.03 ** years) * (1 + rng.normal(0, 0.01, n))
    price = price.round(2)
    revenue = (units * price * (1 - discount)).round(2)

    df = pd.DataFrame({
        "Date": grid["Date"].dt.strftime("%Y-%m-%d"),
        "Product": grid["Product"],
        "Category": grid["Category"],
        "Region": region,
        "UnitsSold": units,
        "Price": price,
        "Revenue": revenue,
        "Discount": discount,
        "MarketingSpend": marketing.round(2),
    })

    # Add missing values
    for col, how_many in [("UnitsSold", 10), ("Price", 12), ("Revenue", 15),
                          ("Discount", 25), ("MarketingSpend", 30), ("Region", 12)]:
        df.loc[rng.choice(n, how_many, replace=False), col] = np.nan
    df["UnitsSold"] = df["UnitsSold"].astype("Float64").astype(float)

    # Add duplicate rows
    df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)
    return df


# ----------------------------------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------------------------------
def add_time_features(df):
    """Create Year, Month (name), Quarter, Day and MonthNumber from the Date column."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.strftime("%b")
    df["Quarter"] = df["Date"].dt.quarter
    df["Day"] = df["Date"].dt.day
    df["MonthNumber"] = df["Date"].dt.month
    return df


def preprocess_data(raw):
    """
    Clean the raw dataset. Returns (clean_dataframe, report_dictionary).

    Steps: remove duplicates -> convert Date -> sort -> fix numbers -> fill missing
           values -> handle outliers -> add date features.
    """
    df = raw.copy()
    report = {"rows_before": len(df), "missing_before": raw.isna().sum()}

    # 1. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    # 2. Convert Date to datetime (rows with an unreadable date are dropped)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    report["bad_dates_removed"] = int(df["Date"].isna().sum())
    df = df.dropna(subset=["Date"])

    # 3. Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    # 4. Make sure numeric columns are numbers
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # If Discount looks like a percentage (0-100) convert it to a fraction (0-1)
    report["discount_rescaled"] = bool(df["Discount"].max() > 1)
    if report["discount_rescaled"]:
        df["Discount"] = df["Discount"] / 100

    # 5. Fill missing text values
    df["Product"] = df["Product"].fillna("Unknown")
    category_of = df.dropna(subset=["Category"]).groupby("Product")["Category"].agg(lambda s: s.mode().iloc[0])
    df["Category"] = df["Category"].fillna(df["Product"].map(category_of)).fillna("Unknown")
    region_mode = df["Region"].mode()
    df["Region"] = df["Region"].fillna(region_mode.iloc[0] if len(region_mode) else "Unknown")

    # 6. Fill missing numbers with the median of the same product
    for col in ["UnitsSold", "Price", "Discount", "MarketingSpend"]:
        df[col] = df[col].fillna(df.groupby("Product")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())
    # Missing revenue can be calculated: units x price x (1 - discount)
    calc = df["UnitsSold"] * df["Price"] * (1 - df["Discount"])
    df["Revenue"] = df["Revenue"].fillna(calc)

    # 7. Outliers - IQR rule on UnitsSold within each product
    q1 = df["Product"].map(df.groupby("Product")["UnitsSold"].quantile(0.25))
    q3 = df["Product"].map(df.groupby("Product")["UnitsSold"].quantile(0.75))
    iqr = q3 - q1
    flagged = (df["UnitsSold"] > q3 + 1.5 * iqr) | (df["UnitsSold"] < q1 - 1.5 * iqr)
    extreme = df["UnitsSold"] > q3 + 3.0 * iqr  # only very extreme values are corrected
    df.loc[extreme, "UnitsSold"] = (q3 + 3.0 * iqr)[extreme].round()
    df.loc[extreme, "Revenue"] = calc.where(~extreme, df["UnitsSold"] * df["Price"] * (1 - df["Discount"]))[extreme]
    report["outliers"] = (
        pd.DataFrame({"Product": df["Product"], "Flagged (1.5 x IQR)": flagged, "Capped (3 x IQR)": extreme})
        .groupby("Product").sum().astype(int).reset_index()
    )

    # 8. Add date features
    df = add_time_features(df)

    report["rows_after"] = len(df)
    report["missing_after"] = df[REQUIRED_COLUMNS].isna().sum()
    return df, report


# ----------------------------------------------------------------------
# 3. PREPARE DATA FOR MACHINE LEARNING
# ----------------------------------------------------------------------
def make_monthly(df):
    """
    Combine daily records into ONE ROW PER MONTH.

    Price and Discount are weighted averages (so busy days count more).
    Incomplete first/last months are dropped because they would look artificially low.
    """
    d = df.copy()
    d["Gross"] = d["UnitsSold"] * d["Price"]            # sales before discount
    d["DiscountAmount"] = d["Gross"] * d["Discount"]
    m = d.groupby(pd.Grouper(key="Date", freq="MS")).agg(
        Revenue=("Revenue", "sum"), UnitsSold=("UnitsSold", "sum"),
        MarketingSpend=("MarketingSpend", "sum"), Gross=("Gross", "sum"),
        DiscountAmount=("DiscountAmount", "sum"), Records=("Revenue", "size"),
    )
    m = m[m["Records"] > 0].reset_index()
    m["Price"] = m["Gross"] / m["UnitsSold"].replace(0, np.nan)
    m["Discount"] = m["DiscountAmount"] / m["Gross"].replace(0, np.nan)

    # Drop partial months at the edges (needs at least 3 months to make sense)
    if len(m) > 3:
        first, last = df["Date"].min(), df["Date"].max()
        if last.days_in_month - last.day > 6:
            m = m.iloc[:-1]
        if first.day > 7:
            m = m.iloc[1:]

    m["Year"] = m["Date"].dt.year
    m["Month"] = m["Date"].dt.month      # month NUMBER (1-12) used as an ML feature
    m["Quarter"] = m["Date"].dt.quarter
    m = m.drop(columns=["Gross", "DiscountAmount", "Records"]).dropna().reset_index(drop=True)
    return m[["Date", "Revenue"] + FEATURES]


# ----------------------------------------------------------------------
# 4. MODEL TRAINING AND EVALUATION
# ----------------------------------------------------------------------
def build_model(name):
    """Create an untrained model by name."""
    if name == "Linear Regression":
        # Scaling makes the coefficients comparable with each other
        return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    return RandomForestRegressor(n_estimators=200, random_state=42)


def feature_importance(model, name):
    """How much each feature matters (0-1, adds up to 1)."""
    if name == "Linear Regression":
        raw = np.abs(model.named_steps["model"].coef_)
    else:
        raw = model.feature_importances_
    imp = pd.DataFrame({"Feature": FEATURES, "Importance": raw / raw.sum()})
    return imp.sort_values("Importance", ascending=False).reset_index(drop=True)


def train_and_evaluate(monthly):
    """
    Train both models on the OLDER months and test them on the most RECENT months
    (a time-based split, because we want to predict the future from the past).
    """
    n = len(monthly)
    n_test = max(3, int(round(n * 0.2)))
    train, test = monthly.iloc[:-n_test], monthly.iloc[-n_test:]

    rows, fitted = [], {}
    for name in MODEL_NAMES:
        model = build_model(name).fit(train[FEATURES], train[TARGET])
        pred = model.predict(test[FEATURES])
        mse = mean_squared_error(test[TARGET], pred)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(test[TARGET], pred),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "R² Score": r2_score(test[TARGET], pred),
        })
        fitted[name] = model

    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best = metrics.loc[0, "Model"]  # lowest RMSE (this is also the highest R²)

    # Actual vs predicted for every month (train months are "fitted", test months are real predictions)
    ap = pd.DataFrame({
        "Date": monthly["Date"],
        "Set": ["Train"] * (n - n_test) + ["Test"] * n_test,
        "Actual Revenue": monthly[TARGET].values,
        "Predicted Revenue": fitted[best].predict(monthly[FEATURES]),
    })
    ap["Error"] = ap["Actual Revenue"] - ap["Predicted Revenue"]
    ap["Abs Error"] = ap["Error"].abs()
    ap["Error %"] = ap["Error"] / ap["Actual Revenue"] * 100

    # For forecasting, re-train the winning model on ALL months
    final = build_model(best).fit(monthly[FEATURES], monthly[TARGET])
    return {
        "metrics": metrics, "best": best, "actual_vs_pred": ap, "final_model": final,
        "importance": feature_importance(final, best),
        "n_train": n - n_test, "n_test": n_test,
        "train_start": train["Date"].min(), "train_end": train["Date"].max(),
        "test_start": test["Date"].min(), "test_end": test["Date"].max(),
    }


# ----------------------------------------------------------------------
# 5. FUTURE FORECASTING
# ----------------------------------------------------------------------
def _month_index(dates, origin):
    """Number of months between each date and the origin date."""
    return (dates.dt.year - origin.year) * 12 + (dates.dt.month - origin.month)


def project_drivers(monthly, n_periods):
    """
    The model needs UnitsSold, Price, Discount and MarketingSpend to predict revenue,
    but we do not know them in the future. So we project each one as:

        future value = straight-line TREND + average SEASONAL effect of that calendar month
    """
    origin = monthly["Date"].min()
    t = _month_index(monthly["Date"], origin).values
    future_dates = pd.date_range(monthly["Date"].max() + pd.offsets.MonthBegin(1), periods=n_periods, freq="MS")
    t_future = _month_index(pd.Series(future_dates), origin).values

    out = pd.DataFrame({"Forecast Date": future_dates})
    for col in ["UnitsSold", "Price", "Discount", "MarketingSpend"]:
        y = monthly[col].values
        slope, intercept = np.polyfit(t, y, 1)
        seasonal = pd.Series(y - (slope * t + intercept)).groupby(monthly["Month"].values).mean()
        out[col] = slope * t_future + intercept + pd.Series(future_dates.month).map(seasonal).fillna(0).values
    out["UnitsSold"] = out["UnitsSold"].clip(lower=0)
    out["Price"] = out["Price"].clip(lower=0)
    out["Discount"] = out["Discount"].clip(0, 0.9)
    out["MarketingSpend"] = out["MarketingSpend"].clip(lower=0)
    out["Year"] = future_dates.year
    out["Month"] = future_dates.month
    out["Quarter"] = future_dates.quarter
    return out


def forecast_future(monthly, model, n_periods):
    """Predict revenue for the next n_periods months."""
    fc = project_drivers(monthly, n_periods)
    fc["Predicted Revenue"] = np.clip(model.predict(fc[FEATURES]), 0, None)
    return fc


def trend_direction(monthly, fc):
    """
    Is the forecast going up or down? We compare it with the same calendar months
    in the past (so normal seasonality does not confuse the answer).
    """
    latest_by_month = monthly.groupby("Month")["Revenue"].last()
    ref = fc["Month"].map(latest_by_month)
    if ref.notna().sum() >= max(1, len(fc) // 2):
        ok = ref.notna()
        pct = fc.loc[ok, "Predicted Revenue"].sum() / ref[ok].sum() - 1
        basis = "the same months last year"
    else:
        pct = fc["Predicted Revenue"].mean() / monthly["Revenue"].tail(len(fc)).mean() - 1
        basis = f"the last {len(fc)} months"
    if pct > 0.02:
        label, arrow = "Upward", "▲"
    elif pct < -0.02:
        label, arrow = "Downward", "▼"
    else:
        label, arrow = "Stable", "►"
    return {"label": label, "arrow": arrow, "pct": pct * 100, "basis": basis}


def backtest_forecast(monthly, model_name):
    """
    Honest test of the WHOLE forecasting method: hide the last few months, forecast them
    from the earlier data only, then compare with what really happened.
    """
    n = len(monthly)
    h = min(6, n // 3)
    if h < 2 or n - h < MIN_MONTHS:
        return None
    train, test = monthly.iloc[:-h], monthly.iloc[-h:]
    model = build_model(model_name).fit(train[FEATURES], train[TARGET])
    fc = forecast_future(train, model, h)
    actual, pred = test[TARGET].values, fc["Predicted Revenue"].values
    return {
        "months": h,
        "mape": float(np.mean(np.abs(actual - pred) / actual) * 100),
        "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
    }
