"""
Predictive Analytics and Future Trend Forecasting Dashboard
============================================================
Streamlit dashboard that explores historical sales data, trains regression models,
evaluates them and forecasts future revenue.

Run with:   streamlit run app.py

Code layout
  models/model.py  -> data cleaning + machine learning (no Streamlit)
  app.py           -> dashboard, charts and automatic insights (this file)
"""

import calendar
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for Streamlit
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from models.model import (
    REQUIRED_COLUMNS, MIN_MONTHS,
    generate_sample_data, preprocess_data, make_monthly,
    train_and_evaluate, forecast_future, trend_direction, backtest_forecast,
)

# ----------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "historical_sales.csv"
BLUE, ORANGE, GREEN, RED, GREY = "#2563EB", "#F59E0B", "#10B981", "#EF4444", "#6B7280"

PAGES = [
    "📊 Overview",
    "📈 Historical Analysis",
    "🤖 Predictive Modeling",
    "🔮 Future Forecast",
    "💡 Business Insights",
    "🗂️ Data & Downloads",
]

sns.set_theme(style="whitegrid", font_scale=0.9)


# ----------------------------------------------------------------------
# 2. CACHED DATA / MODEL FUNCTIONS (so the app stays fast)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes=None):
    """Load an uploaded CSV, the bundled CSV, or (last resort) generate sample data."""
    if uploaded_bytes is not None:
        return pd.read_csv(io.BytesIO(uploaded_bytes))
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    df = generate_sample_data()
    try:
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    except OSError:
        pass
    return df


@st.cache_data(show_spinner=False)
def cached_preprocess(raw):
    return preprocess_data(raw)


@st.cache_data(show_spinner="Training models...")
def cached_models(monthly):
    return train_and_evaluate(monthly)


@st.cache_data(show_spinner=False)
def cached_backtest(monthly, model_name):
    return backtest_forecast(monthly, model_name)


# ----------------------------------------------------------------------
# 3. SMALL HELPERS
# ----------------------------------------------------------------------
def fmt_money(x):
    """Short money format: $1.25M, $830.4K, $95."""
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.1f}K"
    return f"${x:,.0f}"


def money_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_money(v)))


def show(fig):
    """Display a Matplotlib figure in Streamlit and free its memory."""
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def inject_css():
    st.markdown(
        """
        <style>
        .main-title {background: linear-gradient(90deg,#1E3A8A,#2563EB); color:white;
                     padding: 1.1rem 1.4rem; border-radius: 12px; margin-bottom: 1rem;}
        .main-title h1 {color:white; font-size:1.7rem; margin:0; padding:0;}
        .main-title p {color:#DBEAFE; margin:0.2rem 0 0 0; font-size:0.95rem;}
        .kpi {background:#FFFFFF; border:1px solid #E5E7EB; border-left:5px solid #2563EB;
              border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.6rem;
              box-shadow: 0 1px 3px rgba(0,0,0,0.06);}
        .kpi-label {color:#6B7280; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.04em;}
        .kpi-value {color:#111827; font-size:1.45rem; font-weight:700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title, subtitle):
    st.markdown(f'<div class="main-title"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi_row(df, results):
    """Six KPI cards."""
    avg_price = (df["UnitsSold"] * df["Price"]).sum() / df["UnitsSold"].sum()
    n_months = max(1, df["Date"].dt.to_period("M").nunique())
    r2 = f"{results['metrics'].loc[0, 'R² Score']:.3f}" if results else "N/A"
    items = [
        ("Total Revenue", fmt_money(df["Revenue"].sum())),
        ("Avg Monthly Revenue", fmt_money(df["Revenue"].sum() / n_months)),
        ("Total Units Sold", f"{df['UnitsSold'].sum():,.0f}"),
        ("Average Price", f"${avg_price:,.2f}"),
        ("Marketing Spend", fmt_money(df["MarketingSpend"].sum())),
        ("Best Model R² Score", r2),
    ]
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.markdown(
            f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
            unsafe_allow_html=True,
        )


def need_model(results):
    """Show a warning and return False when there is too little data to model."""
    if results is None:
        st.warning(f"Predictive modeling needs at least {MIN_MONTHS} full months of data. "
                   "Please widen the date range or filters in the sidebar.")
        return False
    return True


# ----------------------------------------------------------------------
# 4. CHARTS (Matplotlib + Seaborn)
# ----------------------------------------------------------------------
def chart_time_series(df, col, title, color, as_money=True):
    """Daily values (faint) with a 30-day moving average (bold)."""
    daily = df.groupby("Date")[col].sum()
    smooth = daily.rolling("30D").mean()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(daily.index, daily.values, color=color, alpha=0.25, lw=1, label="Daily")
    ax.plot(smooth.index, smooth.values, color=color, lw=2.2, label="30-day average")
    ax.set_title(title)
    ax.legend(fontsize=8)
    if as_money:
        money_axis(ax)
    fig.autofmt_xdate()
    show(fig)


def chart_monthly_bar(monthly):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(monthly))
    ax.bar(x, monthly["Revenue"], color=BLUE)
    step = max(1, len(monthly) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(monthly["Date"].dt.strftime("%b %y").iloc[::step], rotation=45)
    ax.set_title("Monthly Revenue")
    money_axis(ax)
    show(fig)


def chart_revenue_by(df, col, palette):
    data = df.groupby(col)["Revenue"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.barplot(data=data, x=col, y="Revenue", hue=col, palette=palette, legend=False, ax=ax)
    for bar in ax.patches:
        ax.annotate(fmt_money(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("")
    ax.set_title(f"Revenue by {col}")
    ax.tick_params(axis="x", rotation=25)
    money_axis(ax)
    show(fig)


def chart_marketing(monthly):
    r = monthly["MarketingSpend"].corr(monthly["Revenue"])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.regplot(data=monthly, x="MarketingSpend", y="Revenue", color=BLUE,
                line_kws={"color": RED}, scatter_kws={"s": 40, "alpha": 0.8}, ax=ax)
    ax.set_title(f"Marketing Spend vs Revenue (monthly, correlation = {r:.2f})")
    money_axis(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_money(v)))
    show(fig)


def chart_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.histplot(df["Revenue"], bins=40, kde=True, color=BLUE, ax=ax)
    ax.set_title("Sales Distribution (revenue per record)")
    ax.set_xlabel("Revenue per record ($)")
    show(fig)


def chart_seasonality(monthly):
    avg = monthly.groupby("Month")["Revenue"].mean().reindex(range(1, 13))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    colors = [ORANGE if v == avg.max() else BLUE for v in avg.fillna(0)]
    ax.bar([calendar.month_abbr[m] for m in avg.index], avg.fillna(0).values, color=colors)
    ax.set_title("Average Revenue by Calendar Month (seasonality)")
    money_axis(ax)
    show(fig)


def chart_actual_vs_pred(ap):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ap["Date"], ap["Actual Revenue"], marker="o", ms=4, color=BLUE, label="Actual revenue")
    ax.plot(ap["Date"], ap["Predicted Revenue"], marker="o", ms=4, ls="--", color=ORANGE, label="Predicted revenue")
    test = ap[ap["Set"] == "Test"]
    ax.axvspan(test["Date"].min(), test["Date"].max(), color=GREEN, alpha=0.12, label="Test period (unseen data)")
    ax.set_title("Actual vs Predicted Revenue")
    ax.legend(fontsize=8)
    money_axis(ax)
    fig.autofmt_xdate()
    show(fig)


def chart_errors(test):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    colors = [GREEN if e >= 0 else RED for e in test["Error"]]
    ax.bar(test["Date"].dt.strftime("%b %y"), test["Error"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Prediction Error on Test Months (Actual - Predicted)")
    ax.tick_params(axis="x", rotation=45)
    money_axis(ax)
    show(fig)


def chart_importance(imp, model_name):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.barplot(data=imp, y="Feature", x="Importance", hue="Feature", palette="Blues_r", legend=False, ax=ax)
    ax.set_title(f"What drives revenue? ({model_name})")
    ax.set_ylabel("")
    show(fig)


def chart_forecast(monthly, fc):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly["Date"], monthly["Revenue"], marker="o", ms=3, color=BLUE, label="Historical revenue")
    xs = [monthly["Date"].iloc[-1]] + list(fc["Forecast Date"])
    ys = [monthly["Revenue"].iloc[-1]] + list(fc["Predicted Revenue"])
    ax.plot(xs, ys, marker="o", ms=4, ls="--", color=ORANGE, label="Forecast revenue")
    ax.axvline(monthly["Date"].iloc[-1], color=GREY, ls=":", lw=1)
    ax.set_title("Historical Revenue and Future Forecast")
    ax.legend(fontsize=8)
    money_axis(ax)
    fig.autofmt_xdate()
    show(fig)


# ----------------------------------------------------------------------
# 5. AUTOMATIC INSIGHTS (generated from the data - nothing is hard-coded)
# ----------------------------------------------------------------------
def generate_insights(df, monthly, results, fc, trend, backtest):
    """Return a list of (title, text) insights built from the current data."""
    out = []
    n = len(monthly)
    total = df["Revenue"].sum()

    # 1. Revenue trend over time
    if n >= 4:
        w = min(12, n // 2)
        recent, prev = monthly["Revenue"].tail(w).sum(), monthly["Revenue"].iloc[-2 * w:-w].sum()
        g = (recent / prev - 1) * 100
        word = "increasing" if g > 2 else "decreasing" if g < -2 else "roughly flat"
        extra = "" if w == 12 else " (short windows can be affected by seasonality)"
        out.append(("📈 Revenue trend",
                    f"Revenue is **{word}**: the last {w} months brought in {fmt_money(recent)} compared with "
                    f"{fmt_money(prev)} in the {w} months before ({g:+.1f}%){extra}."))

    # 2. Products, category and region
    prod = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
    cat = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    reg = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
    out.append(("🏆 Top product", f"**{prod.index[0]}** generates the highest revenue ({fmt_money(prod.iloc[0])}, "
                                  f"{prod.iloc[0] / total * 100:.0f}% of the total). "
                                  f"The lowest is **{prod.index[-1]}** ({fmt_money(prod.iloc[-1])})."))
    out.append(("🗂️ Top category", f"**{cat.index[0]}** is the strongest category with {cat.iloc[0] / total * 100:.0f}% of revenue."))
    out.append(("🌍 Best region", f"**{reg.index[0]}** performs best ({fmt_money(reg.iloc[0])}, {reg.iloc[0] / total * 100:.0f}% of revenue), "
                                  f"while **{reg.index[-1]}** is the weakest ({reg.iloc[-1] / total * 100:.0f}%)."))

    # 3. Seasonality
    if n >= 12:
        by_month = monthly.groupby("Month")["Revenue"].mean()
        out.append(("📅 Seasonality", f"**{calendar.month_name[by_month.idxmax()]}** is the strongest month on average "
                                      f"({fmt_money(by_month.max())}) and **{calendar.month_name[by_month.idxmin()]}** the weakest "
                                      f"({fmt_money(by_month.min())}). Plan stock and campaigns around this pattern."))

    # 4. Marketing effect
    if n >= 6 and df["MarketingSpend"].sum() > 0:
        r = monthly["MarketingSpend"].corr(monthly["Revenue"])
        strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        direction = "positive" if r > 0 else "negative"
        roi = total / df["MarketingSpend"].sum()
        out.append(("📣 Marketing effect",
                    f"Monthly marketing spend and revenue have a **{strength} {direction} relationship** (correlation {r:.2f}). "
                    f"Each $1 of marketing is linked to about ${roi:,.1f} of revenue. "
                    "Note: correlation does not prove cause - both usually rise together in peak season."))

    # 5. Discounts
    heavy = df["Discount"] >= 0.15
    out.append(("🏷️ Discounts", f"The average discount is {df['Discount'].mean() * 100:.1f}%. "
                                f"Records with 15%+ discounts account for {df.loc[heavy, 'Revenue'].sum() / total * 100:.0f}% of revenue."))

    # 6. Model quality and forecast
    if results is not None:
        m = results["metrics"].iloc[0]
        pct = m["RMSE"] / monthly["Revenue"].mean() * 100
        text = (f"**{results['best']}** is the best model (R² = {m['R² Score']:.3f}); its typical error (RMSE) is "
                f"{fmt_money(m['RMSE'])}, about {pct:.1f}% of average monthly revenue.")
        if backtest:
            text += (f" When the whole forecasting method was tested on the last {backtest['months']} months, "
                     f"the average error was {backtest['mape']:.1f}%.")
        out.append(("🤖 Model performance", text))
    if fc is not None:
        out.append(("🔮 Expected future revenue",
                    f"Over the next {len(fc)} months revenue is expected to total **{fmt_money(fc['Predicted Revenue'].sum())}** "
                    f"(about {fmt_money(fc['Predicted Revenue'].mean())} per month). The trend is **{trend['label'].lower()} "
                    f"{trend['arrow']}** ({trend['pct']:+.1f}% vs {trend['basis']})."))

    # 7. Areas that need improvement
    todo = []
    if (df["Date"].max() - df["Date"].min()).days >= 700:
        cut = df["Date"].max() - pd.Timedelta(days=365)
        now = df[df["Date"] > cut].groupby("Product")["Revenue"].sum()
        before = df[(df["Date"] <= cut) & (df["Date"] > cut - pd.Timedelta(days=365))].groupby("Product")["Revenue"].sum()
        growth = ((now / before - 1) * 100).dropna().sort_values()
        if len(growth):
            declining = growth[growth < 0]
            if len(declining):
                todo.append("Declining products: " + ", ".join(f"**{p}** ({v:+.1f}%)" for p, v in declining.items()) + " (last 12 months vs the year before).")
            else:
                todo.append(f"Slowest-growing product: **{growth.index[0]}** ({growth.iloc[0]:+.1f}%).")
    todo.append(f"Weakest region: **{reg.index[-1]}** - consider local promotions.")
    mk = df.groupby("Product")[["Revenue", "MarketingSpend"]].sum()
    mk = (mk["Revenue"] / mk["MarketingSpend"]).replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    if len(mk):
        todo.append(f"Lowest marketing efficiency: **{mk.index[0]}** (${mk.iloc[0]:,.1f} revenue per $1 of marketing vs ${mk.iloc[-1]:,.1f} for {mk.index[-1]}).")
    out.append(("🛠️ Areas that need improvement", "\n".join("- " + t for t in todo)))
    return out


# ----------------------------------------------------------------------
# 6. SIDEBAR FILTERS
# ----------------------------------------------------------------------
def sidebar_filters(clean):
    """Draw the filters and return (filtered_dataframe, forecast_months)."""
    st.sidebar.markdown("### 🔎 Filters")
    lo, hi = clean["Date"].min().date(), clean["Date"].max().date()
    picked = st.sidebar.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi)
    start, end = picked if isinstance(picked, (tuple, list)) and len(picked) == 2 else (lo, hi)
    # Round to whole months so the monthly model never sees a half-empty month
    start = pd.Timestamp(start).to_period("M").start_time
    end = pd.Timestamp(end).to_period("M").end_time
    st.sidebar.caption("Dates are rounded to whole months.")

    products = st.sidebar.multiselect("Product", sorted(clean["Product"].unique()), default=sorted(clean["Product"].unique()))
    cats = st.sidebar.multiselect("Category", sorted(clean["Category"].unique()), default=sorted(clean["Category"].unique()))
    regions = st.sidebar.multiselect("Region", sorted(clean["Region"].unique()), default=sorted(clean["Region"].unique()))

    st.sidebar.markdown("### 🔮 Forecast")
    n_periods = st.sidebar.selectbox("Forecast period (months)", [3, 6, 9, 12, 18, 24], index=1)

    mask = (clean["Date"].between(start, end) & clean["Product"].isin(products)
            & clean["Category"].isin(cats) & clean["Region"].isin(regions))
    return clean[mask], n_periods


# ----------------------------------------------------------------------
# 7. DASHBOARD PAGES
# ----------------------------------------------------------------------
def page_overview(df, monthly, results):
    header("Predictive Analytics Dashboard", "Historical sales trends, revenue prediction and future forecasting")
    kpi_row(df, results)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue Over Time")
        chart_time_series(df, "Revenue", "Revenue Over Time", BLUE)
    with c2:
        st.subheader("Units Sold Over Time")
        chart_time_series(df, "UnitsSold", "Units Sold Over Time", GREEN, as_money=False)

    st.subheader("Recent Performance")
    if len(monthly) >= 6:
        last3, prev3 = monthly.tail(3), monthly.iloc[-6:-3]

        def delta(a, b):
            return f"{(a / b - 1) * 100:+.1f}%"
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Revenue (last 3 months)", fmt_money(last3["Revenue"].sum()), delta(last3["Revenue"].sum(), prev3["Revenue"].sum()))
        m2.metric("Units (last 3 months)", f"{last3['UnitsSold'].sum():,.0f}", delta(last3["UnitsSold"].sum(), prev3["UnitsSold"].sum()))
        m3.metric("Avg price", f"${last3['Price'].mean():,.2f}", delta(last3["Price"].mean(), prev3["Price"].mean()))
        m4.metric("Marketing (last 3 months)", fmt_money(last3["MarketingSpend"].sum()), delta(last3["MarketingSpend"].sum(), prev3["MarketingSpend"].sum()))
        st.caption("Changes compare the last 3 months with the 3 months before them.")
    last6 = monthly.tail(6).copy()
    last6["MoM Change"] = monthly["Revenue"].pct_change().tail(6) * 100
    table = pd.DataFrame({
        "Month": last6["Date"].dt.strftime("%b %Y"), "Revenue": last6["Revenue"], "Units Sold": last6["UnitsSold"],
        "Avg Price": last6["Price"], "Avg Discount": last6["Discount"] * 100,
        "Marketing Spend": last6["MarketingSpend"], "MoM Change": last6["MoM Change"],
    })
    st.dataframe(table.style.format({
        "Revenue": "${:,.0f}", "Units Sold": "{:,.0f}", "Avg Price": "${:,.2f}", "Avg Discount": "{:.1f}%",
        "Marketing Spend": "${:,.0f}", "MoM Change": "{:+.1f}%"}, na_rep="-"), hide_index=True)


def page_historical(df, monthly):
    header("Historical Analysis", "What happened in the past? Trends, products, regions and marketing")
    st.subheader("Monthly Revenue")
    chart_monthly_bar(monthly)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue by Product")
        chart_revenue_by(df, "Product", "Blues_r")
        st.caption("Which products bring in the most money.")
    with c2:
        st.subheader("Revenue by Category")
        chart_revenue_by(df, "Category", "Purples_r")
        st.caption("Product groups compared.")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Revenue by Region")
        chart_revenue_by(df, "Region", "Greens_r")
        st.caption("Where customers buy the most.")
    with c4:
        st.subheader("Seasonality")
        if len(monthly) >= 12:
            chart_seasonality(monthly)
            st.caption("The orange bar is the strongest month on average.")
        else:
            st.info("At least 12 months of data are needed to show seasonality.")

    st.subheader("Marketing Spend Analysis")
    c5, c6 = st.columns(2)
    with c5:
        if len(monthly) >= 3:
            chart_marketing(monthly)
            st.caption("Each dot is one month. An upward red line means higher spend goes with higher revenue.")
        else:
            st.info("Not enough months for this chart.")
    with c6:
        chart_distribution(df)
        st.caption("Most records are small sales; a few large sales stretch the right side.")
    roi = df["Revenue"].sum() / max(df["MarketingSpend"].sum(), 1)
    st.metric("Revenue per $1 of marketing", f"${roi:,.2f}")


def page_modeling(monthly, results):
    header("Predictive Modeling", "Two models learn from past months and are tested on recent, unseen months")
    if not need_model(results):
        return
    best, metrics = results["best"], results["metrics"]

    with st.expander("How does the model work?"):
        st.markdown(
            f"""
            * Daily records are combined into **one row per month** ({len(monthly)} months).
            * Features: **UnitsSold, Price, Discount, MarketingSpend, Year, Month, Quarter**. Target: **Revenue**.
            * The **first {results['n_train']} months** ({results['train_start']:%b %Y} - {results['train_end']:%b %Y}) are used for training.
            * The **last {results['n_test']} months** ({results['test_start']:%b %Y} - {results['test_end']:%b %Y}) are hidden and used for testing.
            * Both models are scored on the test months and the better one is selected automatically.
            * Revenue is partly *calculated* from units, price and discount, so high accuracy is expected here.
              The harder part is forecasting those inputs, which the **Future Forecast** page checks separately.
            """
        )

    st.success(f"🏆 Best-performing model: **{best}** - lowest error and highest R² on the {results['n_test']} test months.")

    st.subheader("Model Comparison")
    table = metrics.copy()
    table["Result"] = np.where(table["Model"] == best, "🏆 Best", "")
    st.dataframe(
        table.style.format({"MAE": "${:,.0f}", "MSE": "{:,.0f}", "RMSE": "${:,.0f}", "R² Score": "{:.3f}"}),
        hide_index=True,
    )
    if (metrics["R² Score"] < 0).any():
        st.caption("A negative R² means the model is worse than simply guessing the average. Random Forest cannot predict "
                   "values higher than those it saw in training, so it struggles when revenue keeps growing.")

    b = metrics.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", fmt_money(b["MAE"]))
    m2.metric("MSE", f"{b['MSE']:,.0f}")
    m3.metric("RMSE", fmt_money(b["RMSE"]))
    m4.metric("R² Score", f"{b['R² Score']:.3f}")

    st.subheader("Actual vs Predicted Revenue")
    ap = results["actual_vs_pred"]
    chart_actual_vs_pred(ap)
    st.caption("Blue = what really happened, orange = what the model predicted. The green area marks the test months the model never saw.")

    test = ap[ap["Set"] == "Test"]
    c1, c2 = st.columns(2)
    with c1:
        chart_errors(test)
        st.caption("Green bars: the model predicted too low. Red bars: it predicted too high.")
    with c2:
        chart_importance(results["importance"], best)
    st.subheader("Prediction Error (test months)")
    st.dataframe(
        test[["Date", "Actual Revenue", "Predicted Revenue", "Error", "Error %"]].assign(Date=test["Date"].dt.strftime("%b %Y")).style.format(
            {"Actual Revenue": "${:,.0f}", "Predicted Revenue": "${:,.0f}", "Error": "${:+,.0f}", "Error %": "{:+.2f}%"}),
        hide_index=True,
    )


def page_forecast(monthly, results, fc, trend, backtest, n_periods):
    header("Future Forecast", f"Predicted revenue for the next {n_periods} months")
    if not need_model(results):
        return
    if len(monthly) < 24:
        st.info("With less than 24 months of history, seasonal patterns are estimated from very few observations, so treat this forecast with extra care.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Expected revenue ({n_periods} months)", fmt_money(fc["Predicted Revenue"].sum()))
    k2.metric("Average per month", fmt_money(fc["Predicted Revenue"].mean()))
    k3.metric("Trend direction", f"{trend['arrow']} {trend['label']}", f"{trend['pct']:+.1f}% vs {trend['basis']}")
    k4.metric("Forecast backtest error", f"{backtest['mape']:.1f}%" if backtest else "N/A")

    st.subheader("Forecast Chart")
    chart_forecast(monthly, fc)
    st.caption("The dashed orange line is the forecast. It follows the history's growth trend and seasonal pattern.")

    st.subheader("Future Revenue Forecast")
    table = pd.DataFrame({"Forecast Date": fc["Forecast Date"].dt.strftime("%b %Y"), "Predicted Revenue": fc["Predicted Revenue"]})
    st.dataframe(table.style.format({"Predicted Revenue": "${:,.0f}"}), hide_index=True)

    with st.expander("Inputs assumed for each future month"):
        inputs = fc[["Forecast Date", "UnitsSold", "Price", "Discount", "MarketingSpend"]].copy()
        inputs["Forecast Date"] = inputs["Forecast Date"].dt.strftime("%b %Y")
        inputs["Discount"] = inputs["Discount"] * 100
        st.dataframe(inputs.style.format({"UnitsSold": "{:,.0f}", "Price": "${:,.2f}", "Discount": "{:.1f}%", "MarketingSpend": "${:,.0f}"}), hide_index=True)
        st.caption("Future units, price, discount and marketing are projected from each one's past trend plus its seasonal pattern, then fed to the model.")
    if backtest:
        st.info(f"**How reliable is it?** When the last {backtest['months']} months were hidden and forecast using only earlier data, "
                f"the forecast was off by {backtest['mape']:.1f}% on average. Forecasts further into the future are less certain, "
                "and they assume the future behaves like the past.")


def page_insights(df, monthly, results, fc, trend, backtest):
    header("Business Insights", "Generated automatically from the selected data")
    for title, text in generate_insights(df, monthly, results, fc, trend, backtest):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(text)


def page_data(raw, clean, report, df, results, fc):
    header("Data & Downloads", "Inspect the raw data, cleaning results and download CSV files")
    t1, t2, t3, t4 = st.tabs(["Raw Dataset", "Data Quality", "Processed Data", "Downloads"])

    with t1:
        st.write(f"**{len(raw):,} rows** x **{raw.shape[1]} columns**")
        st.dataframe(raw)

    with t2:
        q1, q2, q3 = st.columns(3)
        q1.metric("Rows before cleaning", f"{report['rows_before']:,}")
        q2.metric("Duplicates removed", f"{report['duplicates_removed']:,}")
        q3.metric("Rows after cleaning", f"{report['rows_after']:,}")
        missing = pd.DataFrame({"Missing before": report["missing_before"], "Missing after": report["missing_after"]}).fillna(0).astype(int)
        st.write("**Missing values** (numbers filled with the product's median; missing revenue recalculated as units x price x (1 - discount); text filled with the most common value):")
        st.dataframe(missing)
        st.write("**Outlier check** (IQR rule on UnitsSold per product; only extreme values beyond 3 x IQR are capped):")
        st.dataframe(report["outliers"], hide_index=True)
        if report["bad_dates_removed"]:
            st.warning(f"{report['bad_dates_removed']} rows with unreadable dates were removed.")
        if report["discount_rescaled"]:
            st.info("Discount values looked like percentages (0-100), so they were converted to fractions (0-1).")

    with t3:
        st.write(f"Processed data for the current filters: **{len(df):,} rows**. New columns: Year, Month, Quarter, Day, MonthNumber.")
        st.dataframe(df)

    with t4:
        st.write("Downloads reflect the filters currently selected in the sidebar.")
        processed = df.assign(Date=df["Date"].dt.strftime("%Y-%m-%d"))
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Processed historical data", to_csv_bytes(processed), "processed_historical_data.csv", "text/csv")
        if results is not None:
            ap = results["actual_vs_pred"].assign(Date=lambda x: x["Date"].dt.strftime("%Y-%m-%d"))
            d2.download_button("⬇️ Actual vs predicted", to_csv_bytes(ap), "actual_vs_predicted.csv", "text/csv")
            out = fc.assign(**{"Forecast Date": fc["Forecast Date"].dt.strftime("%Y-%m-%d")})
            d3.download_button("⬇️ Future forecast", to_csv_bytes(out), "future_forecast.csv", "text/csv")
        else:
            st.info("Actual-vs-predicted and forecast downloads need at least 12 months of data.")


# ----------------------------------------------------------------------
# 8. MAIN APP
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Predictive Analytics", page_icon="📈", layout="wide")
    inject_css()

    # Sidebar: navigation and data source
    st.sidebar.title("📈 Navigation")
    page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("### 📁 Data")
    uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")

    raw = load_data(uploaded.getvalue() if uploaded is not None else None)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        st.error(f"Your CSV is missing these required columns: {', '.join(missing_cols)}")
        st.stop()

    # Clean the data, then apply sidebar filters
    clean, report = cached_preprocess(raw)
    df, n_periods = sidebar_filters(clean)
    if df.empty:
        st.warning("No records match the current filters. Please widen your selection in the sidebar.")
        st.stop()

    # Prepare monthly data, train models and create the forecast
    monthly = make_monthly(df)
    results = fc = trend = backtest = None
    if len(monthly) >= MIN_MONTHS:
        results = cached_models(monthly)
        fc = forecast_future(monthly, results["final_model"], n_periods)
        trend = trend_direction(monthly, fc)
        backtest = cached_backtest(monthly, results["best"])

    # Show the selected page
    if page == PAGES[0]:
        page_overview(df, monthly, results)
    elif page == PAGES[1]:
        page_historical(df, monthly)
    elif page == PAGES[2]:
        page_modeling(monthly, results)
    elif page == PAGES[3]:
        page_forecast(monthly, results, fc, trend, backtest, n_periods)
    elif page == PAGES[4]:
        page_insights(df, monthly, results, fc, trend, backtest)
    else:
        page_data(raw, clean, report, df, results, fc)


if __name__ == "__main__":
    main()
