# 📈 Predictive Analytics and Future Trend Forecasting Dashboard

## Project Objective
Build an interactive dashboard that uses **historical sales data** to
1. identify trends and patterns,
2. train and compare machine-learning models,
3. evaluate how accurate the predictions are, and
4. **forecast future revenue** (3 to 24 months ahead).

## Problem Statement
Businesses need to plan stock, staffing and marketing budgets before the future arrives.
Looking at past sales by eye is slow and unreliable. This project uses regression models to learn how revenue
relates to units sold, price, discounts, marketing and time, then projects revenue into the coming months and explains
the results in plain language.

## Project Structure
```
predictive-analytics/
├── app.py                    # Streamlit dashboard, charts and automatic insights
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   └── historical_sales.csv  # Sample dataset (6,591 rows)
└── models/
    ├── __init__.py
    └── model.py              # Data cleaning, ML models, evaluation and forecasting
```

## Dataset Description
`data/historical_sales.csv` contains **3 years of daily sales (Oct 2023 - Sep 2026)** for 6 products
(6,591 raw rows including 15 duplicates).

| Column | Description |
|--------|-------------|
| Date | Sales date (YYYY-MM-DD) |
| Product | Laptop, Smartphone, Office Chair, Standing Desk, Running Shoes, Backpack |
| Category | Electronics, Furniture, Lifestyle |
| Region | North, South, East, West |
| UnitsSold | Number of units sold |
| Price | Selling price per unit |
| Revenue | UnitsSold x Price x (1 - Discount) |
| Discount | Discount as a fraction (0.10 = 10%) |
| MarketingSpend | Marketing money spent for that record |

The data is **synthetic** but has realistic patterns: yearly growth (some products grow, some decline), a November/December
peak, bigger discounts during the sales season and a small effect from marketing. It also contains missing values,
duplicates and a few extreme values on purpose, so the cleaning step has real work to do.

## Technologies Used
| Tool | Purpose |
|------|---------|
| Python 3.9+ | Programming language |
| Pandas, NumPy | Data cleaning, aggregation, calculations |
| Scikit-learn | Linear Regression, Random Forest, metrics, scaling |
| Matplotlib, Seaborn | Charts |
| Streamlit | Interactive web dashboard |

## Data Preprocessing
Implemented in `models/model.py` (`preprocess_data`):
1. **Load** the CSV and **display** the raw data (Data & Downloads > Raw Dataset).
2. **Check missing values** (Data & Downloads > Data Quality).
3. **Remove duplicate rows**.
4. **Convert Date** to datetime (unreadable dates are dropped) and **sort by date**.
5. **Handle missing values** - numbers use the median of the same product; missing Revenue is recalculated as
   UnitsSold x Price x (1 - Discount); text uses the product's category / most common value.
6. **Check outliers** with the IQR rule on UnitsSold per product. Values beyond 3 x IQR (obvious data errors) are capped and
   Revenue is recalculated. Milder outliers are kept because they are usually real busy days.
7. **Create features**: Year, Month (name), Quarter, Day, MonthNumber.
8. **Prepare for machine learning**: daily records are combined into **one row per month** (`make_monthly`),
   with Price and Discount as weighted averages. Incomplete first/last months are dropped.

## Exploratory Data Analysis
* Revenue over time and Units sold over time (daily values + 30-day moving average)
* Monthly revenue (bar chart)
* Revenue by product, category and region (bar charts)
* Marketing spend vs revenue (scatter plot with trend line)
* Sales distribution (histogram)
* Seasonality: average revenue per calendar month

## Machine Learning Models
**Target:** Revenue. **Features:** UnitsSold, Price, Discount, MarketingSpend, Year, Month, Quarter.

| Model | Idea |
|-------|------|
| Linear Regression | Draws the best straight-line relationship between the features and revenue. Simple and good at following trends. |
| Random Forest | Many decision trees vote on the answer. Handles curved relationships but **cannot predict values higher or lower than it saw during training**. |

**Time-based split:** the model learns from the older 80% of months and is tested on the newest 20% (at least 3 months).
Random shuffling would let the model "peek" at the future, so it is not used for time series.

**Important to understand:** Revenue is partly *calculated* from units, price and discount, so a high R² is expected.
The real challenge is forecasting those inputs, which is why the dashboard also runs a separate **forecast backtest**.

## Model Evaluation
| Metric | Meaning |
|--------|---------|
| MAE | Average size of the error in dollars |
| MSE | Average of squared errors (punishes big mistakes) |
| RMSE | Square root of MSE, in dollars, typical error size |
| R² | Share of revenue variation the model explains (1 = perfect, 0 = no better than the average) |

The model with the **lowest RMSE (and therefore the highest R²)** on the test months is selected automatically.

## Future Forecasting
1. The best model is re-trained on **all** months.
2. The model needs UnitsSold, Price, Discount and MarketingSpend for each future month. These are unknown, so each is
   projected as **straight-line trend + the average seasonal effect of that calendar month**.
3. The model turns those projected inputs into predicted revenue.
4. **Trend direction** compares the forecast with the same calendar months last year, so normal seasonality is not mistaken for growth.
5. **Backtest:** the last 6 months are hidden, forecast from earlier data only, and compared with reality - this gives an honest error figure for the whole method.

The user chooses the forecast period (3, 6, 9, 12, 18 or 24 months) in the sidebar.

## Dashboard Features
* **Sidebar:** page navigation, CSV upload, date range, product / category / region filters, forecast period.
* **KPI cards:** Total Revenue, Avg Monthly Revenue, Total Units Sold, Average Price, Marketing Spend, Best Model R².
* **Pages:** Overview, Historical Analysis, Predictive Modeling, Future Forecast, Business Insights, Data & Downloads.
* **Model comparison table**, actual vs predicted chart, prediction error chart and table, feature importance.
* **Forecast table and chart**, expected revenue, trend direction, backtest error.
* **Downloads (CSV):** processed historical data, actual vs predicted, future forecast.

## Business Insights
Generated from the filtered data - nothing is hard-coded. They cover: revenue trend (recent vs previous period),
top product / category / region, seasonality, marketing relationship and return per marketing dollar, discount share of revenue,
model quality, expected future revenue and trend, and areas that need improvement (declining products, weakest region,
lowest marketing efficiency).

## Installation
```bash
# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Run the Application
```bash
streamlit run app.py
```
The browser opens at http://localhost:8501.

## Use Your Own Dataset
* In the app: sidebar > **Upload your own CSV**, or
* replace `data/historical_sales.csv` with your file.

Required columns (exact names): `Date, Product, Category, Region, UnitsSold, Price, Revenue, Discount, MarketingSpend`.
Notes: Discount can be a fraction (0.10) or a percentage (10) - percentages are converted automatically. At least **12 full months**
of history are needed for modeling; 24+ months give better seasonal forecasts. Data can be daily, weekly or one row per month.

## Expected Output (sample data)
* Raw 6,591 rows, 15 duplicates removed, 6,576 clean records, 36 months of history.
* **Linear Regression** is selected as the best model: R² about 0.97, RMSE about $6.7K (about 1% of average monthly revenue).
* **Random Forest** scores much worse (negative R²) because it cannot predict beyond its training range.
* Forecast shows the November/December peak, about $9.9M expected over the next 12 months, trend **upward**, with a backtest error of about 3%.
* Automatic insights, 15+ charts, tables and three CSV downloads.

## Troubleshooting
* *Command not found: streamlit* - use `python -m streamlit run app.py`.
* *"needs at least 12 full months"* - widen the date range or relax filters in the sidebar.
