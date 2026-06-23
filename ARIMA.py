import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import io
from datetime import datetime, timedelta
import itertools

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Indian Stock Forecaster & Excel Analyzer",
    page_icon="📈",
    layout="wide"
)

# ─── CSS Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1a237e;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .green { color: #2e7d32; font-weight: 600; }
    .red   { color: #c62828; font-weight: 600; }
    .blue  { color: #1565c0; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Indian Stock Forecaster & Excel Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ARIMA Forecasting · Histogram Analysis · Data Quality Check</div>', unsafe_allow_html=True)

# ─── Popular Indian Stocks ───────────────────────────────────────────────────
INDIAN_STOCKS = {
    "Reliance Industries":       "RELIANCE.NS",
    "TCS":                       "TCS.NS",
    "Infosys":                   "INFY.NS",
    "HDFC Bank":                 "HDFCBANK.NS",
    "ICICI Bank":                "ICICIBANK.NS",
    "Wipro":                     "WIPRO.NS",
    "HCL Technologies":          "HCLTECH.NS",
    "Bajaj Finance":             "BAJFINANCE.NS",
    "Kotak Mahindra Bank":       "KOTAKBANK.NS",
    "Larsen & Toubro":           "LT.NS",
    "Axis Bank":                 "AXISBANK.NS",
    "State Bank of India":       "SBIN.NS",
    "Maruti Suzuki":             "MARUTI.NS",
    "Asian Paints":              "ASIANPAINT.NS",
    "Titan Company":             "TITAN.NS",
    "Sun Pharmaceutical":        "SUNPHARMA.NS",
    "Bharti Airtel":             "BHARTIARTL.NS",
    "ITC":                       "ITC.NS",
    "Power Grid":                "POWERGRID.NS",
    "NTPC":                      "NTPC.NS",
    "Tata Motors":               "TATAMOTORS.NS",
    "Tata Steel":                "TATASTEEL.NS",
    "Adani Ports":               "ADANIPORTS.NS",
    "UltraTech Cement":          "ULTRACEMCO.NS",
    "Hindustan Unilever":        "HINDUNILVR.NS",
    "Mahindra & Mahindra":       "M&M.NS",
    "Bajaj Auto":                "BAJAJ-AUTO.NS",
    "Hero MotoCorp":             "HEROMOTOCO.NS",
    "Dr. Reddy's Labs":          "DRREDDY.NS",
    "Cipla":                     "CIPLA.NS",
    "Divis Laboratories":        "DIVISLAB.NS",
    "Tech Mahindra":             "TECHM.NS",
    "JSW Steel":                 "JSWSTEEL.NS",
    "Grasim Industries":         "GRASIM.NS",
    "IndusInd Bank":             "INDUSINDBK.NS",
    "Eicher Motors":             "EICHERMOT.NS",
    "Britannia Industries":      "BRITANNIA.NS",
    "Nestle India":              "NESTLEIND.NS",
    "Shree Cement":              "SHREECEM.NS",
    "BPCL":                      "BPCL.NS",
}

# ─── Helper Functions ────────────────────────────────────────────────────────

def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1]  # p-value

def difference_series(series, d):
    s = series.copy()
    for _ in range(d):
        s = s.diff().dropna()
    return s

def find_best_arima(series, max_p=3, max_d=2, max_q=3):
    best_aic = np.inf
    best_order = (1, 1, 1)
    for p, d, q in itertools.product(range(max_p+1), range(max_d+1), range(max_q+1)):
        try:
            model = ARIMA(series, order=(p, d, q))
            res = model.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_order = (p, d, q)
        except Exception:
            continue
    return best_order, best_aic

def forecast_stock(ticker_symbol, stock_name):
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=5*365)

    with st.spinner(f"Downloading {stock_name} data from Yahoo Finance..."):
        df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)

    if df.empty or len(df) < 60:
        st.error(f"Not enough data for {stock_name} ({ticker_symbol}). Skipping.")
        return None

    close = df["Close"].squeeze().dropna()

    # ── Auto-select ARIMA order ──────────────────────────────────────────────
    with st.spinner("Finding best ARIMA order (this may take a moment)..."):
        order, aic = find_best_arima(close, max_p=2, max_d=2, max_q=2)

    # ── Fit & forecast ───────────────────────────────────────────────────────
    model  = ARIMA(close, order=order)
    result = model.fit()

    # Steps to June 2027
    last_date    = close.index[-1]
    target_date  = pd.Timestamp("2027-06-30")
    business_days = len(pd.bdate_range(last_date, target_date)) - 1
    steps         = max(business_days, 1)

    forecast_obj  = result.get_forecast(steps=steps)
    forecast_mean = forecast_obj.predicted_mean
    conf_int      = forecast_obj.conf_int(alpha=0.05)

    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    forecast_mean.index = forecast_dates
    conf_int.index      = forecast_dates

    # ── June 2027 monthly values ─────────────────────────────────────────────
    june_2027 = forecast_mean[
        (forecast_mean.index >= "2027-06-01") & (forecast_mean.index <= "2027-06-30")
    ]

    return {
        "name":          stock_name,
        "ticker":        ticker_symbol,
        "historical":    close,
        "forecast":      forecast_mean,
        "conf_int":      conf_int,
        "june_2027":     june_2027,
        "order":         order,
        "aic":           aic,
        "last_price":    float(close.iloc[-1]),
        "june_avg":      float(june_2027.mean()) if not june_2027.empty else None,
    }

# ════════════════════════════════════════════════════════════════════════════
#  TAB LAYOUT
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📈 Indian Stock Forecaster (ARIMA)", "📂 Excel File Analyzer"])

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 – STOCK FORECASTER
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Select Indian Stocks to Forecast up to June 2027")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        selected_names = st.multiselect(
            "Choose stocks (select one or more):",
            options=list(INDIAN_STOCKS.keys()),
            default=["TCS", "Infosys", "Reliance Industries"],
        )
    with col_b:
        run_btn = st.button("🚀 Run Forecast", use_container_width=True, type="primary")

    if not selected_names:
        st.info("Please select at least one stock above and click **Run Forecast**.")

    if run_btn and selected_names:
        results = []
        progress_bar = st.progress(0)

        for i, name in enumerate(selected_names):
            ticker = INDIAN_STOCKS[name]
            res    = forecast_stock(ticker, name)
            if res:
                results.append(res)
            progress_bar.progress((i + 1) / len(selected_names))

        progress_bar.empty()

        if not results:
            st.error("No data could be retrieved. Please try again.")
        else:
            # ── Summary table ────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📋 Forecast Summary – June 2027")

            summary_rows = []
            for r in results:
                chg  = ((r["june_avg"] - r["last_price"]) / r["last_price"] * 100) if r["june_avg"] else None
                summary_rows.append({
                    "Stock":            r["name"],
                    "Ticker":           r["ticker"],
                    "ARIMA Order":      str(r["order"]),
                    "Last Close (₹)":   f"₹ {r['last_price']:,.2f}",
                    "June 2027 Avg (₹)": f"₹ {r['june_avg']:,.2f}" if r["june_avg"] else "N/A",
                    "Expected Change":  f"{chg:+.1f}%" if chg else "N/A",
                    "AIC":              f"{r['aic']:.1f}",
                })

            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # ── Download summary ─────────────────────────────────────────────
            csv_buf = io.StringIO()
            summary_df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download Summary CSV",
                data=csv_buf.getvalue(),
                file_name="arima_forecast_june2027.csv",
                mime="text/csv",
            )

            # ── Individual forecast charts ────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Forecast Charts")

            for r in results:
                st.markdown(f"#### {r['name']} ({r['ticker']})")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Last Closing Price", f"₹ {r['last_price']:,.2f}")
                with col2:
                    if r["june_avg"]:
                        chg  = (r["june_avg"] - r["last_price"]) / r["last_price"] * 100
                        st.metric("June 2027 Avg Forecast", f"₹ {r['june_avg']:,.2f}", f"{chg:+.1f}%")
                    else:
                        st.metric("June 2027 Avg Forecast", "N/A")
                with col3:
                    st.metric("ARIMA Order Used", str(r["order"]))

                fig, ax = plt.subplots(figsize=(12, 5))

                # Historical (last 1 year for clarity)
                hist_1y = r["historical"].last("365D")
                ax.plot(hist_1y.index, hist_1y.values, color="#1a237e", linewidth=1.6, label="Historical (1 yr)")

                # Forecast line
                ax.plot(r["forecast"].index, r["forecast"].values, color="#e65100", linewidth=1.8,
                        linestyle="--", label="ARIMA Forecast")

                # Confidence interval
                ax.fill_between(
                    r["conf_int"].index,
                    r["conf_int"].iloc[:, 0],
                    r["conf_int"].iloc[:, 1],
                    alpha=0.18, color="#e65100", label="95% CI"
                )

                # Highlight June 2027
                if not r["june_2027"].empty:
                    ax.axvspan(pd.Timestamp("2027-06-01"), pd.Timestamp("2027-06-30"),
                               alpha=0.12, color="green", label="June 2027")
                    ax.axhline(r["june_avg"], color="green", linewidth=1.2,
                               linestyle=":", label=f"June 2027 Avg ₹{r['june_avg']:,.0f}")

                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.xticks(rotation=45)
                ax.set_title(f"{r['name']} – ARIMA{r['order']} Forecast to June 2027",
                             fontsize=13, fontweight="bold")
                ax.set_ylabel("Price (₹)")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # June 2027 day-by-day table
                if not r["june_2027"].empty:
                    with st.expander(f"📅 Day-by-day June 2027 forecast for {r['name']}"):
                        june_df = r["june_2027"].reset_index()
                        june_df.columns = ["Date", "Forecasted Price (₹)"]
                        june_df["Date"] = june_df["Date"].dt.strftime("%Y-%m-%d")
                        june_df["Forecasted Price (₹)"] = june_df["Forecasted Price (₹)"].map(
                            lambda x: f"₹ {x:,.2f}")
                        st.dataframe(june_df, use_container_width=True, hide_index=True)

                st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 – EXCEL ANALYZER
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload an Excel File for Statistical Analysis")
    uploaded_file = st.file_uploader("Upload your Excel file (.xlsx or .xls)", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            df_excel = pd.read_excel(uploaded_file)
            st.success(f"File loaded: **{uploaded_file.name}** — {df_excel.shape[0]} rows × {df_excel.shape[1]} columns")
            st.dataframe(df_excel.head(20), use_container_width=True)

            num_cols = df_excel.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.warning("No numerical columns found in the uploaded file.")
            else:
                st.markdown(f"#### Numerical columns detected: `{', '.join(num_cols)}`")
                st.markdown("---")

                for col in num_cols:
                    series = df_excel[col].dropna()
                    if len(series) < 5:
                        st.warning(f"Column **{col}** has too few values — skipping.")
                        continue

                    mean_val   = series.mean()
                    median_val = series.median()
                    std_val    = series.std()
                    skewness   = series.skew()
                    rel_diff   = abs(mean_val - median_val) / (abs(mean_val) + 1e-9) * 100

                    if rel_diff <= 10:
                        verdict      = "✅ Mean ≈ Median — Data is approximately symmetric."
                        forecast_msg = "🟢 **This column can be used for forecasting (near-normal distribution).**"
                        verdict_color = "green"
                    else:
                        verdict       = f"⚠️ Mean & Median differ by {rel_diff:.1f}% — Data may be skewed."
                        forecast_msg  = "🔴 **Caution: Skewed data may reduce forecasting reliability. Consider transformations.**"
                        verdict_color = "red"

                    st.markdown(f"### Column: `{col}`")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mean",   f"{mean_val:.4f}")
                    c2.metric("Median", f"{median_val:.4f}")
                    c3.metric("Std Dev", f"{std_val:.4f}")
                    c4.metric("Skewness", f"{skewness:.4f}")

                    st.markdown(f'<p style="color:{verdict_color}; font-size:1rem;">{verdict}</p>',
                                unsafe_allow_html=True)
                    st.markdown(forecast_msg)

                    fig2, ax2 = plt.subplots(figsize=(9, 4))
                    ax2.hist(series, bins=30, color="#1a237e", edgecolor="white", alpha=0.85)
                    ax2.axvline(mean_val,   color="#e65100", linewidth=2, linestyle="--",
                                label=f"Mean = {mean_val:.2f}")
                    ax2.axvline(median_val, color="#2e7d32", linewidth=2, linestyle="-.",
                                label=f"Median = {median_val:.2f}")
                    ax2.set_title(f"Histogram – {col}", fontsize=13, fontweight="bold")
                    ax2.set_xlabel(col)
                    ax2.set_ylabel("Frequency")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
                    st.markdown("---")

        except Exception as e:
            st.error(f"Error reading file: {e}")

    else:
        st.info("Upload an Excel file (.xlsx / .xls) to begin analysis.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
---
<div style='text-align:center; color:#888; font-size:0.85rem;'>
    Built with ❤️ using Streamlit · Yahoo Finance · ARIMA (statsmodels)
</div>
""", unsafe_allow_html=True)
