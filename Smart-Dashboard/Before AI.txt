import streamlit as st
import time
import pandas as pd
import psutil
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import numpy as np
from datetime import datetime, timedelta

# ==================== PAGE CONFIG (must be first) ====================
st.set_page_config(
    page_title="Smart Grid Dashboard",
    page_icon="zap",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 100%;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #1e2a3a;
    }

    section[data-testid="stSidebar"] * {
        color: #94a3b8 !important;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #0f1623 0%, #141d2e 100%);
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease;
    }
    .kpi-card:hover { border-color: #3b82f6; }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 12px 12px 0 0;
    }
    .kpi-card.blue::before   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .kpi-card.green::before  { background: linear-gradient(90deg, #10b981, #34d399); }
    .kpi-card.amber::before  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .kpi-card.red::before    { background: linear-gradient(90deg, #ef4444, #f87171); }
    .kpi-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }

    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b !important;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9 !important;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
    }
    .kpi-unit  { font-size: 0.85rem; color: #64748b; margin-left: 4px; }
    .kpi-delta { font-size: 0.75rem; margin-top: 0.35rem; }
    .kpi-delta.up      { color: #10b981; }
    .kpi-delta.down    { color: #ef4444; }
    .kpi-delta.neutral { color: #64748b; }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge-online  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
    .badge-warning { background: #1c1003; color: #fbbf24; border: 1px solid #92400e; }
    .badge-offline { background: #1c0404; color: #f87171; border: 1px solid #991b1b; }

    /* Section headers */
    .section-header {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #475569;
        border-bottom: 1px solid #1e2a3a;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Alert card */
    .alert-item {
        background: #0f1623;
        border-left: 3px solid #ef4444;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.78rem;
        color: #cbd5e1;
    }
    .alert-item.warning { border-left-color: #f59e0b; }
    .alert-item.info    { border-left-color: #3b82f6; }
    .alert-item.ok      { border-left-color: #10b981; }

    /* Hypothesis info box */
    .hypothesis-box {
        background: linear-gradient(135deg, #0f1a2e, #0d1420);
        border: 1px solid #1e3a5f;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
    }
    .hypothesis-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #3b82f6;
        margin-bottom: 0.35rem;
    }
    .hypothesis-text {
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.6;
    }

    /* ML badge */
    .ml-badge {
        display: inline-flex;
        align-items: center;
        background: #1a0a2e;
        border: 1px solid #6d28d9;
        color: #a78bfa;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-left: 6px;
    }

    /* Sidebar nav */
    div[data-testid="stRadio"] label {
        background: transparent;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1117;
        border-bottom: 1px solid #1e2a3a;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b !important;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 8px 20px;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom-color: #3b82f6 !important;
        background: transparent !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #64748b !important;
    }

    .stDataFrame { border: 1px solid #1e2a3a; border-radius: 8px; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e2a3a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #2d3f55; }

    #MainMenu, footer, header { visibility: hidden; }
    .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ==================== HYPOTHESIS H: AI PREPROCESSOR ====================

class DataPreprocessor:
    """
    Hypothesis H: AI Integration in Data Preprocessing.
    Provides both classical statistical methods (baseline) and real
    ML-based anomaly detection (Isolation Forest, LOF, One-Class SVM,
    PCA reconstruction error).
    """

    # --- Classical methods (kept as baseline) ---

    @staticmethod
    def detect_and_handle_outliers(df, column, method='iqr', threshold=1.5):
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        else:
            mean = df[column].mean()
            std  = df[column].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    @staticmethod
    def check_missing_values(df, columns):
        missing_report = {}
        for col in columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct   = (missing_count / len(df)) * 100
                missing_report[col] = {
                    "missing_count":      int(missing_count),
                    "missing_percentage": missing_pct,
                    "has_missing":        missing_count > 0
                }
        return missing_report

    @staticmethod
    def interpolate_missing_values(df, columns, method='linear', order=2):
        df_processed = df.copy()
        for col in columns:
            if col in df_processed.columns:
                try:
                    if method in ['polynomial', 'spline']:
                        df_processed[col] = df_processed[col].interpolate(
                            method=method, order=order, limit_direction='both')
                    else:
                        df_processed[col] = df_processed[col].interpolate(
                            method=method, limit_direction='both')
                except Exception as e:
                    st.warning(f"Could not interpolate {col}: {e}")
        return df_processed

    @staticmethod
    def normalize_data(df, columns, method='standard'):
        df_normalized = df.copy()
        if method == 'standard':
            scaler = StandardScaler()
            df_normalized[columns] = scaler.fit_transform(df[columns])
        elif method == 'minmax':
            for col in columns:
                mn, mx = df[col].min(), df[col].max()
                df_normalized[col] = (df[col] - mn) / (mx - mn) if mx != mn else 0
        elif method == 'robust':
            for col in columns:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR    = Q3 - Q1
                df_normalized[col] = (df[col] - df[col].median()) / IQR if IQR != 0 else 0
        elif method == 'log':
            for col in columns:
                if (df[col] > 0).all():
                    df_normalized[col] = np.log(df[col])
        elif method == 'zscore':
            for col in columns:
                mean, std = df[col].mean(), df[col].std()
                df_normalized[col] = (df[col] - mean) / std if std != 0 else 0
        return df_normalized

    @staticmethod
    def detect_anomalies(df, column, window=5, threshold=2):
        """Classical rolling-window anomaly detection (baseline)."""
        rolling_mean = df[column].rolling(window=window).mean()
        rolling_std  = df[column].rolling(window=window).std()
        return (df[column] - rolling_mean).abs() > (threshold * rolling_std)

    # --- AI / ML methods ---

    @staticmethod
    def isolation_forest(df, columns, contamination=0.05, n_estimators=100):
        """
        Isolation Forest: unsupervised ML that isolates anomalies by
        randomly partitioning the feature space. Anomalies require fewer
        splits to isolate — shorter path → lower score.
        """
        X      = df[columns].dropna()
        model  = IsolationForest(n_estimators=n_estimators,
                                  contamination=contamination, random_state=42)
        preds  = model.fit_predict(X)         # -1 = anomaly
        scores = model.score_samples(X)       # lower = more anomalous
        mask         = pd.Series(False, index=df.index)
        score_series = pd.Series(np.nan,  index=df.index)
        mask[X.index]         = preds == -1
        score_series[X.index] = scores
        return mask, score_series, model

    @staticmethod
    def local_outlier_factor(df, columns, n_neighbors=20, contamination=0.05):
        """
        Local Outlier Factor: compares local density of each point to its
        neighbours. Points in low-density regions relative to neighbours
        are flagged as outliers.
        """
        X      = df[columns].dropna()
        model  = LocalOutlierFactor(n_neighbors=n_neighbors,
                                     contamination=contamination)
        preds  = model.fit_predict(X)
        scores = model.negative_outlier_factor_
        mask         = pd.Series(False, index=df.index)
        score_series = pd.Series(np.nan,  index=df.index)
        mask[X.index]         = preds == -1
        score_series[X.index] = scores
        return mask, score_series

    @staticmethod
    def one_class_svm(df, columns, nu=0.05, kernel="rbf"):
        """
        One-Class SVM: learns a decision boundary in kernel space
        around normal data. Points outside the boundary are anomalies.
        """
        X        = df[columns].dropna()
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model    = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")
        preds    = model.fit_predict(X_scaled)
        scores   = model.decision_function(X_scaled)
        mask         = pd.Series(False, index=df.index)
        score_series = pd.Series(np.nan,  index=df.index)
        mask[X.index]         = preds == -1
        score_series[X.index] = scores
        return mask, score_series, model

    @staticmethod
    def pca_reconstruction_error(df, columns, n_components=None, threshold_sigma=2.5):
        """
        PCA Reconstruction: learns principal components of normal operation.
        High reconstruction error = anomaly — the point cannot be
        represented by the normal-data components.
        """
        X         = df[columns].dropna()
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(X)
        n_comp    = n_components or max(1, int(len(columns) * 0.6))
        pca       = PCA(n_components=n_comp, random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
        X_recon   = pca.inverse_transform(X_reduced)
        errors    = np.mean((X_scaled - X_recon) ** 2, axis=1)
        threshold = errors.mean() + threshold_sigma * errors.std()
        mask         = pd.Series(False, index=df.index)
        error_series = pd.Series(np.nan,  index=df.index)
        mask[X.index]         = errors > threshold
        error_series[X.index] = errors
        return mask, error_series, threshold, pca.explained_variance_ratio_, pca


# ==================== HYPOTHESIS H1: CONTROL PANEL VALIDATOR ====================

class ControlPanelValidator:
    """
    Hypothesis H1: Verification of control panel elements.
    Technology-agnostic: these rules apply to any data source —
    CSV, PLC, SCADA, MQTT, REST API, etc.
    """

    @staticmethod
    def validate_data_range(df, column, min_val, max_val):
        invalid_rows  = df[(df[column] < min_val) | (df[column] > max_val)]
        validity_score = (1 - len(invalid_rows) / len(df)) * 100
        return {
            "valid":            len(invalid_rows) == 0,
            "invalid_count":    len(invalid_rows),
            "validity_score":   validity_score,
            "out_of_range_rows": invalid_rows
        }

    @staticmethod
    def validate_data_consistency(df, column_pairs):
        inconsistencies = []
        for col1, col2, condition in column_pairs:
            if col1 in df.columns and col2 in df.columns:
                invalid = df[~condition(df[col1], df[col2])]
                if len(invalid) > 0:
                    inconsistencies.append({
                        "columns":           f"{col1} vs {col2}",
                        "inconsistent_count": len(invalid)
                    })
        return inconsistencies

    @staticmethod
    def validate_sensor_health(df, sensor_columns):
        health_report = {}
        for col in sensor_columns:
            if col in df.columns:
                missing = df[col].isna().sum()
                health_report[col] = {
                    "total_readings":   len(df),
                    "missing_readings": int(missing),
                    "data_availability": (1 - missing / len(df)) * 100
                }
        return health_report


# ==================== HYPOTHESIS H2: QUALITY ASSESSMENT ====================

class QualityAssessment:
    """
    Hypothesis H2: Quality assessment in correction of the control panel.
    Technology-agnostic weighted scoring framework (Completeness,
    Accuracy, Stability) applicable to any monitoring system.
    """

    @staticmethod
    def calculate_correction_impact(df_original, df_corrected, columns):
        impact_report = {}
        for col in columns:
            if col in df_original.columns and col in df_corrected.columns:
                changes = (df_corrected[col] != df_original[col]).sum()
                mae     = np.mean(np.abs(df_original[col] - df_corrected[col]))
                impact_report[col] = {
                    "rows_changed":      int(changes),
                    "percent_changed":   (changes / len(df_original)) * 100,
                    "mean_absolute_error": mae
                }
        return impact_report

    @staticmethod
    def assess_data_quality_score(df, sensor_columns):
        scores = []

        completeness = (1 - df[sensor_columns].isna().sum().sum() /
                        (len(df) * len(sensor_columns))) * 100
        scores.append(("Completeness", completeness, 0.3))

        outlier_score = 100
        for col in sensor_columns:
            if col in df.columns and df[col].std() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_score -= (z_scores > 3).sum() / len(df) * 100
        scores.append(("Outlier Detection", max(outlier_score, 0), 0.3))

        stability = 100
        for col in sensor_columns:
            if col in df.columns and df[col].mean() != 0:
                cv = df[col].std() / df[col].mean()
                if cv > 0.5:
                    stability -= cv * 10
        scores.append(("Stability", max(stability, 0), 0.4))

        overall = sum(s * w for _, s, w in scores)
        return overall, scores

    @staticmethod
    def generate_quality_report(overall_score):
        if overall_score >= 90:
            return {"status": "Excellent", "color": "#10b981", "score": overall_score,
                    "recommendation": "Data quality is excellent. Continue monitoring."}
        elif overall_score >= 75:
            return {"status": "Good",  "color": "#3b82f6", "score": overall_score,
                    "recommendation": "Data quality is good. Minor improvements recommended."}
        elif overall_score >= 60:
            return {"status": "Fair",  "color": "#f59e0b", "score": overall_score,
                    "recommendation": "Data quality needs attention. Review outliers and missing values."}
        else:
            return {"status": "Poor",  "color": "#ef4444", "score": overall_score,
                    "recommendation": "Data quality is poor. Immediate action required."}


# ==================== CONFIGURATION ====================

DATA_FILE = "demo_data.csv"

RELEVANT_COLUMNS = [
    "Voltage", "Current", "Measured_Frequency", "Active_Power",
    "Reactive_Power", "Apperent_Power", "Phase_Voltage_Angle",
    "Cos_Phi", "Power_Factor"
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#1e2a3a", showline=False, zeroline=False),
    yaxis=dict(gridcolor="#1e2a3a", showline=False, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hovermode="x unified",
)

# ==================== HELPERS ====================

# How many rows the live chart window keeps
LIVE_BUFFER_SIZE = 200

def _make_synthetic_base():
    """Generate a fixed synthetic baseline dataset (called once at startup)."""
    n = 300
    np.random.seed(42)
    times = [datetime.now() - timedelta(seconds=i * 3) for i in range(n, 0, -1)]
    df = pd.DataFrame({
        "Time":                times,
        "Voltage":             np.random.normal(230, 5, n),
        "Current":             np.random.normal(10, 1.5, n),
        "Measured_Frequency":  np.random.normal(50, 0.1, n),
        "Active_Power":        np.random.normal(2200, 150, n),
        "Reactive_Power":      np.random.normal(400, 80, n),
        "Apperent_Power":      np.random.normal(2300, 160, n),
        "Phase_Voltage_Angle": np.random.normal(0, 2, n),
        "Cos_Phi":             np.random.uniform(0.9, 1.0, n),
        "Power_Factor":        np.random.uniform(0.88, 0.99, n),
    })
    df.loc[np.random.choice(n, 8), "Voltage"]      = np.random.uniform(260, 285, 8)
    df.loc[np.random.choice(n, 5), "Current"]      = np.random.uniform(18, 24, 5)
    df.loc[np.random.choice(n, 4), "Active_Power"] = np.random.uniform(2800, 3200, 4)
    return df


def load_data(file_path):
    """
    Load fresh data from CSV every call — NO cache so charts always
    reflect the latest rows written to disk.
    Falls back to synthetic data when the file is not found.
    """
    try:
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        return df
    except FileNotFoundError:
        # Return (or re-use) the synthetic base stored in session_state
        if "synthetic_base" not in st.session_state:
            st.session_state["synthetic_base"] = _make_synthetic_base()
        return st.session_state["synthetic_base"]
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def _new_live_row(base_row: pd.Series) -> pd.Series:
    """
    Generate one new live reading by walking the last known values
    with a small random drift — simulates a real sensor tick.
    """
    row = base_row.copy()
    # Each column drifts slightly from its previous value
    drifts = {
        "Voltage":             np.random.normal(0, 0.4),
        "Current":             np.random.normal(0, 0.08),
        "Measured_Frequency":  np.random.normal(0, 0.008),
        "Active_Power":        np.random.normal(0, 12),
        "Reactive_Power":      np.random.normal(0, 6),
        "Apperent_Power":      np.random.normal(0, 12),
        "Phase_Voltage_Angle": np.random.normal(0, 0.05),
        "Cos_Phi":             np.random.normal(0, 0.001),
        "Power_Factor":        np.random.normal(0, 0.001),
    }
    for col, drift in drifts.items():
        if col in row.index:
            row[col] = float(row[col]) + drift
    # Clamp reasonable bounds so the simulation stays physical
    row["Voltage"]            = np.clip(row["Voltage"],            180, 280)
    row["Current"]            = np.clip(row["Current"],              0,  30)
    row["Measured_Frequency"] = np.clip(row["Measured_Frequency"],  49,  51)
    row["Cos_Phi"]            = np.clip(row["Cos_Phi"],            0.7, 1.0)
    row["Power_Factor"]       = np.clip(row["Power_Factor"],       0.7, 1.0)
    row["Time"] = datetime.now()
    return row


def get_live_df(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maintain a rolling buffer in session_state.
    On every call: append one new simulated row (or the latest CSV row),
    then trim to LIVE_BUFFER_SIZE so the chart window scrolls forward.
    When reading from a real CSV, the buffer grows as new rows are written.
    """
    # Initialise buffer from the base dataset on first run
    if "live_buffer" not in st.session_state:
        st.session_state["live_buffer"] = base_df.tail(LIVE_BUFFER_SIZE).copy().reset_index(drop=True)

    buf: pd.DataFrame = st.session_state["live_buffer"]

    # --- Real CSV path: check whether new rows have arrived ---
    csv_last_time = base_df["Time"].iloc[-1] if "Time" in base_df.columns else None
    buf_last_time = buf["Time"].iloc[-1]     if "Time" in buf.columns     else None

    if csv_last_time is not None and csv_last_time != buf_last_time:
        # New rows exist in the CSV — append them
        new_rows = base_df[base_df["Time"] > buf_last_time]
        if len(new_rows):
            buf = pd.concat([buf, new_rows], ignore_index=True)
    else:
        # Synthetic / no new CSV rows — generate one simulated tick
        new_row = _new_live_row(buf.iloc[-1])
        buf = pd.concat([buf, pd.DataFrame([new_row])], ignore_index=True)

    # Keep only the latest LIVE_BUFFER_SIZE rows (scrolling window)
    buf = buf.tail(LIVE_BUFFER_SIZE).reset_index(drop=True)
    st.session_state["live_buffer"] = buf
    return buf


def kpi_card(label, value, unit, delta_text, delta_dir, color_class):
    arrow = "+" if delta_dir == "up" else "-" if delta_dir == "down" else ""
    return f"""
    <div class="kpi-card {color_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
        <div class="kpi-delta {delta_dir}">{arrow} {delta_text}</div>
    </div>
    """


def status_badge(status, label):
    cls = {"Online": "badge-online", "Warning": "badge-warning",
           "Offline": "badge-offline"}.get(status, "badge-online")
    dot = {"Online": "#4ade80", "Warning": "#fbbf24",
           "Offline": "#f87171"}.get(status, "#4ade80")
    return (f'<span class="status-badge {cls}">'
            f'<svg width="7" height="7" viewBox="0 0 7 7">'
            f'<circle cx="3.5" cy="3.5" r="3.5" fill="{dot}"/></svg>'
            f'{label}</span>')


# ==================== DASHBOARD PAGES ====================

def page_realtime(df):
    # Build / extend the rolling live buffer — this is what every chart reads
    live_df = get_live_df(df)
    reading = live_df.iloc[-1]                           # latest row
    prev    = live_df.iloc[-2] if len(live_df) > 1 else reading

    col_title, col_status, col_refresh = st.columns([3, 2, 1])
    with col_title:
        st.markdown("### Live Grid Monitor")
        st.markdown(
            f"<span style='font-size:0.75rem;color:#475569;font-family:JetBrains Mono,monospace;'>"
            f"Last updated: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;buffer: {len(live_df)} rows</span>",
            unsafe_allow_html=True)
    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        v = float(reading.get("Voltage", 230))
        s = "Online" if 210 <= v <= 250 else "Warning"
        st.markdown(status_badge(s, f"Grid {s}") + "&nbsp;&nbsp;" +
                    status_badge("Online", "Sensors OK"), unsafe_allow_html=True)
    with col_refresh:
        auto_refresh = st.checkbox("Auto-refresh", value=False)

    st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    def delta(col, fmt=".1f"):
        diff = float(reading.get(col, 0)) - float(prev.get(col, 0))
        d = "up" if diff > 0 else ("down" if diff < 0 else "neutral")
        return f"{abs(diff):{fmt}}", d

    with c1:
        dv, dd = delta("Voltage")
        st.markdown(kpi_card("Voltage", f"{reading.get('Voltage', 0):.1f}", "V", f"{dv} V", dd, "blue"), unsafe_allow_html=True)
    with c2:
        dc, dd = delta("Current")
        st.markdown(kpi_card("Current", f"{reading.get('Current', 0):.2f}", "A", f"{dc} A", dd, "green"), unsafe_allow_html=True)
    with c3:
        df_val, dd = delta("Measured_Frequency", ".3f")
        st.markdown(kpi_card("Frequency", f"{reading.get('Measured_Frequency', 0):.3f}", "Hz", f"{df_val} Hz", dd, "purple"), unsafe_allow_html=True)
    with c4:
        dp, dd = delta("Active_Power")
        st.markdown(kpi_card("Active Power", f"{reading.get('Active_Power', 0):.0f}", "W", f"{dp} W", dd, "amber"), unsafe_allow_html=True)
    with c5:
        dcosphi, dd = delta("Cos_Phi", ".4f")
        st.markdown(kpi_card("Power Factor", f"{reading.get('Power_Factor', 0):.3f}", "", f"{dcosphi}", dd, "red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Waveform Trends</div>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        # ---- Uses live_df — scrolls forward every refresh ----
        window = live_df.tail(120)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=window["Time"], y=window["Voltage"],
                                  name="Voltage (V)", mode="lines",
                                  line=dict(color="#3b82f6", width=2),
                                  fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=window["Time"], y=window["Current"],
                                  name="Current (A)", mode="lines",
                                  line=dict(color="#10b981", width=2, dash="dot")),
                      secondary_y=True)
        fig.update_layout(title=dict(text="Voltage & Current — Last 120 Readings",
                                     font=dict(size=13, color="#94a3b8")),
                          height=300, **PLOTLY_LAYOUT)
        fig.update_yaxes(title_text="Voltage (V)", gridcolor="#1e2a3a", secondary_y=False)
        fig.update_yaxes(title_text="Current (A)", gridcolor="#1e2a3a", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        # ---- Gauge reads the live latest value ----
        pf_val = float(reading.get("Power_Factor", 0.95))
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pf_val,
            delta={"reference": float(prev.get("Power_Factor", 0.95)), "valueformat": ".4f"},
            number={"valueformat": ".4f", "font": {"color": "#f1f5f9", "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0.7, 1.0], "tickcolor": "#475569", "tickwidth": 1},
                "bar": {"color": "#3b82f6", "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)", "bordercolor": "#1e2a3a",
                "steps": [
                    {"range": [0.7,  0.85], "color": "rgba(239,68,68,0.15)"},
                    {"range": [0.85, 0.93], "color": "rgba(245,158,11,0.15)"},
                    {"range": [0.93, 1.0],  "color": "rgba(16,185,129,0.15)"},
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 2}, "value": 0.9}
            },
            title={"text": "Power Factor", "font": {"color": "#94a3b8", "size": 13}}
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                                height=300, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    chart_col3, chart_col4, chart_col5 = st.columns([2, 2, 1])

    with chart_col3:
        # ---- Power components — live_df ----
        window = live_df.tail(80)
        fig2   = go.Figure()
        for col, color, label in [
            ("Active_Power",   "#3b82f6", "Active (W)"),
            ("Reactive_Power", "#8b5cf6", "Reactive (VAR)"),
            ("Apperent_Power", "#f59e0b", "Apparent (VA)"),
        ]:
            if col in window.columns:
                fig2.add_trace(go.Scatter(x=window["Time"], y=window[col],
                                           name=label, mode="lines",
                                           line=dict(color=color, width=1.8)))
        fig2.update_layout(title=dict(text="Power Components", font=dict(size=13, color="#94a3b8")),
                           height=280, **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with chart_col4:
        # ---- Frequency — live_df ----
        window = live_df.tail(100)
        fig3   = go.Figure()
        fig3.add_hrect(y0=49.8, y1=50.2, fillcolor="rgba(16,185,129,0.08)", line_width=0)
        fig3.add_hline(y=50.0, line_dash="dash", line_color="#475569", line_width=1)
        if "Measured_Frequency" in window.columns:
            fig3.add_trace(go.Scatter(x=window["Time"], y=window["Measured_Frequency"],
                                      mode="lines", line=dict(color="#a78bfa", width=2),
                                      fill="tozeroy", fillcolor="rgba(139,92,246,0.05)",
                                      name="Frequency (Hz)"))
        fig3.update_layout(title=dict(text="Frequency Stability", font=dict(size=13, color="#94a3b8")),
                           height=280, **PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with chart_col5:
        # ---- Live sensor values — latest reading ----
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Live Sensors</div>", unsafe_allow_html=True)
        for col in RELEVANT_COLUMNS[:7]:
            if col in reading.index:
                val = reading[col]
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"padding:5px 0;border-bottom:1px solid #1e2a3a;font-size:0.75rem;'>"
                    f"<span style='color:#64748b;'>{col.replace('_',' ')}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace;color:#e2e8f0;font-weight:600;'>"
                    f"{val:.3f}</span></div>",
                    unsafe_allow_html=True)

    st.markdown("<br><div class='section-header'>System Alerts</div>", unsafe_allow_html=True)
    alerts = []
    v  = float(reading.get("Voltage", 230))
    f  = float(reading.get("Measured_Frequency", 50))
    pf = float(reading.get("Power_Factor", 0.95))

    if v > 250 or v < 210:
        alerts.append(("critical", f"Voltage out of nominal range: {v:.1f} V (expected 210-250 V)"))
    if abs(f - 50) > 0.2:
        alerts.append(("warning", f"Frequency deviation detected: {f:.3f} Hz (nominal 50 Hz)"))
    if pf < 0.9:
        alerts.append(("warning", f"Low power factor: {pf:.3f} - Consider reactive power compensation"))

    # Run anomaly detection on the live buffer
    anomaly_mask  = DataPreprocessor.detect_anomalies(live_df, "Voltage", window=10, threshold=2.5)
    anomaly_count = int(anomaly_mask.sum()) if anomaly_mask is not None else 0
    if anomaly_count > 0:
        alerts.append(("warning", f"{anomaly_count} voltage anomalies in live buffer ({len(live_df)} rows)"))
    if not alerts:
        alerts.append(("ok", "All parameters within nominal operating ranges."))

    alert_colors = {"critical": "", "warning": "warning", "ok": "ok", "info": "info"}
    cols_alert   = st.columns(min(len(alerts), 3))
    for i, (level, msg) in enumerate(alerts[:3]):
        with cols_alert[i % 3]:
            st.markdown(f"<div class='alert-item {alert_colors.get(level, '')}'>{msg}</div>",
                        unsafe_allow_html=True)
    if auto_refresh:
        time.sleep(3)
        st.rerun()


def page_analysis(df):
    st.markdown("### Historical Analysis")
    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Correlation Matrix", "Statistical Summary"])

    with tab1:
        col_sel = st.selectbox("Select column:", [c for c in RELEVANT_COLUMNS if c in df.columns])
        col_l, col_r = st.columns(2)
        with col_l:
            fig_hist = px.histogram(df, x=col_sel, nbins=40,
                                    color_discrete_sequence=["#3b82f6"],
                                    title=f"Distribution: {col_sel}")
            fig_hist.update_layout(height=320, **PLOTLY_LAYOUT)
            fig_hist.update_traces(marker_line_color="#1e2a3a", marker_line_width=0.5)
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
        with col_r:
            fig_box = px.box(df, y=col_sel, color_discrete_sequence=["#8b5cf6"],
                             title=f"Box Plot: {col_sel}")
            fig_box.update_layout(height=320, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=df["Time"], y=df[col_sel], mode="lines",
                                    line=dict(color="#10b981", width=1.5), name=col_sel))
        fig_ts.add_trace(go.Scatter(x=df["Time"], y=df[col_sel].rolling(20).mean(), mode="lines",
                                    line=dict(color="#f59e0b", width=2, dash="dash"),
                                    name="20-pt Rolling Mean"))
        fig_ts.update_layout(title=dict(text=f"{col_sel} Over Time", font=dict(size=13, color="#94a3b8")),
                             height=300, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        available = [c for c in RELEVANT_COLUMNS if c in df.columns]
        fig_corr  = px.imshow(df[available].corr(),
                              color_continuous_scale=[[0,"#ef4444"],[0.5,"#1e2a3a"],[1,"#3b82f6"]],
                              zmin=-1, zmax=1, title="Pearson Correlation Matrix", text_auto=".2f")
        fig_corr.update_layout(height=500, **PLOTLY_LAYOUT)
        fig_corr.update_coloraxes(colorbar_tickfont_color="#94a3b8")
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        available = [c for c in RELEVANT_COLUMNS if c in df.columns]
        stats     = df[available].describe().T
        stats["cv%"] = (stats["std"] / stats["mean"] * 100).round(2)
        # Plain dataframe — no matplotlib dependency
        st.dataframe(stats, use_container_width=True)


def page_preprocessing(df):
    """Hypothesis H: AI Integration in Data Preprocessing."""
    st.markdown("### AI Data Preprocessing  —  Hypothesis H")

    st.markdown(
        "<div class='hypothesis-box'>"
        "<div class='hypothesis-title'>Hypothesis H</div>"
        "<div class='hypothesis-text'>"
        "It is possible to integrate AI in data preprocessing for the control panel. "
        "This page shows both the classical statistical baseline (IQR, z-score, rolling window) "
        "and genuine machine learning models — Isolation Forest, Local Outlier Factor, "
        "One-Class SVM, and PCA Reconstruction — that learn normal operating patterns from "
        "the data and flag deviations without any manually defined thresholds."
        "</div></div>",
        unsafe_allow_html=True)

    preprocessor = DataPreprocessor()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Outlier Detection", "Missing Values", "Normalization",
        "🤖 ML Anomaly Detection", "🤖 PCA Reconstruction"
    ])

    # ---- Tab 1: Classical outlier detection ----
    with tab1:
        st.markdown("<div class='section-header'>Statistical Outlier Detection (Baseline)</div>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: selected_col = st.selectbox("Column:", [c for c in RELEVANT_COLUMNS if c in df.columns])
        with c2: method       = st.radio("Method:", ["IQR", "Z-Score"], horizontal=True)
        with c3: threshold    = st.slider("Threshold:", 0.5, 5.0, 1.5)

        if st.button("Run Outlier Detection", type="primary"):
            outliers, lower, upper = preprocessor.detect_and_handle_outliers(
                df, selected_col, 'iqr' if method == "IQR" else 'zscore', threshold)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Outliers Found", len(outliers))
            mc2.metric("Lower Bound",    f"{lower:.3f}")
            mc3.metric("Upper Bound",    f"{upper:.3f}")
            fig_out = go.Figure()
            fig_out.add_trace(go.Scatter(x=df.index, y=df[selected_col], mode="lines",
                                          name=selected_col, line=dict(color="#3b82f6", width=1.5)))
            if len(outliers) > 0:
                fig_out.add_trace(go.Scatter(x=outliers.index, y=outliers[selected_col],
                                              mode="markers", name="Outliers",
                                              marker=dict(color="#ef4444", size=8, symbol="x")))
            fig_out.add_hrect(y0=lower, y1=upper, fillcolor="rgba(16,185,129,0.07)", line_width=0)
            fig_out.update_layout(title=f"Outlier Detection: {selected_col}", height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_out, use_container_width=True, config={"displayModeBar": False})

    # ---- Tab 2: Missing values ----
    with tab2:
        st.markdown("<div class='section-header'>Missing Value Interpolation</div>", unsafe_allow_html=True)
        interp_method = st.selectbox("Method:", ["linear", "polynomial", "spline"])
        order = 2
        if interp_method in ["polynomial", "spline"]:
            order = st.slider("Order:", 1, 5, 2)
        cols_to_interp = st.multiselect("Columns:", [c for c in RELEVANT_COLUMNS if c in df.columns])
        c1, c2 = st.columns(2)
        if c1.button("Check Missing Values"):
            if cols_to_interp:
                report = preprocessor.check_missing_values(df, cols_to_interp)
                data = [{"Column": k, "Missing Count": v["missing_count"],
                         "Missing %": f"{v['missing_percentage']:.2f}%",
                         "Has Missing": "Yes" if v["has_missing"] else "No"}
                        for k, v in report.items()]
                st.dataframe(pd.DataFrame(data), use_container_width=True)
        if c2.button("Interpolate"):
            if cols_to_interp:
                df_proc = preprocessor.interpolate_missing_values(df, cols_to_interp, interp_method, order)
                st.metric("Missing Before", int(df[cols_to_interp].isna().sum().sum()))
                st.metric("Missing After",  int(df_proc[cols_to_interp].isna().sum().sum()))
                st.success("Interpolation complete.")

    # ---- Tab 3: Normalization ----
    with tab3:
        st.markdown("<div class='section-header'>Data Normalization</div>", unsafe_allow_html=True)
        norm_method  = st.selectbox("Method:", ["standard", "minmax", "robust", "log", "zscore"])
        cols_to_norm = st.multiselect("Columns:", [c for c in RELEVANT_COLUMNS if c in df.columns], key="norm_cols")
        if st.button("Normalize", type="primary"):
            if cols_to_norm:
                df_norm  = preprocessor.normalize_data(df, cols_to_norm, norm_method)
                c1, c2   = st.columns(2)
                with c1:
                    st.caption("Original Statistics")
                    st.dataframe(df[cols_to_norm].describe(), use_container_width=True)
                with c2:
                    st.caption("Normalized Statistics")
                    st.dataframe(df_norm[cols_to_norm].describe(), use_container_width=True)
                col_show = cols_to_norm[0]
                fig_n = go.Figure()
                fig_n.add_trace(go.Scatter(y=df[col_show].values, mode="lines",
                                           name="Original",   line=dict(color="#3b82f6")))
                fig_n.add_trace(go.Scatter(y=df_norm[col_show].values, mode="lines",
                                           name="Normalized", line=dict(color="#10b981")))
                fig_n.update_layout(title=f"Normalization Effect: {col_show}", height=300, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_n, use_container_width=True, config={"displayModeBar": False})

    # ---- Tab 4: ML anomaly detection ----
    with tab4:
        st.markdown(
            "<div class='section-header'>ML Anomaly Detection "
            "<span class='ml-badge'>AI</span></div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.8rem;color:#64748b;margin-bottom:1rem;'>"
            "Three unsupervised ML models learn the normal operating envelope from the data — "
            "no manually defined thresholds needed. Select features and run any algorithm.</div>",
            unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            ml_cols = st.multiselect(
                "Features (multivariate):",
                [c for c in RELEVANT_COLUMNS if c in df.columns],
                default=[c for c in ["Voltage", "Current", "Active_Power"] if c in df.columns])
        with c2:
            contamination = st.slider("Expected anomaly rate:", 0.01, 0.20, 0.05, 0.01,
                                       help="Fraction of data expected to be anomalies")
        with c3:
            ml_method = st.selectbox("Algorithm:",
                                      ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"])

        if st.button("Run ML Detection", type="primary") and ml_cols:
            with st.spinner(f"Training {ml_method}..."):
                if ml_method == "Isolation Forest":
                    mask, scores, _ = preprocessor.isolation_forest(df, ml_cols, contamination)
                    score_label = "Anomaly Score (lower = more anomalous)"
                elif ml_method == "Local Outlier Factor":
                    mask, scores  = preprocessor.local_outlier_factor(df, ml_cols,
                                                                        contamination=contamination)
                    score_label = "LOF Score (more negative = more anomalous)"
                else:
                    mask, scores, _ = preprocessor.one_class_svm(df, ml_cols, nu=contamination)
                    score_label = "SVM Decision Score (more negative = more anomalous)"

            count = mask.sum()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Anomalies Found",  int(count))
            mc2.metric("Anomaly Rate",     f"{count/len(df)*100:.2f}%")
            mc3.metric("Features Used",    len(ml_cols))

            plot_col = ml_cols[0]
            fig_ml   = go.Figure()
            fig_ml.add_trace(go.Scatter(x=df.index, y=df[plot_col], mode="lines",
                                         name=plot_col, line=dict(color="#8b5cf6", width=1.5)))
            if count:
                fig_ml.add_trace(go.Scatter(x=df[mask].index, y=df[mask][plot_col],
                                             mode="markers", name=f"{ml_method} Anomalies",
                                             marker=dict(color="#ef4444", size=9, symbol="x-open-dot")))
            fig_ml.update_layout(title=f"{ml_method} — {plot_col}", height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_ml, use_container_width=True, config={"displayModeBar": False})

            # Score distribution
            valid_scores = scores.dropna()
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=valid_scores[~mask[valid_scores.index]],
                                         name="Normal",  marker_color="#3b82f6", opacity=0.7, nbinsx=40))
            fig2.add_trace(go.Histogram(x=valid_scores[mask[valid_scores.index]],
                                         name="Anomaly", marker_color="#ef4444", opacity=0.7, nbinsx=20))
            fig2.update_layout(title=score_label, barmode="overlay", height=280, **PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ---- Tab 5: PCA reconstruction ----
    with tab5:
        st.markdown(
            "<div class='section-header'>PCA Reconstruction Error "
            "<span class='ml-badge'>AI</span></div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.8rem;color:#64748b;margin-bottom:1rem;'>"
            "PCA learns the principal components of normal multi-sensor behaviour. "
            "Points with high reconstruction error deviate from the normal pattern "
            "across multiple sensors simultaneously.</div>",
            unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            pca_cols = st.multiselect(
                "Features:", [c for c in RELEVANT_COLUMNS if c in df.columns],
                default=[c for c in RELEVANT_COLUMNS if c in df.columns], key="pca_cols")
        with c2:
            n_comp = st.slider("PCA components:", 1,
                               max(1, len([c for c in RELEVANT_COLUMNS if c in df.columns]) - 1), 4)
        with c3:
            sigma  = st.slider("Anomaly threshold (σ):", 1.5, 4.0, 2.5, 0.1)

        if st.button("Run PCA Analysis", type="primary") and pca_cols:
            with st.spinner("Running PCA reconstruction..."):
                mask, errors, threshold, explained, _ = preprocessor.pca_reconstruction_error(
                    df, pca_cols, n_components=n_comp, threshold_sigma=sigma)

            count = mask.sum()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Anomalies Found",   int(count))
            mc2.metric("Variance Explained", f"{explained.sum()*100:.1f}%")
            mc3.metric("Error Threshold",    f"{threshold:.4f}")

            fig_pca = go.Figure()
            fig_pca.add_trace(go.Scatter(x=df.index, y=errors, mode="lines",
                                          name="Reconstruction Error",
                                          line=dict(color="#a78bfa", width=1.5)))
            fig_pca.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
                              annotation_text=f"Threshold (μ+{sigma}σ)",
                              annotation_font_color="#ef4444")
            if count:
                fig_pca.add_trace(go.Scatter(x=df[mask].index, y=errors[mask],
                                              mode="markers", name="Anomalies",
                                              marker=dict(color="#ef4444", size=8, symbol="x")))
            fig_pca.update_layout(title="PCA Reconstruction Error Over Time",
                                  height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_pca, use_container_width=True, config={"displayModeBar": False})

            fig_var = go.Figure(go.Bar(
                x=[f"PC{i+1}" for i in range(len(explained))],
                y=explained * 100,
                marker_color="#6d28d9",
                text=[f"{v*100:.1f}%" for v in explained], textposition="outside"))
            fig_var.update_layout(title="Explained Variance per Principal Component",
                                  yaxis_title="Variance Explained (%)",
                                  height=280, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_var, use_container_width=True, config={"displayModeBar": False})


def page_validation(df):
    """Hypothesis H1: Control Panel Validation."""
    st.markdown("### Control Panel Validation  —  Hypothesis H1")

    st.markdown(
        "<div class='hypothesis-box'>"
        "<div class='hypothesis-title'>Hypothesis H1</div>"
        "<div class='hypothesis-text'>"
        "It is possible to verify the elements of a control panel. "
        "The methodology (range checking, logical consistency rules, sensor availability scoring) "
        "is technology-agnostic — the same approach works whether data comes from a CSV file, "
        "a PLC, SCADA, MQTT broker, or REST API."
        "</div></div>",
        unsafe_allow_html=True)

    validator = ControlPanelValidator()
    tab1, tab2, tab3 = st.tabs(["Range Validation", "Consistency Check", "Sensor Health"])

    with tab1:
        st.markdown("<div class='section-header'>Data Range Validation</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            val_col = st.selectbox("Column:", [c for c in RELEVANT_COLUMNS if c in df.columns], key="val_col")
        with c2:
            min_v = st.number_input("Min acceptable:",
                                     value=float(df[val_col].mean() - 3 * df[val_col].std()))
        with c3:
            max_v = st.number_input("Max acceptable:",
                                     value=float(df[val_col].mean() + 3 * df[val_col].std()))

        if st.button("Validate Range", type="primary"):
            result = validator.validate_data_range(df, val_col, min_v, max_v)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Validity Score",  f"{result['validity_score']:.2f}%")
            mc2.metric("Invalid Rows",    result['invalid_count'])
            mc3.metric("Status", "✅ Valid" if result['valid'] else "❌ Invalid")
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=df.index, y=df[val_col], mode="lines",
                                       line=dict(color="#3b82f6", width=1.5), name=val_col))
            fig_v.add_hrect(y0=min_v, y1=max_v, fillcolor="rgba(16,185,129,0.07)",
                            line_width=0, annotation_text="Valid Range")
            if result['invalid_count'] > 0:
                inv = result['out_of_range_rows']
                fig_v.add_trace(go.Scatter(x=inv.index, y=inv[val_col], mode="markers",
                                           marker=dict(color="#ef4444", size=7), name="Out of Range"))
            fig_v.update_layout(title=f"Range Validation: {val_col}", height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_v, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        st.markdown("<div class='section-header'>Logical Consistency Checks</div>", unsafe_allow_html=True)
        st.info("Validates that: Voltage > 0 and Current >= 0; Active Power <= Apparent Power")
        if st.button("Run Consistency Check", type="primary"):
            checks = [
                ("Voltage", "Current",        lambda v, c: (v > 0) & (c >= 0)),
                ("Active_Power", "Apperent_Power", lambda ap, app: ap <= app),
            ]
            incons = validator.validate_data_consistency(df, checks)
            if incons:
                for inc in incons:
                    st.markdown(
                        f"<div class='alert-item warning'>{inc['columns']}: "
                        f"{inc['inconsistent_count']} inconsistent rows</div>",
                        unsafe_allow_html=True)
            else:
                st.markdown("<div class='alert-item ok'>All consistency checks passed.</div>",
                            unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='section-header'>Sensor Availability</div>", unsafe_allow_html=True)
        if st.button("Run Sensor Health Check", type="primary"):
            health  = validator.validate_sensor_health(df, RELEVANT_COLUMNS)
            avail   = [h["data_availability"] for h in health.values()]
            sensors = list(health.keys())
            fig_health = go.Figure(go.Bar(
                x=sensors, y=avail,
                marker_color=["#10b981" if a>=95 else "#f59e0b" if a>=80 else "#ef4444" for a in avail],
                text=[f"{a:.1f}%" for a in avail], textposition="outside"))
            fig_health.update_layout(title="Sensor Data Availability (%)",
                                     yaxis=dict(range=[0, 110]), height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_health, use_container_width=True, config={"displayModeBar": False})
            st.metric("Average Availability", f"{np.mean(avail):.2f}%")


def page_quality(df):
    """Hypothesis H2: Quality Assessment."""
    st.markdown("### Quality Assessment  —  Hypothesis H2")

    st.markdown(
        "<div class='hypothesis-box'>"
        "<div class='hypothesis-title'>Hypothesis H2</div>"
        "<div class='hypothesis-text'>"
        "It is possible to assess the quality of control panel data using a technology-agnostic "
        "weighted framework. Three dimensions are scored independently: "
        "Completeness (30%), Accuracy / Outlier Detection (30%), and Stability (40%). "
        "The same methodology applies to any monitoring system regardless of technology stack."
        "</div></div>",
        unsafe_allow_html=True)

    quality = QualityAssessment()
    tab1, tab2 = st.tabs(["Quality Score", "Correction Impact"])

    with tab1:
        if st.button("Calculate Quality Score", type="primary"):
            available        = [c for c in RELEVANT_COLUMNS if c in df.columns]
            score, components = quality.assess_data_quality_score(df, available)
            report           = quality.generate_quality_report(score)

            c1, c2 = st.columns([1, 2])
            with c1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"suffix": "/100", "font": {"color": "#f1f5f9", "family": "JetBrains Mono", "size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#475569"},
                        "bar": {"color": report["color"], "thickness": 0.3},
                        "bgcolor": "rgba(0,0,0,0)", "bordercolor": "#1e2a3a",
                        "steps": [
                            {"range": [0,  60],  "color": "rgba(239,68,68,0.1)"},
                            {"range": [60, 75],  "color": "rgba(245,158,11,0.1)"},
                            {"range": [75, 90],  "color": "rgba(59,130,246,0.1)"},
                            {"range": [90, 100], "color": "rgba(16,185,129,0.1)"},
                        ],
                    },
                    title={"text": f"Overall Quality: {report['status']}",
                           "font": {"color": "#94a3b8", "size": 13}}
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280,
                                        margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            with c2:
                st.markdown(f"<div class='alert-item info'>{report['recommendation']}</div>",
                            unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                for name, val, weight in components:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;"
                        f"font-size:0.8rem;'>"
                        f"<span style='color:#94a3b8;'>{name}</span>"
                        f"<span style='color:#f1f5f9;font-family:JetBrains Mono,monospace;'>"
                        f"{val:.1f} / 100</span></div>",
                        unsafe_allow_html=True)
                    st.progress(min(val / 100, 1.0))
                    st.markdown(
                        f"<div style='font-size:0.7rem;color:#475569;margin-bottom:8px;'>"
                        f"Weight: {weight*100:.0f}%</div>",
                        unsafe_allow_html=True)

    with tab2:
        if st.button("Analyze Correction Impact", type="primary"):
            available    = [c for c in RELEVANT_COLUMNS if c in df.columns]
            preprocessor = DataPreprocessor()
            df_corrected = preprocessor.interpolate_missing_values(df.copy(), available)
            impact       = quality.calculate_correction_impact(df, df_corrected, available)
            impact_data  = pd.DataFrame([{
                "Column":       col,
                "Rows Changed": m["rows_changed"],
                "% Changed":    round(m["percent_changed"], 3),
                "MAE":          round(m["mean_absolute_error"], 6)
            } for col, m in impact.items()])
            fig_impact = px.bar(impact_data, x="Column", y="% Changed", color="MAE",
                                color_continuous_scale=[[0,"#10b981"],[0.5,"#3b82f6"],[1,"#ef4444"]],
                                title="Correction Impact by Column")
            fig_impact.update_layout(height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_impact, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(impact_data, use_container_width=True)


# ==================== MAIN ====================

def main():
    with st.sidebar:
        st.markdown(
            "<div style='padding:1rem 0 1.5rem;'>"
            "<div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.02em;'>SmartGrid</div>"
            "<div style='font-size:0.7rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;"
            "margin-top:2px;'>AI Dashboard</div>"
            "</div>",
            unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Navigation</div>", unsafe_allow_html=True)
        page = st.radio(
            label="",
            options=["Live Monitor", "Historical Analysis", "H: Preprocessing",
                     "H1: Validation", "H2: Quality"],
            label_visibility="collapsed")

        st.markdown("<br><div class='section-header'>System Info</div>", unsafe_allow_html=True)
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        st.progress(cpu / 100)
        st.markdown(f"<div style='font-size:0.72rem;color:#475569;margin-top:-8px;'>CPU: {cpu:.1f}%</div>",
                    unsafe_allow_html=True)
        st.progress(mem / 100)
        st.markdown(f"<div style='font-size:0.72rem;color:#475569;margin-top:-8px;'>Memory: {mem:.1f}%</div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:0.65rem;color:#334155;'>"
            f"Dataset: demo_data.csv<br>"
            f"Refreshed: {datetime.now().strftime('%H:%M:%S')}</div>",
            unsafe_allow_html=True)

    df = load_data(DATA_FILE)
    if df is None:
        st.stop()

    if page == "Live Monitor":        page_realtime(df)
    elif page == "Historical Analysis": page_analysis(df)
    elif page == "H: Preprocessing":   page_preprocessing(df)
    elif page == "H1: Validation":     page_validation(df)
    elif page == "H2: Quality":        page_quality(df)


if __name__ == "__main__":
    main()