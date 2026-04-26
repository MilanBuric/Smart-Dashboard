import streamlit as st
import time
import pandas as pd
import requests
import psutil
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime, timedelta
import random

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

    /* Dark background */
    .stApp {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 100%;
    }

    /* Sidebar */
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
    .kpi-card.blue::before  { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .kpi-card.green::before { background: linear-gradient(90deg, #10b981, #34d399); }
    .kpi-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .kpi-card.red::before   { background: linear-gradient(90deg, #ef4444, #f87171); }
    .kpi-card.purple::before{ background: linear-gradient(90deg, #8b5cf6, #a78bfa); }

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
    .kpi-unit {
        font-size: 0.85rem;
        color: #64748b;
        margin-left: 4px;
    }
    .kpi-delta {
        font-size: 0.75rem;
        margin-top: 0.35rem;
    }
    .kpi-delta.up   { color: #10b981; }
    .kpi-delta.down { color: #ef4444; }
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

    /* Chart container */
    .chart-container {
        background: #0d1420;
        border: 1px solid #1e2a3a;
        border-radius: 12px;
        padding: 1rem;
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

    /* Sidebar nav button */
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

    /* Metric overrides */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #64748b !important;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #1e2a3a;
        border-radius: 8px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e2a3a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #2d3f55; }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Plotly chart background fix */
    .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ==================== HYPOTHESIS IMPLEMENTATIONS ====================

class DataPreprocessor:
    """H: AI Integration in Data Preprocessing"""

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
            std = df[column].std()
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
                missing_pct = (missing_count / len(df)) * 100
                missing_report[col] = {
                    "missing_count": int(missing_count),
                    "missing_percentage": missing_pct,
                    "has_missing": missing_count > 0
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
                min_val, max_val = df[col].min(), df[col].max()
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0
        elif method == 'robust':
            for col in columns:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                median = df[col].median()
                df_normalized[col] = (df[col] - median) / IQR if IQR != 0 else 0
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
        rolling_mean = df[column].rolling(window=window).mean()
        rolling_std = df[column].rolling(window=window).std()
        return (df[column] - rolling_mean).abs() > (threshold * rolling_std)


class ControlPanelValidator:
    @staticmethod
    def validate_data_range(df, column, min_val, max_val):
        invalid_rows = df[(df[column] < min_val) | (df[column] > max_val)]
        validity_score = (1 - len(invalid_rows) / len(df)) * 100
        return {
            "valid": len(invalid_rows) == 0,
            "invalid_count": len(invalid_rows),
            "validity_score": validity_score,
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
                        "columns": f"{col1} vs {col2}",
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
                    "total_readings": len(df),
                    "missing_readings": int(missing),
                    "data_availability": (1 - missing / len(df)) * 100
                }
        return health_report


class QualityAssessment:
    @staticmethod
    def calculate_correction_impact(df_original, df_corrected, columns):
        impact_report = {}
        for col in columns:
            if col in df_original.columns and col in df_corrected.columns:
                changes = (df_corrected[col] != df_original[col]).sum()
                mae = np.mean(np.abs(df_original[col] - df_corrected[col]))
                impact_report[col] = {
                    "rows_changed": int(changes),
                    "percent_changed": (changes / len(df_original)) * 100,
                    "mean_absolute_error": mae
                }
        return impact_report

    @staticmethod
    def assess_data_quality_score(df, sensor_columns):
        scores = []
        completeness = (1 - df[sensor_columns].isna().sum().sum() / (len(df) * len(sensor_columns))) * 100
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
            return {"status": "Good", "color": "#3b82f6", "score": overall_score,
                    "recommendation": "Data quality is good. Minor improvements recommended."}
        elif overall_score >= 60:
            return {"status": "Fair", "color": "#f59e0b", "score": overall_score,
                    "recommendation": "Data quality needs attention. Review outliers and missing values."}
        else:
            return {"status": "Poor", "color": "#ef4444", "score": overall_score,
                    "recommendation": "Data quality is poor. Immediate action required."}


# ==================== CONFIGURATION ====================

DATA_FILE = "demo_data.csv"
CITIES = {
    "Zrenjanin, Serbia":              {"latitude": 45.3755, "longitude": 20.4020},
    "Belgrade, Serbia":               {"latitude": 44.8176, "longitude": 20.4633},
    "Novi Sad, Serbia":               {"latitude": 45.2671, "longitude": 19.8335},
    "Banja Luka, Bosnia":             {"latitude": 44.7722, "longitude": 17.1910},
    "Sarajevo, Bosnia":               {"latitude": 43.8486, "longitude": 18.3564},
    "Zagreb, Croatia":                {"latitude": 45.8125, "longitude": 15.9780},
}

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

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        return df
    except FileNotFoundError:
        # Generate synthetic data for demo
        n = 500
        np.random.seed(42)
        times = [datetime.now() - timedelta(minutes=i) for i in range(n, 0, -1)]
        df = pd.DataFrame({
            "Time": times,
            "Voltage":              np.random.normal(230, 5, n),
            "Current":              np.random.normal(10, 1.5, n),
            "Measured_Frequency":   np.random.normal(50, 0.1, n),
            "Active_Power":         np.random.normal(2200, 150, n),
            "Reactive_Power":       np.random.normal(400, 80, n),
            "Apperent_Power":       np.random.normal(2300, 160, n),
            "Phase_Voltage_Angle":  np.random.normal(0, 2, n),
            "Cos_Phi":              np.random.uniform(0.9, 1.0, n),
            "Power_Factor":         np.random.uniform(0.88, 0.99, n),
        })
        # Introduce some anomalies
        df.loc[np.random.choice(n, 5), "Voltage"] = np.random.uniform(260, 280, 5)
        df.loc[np.random.choice(n, 3), "Current"] = np.random.uniform(18, 22, 3)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def simulate_live_reading(df):
    """Return the latest simulated reading with slight noise."""
    latest = df.iloc[-1].copy()
    for col in RELEVANT_COLUMNS:
        if col in latest.index:
            noise = latest[col] * np.random.uniform(-0.005, 0.005)
            latest[col] = latest[col] + noise
    latest["Time"] = datetime.now()
    return latest


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
    cls = {"Online": "badge-online", "Warning": "badge-warning", "Offline": "badge-offline"}.get(status, "badge-online")
    dot = {"Online": "#4ade80", "Warning": "#fbbf24", "Offline": "#f87171"}.get(status, "#4ade80")
    return f"""<span class="status-badge {cls}">
        <svg width="7" height="7" viewBox="0 0 7 7"><circle cx="3.5" cy="3.5" r="3.5" fill="{dot}"/></svg>
        {label}
    </span>"""


# ==================== DASHBOARD PAGES ====================

def page_realtime(df):
    """Main real-time monitoring page."""
    reading = simulate_live_reading(df)

    # ---- Header row ----
    col_title, col_status, col_refresh = st.columns([3, 2, 1])
    with col_title:
        st.markdown("### Live Grid Monitor")
        st.markdown(
            f"<span style='font-size:0.75rem;color:#475569;font-family:JetBrains Mono,monospace;'>"
            f"Last updated: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}</span>",
            unsafe_allow_html=True
        )
    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        v = reading.get("Voltage", 230)
        s = "Online" if 210 <= v <= 250 else "Warning"
        st.markdown(status_badge(s, f"Grid {s}") + "&nbsp;&nbsp;" +
                    status_badge("Online", "Sensors OK"), unsafe_allow_html=True)
    with col_refresh:
        auto_refresh = st.checkbox("Auto-refresh", value=False)

    st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)

    # ---- KPI Cards ----
    c1, c2, c3, c4, c5 = st.columns(5)
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]

    def delta(col, fmt=".1f"):
        diff = reading.get(col, 0) - prev.get(col, 0)
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

    # ---- Charts Row 1 ----
    st.markdown("<div class='section-header'>Waveform Trends</div>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        # Voltage & Current over time
        window = df.tail(120).copy()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=window["Time"], y=window["Voltage"],
                name="Voltage (V)", mode="lines",
                line=dict(color="#3b82f6", width=2),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"
            ), secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=window["Time"], y=window["Current"],
                name="Current (A)", mode="lines",
                line=dict(color="#10b981", width=2, dash="dot")
            ), secondary_y=True
        )
        fig.update_layout(
            title=dict(text="Voltage & Current — Last 120 Readings", font=dict(size=13, color="#94a3b8")),
            height=300, **PLOTLY_LAYOUT
        )
        fig.update_yaxes(title_text="Voltage (V)", gridcolor="#1e2a3a", secondary_y=False)
        fig.update_yaxes(title_text="Current (A)", gridcolor="#1e2a3a", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        # Power Gauge
        pf_val = float(reading.get("Power_Factor", 0.95))
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pf_val,
            delta={"reference": float(prev.get("Power_Factor", 0.95)), "valueformat": ".4f"},
            number={"valueformat": ".4f", "font": {"color": "#f1f5f9", "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0.7, 1.0], "tickcolor": "#475569", "tickwidth": 1},
                "bar": {"color": "#3b82f6", "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "#1e2a3a",
                "steps": [
                    {"range": [0.7, 0.85], "color": "rgba(239,68,68,0.15)"},
                    {"range": [0.85, 0.93], "color": "rgba(245,158,11,0.15)"},
                    {"range": [0.93, 1.0],  "color": "rgba(16,185,129,0.15)"},
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 2}, "value": 0.9}
            },
            title={"text": "Power Factor", "font": {"color": "#94a3b8", "size": 13}}
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"), height=300, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # ---- Charts Row 2 ----
    chart_col3, chart_col4, chart_col5 = st.columns([2, 2, 1])

    with chart_col3:
        # Active vs Reactive vs Apparent Power
        window = df.tail(80)
        fig2 = go.Figure()
        power_cols = {
            "Active_Power":   ("#3b82f6", "Active (W)"),
            "Reactive_Power": ("#8b5cf6", "Reactive (VAR)"),
            "Apperent_Power": ("#f59e0b", "Apparent (VA)"),
        }
        for col, (color, label) in power_cols.items():
            if col in window.columns:
                fig2.add_trace(go.Scatter(
                    x=window["Time"], y=window[col],
                    name=label, mode="lines",
                    line=dict(color=color, width=1.8)
                ))
        fig2.update_layout(
            title=dict(text="Power Components", font=dict(size=13, color="#94a3b8")),
            height=280, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with chart_col4:
        # Frequency stability
        window = df.tail(100)
        fig3 = go.Figure()
        fig3.add_hrect(y0=49.8, y1=50.2, fillcolor="rgba(16,185,129,0.08)", line_width=0)
        fig3.add_hline(y=50.0, line_dash="dash", line_color="#475569", line_width=1)
        if "Measured_Frequency" in window.columns:
            fig3.add_trace(go.Scatter(
                x=window["Time"], y=window["Measured_Frequency"],
                mode="lines",
                line=dict(color="#a78bfa", width=2),
                fill="tozeroy", fillcolor="rgba(139,92,246,0.05)",
                name="Frequency (Hz)"
            ))
        fig3.update_layout(
            title=dict(text="Frequency Stability", font=dict(size=13, color="#94a3b8")),
            height=280, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with chart_col5:
        # Mini status table
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
                    unsafe_allow_html=True
                )

    # ---- Alerts ----
    st.markdown("<br><div class='section-header'>System Alerts</div>", unsafe_allow_html=True)
    alerts = []
    v = float(reading.get("Voltage", 230))
    f = float(reading.get("Measured_Frequency", 50))
    pf = float(reading.get("Power_Factor", 0.95))

    if v > 250 or v < 210:
        alerts.append(("critical", f"Voltage out of nominal range: {v:.1f} V (expected 210-250 V)"))
    if abs(f - 50) > 0.2:
        alerts.append(("warning", f"Frequency deviation detected: {f:.3f} Hz (nominal 50 Hz)"))
    if pf < 0.9:
        alerts.append(("warning", f"Low power factor: {pf:.3f} - Consider reactive power compensation"))

    anomaly_mask = DataPreprocessor.detect_anomalies(df, "Voltage", window=10, threshold=2.5)
    anomaly_count = anomaly_mask.sum() if anomaly_mask is not None else 0
    if anomaly_count > 0:
        alerts.append(("warning", f"{anomaly_count} voltage anomalies detected in last {len(df)} readings"))

    if not alerts:
        alerts.append(("ok", "All parameters within nominal operating ranges."))

    alert_colors = {"critical": "", "warning": "warning", "ok": "ok", "info": "info"}
    cols_alert = st.columns(min(len(alerts), 3))
    for i, (level, msg) in enumerate(alerts[:3]):
        with cols_alert[i % 3]:
            st.markdown(f"<div class='alert-item {alert_colors.get(level, '')}'>{msg}</div>", unsafe_allow_html=True)

    # Auto-refresh
    if auto_refresh:
        time.sleep(3)
        st.rerun()


def page_analysis(df):
    """Historical analysis and trends."""
    st.markdown("### Historical Analysis")

    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Correlation Matrix", "Statistical Summary"])

    with tab1:
        col_sel = st.selectbox("Select column:", [c for c in RELEVANT_COLUMNS if c in df.columns])
        col_l, col_r = st.columns(2)

        with col_l:
            fig_hist = px.histogram(
                df, x=col_sel, nbins=40,
                color_discrete_sequence=["#3b82f6"],
                title=f"Distribution: {col_sel}"
            )
            fig_hist.update_layout(height=320, **PLOTLY_LAYOUT)
            fig_hist.update_traces(marker_line_color="#1e2a3a", marker_line_width=0.5)
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

        with col_r:
            fig_box = px.box(
                df, y=col_sel,
                color_discrete_sequence=["#8b5cf6"],
                title=f"Box Plot: {col_sel}"
            )
            fig_box.update_layout(height=320, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

        # Time series for selected column
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df["Time"], y=df[col_sel],
            mode="lines",
            line=dict(color="#10b981", width=1.5),
            name=col_sel
        ))
        # Rolling mean
        rolling = df[col_sel].rolling(20).mean()
        fig_ts.add_trace(go.Scatter(
            x=df["Time"], y=rolling,
            mode="lines",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            name="20-pt Rolling Mean"
        ))
        fig_ts.update_layout(
            title=dict(text=f"{col_sel} Over Time", font=dict(size=13, color="#94a3b8")),
            height=300, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        available = [c for c in RELEVANT_COLUMNS if c in df.columns]
        corr = df[available].corr()

        fig_corr = px.imshow(
            corr,
            color_continuous_scale=[[0, "#ef4444"], [0.5, "#1e2a3a"], [1, "#3b82f6"]],
            zmin=-1, zmax=1,
            title="Pearson Correlation Matrix",
            text_auto=".2f"
        )
        fig_corr.update_layout(height=500, **PLOTLY_LAYOUT)
        fig_corr.update_coloraxes(colorbar_tickfont_color="#94a3b8")
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        available = [c for c in RELEVANT_COLUMNS if c in df.columns]
        stats = df[available].describe().T
        stats["cv%"] = (stats["std"] / stats["mean"] * 100).round(2)
        st.dataframe(
            stats.style.background_gradient(cmap="Blues", subset=["mean", "std"]),
            use_container_width=True
        )


def page_preprocessing(df):
    """AI Preprocessing Tools (H)."""
    st.markdown("### AI Data Preprocessing  —  Hypothesis H")
    preprocessor = DataPreprocessor()

    tab1, tab2, tab3, tab4 = st.tabs(["Outlier Detection", "Missing Values", "Normalization", "Anomaly Detection"])

    with tab1:
        st.markdown("<div class='section-header'>Outlier Detection</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            selected_col = st.selectbox("Column:", [c for c in RELEVANT_COLUMNS if c in df.columns])
        with c2:
            method = st.radio("Method:", ["IQR", "Z-Score"], horizontal=True)
        with c3:
            threshold = st.slider("Threshold:", 0.5, 5.0, 1.5)

        if st.button("Run Outlier Detection", type="primary"):
            outliers, lower, upper = preprocessor.detect_and_handle_outliers(
                df, selected_col, 'iqr' if method == "IQR" else 'zscore', threshold)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Outliers Found", len(outliers))
            mc2.metric("Lower Bound", f"{lower:.3f}")
            mc3.metric("Upper Bound", f"{upper:.3f}")

            fig_out = go.Figure()
            fig_out.add_trace(go.Scatter(
                x=df.index, y=df[selected_col],
                mode="lines", name=selected_col,
                line=dict(color="#3b82f6", width=1.5)
            ))
            if len(outliers) > 0:
                fig_out.add_trace(go.Scatter(
                    x=outliers.index, y=outliers[selected_col],
                    mode="markers", name="Outliers",
                    marker=dict(color="#ef4444", size=8, symbol="x")
                ))
            fig_out.add_hrect(y0=lower, y1=upper, fillcolor="rgba(16,185,129,0.07)", line_width=0)
            fig_out.update_layout(title=f"Outlier Detection: {selected_col}", height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_out, use_container_width=True, config={"displayModeBar": False})

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
                         "Has Missing": "Yes" if v["has_missing"] else "No"} for k, v in report.items()]
                st.dataframe(pd.DataFrame(data), use_container_width=True)

        if c2.button("Interpolate"):
            if cols_to_interp:
                df_proc = preprocessor.interpolate_missing_values(df, cols_to_interp, interp_method, order)
                before = df[cols_to_interp].isna().sum().sum()
                after = df_proc[cols_to_interp].isna().sum().sum()
                st.metric("Missing Before", int(before))
                st.metric("Missing After", int(after))
                st.success("Interpolation complete.")

    with tab3:
        st.markdown("<div class='section-header'>Data Normalization</div>", unsafe_allow_html=True)
        norm_method = st.selectbox("Method:", ["standard", "minmax", "robust", "log", "zscore"])
        cols_to_norm = st.multiselect("Columns:", [c for c in RELEVANT_COLUMNS if c in df.columns], key="norm_cols")

        if st.button("Normalize", type="primary"):
            if cols_to_norm:
                df_norm = preprocessor.normalize_data(df, cols_to_norm, norm_method)
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original Statistics")
                    st.dataframe(df[cols_to_norm].describe(), use_container_width=True)
                with c2:
                    st.caption("Normalized Statistics")
                    st.dataframe(df_norm[cols_to_norm].describe(), use_container_width=True)

                # Overlay chart
                col_show = cols_to_norm[0]
                fig_n = go.Figure()
                fig_n.add_trace(go.Scatter(y=df[col_show].values, mode="lines",
                                           name="Original", line=dict(color="#3b82f6")))
                fig_n.add_trace(go.Scatter(y=df_norm[col_show].values, mode="lines",
                                           name="Normalized", line=dict(color="#10b981")))
                fig_n.update_layout(title=f"Normalization Effect: {col_show}", height=300, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_n, use_container_width=True, config={"displayModeBar": False})

    with tab4:
        st.markdown("<div class='section-header'>Anomaly Detection</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            anom_col = st.selectbox("Column:", [c for c in RELEVANT_COLUMNS if c in df.columns], key="anom_col")
        with c2:
            window = st.slider("Window:", 3, 20, 5)
        with c3:
            anom_thresh = st.slider("Std Dev Threshold:", 1.0, 5.0, 2.0)

        if st.button("Detect Anomalies", type="primary"):
            anomalies = preprocessor.detect_anomalies(df, anom_col, window, anom_thresh)
            count = anomalies.sum()

            mc1, mc2 = st.columns(2)
            mc1.metric("Anomalies Detected", int(count))
            mc2.metric("Anomaly Rate", f"{(count / len(df) * 100):.2f}%")

            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=df.index, y=df[anom_col], mode="lines",
                                       name=anom_col, line=dict(color="#8b5cf6", width=1.5)))
            if count > 0:
                fig_a.add_trace(go.Scatter(
                    x=df[anomalies].index, y=df[anomalies][anom_col],
                    mode="markers", name="Anomaly",
                    marker=dict(color="#ef4444", size=8, symbol="x-open-dot")))
            fig_a.update_layout(title=f"Anomaly Detection: {anom_col}", height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_a, use_container_width=True, config={"displayModeBar": False})


def page_validation(df):
    """H1: Control Panel Validation."""
    st.markdown("### Control Panel Validation  —  Hypothesis H1")
    validator = ControlPanelValidator()

    tab1, tab2, tab3 = st.tabs(["Range Validation", "Consistency Check", "Sensor Health"])

    with tab1:
        st.markdown("<div class='section-header'>Data Range Validation</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            val_col = st.selectbox("Column:", [c for c in RELEVANT_COLUMNS if c in df.columns], key="val_col")
        with c2:
            min_v = st.number_input("Min acceptable:", value=float(df[val_col].mean() - 3 * df[val_col].std()))
        with c3:
            max_v = st.number_input("Max acceptable:", value=float(df[val_col].mean() + 3 * df[val_col].std()))

        if st.button("Validate Range", type="primary"):
            result = validator.validate_data_range(df, val_col, min_v, max_v)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Validity Score", f"{result['validity_score']:.2f}%")
            mc2.metric("Invalid Rows", result['invalid_count'])
            mc3.metric("Status", "Valid" if result['valid'] else "Invalid")

            # Visualization
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=df.index, y=df[val_col], mode="lines",
                                       line=dict(color="#3b82f6", width=1.5), name=val_col))
            fig_v.add_hrect(y0=min_v, y1=max_v, fillcolor="rgba(16,185,129,0.07)", line_width=0,
                            annotation_text="Valid Range")
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
                ("Voltage", "Current", lambda v, c: (v > 0) & (c >= 0)),
                ("Active_Power", "Apperent_Power", lambda ap, app: ap <= app),
            ]
            incons = validator.validate_data_consistency(df, checks)
            if incons:
                for inc in incons:
                    st.markdown(
                        f"<div class='alert-item warning'>{inc['columns']}: "
                        f"{inc['inconsistent_count']} inconsistent rows</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("<div class='alert-item ok'>All consistency checks passed.</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='section-header'>Sensor Availability</div>", unsafe_allow_html=True)
        if st.button("Run Sensor Health Check", type="primary"):
            health = validator.validate_sensor_health(df, RELEVANT_COLUMNS)
            avail = [h["data_availability"] for h in health.values()]
            sensors = list(health.keys())

            fig_health = go.Figure(go.Bar(
                x=sensors,
                y=avail,
                marker_color=["#10b981" if a >= 95 else "#f59e0b" if a >= 80 else "#ef4444" for a in avail],
                text=[f"{a:.1f}%" for a in avail],
                textposition="outside",
            ))
            fig_health.update_layout(
                title="Sensor Data Availability (%)",
                yaxis=dict(range=[0, 110]),
                height=350, **PLOTLY_LAYOUT
            )
            st.plotly_chart(fig_health, use_container_width=True, config={"displayModeBar": False})
            avg = np.mean(avail)
            st.metric("Average Availability", f"{avg:.2f}%")


def page_quality(df):
    """H2: Quality Assessment."""
    st.markdown("### Quality Assessment  —  Hypothesis H2")
    quality = QualityAssessment()

    tab1, tab2 = st.tabs(["Quality Score", "Correction Impact"])

    with tab1:
        if st.button("Calculate Quality Score", type="primary"):
            available = [c for c in RELEVANT_COLUMNS if c in df.columns]
            score, components = quality.assess_data_quality_score(df, available)
            report = quality.generate_quality_report(score)

            c1, c2 = st.columns([1, 2])
            with c1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"suffix": "/100", "font": {"color": "#f1f5f9", "family": "JetBrains Mono", "size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#475569"},
                        "bar": {"color": report["color"], "thickness": 0.3},
                        "bgcolor": "rgba(0,0,0,0)",
                        "bordercolor": "#1e2a3a",
                        "steps": [
                            {"range": [0, 60],  "color": "rgba(239,68,68,0.1)"},
                            {"range": [60, 75], "color": "rgba(245,158,11,0.1)"},
                            {"range": [75, 90], "color": "rgba(59,130,246,0.1)"},
                            {"range": [90, 100], "color": "rgba(16,185,129,0.1)"},
                        ],
                    },
                    title={"text": f"Overall Quality: {report['status']}", "font": {"color": "#94a3b8", "size": 13}}
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            with c2:
                st.markdown(f"<div class='alert-item info'>{report['recommendation']}</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                for name, val, weight in components:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;font-size:0.8rem;'>"
                        f"<span style='color:#94a3b8;'>{name}</span>"
                        f"<span style='color:#f1f5f9;font-family:JetBrains Mono,monospace;'>{val:.1f} / 100</span></div>",
                        unsafe_allow_html=True
                    )
                    st.progress(min(val / 100, 1.0))
                    st.markdown(f"<div style='font-size:0.7rem;color:#475569;margin-bottom:8px;'>Weight: {weight*100:.0f}%</div>",
                                unsafe_allow_html=True)

    with tab2:
        if st.button("Analyze Correction Impact", type="primary"):
            available = [c for c in RELEVANT_COLUMNS if c in df.columns]
            preprocessor = DataPreprocessor()
            df_corrected = preprocessor.interpolate_missing_values(df.copy(), available)
            impact = quality.calculate_correction_impact(df, df_corrected, available)

            impact_data = pd.DataFrame([{
                "Column": col,
                "Rows Changed": m["rows_changed"],
                "% Changed": round(m["percent_changed"], 3),
                "MAE": round(m["mean_absolute_error"], 6)
            } for col, m in impact.items()])

            fig_impact = px.bar(
                impact_data, x="Column", y="% Changed",
                color="MAE",
                color_continuous_scale=[[0, "#10b981"], [0.5, "#3b82f6"], [1, "#ef4444"]],
                title="Correction Impact by Column",
            )
            fig_impact.update_layout(height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_impact, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(impact_data, use_container_width=True)


# ==================== MAIN ====================

def main():
    # ---- Sidebar ----
    with st.sidebar:
        st.markdown(
            "<div style='padding:1rem 0 1.5rem;'>"
            "<div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.02em;'>SmartGrid</div>"
            "<div style='font-size:0.7rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;margin-top:2px;'>AI Dashboard</div>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='section-header'>Navigation</div>", unsafe_allow_html=True)
        page = st.radio(
            label="",
            options=["Live Monitor", "Historical Analysis", "H: Preprocessing",
                     "H1: Validation", "H2: Quality"],
            label_visibility="collapsed"
        )

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
            unsafe_allow_html=True
        )

    # ---- Load Data ----
    df = load_data(DATA_FILE)
    if df is None:
        st.stop()

    # ---- Route ----
    if page == "Live Monitor":
        page_realtime(df)
    elif page == "Historical Analysis":
        page_analysis(df)
    elif page == "H: Preprocessing":
        page_preprocessing(df)
    elif page == "H1: Validation":
        page_validation(df)
    elif page == "H2: Quality":
        page_quality(df)


if __name__ == "__main__":
    main()
