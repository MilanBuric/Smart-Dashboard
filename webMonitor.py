# =============================================================================
# webMonitor.py  —  SmartGrid AI Dashboard
# =============================================================================
# A single-file Streamlit application that reads electrical grid sensor data
# from a CSV file and provides five interactive pages:
#
#   1. Live Monitor      — real-time KPIs, charts and anomaly alerts
#   2. Historical        — distribution, correlation and statistics on the full dataset
#   3. H:  Preprocessing — classical + AI/ML anomaly detection (Hypothesis H)
#   4. H1: Validation    — control panel element verification (Hypothesis H1)
#   5. H2: Quality       — weighted data quality scoring (Hypothesis H2)
#
# Run with:  streamlit run webMonitor.py
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# All external libraries used throughout the file are imported here at the top
# so Python can resolve them before any code runs.
# -----------------------------------------------------------------------------

import streamlit as st          # The web-application framework — every UI element comes from here
import time                     # Standard library time utilities (kept for reference, not actively used after the autorefresh fix)
import pandas as pd             # DataFrame library — used for all tabular data handling
import psutil                   # System metrics library — reads CPU and RAM usage for the sidebar
import plotly.graph_objects as go   # Low-level Plotly API for building custom chart figures
import plotly.express as px         # High-level Plotly API for quick charts (histogram, box, etc.)
from plotly.subplots import make_subplots  # Used to create a chart with two Y-axes (Voltage + Current)
from sklearn.preprocessing import StandardScaler  # Normalises features before feeding them to ML models
from sklearn.ensemble import IsolationForest       # ML model 1: tree-based anomaly detection
from sklearn.neighbors import LocalOutlierFactor   # ML model 2: density-based anomaly detection
from sklearn.svm import OneClassSVM                # ML model 3: kernel boundary anomaly detection
from sklearn.decomposition import PCA              # ML model 4: dimensionality reduction for reconstruction-error anomaly detection
import numpy as np                        # Numerical computing — used for random numbers, statistics, array operations
from datetime import datetime, timedelta  # Date/time types used to construct and compare timestamps
from streamlit_autorefresh import st_autorefresh  # Third-party component that triggers a page rerun from the browser on a timer


# =============================================================================
# PAGE CONFIGURATION
# This MUST be the first Streamlit call in the script — Streamlit raises an
# error if any other st.* call appears before it.
# =============================================================================

st.set_page_config(
    page_title="Smart Grid Dashboard",  # Text shown in the browser tab
    page_icon="zap",                    # Emoji or path shown as the favicon
    layout="wide",                      # Uses the full browser width instead of the default centred narrow column
    initial_sidebar_state="expanded"    # Sidebar is open when the page first loads
)


# =============================================================================
# CUSTOM CSS
# Streamlit's default styling is overridden here with a dark-theme design.
# The CSS is injected as raw HTML using st.markdown with unsafe_allow_html=True.
# Every selector targets either a Streamlit internal element (e.g. stApp,
# stSidebar) or a custom class used later in the HTML strings we build.
# =============================================================================

st.markdown("""
<style>
    /* Load two typefaces from Google Fonts:
       - Inter for all body text (modern, readable)
       - JetBrains Mono for numbers and code (fixed-width, clean) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Apply Inter as the default font to every element on the page */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark navy background and light text for the whole app */
    .stApp {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    /* Reduce default padding around the main content area and allow full width */
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 100%;
    }

    /* Sidebar: slightly lighter dark background with a subtle right border */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #1e2a3a;
    }

    /* Force all text inside the sidebar to a muted blue-grey colour */
    section[data-testid="stSidebar"] * {
        color: #94a3b8 !important;
    }

    /* -------------------------------------------------------------------------
       KPI CARDS
       Each KPI (Voltage, Current, etc.) is rendered as a dark card with a
       coloured top stripe.  The card itself is a <div class="kpi-card COLOUR">
       and the stripe is drawn using the ::before pseudo-element.
    -------------------------------------------------------------------------- */
    .kpi-card {
        background: linear-gradient(135deg, #0f1623 0%, #141d2e 100%);
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        position: relative;   /* Required so ::before can be positioned inside */
        overflow: hidden;
        transition: border-color 0.3s ease;  /* Smooth colour change on hover */
    }
    /* Highlight the card border to blue when the user hovers over it */
    .kpi-card:hover { border-color: #3b82f6; }

    /* The thin coloured top stripe is a zero-content pseudo-element positioned
       at the top of the card and given a gradient fill via the colour classes */
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 12px 12px 0 0;
    }
    /* Each colour class applies a different gradient to the stripe */
    .kpi-card.blue::before   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .kpi-card.green::before  { background: linear-gradient(90deg, #10b981, #34d399); }
    .kpi-card.amber::before  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .kpi-card.red::before    { background: linear-gradient(90deg, #ef4444, #f87171); }
    .kpi-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }

    /* Small uppercase label above the number (e.g. "VOLTAGE") */
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b !important;
        margin-bottom: 0.4rem;
    }
    /* The large numeric value displayed in monospace */
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9 !important;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
    }
    /* Smaller unit text shown after the number (e.g. "V", "Hz") */
    .kpi-unit  { font-size: 0.85rem; color: #64748b; margin-left: 4px; }

    /* The small delta line below the value showing change since last reading */
    .kpi-delta { font-size: 0.75rem; margin-top: 0.35rem; }
    .kpi-delta.up      { color: #10b981; }  /* Green when value went up */
    .kpi-delta.down    { color: #ef4444; }  /* Red when value went down */
    .kpi-delta.neutral { color: #64748b; }  /* Grey when unchanged */

    /* -------------------------------------------------------------------------
       STATUS BADGES
       Pill-shaped labels used in the Live Monitor header to show grid status.
       The coloured dot inside is an inline SVG circle.
    -------------------------------------------------------------------------- */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;  /* Makes it fully pill-shaped */
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    /* Three colour variants: green (online), amber (warning), red (offline) */
    .badge-online  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
    .badge-warning { background: #1c1003; color: #fbbf24; border: 1px solid #92400e; }
    .badge-offline { background: #1c0404; color: #f87171; border: 1px solid #991b1b; }

    /* -------------------------------------------------------------------------
       SECTION HEADERS
       Thin uppercase divider lines used to separate groups of content
       (e.g. "KEY PERFORMANCE INDICATORS", "WAVEFORM TRENDS")
    -------------------------------------------------------------------------- */
    .section-header {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #475569;
        border-bottom: 1px solid #1e2a3a;  /* Horizontal rule below the text */
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* -------------------------------------------------------------------------
       ALERT CARDS
       Used in the System Alerts strip at the bottom of the Live Monitor.
       The default (no modifier class) is critical (red left border).
       Modifier classes shift the border colour for other severity levels.
    -------------------------------------------------------------------------- */
    .alert-item {
        background: #0f1623;
        border-left: 3px solid #ef4444;  /* Red = critical */
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.78rem;
        color: #cbd5e1;
    }
    .alert-item.warning { border-left-color: #f59e0b; }  /* Amber = warning */
    .alert-item.info    { border-left-color: #3b82f6; }  /* Blue  = informational */
    .alert-item.ok      { border-left-color: #10b981; }  /* Green = all good */

    /* -------------------------------------------------------------------------
       HYPOTHESIS INFO BOX
       The blue-outlined description box shown at the top of pages H, H1, H2.
    -------------------------------------------------------------------------- */
    .hypothesis-box {
        background: linear-gradient(135deg, #0f1a2e, #0d1420);
        border: 1px solid #1e3a5f;
        border-left: 4px solid #3b82f6;  /* Thick blue left accent */
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
    }
    /* Small blue uppercase label ("HYPOTHESIS H") */
    .hypothesis-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #3b82f6;
        margin-bottom: 0.35rem;
    }
    /* Body text of the hypothesis description */
    .hypothesis-text {
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.6;
    }

    /* -------------------------------------------------------------------------
       ML BADGE
       Tiny purple "AI" label appended to tab titles to flag ML-powered content.
    -------------------------------------------------------------------------- */
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

    /* Give the sidebar radio navigation items some padding */
    div[data-testid="stRadio"] label {
        background: transparent;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }

    /* -------------------------------------------------------------------------
       TAB STYLING
       Overrides Streamlit's default white tab bar with the dark theme.
       The selected tab gets a blue bottom border and blue text.
    -------------------------------------------------------------------------- */
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

    /* Override Streamlit's st.metric colours to match the dark theme */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #64748b !important;
    }

    /* Add a subtle border around data tables */
    .stDataFrame { border: 1px solid #1e2a3a; border-radius: 8px; }

    /* Custom thin dark scrollbar so the scrollbar blends with the theme */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e2a3a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #2d3f55; }

    /* Hide Streamlit's hamburger menu and footer branding.
       IMPORTANT: We do NOT hide the full <header> element because the
       sidebar toggle arrow lives inside it — hiding the header would make
       it impossible to reopen a collapsed sidebar.
       Instead we keep the header but strip its background so it is
       invisible while the toggle button remains clickable. */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
    }

    /* Force all Plotly chart backgrounds to transparent so they blend
       with the dark page background rather than showing a white box */
    .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CLASS: DataPreprocessor  (Hypothesis H)
# =============================================================================
# Implements both classical statistical preprocessing methods (baseline) and
# genuine machine-learning anomaly detection models.
#
# The key distinction between the two groups:
#   Classical methods — require the developer to manually specify a threshold
#                       (e.g. "flag anything beyond 1.5 × IQR").
#   ML methods        — learn what "normal" looks like from the data itself
#                       and flag deviations automatically, with no manual threshold.
# =============================================================================

class DataPreprocessor:
    """
    Hypothesis H: AI Integration in Data Preprocessing.
    All methods are @staticmethod — they do not need an instance to be called,
    they just operate on whatever DataFrame is passed in.
    """

    # -------------------------------------------------------------------------
    # CLASSICAL / STATISTICAL METHODS  (baseline comparison group)
    # -------------------------------------------------------------------------

    @staticmethod
    def detect_and_handle_outliers(df, column, method='iqr', threshold=1.5):
        """
        Detects outliers in a single column using either IQR or Z-Score.

        IQR (Interquartile Range):
            Computes Q1 (25th percentile) and Q3 (75th percentile).
            IQR = Q3 - Q1.
            Lower bound = Q1 - threshold × IQR
            Upper bound = Q3 + threshold × IQR
            Anything outside those bounds is an outlier.
            The default threshold of 1.5 is the standard Tukey rule.

        Z-Score:
            Computes how many standard deviations each value is from the mean.
            Lower bound = mean - threshold × std
            Upper bound = mean + threshold × std
            Threshold of 2.0 catches ~95% of normally distributed data.

        Returns: (outlier_rows DataFrame, lower_bound float, upper_bound float)
        """
        if method == 'iqr':
            Q1  = df[column].quantile(0.25)   # 25th percentile
            Q3  = df[column].quantile(0.75)   # 75th percentile
            IQR = Q3 - Q1                     # Interquartile range
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        else:
            # Z-Score method: bounds are multiples of the standard deviation
            mean = df[column].mean()
            std  = df[column].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std

        # Select all rows where the value falls outside the calculated bounds
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    @staticmethod
    def check_missing_values(df, columns):
        """
        Counts missing (NaN) values for each column in the provided list.
        Returns a dictionary keyed by column name with:
          - missing_count      : absolute number of NaN rows
          - missing_percentage : what fraction of the total rows are NaN
          - has_missing        : boolean convenience flag
        """
        missing_report = {}
        for col in columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()                    # Count NaN values
                missing_pct   = (missing_count / len(df)) * 100        # Convert to percentage
                missing_report[col] = {
                    "missing_count":      int(missing_count),
                    "missing_percentage": missing_pct,
                    "has_missing":        missing_count > 0
                }
        return missing_report

    @staticmethod
    def interpolate_missing_values(df, columns, method='linear', order=2):
        """
        Fills NaN gaps in each column by interpolating between surrounding values.

        method='linear'     — straight line between adjacent known values (fastest, good default)
        method='polynomial' — fits a polynomial curve of the given order
        method='spline'     — fits a smooth spline curve of the given order

        limit_direction='both' means interpolation works forwards AND backwards,
        filling gaps even at the start or end of the series.

        Works on a copy so the original DataFrame is never modified.
        Warnings are shown via st.warning if a column cannot be interpolated.
        """
        df_processed = df.copy()
        for col in columns:
            if col in df_processed.columns:
                try:
                    if method in ['polynomial', 'spline']:
                        # Polynomial and spline methods need an explicit order parameter
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
        """
        Rescales the values in each column so they are on a comparable scale.
        Required before distance-based ML algorithms (SVM, LOF) so that columns
        with large values (e.g. Active Power ~2200 W) do not dominate columns
        with small values (e.g. Cos_Phi ~0.95).

        standard — zero mean, unit variance: (x - mean) / std
                   Useful when data is roughly Gaussian.

        minmax   — maps values to [0, 1]: (x - min) / (max - min)
                   Useful when you need bounded output.

        robust   — uses median and IQR instead of mean/std: (x - median) / IQR
                   Less affected by extreme outliers than standard scaling.

        log      — natural logarithm ln(x)
                   Only applied if all values in the column are positive.
                   Compresses large ranges and makes skewed data more symmetrical.

        zscore   — identical to 'standard'; included as an alias for clarity.
        """
        df_normalized = df.copy()
        if method == 'standard':
            scaler = StandardScaler()
            # fit_transform computes mean/std from the data and applies the scaling
            df_normalized[columns] = scaler.fit_transform(df[columns])
        elif method == 'minmax':
            for col in columns:
                mn, mx = df[col].min(), df[col].max()
                # Guard against division by zero if all values are identical
                df_normalized[col] = (df[col] - mn) / (mx - mn) if mx != mn else 0
        elif method == 'robust':
            for col in columns:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR    = Q3 - Q1
                df_normalized[col] = (df[col] - df[col].median()) / IQR if IQR != 0 else 0
        elif method == 'log':
            for col in columns:
                # Only apply log if every value is strictly positive (log(0) is undefined)
                if (df[col] > 0).all():
                    df_normalized[col] = np.log(df[col])
        elif method == 'zscore':
            for col in columns:
                mean, std = df[col].mean(), df[col].std()
                df_normalized[col] = (df[col] - mean) / std if std != 0 else 0
        return df_normalized

    @staticmethod
    def detect_anomalies(df, column, window=5, threshold=2):
        """
        Classical rolling-window anomaly detection used in the Live Monitor
        alert strip (not one of the four ML models — this is the baseline).

        For each row, computes the mean and standard deviation of the surrounding
        `window` readings. If the current value deviates more than
        `threshold` standard deviations from that local mean, it is an anomaly.

        Returns a boolean Series (True = anomaly) aligned to df's index.
        NaN is returned for the first `window-1` rows because there are not
        enough preceding values to compute a rolling statistic.
        """
        rolling_mean = df[column].rolling(window=window).mean()
        rolling_std  = df[column].rolling(window=window).std()
        return (df[column] - rolling_mean).abs() > (threshold * rolling_std)

    # -------------------------------------------------------------------------
    # AI / ML ANOMALY DETECTION METHODS
    # None of these require a manually defined threshold.
    # They learn the normal operating pattern from the data itself.
    # -------------------------------------------------------------------------

    @staticmethod
    def isolation_forest(df, columns, contamination=0.05, n_estimators=100):
        """
        Isolation Forest — unsupervised ensemble anomaly detection.

        HOW IT WORKS:
        Builds `n_estimators` random decision trees (an "isolation forest").
        Each tree repeatedly picks a random feature and a random split point
        to partition the data. Anomalous points are unusual — they get
        isolated (separated from all other points) after very few splits.
        Normal points are surrounded by similar points and require many splits
        to isolate. The anomaly score is based on average path length across
        all trees: shorter path = more anomalous.

        PARAMETERS:
        contamination — expected fraction of anomalies in the data (0.05 = 5%).
                        The model uses this to set the decision boundary.
        n_estimators  — number of trees in the forest. More trees = more stable
                        results but slower to train.
        random_state  — fixed seed so results are reproducible.

        OUTPUT:
        preds  : array of +1 (normal) or -1 (anomaly) for every row
        scores : raw anomaly score (lower = more anomalous)
        mask   : boolean pandas Series aligned to df's full index
        """
        X      = df[columns].dropna()   # Remove rows with NaN — sklearn cannot handle them
        model  = IsolationForest(n_estimators=n_estimators,
                                  contamination=contamination, random_state=42)
        preds  = model.fit_predict(X)   # Trains the forest and returns +1/-1 labels
        scores = model.score_samples(X) # Returns the raw score for each point

        # Create output Series aligned to the original df index (including rows dropped by dropna)
        mask         = pd.Series(False, index=df.index)
        score_series = pd.Series(np.nan,  index=df.index)
        mask[X.index]         = preds == -1   # Mark anomalies
        score_series[X.index] = scores        # Fill in scores for rows that had no NaN
        return mask, score_series, model

    @staticmethod
    def local_outlier_factor(df, columns, n_neighbors=20, contamination=0.05):
        """
        Local Outlier Factor (LOF) — density-based anomaly detection.

        HOW IT WORKS:
        For each point, LOF compares its local density (how tightly packed the
        points around it are) to the densities of its `n_neighbors` nearest
        neighbours. A point in a much lower-density region than its neighbours
        gets a high LOF score and is flagged as an outlier. This means LOF can
        detect local anomalies that a global method like Isolation Forest might
        miss — for example, a cluster of points that is itself unusual relative
        to its local neighbourhood.

        PARAMETERS:
        n_neighbors   — number of neighbours used to estimate local density.
                        Higher values make the algorithm more global.
        contamination — expected fraction of anomalies (used to set the threshold).

        NOTE: LOF does not return a trained model that can be applied to new data
        (unlike Isolation Forest). The scores are computed in a single pass during
        fit_predict and stored in negative_outlier_factor_ (negative so that
        lower = more anomalous, consistent with sklearn conventions).
        """
        X      = df[columns].dropna()
        model  = LocalOutlierFactor(n_neighbors=n_neighbors,
                                     contamination=contamination)
        preds  = model.fit_predict(X)             # -1 = anomaly
        scores = model.negative_outlier_factor_   # More negative = more anomalous

        mask         = pd.Series(False, index=df.index)
        score_series = pd.Series(np.nan,  index=df.index)
        mask[X.index]         = preds == -1
        score_series[X.index] = scores
        return mask, score_series

    @staticmethod
    def one_class_svm(df, columns, nu=0.05, kernel="rbf"):
        """
        One-Class SVM — kernel boundary anomaly detection.

        HOW IT WORKS:
        Learns a tight decision boundary (hyperplane) in a high-dimensional
        kernel space that encloses the majority of the training data. Points
        outside the boundary are classified as anomalies. The RBF (Radial Basis
        Function) kernel maps the data into a space where a linear boundary can
        separate normal from anomalous points even if the original data has
        complex non-linear structure.

        PARAMETERS:
        nu     — an upper bound on the fraction of anomalies (similar to contamination).
                 Also controls how tight the boundary is around the normal data.
        kernel — 'rbf' (Radial Basis Function) is the standard choice.

        WHY STANDARDISE:
        SVMs are sensitive to feature scale because they work with distances.
        Without standardisation, a column like Active_Power (~2200) would dominate
        Cos_Phi (~0.95) and the model would effectively ignore the smaller columns.
        StandardScaler is applied before training and the same scale is used for scoring.
        """
        X        = df[columns].dropna()
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)   # Standardise: mean=0, std=1 per column
        model    = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")
        preds    = model.fit_predict(X_scaled)       # -1 = anomaly
        scores   = model.decision_function(X_scaled) # Signed distance from decision boundary

        mask         = pd.Series(False, index=df.index)
        score_series = pd.Series(np.nan,  index=df.index)
        mask[X.index]         = preds == -1
        score_series[X.index] = scores
        return mask, score_series, model

    @staticmethod
    def pca_reconstruction_error(df, columns, n_components=None, threshold_sigma=2.5):
        """
        PCA Reconstruction Error — multi-sensor pattern anomaly detection.

        HOW IT WORKS:
        PCA (Principal Component Analysis) finds the directions of maximum
        variance in the data — the "principal components" that capture the
        dominant patterns of normal multi-sensor behaviour.

        Step 1 — Compress: project every row down to `n_components` dimensions.
                 This discards the minor variance directions that capture noise.
        Step 2 — Reconstruct: project back up to the original number of dimensions.
        Step 3 — Error: compute the mean squared difference between the original
                 and reconstructed values for each row.

        Normal rows can be accurately reconstructed because they fit the learned
        patterns. Anomalous rows — which deviate from normal operation across
        multiple sensors simultaneously — cannot be well reconstructed, so they
        have high reconstruction error.

        PARAMETERS:
        n_components     — how many principal components to keep. Defaults to
                           60% of the number of features if not specified.
                           More components = less compression = lower reconstruction error.
        threshold_sigma  — how many standard deviations above the mean error
                           counts as an anomaly. Higher σ = fewer anomalies flagged.

        RETURNS:
        mask             — boolean Series (True = anomaly)
        error_series     — the reconstruction MSE for each row
        threshold        — the actual error value used as the cutoff
        explained_var    — fraction of total variance captured by each component
                           (used to draw the explained variance bar chart)
        pca              — the trained PCA object (kept in case the caller needs it)
        """
        X         = df[columns].dropna()
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(X)   # Standardise before PCA (required)

        # Default: use 60% of the number of features as components
        n_comp    = n_components or max(1, int(len(columns) * 0.6))
        pca       = PCA(n_components=n_comp, random_state=42)

        X_reduced = pca.fit_transform(X_scaled)    # Compress to n_comp dimensions
        X_recon   = pca.inverse_transform(X_reduced)  # Reconstruct back to original dimensions

        # Mean squared error between original and reconstructed values, one value per row
        errors    = np.mean((X_scaled - X_recon) ** 2, axis=1)

        # Anomaly threshold: rows whose error exceeds (mean + sigma × std) are flagged
        threshold = errors.mean() + threshold_sigma * errors.std()

        mask         = pd.Series(False, index=df.index)
        error_series = pd.Series(np.nan,  index=df.index)
        mask[X.index]         = errors > threshold
        error_series[X.index] = errors
        return mask, error_series, threshold, pca.explained_variance_ratio_, pca


# =============================================================================
# CLASS: ControlPanelValidator  (Hypothesis H1)
# =============================================================================
# Verifies that control panel data meets expected rules.
# All checks are technology-agnostic — the same logic works whether the data
# arrives from a CSV file, a PLC, SCADA system, MQTT broker, or REST API.
# =============================================================================

class ControlPanelValidator:
    """
    Hypothesis H1: It is possible to verify the elements of a control panel.
    Three verification dimensions: range compliance, logical consistency,
    and sensor data availability.
    """

    @staticmethod
    def validate_data_range(df, column, min_val, max_val):
        """
        Checks that every value in `column` falls within [min_val, max_val].
        Any row outside these bounds is considered out-of-range.

        Returns a dict containing:
          valid            — True if zero rows are out of range
          invalid_count    — absolute number of out-of-range rows
          validity_score   — percentage of rows that ARE within range (0-100)
          out_of_range_rows — the actual DataFrame rows that failed, so they
                              can be plotted as red markers on the chart
        """
        invalid_rows   = df[(df[column] < min_val) | (df[column] > max_val)]
        validity_score = (1 - len(invalid_rows) / len(df)) * 100
        return {
            "valid":             len(invalid_rows) == 0,
            "invalid_count":     len(invalid_rows),
            "validity_score":    validity_score,
            "out_of_range_rows": invalid_rows
        }

    @staticmethod
    def validate_data_consistency(df, column_pairs):
        """
        Checks logical relationships between pairs of columns.
        Each entry in column_pairs is a tuple: (col1, col2, condition_function).
        The condition function receives the two column Series and returns a
        boolean Series — True where the relationship holds, False where it fails.

        Example rule: Active_Power should always be <= Apparent_Power.
        If Active_Power > Apparent_Power for any row, that row is inconsistent.

        Returns a list of dicts, one per violated rule, with:
          columns            — string describing which pair failed
          inconsistent_count — number of rows that violated the rule
        An empty list means all checks passed.
        """
        inconsistencies = []
        for col1, col2, condition in column_pairs:
            if col1 in df.columns and col2 in df.columns:
                # ~ inverts the boolean: selects rows where condition is FALSE
                invalid = df[~condition(df[col1], df[col2])]
                if len(invalid) > 0:
                    inconsistencies.append({
                        "columns":            f"{col1} vs {col2}",
                        "inconsistent_count": len(invalid)
                    })
        return inconsistencies

    @staticmethod
    def validate_sensor_health(df, sensor_columns):
        """
        Measures data availability for each sensor column.
        A sensor with many NaN values is either malfunctioning or offline.

        For each column returns:
          total_readings    — total number of rows in the dataset
          missing_readings  — number of NaN rows for this sensor
          data_availability — percentage of non-NaN rows (100% = fully healthy)
        """
        health_report = {}
        for col in sensor_columns:
            if col in df.columns:
                missing = df[col].isna().sum()
                health_report[col] = {
                    "total_readings":    len(df),
                    "missing_readings":  int(missing),
                    "data_availability": (1 - missing / len(df)) * 100
                }
        return health_report


# =============================================================================
# CLASS: QualityAssessment  (Hypothesis H2)
# =============================================================================
# Produces a single weighted quality score for the entire dataset based on
# three independently measured dimensions.  The same scoring framework can
# be applied to any monitoring system regardless of technology stack.
# =============================================================================

class QualityAssessment:
    """
    Hypothesis H2: It is possible to assess the quality of control panel data.
    Weighted framework: Completeness 30%, Accuracy/Outliers 30%, Stability 40%.
    """

    @staticmethod
    def calculate_correction_impact(df_original, df_corrected, columns):
        """
        Compares two versions of the dataset — before and after a correction
        step (e.g. interpolation) — to measure how much was changed.

        For each column returns:
          rows_changed        — how many rows differ between original and corrected
          percent_changed     — that count as a fraction of total rows
          mean_absolute_error — average absolute difference per changed value
                                (gives a sense of how large the corrections were)
        """
        impact_report = {}
        for col in columns:
            if col in df_original.columns and col in df_corrected.columns:
                # Count rows where the value changed after correction
                changes = (df_corrected[col] != df_original[col]).sum()
                # Average magnitude of the corrections
                mae     = np.mean(np.abs(df_original[col] - df_corrected[col]))
                impact_report[col] = {
                    "rows_changed":       int(changes),
                    "percent_changed":    (changes / len(df_original)) * 100,
                    "mean_absolute_error": mae
                }
        return impact_report

    @staticmethod
    def assess_data_quality_score(df, sensor_columns):
        """
        Computes three sub-scores and combines them into an overall quality score.

        COMPLETENESS (weight 30%):
            What fraction of expected readings are actually present (not NaN)?
            Formula: (1 - total_NaN / total_cells) × 100
            100% means no missing values anywhere.

        OUTLIER DETECTION / ACCURACY (weight 30%):
            Starts at 100 and deducts points for each column that contains
            values more than 3 standard deviations from the mean.
            These extreme values suggest sensor errors or data corruption.
            Formula: score -= (count_of_extreme_values / total_rows) × 100

        STABILITY (weight 40%):
            Measures how much each sensor fluctuates relative to its average
            using the Coefficient of Variation (CV = std / mean).
            High CV (> 0.5) means the signal is very noisy or unstable.
            Formula: score -= CV × 10  for each column where CV > 0.5

        OVERALL:
            Weighted sum: 0.30 × completeness + 0.30 × accuracy + 0.40 × stability
            Result is a score between 0 and 100.
        """
        scores = []

        # --- Completeness ---
        completeness = (1 - df[sensor_columns].isna().sum().sum() /
                        (len(df) * len(sensor_columns))) * 100
        scores.append(("Completeness", completeness, 0.3))

        # --- Accuracy / Outlier Detection ---
        outlier_score = 100
        for col in sensor_columns:
            if col in df.columns and df[col].std() > 0:
                # Z-score: how many standard deviations from the mean
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                # Deduct based on the fraction of extreme (>3σ) values
                outlier_score -= (z_scores > 3).sum() / len(df) * 100
        scores.append(("Outlier Detection", max(outlier_score, 0), 0.3))

        # --- Stability ---
        stability = 100
        for col in sensor_columns:
            if col in df.columns and df[col].mean() != 0:
                cv = df[col].std() / df[col].mean()  # Coefficient of Variation
                if cv > 0.5:
                    stability -= cv * 10
        scores.append(("Stability", max(stability, 0), 0.4))

        # Weighted sum: multiply each score by its weight and add them up
        overall = sum(s * w for _, s, w in scores)
        return overall, scores

    @staticmethod
    def generate_quality_report(overall_score):
        """
        Converts the numeric overall score (0-100) into a human-readable
        status label, a display colour, and a plain-English recommendation.
        Used to fill the gauge chart title and the recommendation card.
        """
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


# =============================================================================
# CONFIGURATION CONSTANTS
# Defined here so they can be changed in one place and immediately take
# effect everywhere in the file that references them.
# =============================================================================

# Path to the CSV data file.  If the file is not found, synthetic data is used.
DATA_FILE = "demo_data.csv"

# The sensor columns expected in the CSV.  Any column not in this list is
# ignored by the preprocessing, validation and quality pages.
RELEVANT_COLUMNS = [
    "Voltage",
    "Current",
    "Measured_Frequency",
    "Active_Power",
    "Reactive_Power",
    "Apperent_Power",       # Note: intentional typo preserved from original dataset
    "Phase_Voltage_Angle",
    "Cos_Phi",
    "Power_Factor"
]

# A shared Plotly layout dictionary applied to every chart in the app.
# Using **PLOTLY_LAYOUT in update_layout() spreads these key-value pairs
# into the function call so we don't repeat them for every figure.
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",   # Transparent chart outer background
    plot_bgcolor="rgba(0,0,0,0)",    # Transparent chart inner (plot area) background
    font=dict(color="#94a3b8", family="Inter"),   # Default axis/legend text style
    margin=dict(l=10, r=10, t=30, b=10),          # Tight margins
    xaxis=dict(gridcolor="#1e2a3a", showline=False, zeroline=False),  # Dark grid, no axis lines
    yaxis=dict(gridcolor="#1e2a3a", showline=False, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),  # Transparent legend background
    hovermode="x unified",   # All traces show their value in a single hover tooltip
)


# =============================================================================
# DATA HELPERS
# Functions that handle loading the CSV, simulating live sensor ticks,
# and maintaining the scrolling live buffer.
# =============================================================================

# Maximum number of rows kept in the live buffer at any time.
# Older rows are dropped as new ones arrive to keep memory usage constant.
LIVE_BUFFER_SIZE = 200


def _make_synthetic_base():
    """
    Generates a realistic-looking synthetic dataset for use when no CSV file
    is present.  Called once and stored in session_state so it does not change
    between reruns (which would cause the live buffer to reset).

    Uses numpy's seeded random generator for reproducibility:
      np.random.seed(42) ensures the same data every run.

    The data simulates 300 readings spread over the last 15 minutes
    (one reading every 3 seconds).  Voltage/Current/Frequency/Power values
    are drawn from normal distributions centred on typical European grid values.
    A handful of artificially injected outliers give the ML models something
    interesting to detect.
    """
    n  = 300
    np.random.seed(42)   # Fix the random seed so synthetic data is reproducible
    # Create timestamps going from 15 minutes ago to now, one per row
    times = [datetime.now() - timedelta(seconds=i * 3) for i in range(n, 0, -1)]
    df = pd.DataFrame({
        "Time":                times,
        "Voltage":             np.random.normal(230, 5, n),     # European grid: 230V nominal
        "Current":             np.random.normal(10, 1.5, n),
        "Measured_Frequency":  np.random.normal(50, 0.1, n),   # European grid: 50Hz nominal
        "Active_Power":        np.random.normal(2200, 150, n),
        "Reactive_Power":      np.random.normal(400, 80, n),
        "Apperent_Power":      np.random.normal(2300, 160, n),
        "Phase_Voltage_Angle": np.random.normal(0, 2, n),
        "Cos_Phi":             np.random.uniform(0.9, 1.0, n),
        "Power_Factor":        np.random.uniform(0.88, 0.99, n),
    })
    # Inject synthetic outliers so the anomaly detection tools have something to find
    df.loc[np.random.choice(n, 8), "Voltage"]      = np.random.uniform(260, 285, 8)
    df.loc[np.random.choice(n, 5), "Current"]      = np.random.uniform(18, 24, 5)
    df.loc[np.random.choice(n, 4), "Active_Power"] = np.random.uniform(2800, 3200, 4)
    return df


def load_data(file_path):
    """
    Reads the CSV file every time it is called (no caching) so new rows
    written to disk by an external process are always picked up.

    Handles two possible CSV layouts:
      Layout A — Single 'Time' column containing a full datetime string
                 e.g.  2026-04-28 14:59:17
      Layout B — Separate 'Date' and 'Time' columns
                 e.g.  Date=28.04.2026  Time=14:59:17
                 These are combined into a single 'Time' datetime column.

    format='mixed' tells pandas to infer the format from each value
    individually rather than assuming all rows use the same format.
    This avoids the dateutil UserWarning that appears with older default behaviour.
    dayfirst=True interprets ambiguous dates like 04/05 as 4th May (not May 4th),
    which is correct for European date formatting.

    Falls back to the synthetic dataset if the file is not found.
    Returns None if an unexpected error occurs (triggers st.stop() in main).
    """
    try:
        df = pd.read_csv(file_path)

        if 'Date' in df.columns and 'Time' in df.columns:
            # Layout B: merge the two separate date and time columns
            combined   = df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip()
            df['Time'] = pd.to_datetime(combined, format='mixed', dayfirst=True, errors='coerce')
            df.drop(columns=['Date'], inplace=True)   # Remove the now-redundant Date column
        elif 'Time' in df.columns:
            # Layout A: just parse the single Time column
            df['Time'] = pd.to_datetime(df['Time'], format='mixed', errors='coerce')
        else:
            st.warning("CSV has no 'Time' or 'Date' column — timestamps will be missing.")

        return df

    except FileNotFoundError:
        # No CSV found — fall back to synthetic data stored in session_state
        if "synthetic_base" not in st.session_state:
            st.session_state["synthetic_base"] = _make_synthetic_base()
        return st.session_state["synthetic_base"]

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def _new_live_row(base_row: pd.Series) -> pd.Series:
    """
    Generates one simulated sensor reading by adding a tiny random drift
    to each column of the last known row.

    This simulates the natural small fluctuations of a real power grid:
    voltage drifts by ~0.4V per tick, frequency by ~0.008Hz, etc.

    After applying the drift, each column is clamped to a physically
    realistic range using np.clip so the simulation never produces
    impossible values (e.g. negative current or 55Hz frequency).

    The timestamp is set to datetime.now() so the live chart X-axis
    advances in real wall-clock time.
    """
    row    = base_row.copy()  # Never modify the original row in-place
    drifts = {
        "Voltage":             np.random.normal(0, 0.4),    # ±0.4V drift
        "Current":             np.random.normal(0, 0.08),   # ±0.08A drift
        "Measured_Frequency":  np.random.normal(0, 0.008),  # ±0.008Hz drift
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

    # Clamp to physical bounds to prevent the simulation from drifting away
    row["Voltage"]            = np.clip(row["Voltage"],            180, 280)
    row["Current"]            = np.clip(row["Current"],              0,  30)
    row["Measured_Frequency"] = np.clip(row["Measured_Frequency"],  49,  51)
    row["Cos_Phi"]            = np.clip(row["Cos_Phi"],            0.7, 1.0)
    row["Power_Factor"]       = np.clip(row["Power_Factor"],       0.7, 1.0)
    row["Time"]               = datetime.now()
    return row


def get_live_df(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maintains a scrolling window of the most recent LIVE_BUFFER_SIZE rows.
    Called once per page rerun — every rerun produces exactly one new row.

    HOW IT WORKS:
    1. On the very first call (when "live_buffer" is not in session_state),
       the buffer is seeded from the last LIVE_BUFFER_SIZE rows of the CSV.
       The timestamp of the last CSV row is saved as "csv_last_seen".

    2. On every subsequent call:
       a. Check whether the CSV has new rows with timestamps newer than
          csv_last_seen.  If yes, append those real rows and update csv_last_seen.
       b. If no new CSV rows arrived, generate one synthetic tick via
          _new_live_row and append it.

    3. Always trim the buffer to LIVE_BUFFER_SIZE rows (drop the oldest).

    WHY THIS APPROACH (and what was wrong before):
    The original code compared the buffer's last timestamp to the CSV's last
    timestamp.  Once the buffer accumulated synthetic rows (which have
    datetime.now() timestamps in the future relative to the CSV), the
    comparison always said "new rows exist" but
    base_df[base_df["Time"] > buf_last_time] returned ZERO rows because
    the CSV was in the past.  This silently stopped all updates after the
    very first rerun.

    The fix: track csv_last_seen independently in session_state so the
    synthetic-tick clock and the CSV clock never interfere with each other.
    The synthetic tick path runs unconditionally when no real data arrives,
    guaranteeing every rerun produces a new data point.
    """
    # --- First call: initialise the buffer ---
    if "live_buffer" not in st.session_state:
        st.session_state["live_buffer"]     = (
            base_df.tail(LIVE_BUFFER_SIZE).copy().reset_index(drop=True)
        )
        # Record the last CSV timestamp we have consumed so we can detect new rows later
        st.session_state["csv_last_seen"] = (
            base_df["Time"].iloc[-1] if "Time" in base_df.columns else None
        )

    buf           = st.session_state["live_buffer"]
    csv_last_seen = st.session_state.get("csv_last_seen")
    ingested_real = False   # Flag: did we get real data this tick?

    # --- Try to ingest genuinely new CSV rows ---
    if "Time" in base_df.columns and csv_last_seen is not None:
        new_rows = base_df[base_df["Time"] > csv_last_seen]  # Rows newer than what we last saw
        if len(new_rows) > 0:
            buf = pd.concat([buf, new_rows], ignore_index=True)
            st.session_state["csv_last_seen"] = base_df["Time"].iloc[-1]  # Advance the cursor
            ingested_real = True

    # --- Fall back to synthetic tick if no real data arrived ---
    if not ingested_real:
        new_row = _new_live_row(buf.iloc[-1])   # Drift from the last known row
        buf     = pd.concat([buf, pd.DataFrame([new_row])], ignore_index=True)

    # --- Trim to the rolling window size and save ---
    buf = buf.tail(LIVE_BUFFER_SIZE).reset_index(drop=True)
    st.session_state["live_buffer"] = buf
    return buf


# =============================================================================
# UI HELPER FUNCTIONS
# Small functions that build and return HTML strings.
# They are kept separate so the page functions stay readable.
# =============================================================================

def kpi_card(label, value, unit, delta_text, delta_dir, color_class):
    """
    Builds the HTML string for a KPI card widget.

    Parameters:
      label       — small uppercase label (e.g. "Voltage")
      value       — the formatted numeric string (e.g. "229.4")
      unit        — the unit string shown after the number (e.g. "V")
      delta_text  — the absolute change since the last reading (e.g. "0.3 V")
      delta_dir   — "up", "down" or "neutral" — controls the delta colour
      color_class — "blue", "green", "amber", "red" or "purple"
                    — controls which CSS gradient is used for the top stripe

    The arrow (+/-) is computed here so the delta line reads e.g. "+ 0.3 V".
    """
    arrow = "+" if delta_dir == "up" else "-" if delta_dir == "down" else ""
    return f"""
    <div class="kpi-card {color_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
        <div class="kpi-delta {delta_dir}">{arrow} {delta_text}</div>
    </div>
    """


def status_badge(status, label):
    """
    Builds the HTML for a pill-shaped status badge with a coloured dot.

    Parameters:
      status — "Online", "Warning" or "Offline"
              selects the CSS class (badge-online, badge-warning, badge-offline)
              which sets the background and text colour.
      label  — the text shown inside the badge (e.g. "Grid Online")

    The dot is a tiny inline SVG circle whose fill colour matches the badge theme.
    """
    cls = {"Online": "badge-online", "Warning": "badge-warning",
           "Offline": "badge-offline"}.get(status, "badge-online")
    dot = {"Online": "#4ade80", "Warning": "#fbbf24",
           "Offline": "#f87171"}.get(status, "#4ade80")
    return (f'<span class="status-badge {cls}">'
            f'<svg width="7" height="7" viewBox="0 0 7 7">'
            f'<circle cx="3.5" cy="3.5" r="3.5" fill="{dot}"/></svg>'
            f'{label}</span>')


# =============================================================================
# PAGE FUNCTIONS
# Each page is implemented as a separate function that receives the full
# historical DataFrame `df` as its only argument.
# The live buffer (get_live_df) is called inside page_realtime only.
# =============================================================================

def page_realtime(df):
    """
    Live Monitor page — shows real-time KPIs, charts, and system alerts.

    AUTO-REFRESH:
    st_autorefresh(interval=3000) tells the browser to trigger a full page
    rerun every 3000 ms (3 seconds) when the checkbox is ticked.
    This is completely different from the old time.sleep(3) + st.rerun() approach:
      - time.sleep blocks Streamlit's server thread, freezing the browser UI
        for the full sleep duration before the rerun can happen.
      - st_autorefresh fires a lightweight JavaScript timer in the browser;
        the server is never blocked and the UI stays fully responsive.

    The autorefresh call is placed BEFORE any data is read from the live buffer
    so that the very first rerun after ticking the checkbox already sees fresh data.
    """
    # --- Header row with title, status badges, and auto-refresh toggle ---
    col_title, col_status, col_refresh = st.columns([3, 2, 1])
    with col_refresh:
        auto_refresh = st.checkbox("Auto-refresh", value=False)

    # Activate the browser-side refresh timer only when the checkbox is ticked
    if auto_refresh:
        st_autorefresh(interval=3000, limit=None, key="live_refresh")

    # Extend the rolling live buffer with one new data point and get the result
    live_df = get_live_df(df)
    reading = live_df.iloc[-1]                           # Most recent row
    prev    = live_df.iloc[-2] if len(live_df) > 1 else reading  # Row before that (for delta)

    with col_title:
        st.markdown("### Live Grid Monitor")
        # Show the last-updated timestamp and current buffer size
        st.markdown(
            f"<span style='font-size:0.75rem;color:#475569;font-family:JetBrains Mono,monospace;'>"
            f"Last updated: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;buffer: {len(live_df)} rows</span>",
            unsafe_allow_html=True)
    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        v = float(reading.get("Voltage", 230))
        # Grid status is determined by whether voltage is in the 210-250V nominal range
        s = "Online" if 210 <= v <= 250 else "Warning"
        st.markdown(status_badge(s, f"Grid {s}") + "&nbsp;&nbsp;" +
                    status_badge("Online", "Sensors OK"), unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)

    # Five equal-width columns, one per KPI card
    c1, c2, c3, c4, c5 = st.columns(5)

    def delta(col, fmt=".1f"):
        """
        Helper: computes the absolute change between the current and previous reading
        for column `col` and returns (formatted_string, direction).
        direction is 'up', 'down', or 'neutral' — used to colour the delta text.
        fmt controls how many decimal places are shown.
        """
        diff = float(reading.get(col, 0)) - float(prev.get(col, 0))
        d    = "up" if diff > 0 else ("down" if diff < 0 else "neutral")
        return f"{abs(diff):{fmt}}", d

    # Render one KPI card per column
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

    # --- Row 1: Voltage/Current line chart (left) + Power Factor gauge (right) ---
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        # Use the last 120 rows from the live buffer for the waveform chart
        window = live_df.tail(120)
        # make_subplots with secondary_y=True creates a chart with two Y-axes
        # so Voltage (left axis) and Current (right axis) can be shown together
        # despite having very different scales (230V vs 10A)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=window["Time"], y=window["Voltage"],
                                  name="Voltage (V)", mode="lines",
                                  line=dict(color="#3b82f6", width=2),
                                  fill="tozeroy",                        # Fill area under the line
                                  fillcolor="rgba(59,130,246,0.06)"),    # Very faint blue fill
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=window["Time"], y=window["Current"],
                                  name="Current (A)", mode="lines",
                                  line=dict(color="#10b981", width=2, dash="dot")),
                      secondary_y=True)   # Current goes on the right Y-axis
        fig.update_layout(title=dict(text="Voltage & Current — Last 120 Readings",
                                     font=dict(size=13, color="#94a3b8")),
                          height=300, **PLOTLY_LAYOUT)
        fig.update_yaxes(title_text="Voltage (V)", gridcolor="#1e2a3a", secondary_y=False)
        fig.update_yaxes(title_text="Current (A)", gridcolor="#1e2a3a", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        # Gauge chart showing Power Factor with colour-coded zones
        pf_val    = float(reading.get("Power_Factor", 0.95))
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",      # Show gauge + numeric value + change from previous
            value=pf_val,
            delta={"reference": float(prev.get("Power_Factor", 0.95)), "valueformat": ".4f"},
            number={"valueformat": ".4f", "font": {"color": "#f1f5f9", "family": "JetBrains Mono"}},
            gauge={
                "axis":    {"range": [0.7, 1.0], "tickcolor": "#475569", "tickwidth": 1},
                "bar":     {"color": "#3b82f6", "thickness": 0.25},   # Blue indicator bar
                "bgcolor": "rgba(0,0,0,0)", "bordercolor": "#1e2a3a",
                "steps":   [
                    # Colour-coded background zones: red (poor), amber (acceptable), green (good)
                    {"range": [0.7,  0.85], "color": "rgba(239,68,68,0.15)"},
                    {"range": [0.85, 0.93], "color": "rgba(245,158,11,0.15)"},
                    {"range": [0.93, 1.0],  "color": "rgba(16,185,129,0.15)"},
                ],
                # Yellow threshold line at 0.9 — industry standard minimum power factor
                "threshold": {"line": {"color": "#f59e0b", "width": 2}, "value": 0.9}
            },
            title={"text": "Power Factor", "font": {"color": "#94a3b8", "size": 13}}
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                                height=300, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # --- Row 2: Power components chart + Frequency stability + Live sensor values ---
    chart_col3, chart_col4, chart_col5 = st.columns([2, 2, 1])

    with chart_col3:
        # Active, Reactive and Apparent Power overlaid on the same chart
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
        # Frequency chart with a green band showing the nominal 49.8-50.2Hz acceptable range
        window = live_df.tail(100)
        fig3   = go.Figure()
        # add_hrect draws a horizontal band across the full X range — used for the green safe zone
        fig3.add_hrect(y0=49.8, y1=50.2, fillcolor="rgba(16,185,129,0.08)", line_width=0)
        # Dashed reference line at exactly 50.0Hz
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
        # Compact vertical list showing the latest raw value of each sensor
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Live Sensors</div>", unsafe_allow_html=True)
        for col in RELEVANT_COLUMNS[:7]:    # Show the first 7 to fit in the narrow column
            if col in reading.index:
                val = reading[col]
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"padding:5px 0;border-bottom:1px solid #1e2a3a;font-size:0.75rem;'>"
                    f"<span style='color:#64748b;'>{col.replace('_',' ')}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace;color:#e2e8f0;font-weight:600;'>"
                    f"{val:.3f}</span></div>",
                    unsafe_allow_html=True)

    # --- Alert strip ---
    st.markdown("<br><div class='section-header'>System Alerts</div>", unsafe_allow_html=True)
    alerts = []  # Will be filled with (severity, message) tuples

    v  = float(reading.get("Voltage", 230))
    f  = float(reading.get("Measured_Frequency", 50))
    pf = float(reading.get("Power_Factor", 0.95))

    # Rule-based threshold checks on the current reading
    if v > 250 or v < 210:
        alerts.append(("critical", f"Voltage out of nominal range: {v:.1f} V (expected 210-250 V)"))
    if abs(f - 50) > 0.2:
        alerts.append(("warning", f"Frequency deviation detected: {f:.3f} Hz (nominal 50 Hz)"))
    if pf < 0.9:
        alerts.append(("warning", f"Low power factor: {pf:.3f} - Consider reactive power compensation"))

    # Rolling-window anomaly check on the live buffer (classical method used in real-time)
    anomaly_mask  = DataPreprocessor.detect_anomalies(live_df, "Voltage", window=10, threshold=2.5)
    anomaly_count = int(anomaly_mask.sum()) if anomaly_mask is not None else 0
    if anomaly_count > 0:
        alerts.append(("warning", f"{anomaly_count} voltage anomalies in live buffer ({len(live_df)} rows)"))

    # If no alerts were triggered, show a green all-clear message
    if not alerts:
        alerts.append(("ok", "All parameters within nominal operating ranges."))

    # Map severity names to the CSS modifier classes defined in the stylesheet
    alert_colors = {"critical": "", "warning": "warning", "ok": "ok", "info": "info"}
    # Display up to 3 alerts side by side in equal columns
    cols_alert   = st.columns(min(len(alerts), 3))
    for i, (level, msg) in enumerate(alerts[:3]):
        with cols_alert[i % 3]:
            st.markdown(f"<div class='alert-item {alert_colors.get(level, '')}'>{msg}</div>",
                        unsafe_allow_html=True)
    # NOTE: time.sleep + st.rerun removed — st_autorefresh above handles refresh cleanly


def page_analysis(df):
    """
    Historical Analysis page — explores the full static CSV dataset.
    Three tabs: distribution plots, correlation matrix, statistical summary.
    """
    st.markdown("### Historical Analysis")
    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Correlation Matrix", "Statistical Summary"])

    with tab1:
        # Dropdown to choose which column to analyse
        col_sel = st.selectbox("Select column:", [c for c in RELEVANT_COLUMNS if c in df.columns])
        col_l, col_r = st.columns(2)

        with col_l:
            # Histogram: shows the distribution shape (is the data bell-curved, skewed, bimodal?)
            fig_hist = px.histogram(df, x=col_sel, nbins=40,
                                    color_discrete_sequence=["#3b82f6"],
                                    title=f"Distribution: {col_sel}")
            fig_hist.update_layout(height=320, **PLOTLY_LAYOUT)
            fig_hist.update_traces(marker_line_color="#1e2a3a", marker_line_width=0.5)
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

        with col_r:
            # Box plot: shows median, IQR, and outliers in a compact form
            fig_box = px.box(df, y=col_sel, color_discrete_sequence=["#8b5cf6"],
                             title=f"Box Plot: {col_sel}")
            fig_box.update_layout(height=320, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

        # Time series with a 20-point rolling mean overlay
        # The rolling mean smooths out noise to reveal longer-term trends
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
        # Pearson correlation heatmap: -1 = perfect negative correlation,
        # 0 = no correlation, +1 = perfect positive correlation.
        # Useful for understanding which sensors move together (e.g. Voltage and Power Factor).
        available = [c for c in RELEVANT_COLUMNS if c in df.columns]
        fig_corr  = px.imshow(df[available].corr(),
                              color_continuous_scale=[[0,"#ef4444"],[0.5,"#1e2a3a"],[1,"#3b82f6"]],
                              zmin=-1, zmax=1, title="Pearson Correlation Matrix", text_auto=".2f")
        fig_corr.update_layout(height=500, **PLOTLY_LAYOUT)
        fig_corr.update_coloraxes(colorbar_tickfont_color="#94a3b8")
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        # pandas .describe() produces standard statistics: count, mean, std,
        # min, 25%, 50%, 75%, max.  The .T transposes so columns become rows.
        # We add a custom cv% (Coefficient of Variation) column = std/mean × 100
        # which shows relative variability regardless of the column's scale.
        available = [c for c in RELEVANT_COLUMNS if c in df.columns]
        stats     = df[available].describe().T
        stats["cv%"] = (stats["std"] / stats["mean"] * 100).round(2)
        st.dataframe(stats, use_container_width=True)


def page_preprocessing(df):
    """
    Hypothesis H page — demonstrates AI integration in data preprocessing.
    Five tabs covering: classical outlier detection, missing value handling,
    normalisation, ML anomaly detection, and PCA reconstruction.
    """
    st.markdown("### AI Data Preprocessing  —  Hypothesis H")

    # Hypothesis description box at the top of the page
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

    # Single DataPreprocessor instance used across all five tabs
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
            # Run the selected method and get back outlier rows plus the computed bounds
            outliers, lower, upper = preprocessor.detect_and_handle_outliers(
                df, selected_col, 'iqr' if method == "IQR" else 'zscore', threshold)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Outliers Found", len(outliers))
            mc2.metric("Lower Bound",    f"{lower:.3f}")
            mc3.metric("Upper Bound",    f"{upper:.3f}")

            # Chart: normal data as a blue line, outliers as red X markers
            fig_out = go.Figure()
            fig_out.add_trace(go.Scatter(x=df.index, y=df[selected_col], mode="lines",
                                          name=selected_col, line=dict(color="#3b82f6", width=1.5)))
            if len(outliers) > 0:
                fig_out.add_trace(go.Scatter(x=outliers.index, y=outliers[selected_col],
                                              mode="markers", name="Outliers",
                                              marker=dict(color="#ef4444", size=8, symbol="x")))
            # Green shaded band showing the acceptable range
            fig_out.add_hrect(y0=lower, y1=upper, fillcolor="rgba(16,185,129,0.07)", line_width=0)
            fig_out.update_layout(title=f"Outlier Detection: {selected_col}", height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_out, use_container_width=True, config={"displayModeBar": False})

    # ---- Tab 2: Missing values ----
    with tab2:
        st.markdown("<div class='section-header'>Missing Value Interpolation</div>", unsafe_allow_html=True)
        interp_method = st.selectbox("Method:", ["linear", "polynomial", "spline"])
        order = 2
        # Only show the order slider for methods that need it
        if interp_method in ["polynomial", "spline"]:
            order = st.slider("Order:", 1, 5, 2)
        cols_to_interp = st.multiselect("Columns:", [c for c in RELEVANT_COLUMNS if c in df.columns])

        c1, c2 = st.columns(2)
        if c1.button("Check Missing Values"):
            if cols_to_interp:
                # Get a per-column missing value report and display it as a table
                report = preprocessor.check_missing_values(df, cols_to_interp)
                data = [{"Column": k, "Missing Count": v["missing_count"],
                         "Missing %": f"{v['missing_percentage']:.2f}%",
                         "Has Missing": "Yes" if v["has_missing"] else "No"}
                        for k, v in report.items()]
                st.dataframe(pd.DataFrame(data), use_container_width=True)

        if c2.button("Interpolate"):
            if cols_to_interp:
                # Run interpolation and compare missing counts before and after
                df_proc = preprocessor.interpolate_missing_values(df, cols_to_interp, interp_method, order)
                st.metric("Missing Before", int(df[cols_to_interp].isna().sum().sum()))
                st.metric("Missing After",  int(df_proc[cols_to_interp].isna().sum().sum()))
                st.success("Interpolation complete.")

    # ---- Tab 3: Normalisation ----
    with tab3:
        st.markdown("<div class='section-header'>Data Normalization</div>", unsafe_allow_html=True)
        norm_method  = st.selectbox("Method:", ["standard", "minmax", "robust", "log", "zscore"])
        cols_to_norm = st.multiselect("Columns:", [c for c in RELEVANT_COLUMNS if c in df.columns], key="norm_cols")

        if st.button("Normalize", type="primary"):
            if cols_to_norm:
                df_norm = preprocessor.normalize_data(df, cols_to_norm, norm_method)

                # Show statistics before and after normalisation side by side
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original Statistics")
                    st.dataframe(df[cols_to_norm].describe(), use_container_width=True)
                with c2:
                    st.caption("Normalized Statistics")
                    st.dataframe(df_norm[cols_to_norm].describe(), use_container_width=True)

                # Visual comparison: original vs normalised values for the first selected column
                col_show = cols_to_norm[0]
                fig_n = go.Figure()
                fig_n.add_trace(go.Scatter(y=df[col_show].values, mode="lines",
                                           name="Original",   line=dict(color="#3b82f6")))
                fig_n.add_trace(go.Scatter(y=df_norm[col_show].values, mode="lines",
                                           name="Normalized", line=dict(color="#10b981")))
                fig_n.update_layout(title=f"Normalization Effect: {col_show}", height=300, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_n, use_container_width=True, config={"displayModeBar": False})

    # ---- Tab 4: ML Anomaly Detection ----
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
            # Multi-select: allows testing on any combination of sensor columns
            ml_cols = st.multiselect(
                "Features (multivariate):",
                [c for c in RELEVANT_COLUMNS if c in df.columns],
                default=[c for c in ["Voltage", "Current", "Active_Power"] if c in df.columns])
        with c2:
            # Contamination controls what fraction of data the model assumes are anomalies
            contamination = st.slider("Expected anomaly rate:", 0.01, 0.20, 0.05, 0.01,
                                       help="Fraction of data expected to be anomalies")
        with c3:
            ml_method = st.selectbox("Algorithm:",
                                      ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"])

        if st.button("Run ML Detection", type="primary") and ml_cols:
            with st.spinner(f"Training {ml_method}..."):
                # Route to the appropriate ML method based on the dropdown selection
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

            count = mask.sum()   # Total number of rows flagged as anomalies
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Anomalies Found",  int(count))
            mc2.metric("Anomaly Rate",     f"{count/len(df)*100:.2f}%")
            mc3.metric("Features Used",    len(ml_cols))

            # Time series chart: normal data in purple, anomalies as red X markers
            plot_col = ml_cols[0]   # Use the first selected feature for the time series
            fig_ml   = go.Figure()
            fig_ml.add_trace(go.Scatter(x=df.index, y=df[plot_col], mode="lines",
                                         name=plot_col, line=dict(color="#8b5cf6", width=1.5)))
            if count:
                fig_ml.add_trace(go.Scatter(x=df[mask].index, y=df[mask][plot_col],
                                             mode="markers", name=f"{ml_method} Anomalies",
                                             marker=dict(color="#ef4444", size=9, symbol="x-open-dot")))
            fig_ml.update_layout(title=f"{ml_method} — {plot_col}", height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_ml, use_container_width=True, config={"displayModeBar": False})

            # Score distribution histogram: overlaid normal vs anomaly distributions
            # Shows whether the model has cleanly separated the two groups
            valid_scores = scores.dropna()
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=valid_scores[~mask[valid_scores.index]],
                                         name="Normal",  marker_color="#3b82f6", opacity=0.7, nbinsx=40))
            fig2.add_trace(go.Histogram(x=valid_scores[mask[valid_scores.index]],
                                         name="Anomaly", marker_color="#ef4444", opacity=0.7, nbinsx=20))
            fig2.update_layout(title=score_label, barmode="overlay", height=280, **PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ---- Tab 5: PCA Reconstruction ----
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
            # Number of components to keep: more = less compression = lower reconstruction error
            n_comp = st.slider("PCA components:", 1,
                               max(1, len([c for c in RELEVANT_COLUMNS if c in df.columns]) - 1), 4)
        with c3:
            # Sigma threshold: higher = fewer anomalies flagged
            sigma  = st.slider("Anomaly threshold (σ):", 1.5, 4.0, 2.5, 0.1)

        if st.button("Run PCA Analysis", type="primary") and pca_cols:
            with st.spinner("Running PCA reconstruction..."):
                mask, errors, threshold, explained, _ = preprocessor.pca_reconstruction_error(
                    df, pca_cols, n_components=n_comp, threshold_sigma=sigma)

            count = mask.sum()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Anomalies Found",    int(count))
            mc2.metric("Variance Explained", f"{explained.sum()*100:.1f}%")
            mc3.metric("Error Threshold",    f"{threshold:.4f}")

            # Reconstruction error over time with the threshold shown as a dashed red line
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

            # Bar chart showing how much variance each principal component explains.
            # e.g. PC1 might explain 60%, PC2 20%, PC3 10% — total captured: 90%.
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
    """
    Hypothesis H1 page — verifies control panel element correctness.
    Three tabs: range validation, logical consistency checks, sensor health.
    """
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
            # Default min = mean - 3σ (standard statistical lower fence)
            min_v = st.number_input("Min acceptable:",
                                     value=float(df[val_col].mean() - 3 * df[val_col].std()))
        with c3:
            # Default max = mean + 3σ
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
            # Green band showing the user-defined acceptable range
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
            # Each rule is a tuple: (col1, col2, lambda that returns True where rule holds)
            checks = [
                ("Voltage",      "Current",       lambda v, c: (v > 0) & (c >= 0)),
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

            # Bar chart with colour-coded bars: green (≥95%), amber (≥80%), red (<80%)
            fig_health = go.Figure(go.Bar(
                x=sensors, y=avail,
                marker_color=["#10b981" if a>=95 else "#f59e0b" if a>=80 else "#ef4444" for a in avail],
                text=[f"{a:.1f}%" for a in avail], textposition="outside"))
            fig_health.update_layout(title="Sensor Data Availability (%)",
                                     yaxis=dict(range=[0, 110]), height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_health, use_container_width=True, config={"displayModeBar": False})
            st.metric("Average Availability", f"{np.mean(avail):.2f}%")


def page_quality(df):
    """
    Hypothesis H2 page — assesses overall data quality with a weighted score.
    Two tabs: overall quality gauge + component scores, and correction impact.
    """
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
                # Gauge chart: needle shows the overall score, zones colour-code the quality bands
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"suffix": "/100", "font": {"color": "#f1f5f9", "family": "JetBrains Mono", "size": 36}},
                    gauge={
                        "axis":    {"range": [0, 100], "tickcolor": "#475569"},
                        "bar":     {"color": report["color"], "thickness": 0.3},
                        "bgcolor": "rgba(0,0,0,0)", "bordercolor": "#1e2a3a",
                        "steps":   [
                            {"range": [0,  60],  "color": "rgba(239,68,68,0.1)"},   # Red zone: poor
                            {"range": [60, 75],  "color": "rgba(245,158,11,0.1)"},  # Amber: fair
                            {"range": [75, 90],  "color": "rgba(59,130,246,0.1)"},  # Blue: good
                            {"range": [90, 100], "color": "rgba(16,185,129,0.1)"},  # Green: excellent
                        ],
                    },
                    title={"text": f"Overall Quality: {report['status']}",
                           "font": {"color": "#94a3b8", "size": 13}}
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280,
                                        margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            with c2:
                # Recommendation message
                st.markdown(f"<div class='alert-item info'>{report['recommendation']}</div>",
                            unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # Progress bars for each of the three quality dimensions
                for name, val, weight in components:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;"
                        f"font-size:0.8rem;'>"
                        f"<span style='color:#94a3b8;'>{name}</span>"
                        f"<span style='color:#f1f5f9;font-family:JetBrains Mono,monospace;'>"
                        f"{val:.1f} / 100</span></div>",
                        unsafe_allow_html=True)
                    st.progress(min(val / 100, 1.0))  # Clamp to 1.0 so progress bar never overflows
                    st.markdown(
                        f"<div style='font-size:0.7rem;color:#475569;margin-bottom:8px;'>"
                        f"Weight: {weight*100:.0f}%</div>",
                        unsafe_allow_html=True)

    with tab2:
        if st.button("Analyze Correction Impact", type="primary"):
            available    = [c for c in RELEVANT_COLUMNS if c in df.columns]
            preprocessor = DataPreprocessor()
            # Run linear interpolation on a copy of the dataset to simulate a correction step
            df_corrected = preprocessor.interpolate_missing_values(df.copy(), available)
            # Compare original vs corrected to see what changed and by how much
            impact       = quality.calculate_correction_impact(df, df_corrected, available)
            impact_data  = pd.DataFrame([{
                "Column":       col,
                "Rows Changed": m["rows_changed"],
                "% Changed":    round(m["percent_changed"], 3),
                "MAE":          round(m["mean_absolute_error"], 6)
            } for col, m in impact.items()])

            # Bar chart: height = % of rows changed, colour = magnitude of changes (MAE)
            fig_impact = px.bar(impact_data, x="Column", y="% Changed", color="MAE",
                                color_continuous_scale=[[0,"#10b981"],[0.5,"#3b82f6"],[1,"#ef4444"]],
                                title="Correction Impact by Column")
            fig_impact.update_layout(height=350, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_impact, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(impact_data, use_container_width=True)


# =============================================================================
# MAIN FUNCTION
# Entry point — builds the sidebar and routes to the correct page function.
# Streamlit calls main() on every page rerun (every user interaction,
# every auto-refresh tick, every page load).
# =============================================================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        # App title and subtitle
        st.markdown(
            "<div style='padding:1rem 0 1.5rem;'>"
            "<div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.02em;'>SmartGrid</div>"
            "<div style='font-size:0.7rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;"
            "margin-top:2px;'>AI Dashboard</div>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Navigation</div>", unsafe_allow_html=True)

        # Radio button navigation — label is provided (not empty) but hidden
        # with label_visibility="collapsed" to avoid the accessibility warning
        page = st.radio(
            label="Navigation",
            options=["Live Monitor", "Historical Analysis", "H: Preprocessing",
                     "H1: Validation", "H2: Quality"],
            label_visibility="collapsed")

        # --- System resource meters ---
        st.markdown("<br><div class='section-header'>System Info</div>", unsafe_allow_html=True)

        # psutil.cpu_percent reads actual CPU usage at this moment
        cpu = psutil.cpu_percent(interval=0.1)   # interval=0.1 gives a 100ms measurement window
        mem = psutil.virtual_memory().percent     # RAM usage as a percentage

        # Progress bars show a visual 0-100% fill
        st.progress(cpu / 100)
        st.markdown(f"<div style='font-size:0.72rem;color:#475569;margin-top:-8px;'>CPU: {cpu:.1f}%</div>",
                    unsafe_allow_html=True)
        st.progress(mem / 100)
        st.markdown(f"<div style='font-size:0.72rem;color:#475569;margin-top:-8px;'>Memory: {mem:.1f}%</div>",
                    unsafe_allow_html=True)

        # Footer: dataset name and last-refreshed timestamp
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:0.65rem;color:#334155;'>"
            f"Dataset: demo_data.csv<br>"
            f"Refreshed: {datetime.now().strftime('%H:%M:%S')}</div>",
            unsafe_allow_html=True)

    # --- Load data ---
    df = load_data(DATA_FILE)
    if df is None:
        # load_data already showed an error message; stop rendering the rest of the page
        st.stop()

    # --- Route to the selected page ---
    # Each branch calls one page function and passes the loaded DataFrame
    if   page == "Live Monitor":          page_realtime(df)
    elif page == "Historical Analysis":   page_analysis(df)
    elif page == "H: Preprocessing":      page_preprocessing(df)
    elif page == "H1: Validation":        page_validation(df)
    elif page == "H2: Quality":           page_quality(df)


# Standard Python entry-point guard.
# When Streamlit runs this file it executes it as a module (not __main__),
# but when you run it directly with `python webMonitor.py` this block
# calls main() explicitly.  In normal Streamlit usage, Streamlit itself
# calls main() on every rerun — this guard just ensures it also works
# if someone runs the file directly.
if __name__ == "__main__":
    main()