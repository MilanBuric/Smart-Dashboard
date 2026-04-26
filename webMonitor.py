import streamlit as st
import time
import pandas as pd
import requests
import psutil
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import numpy as np

# ==================== HYPOTHESIS IMPLEMENTATIONS ====================

class DataPreprocessor:
    """H: AI Integration in Data Preprocessing"""
    
    @staticmethod
    def detect_and_handle_outliers(df, column, method='iqr', threshold=1.5):
        """Detect outliers using IQR or Z-score method"""
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        else:  # z-score
            mean = df[column].mean()
            std = df[column].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    @staticmethod
    def check_missing_values(df, columns):
        """Check which columns actually have missing values"""
        missing_report = {}
        for col in columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                missing_report[col] = {
                    "missing_count": missing_count,
                    "missing_percentage": missing_pct,
                    "has_missing": missing_count > 0
                }
        return missing_report
    
    @staticmethod
    def interpolate_missing_values(df, columns, method='linear', order=2):
        """Fill missing values using interpolation"""
        df_processed = df.copy()
        for col in columns:
            if col in df_processed.columns:
                try:
                    if method in ['polynomial', 'spline']:
                        df_processed[col] = df_processed[col].interpolate(method=method, order=order, limit_direction='both')
                    else:
                        df_processed[col] = df_processed[col].interpolate(method=method, limit_direction='both')
                except Exception as e:
                    print(f"Warning: Could not interpolate {col}: {e}")
        return df_processed
    
    @staticmethod
    def normalize_data(df, columns, method='standard'):
        """Normalize data using different methods"""
        df_normalized = df.copy()
        
        if method == 'standard':  # (x - mean) / std
            scaler = StandardScaler()
            df_normalized[columns] = scaler.fit_transform(df[columns])
        
        elif method == 'minmax':  # (x - min) / (max - min) -> [0, 1]
            for col in columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val - min_val != 0:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df_normalized[col] = 0  # Ako nema varijacije, ostavi 0
        
        elif method == 'robust':  # (x - median) / IQR -> otporno na outliere
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                median = df[col].median()
                if IQR != 0:
                    df_normalized[col] = (df[col] - median) / IQR
                else:
                    df_normalized[col] = 0
        
        elif method == 'log':  # Log transformacija (za pozitivne vrednosti)
            for col in columns:
                if (df[col] > 0).all():
                    df_normalized[col] = np.log(df[col])
                else:
                    st.warning(f"Column {col} has non-positive values. Cannot apply log transformation.")
        
        elif method == 'zscore':  # Z-score normalizacija
            for col in columns:
                mean = df[col].mean()
                std = df[col].std()
                if std != 0:
                    df_normalized[col] = (df[col] - mean) / std
                else:
                    df_normalized[col] = 0
        
        return df_normalized
    
    @staticmethod
    def detect_anomalies(df, column, window=5, threshold=2):
        """Detect anomalies using rolling statistics"""
        rolling_mean = df[column].rolling(window=window).mean()
        rolling_std = df[column].rolling(window=window).std()
        
        anomalies = (df[column] - rolling_mean).abs() > (threshold * rolling_std)
        return anomalies


class ControlPanelValidator:
    """H1: Verification of Control Panel Elements"""
    
    @staticmethod
    def validate_data_range(df, column, min_val, max_val):
        """Check if column values are within acceptable range"""
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
        """Check logical consistency between related columns"""
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
        """Check if sensors are providing data"""
        health_report = {}
        for col in sensor_columns:
            if col in df.columns:
                missing = df[col].isna().sum()
                health_report[col] = {
                    "total_readings": len(df),
                    "missing_readings": missing,
                    "data_availability": (1 - missing / len(df)) * 100
                }
        return health_report


class QualityAssessment:
    """H2: Quality Assessment of Control Panel Corrections"""
    
    @staticmethod
    def calculate_correction_impact(df_original, df_corrected, columns):
        """Measure the impact of corrections"""
        impact_report = {}
        for col in columns:
            if col in df_original.columns and col in df_corrected.columns:
                changes = (df_corrected[col] != df_original[col]).sum()
                mae = np.mean(np.abs(df_original[col] - df_corrected[col]))
                impact_report[col] = {
                    "rows_changed": changes,
                    "percent_changed": (changes / len(df_original)) * 100,
                    "mean_absolute_error": mae
                }
        return impact_report
    
    @staticmethod
    def assess_data_quality_score(df, sensor_columns):
        """Calculate overall data quality score (0-100)"""
        scores = []
        
        # Check for completeness
        completeness = (1 - df[sensor_columns].isna().sum().sum() / (len(df) * len(sensor_columns))) * 100
        scores.append(("Completeness", completeness, 0.3))
        
        # Check for consistency (no extreme outliers)
        outlier_score = 100
        for col in sensor_columns:
            if df[col].std() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_pct = (z_scores > 3).sum() / len(df) * 100
                outlier_score -= outliers_pct
        scores.append(("Outlier Detection", max(outlier_score, 0), 0.3))
        
        # Check for stability (low variance in readings)
        stability = 100
        for col in sensor_columns:
            cv = (df[col].std() / df[col].mean()) if df[col].mean() != 0 else 0
            if cv > 0.5:
                stability -= cv * 10
        scores.append(("Stability", max(stability, 0), 0.4))
        
        overall_score = sum(score * weight for _, score, weight in scores)
        return overall_score, scores
    
    @staticmethod
    def generate_quality_report(overall_score):
        """Generate quality assessment report"""
        if overall_score >= 90:
            status = "🟢 Excellent"
            recommendation = "Data quality is excellent. Continue monitoring."
        elif overall_score >= 75:
            status = "🟡 Good"
            recommendation = "Data quality is good. Minor improvements recommended."
        elif overall_score >= 60:
            status = "🟠 Fair"
            recommendation = "Data quality needs attention. Review outliers and missing values."
        else:
            status = "🔴 Poor"
            recommendation = "Data quality is poor. Immediate action required."
        
        return {"status": status, "score": overall_score, "recommendation": recommendation}


# ==================== CONFIGURATION ====================

DATA_FILE = "demo_data.csv"
CITIES = {
    "Zrenjanin, Serbia": {"latitude": 45.3755, "longitude": 20.4020},
    "Belgrade, Serbia": {"latitude": 44.8176, "longitude": 20.4633},
    "Novi Sad, Serbia": {"latitude": 45.2671, "longitude": 19.8335},
    "Banja Luka, Bosnia and Herzegovina": {"latitude": 44.7722, "longitude": 17.1910},
    "Sarajevo, Bosnia and Herzegovina": {"latitude": 43.8486, "longitude": 18.3564},
    "Zagreb, Croatia": {"latitude": 45.8125, "longitude": 15.978}
}

RELEVANT_COLUMNS = [
    "Voltage", "Current", "Measured_Frequency", "Active_Power",
    "Reactive_Power", "Apperent_Power", "Phase_Voltage_Angle",
    "Cos_Phi", "Power_Factor"
]

# ==================== HELPER FUNCTIONS ====================

def calculate_dynamic_threshold(series, factor=2.0):
    return series.mean() + factor * series.std()

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce').dt.time
        return df
    except FileNotFoundError:
        st.error(f"Data file '{file_path}' not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

def display_hypothesis_h(df):
    """H: AI Integration in Data Preprocessing"""
    st.markdown("## 🤖 H: AI Integration in Data Preprocessing")
    
    with st.expander("Data Preprocessing Options", expanded=True):
        preprocessor = DataPreprocessor()
        
        # Tab-based UI
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Outlier Detection", "Missing Values", "Normalization", "Anomaly Detection"]
        )
        
        with tab1:
            st.subheader("Outlier Detection & Handling")
            selected_col = st.selectbox("Select column for outlier detection:", 
                                       [col for col in RELEVANT_COLUMNS if col in df.columns])
            method = st.radio("Select method:", ["IQR", "Z-Score"])
            threshold = st.slider("Threshold:", 0.5, 5.0, 1.5)
            
            if st.button("Detect Outliers"):
                try:
                    outliers, lower, upper = preprocessor.detect_and_handle_outliers(
                        df, selected_col, 
                        method='iqr' if method == "IQR" else 'zscore', 
                        threshold=threshold
                    )
                    st.write(f"**Found {len(outliers)} outliers**")
                    st.write(f"Expected range: [{lower:.2f}, {upper:.2f}]")
                    if len(outliers) > 0:
                        st.dataframe(outliers[[selected_col, "Time"]].head(10))
                    else:
                        st.success("✅ No outliers detected!")
                except Exception as e:
                    st.error(f"Error during outlier detection: {e}")
        
        with tab2:
            st.subheader("Missing Values Interpolation")
            method = st.selectbox("Interpolation method:", ["linear", "polynomial", "spline"])
            
            # Slider za order SAMO za polynomial/spline
            order = 2
            if method in ["polynomial", "spline"]:
                order = st.slider("Interpolation order:", 1, 5, 2)
            
            columns_to_interpolate = st.multiselect(
                "Select columns:", 
                [col for col in RELEVANT_COLUMNS if col in df.columns]
            )
            
            # Prvo prikaži koliko nedostajecih vrednosti ima
            if st.button("Check Missing Values First"):
                if columns_to_interpolate:
                    missing_report = preprocessor.check_missing_values(df, columns_to_interpolate)
                    
                    st.subheader("Missing Values Report:")
                    report_data = []
                    for col, metrics in missing_report.items():
                        report_data.append({
                            "Column": col,
                            "Missing Count": metrics['missing_count'],
                            "Missing %": f"{metrics['missing_percentage']:.2f}%",
                            "Has Missing": "✅ Yes" if metrics['has_missing'] else "❌ No"
                        })
                    
                    st.dataframe(pd.DataFrame(report_data))
                else:
                    st.warning("Please select at least one column")
            
            if st.button("Interpolate Missing Values"):
                if columns_to_interpolate:
                    try:
                        df_processed = preprocessor.interpolate_missing_values(
                            df, columns_to_interpolate, method=method, order=order
                        )
                        st.success("✅ Interpolation complete!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original missing values", df[columns_to_interpolate].isna().sum().sum())
                        with col2:
                            st.metric("After interpolation", df_processed[columns_to_interpolate].isna().sum().sum())
                    except Exception as e:
                        st.error(f"Error during interpolation: {e}")
                else:
                    st.warning("Please select at least one column")
        
        with tab3:
            st.subheader("Data Normalization")
            
            # Odabir metode normalizacije
            norm_method = st.selectbox(
                "Select normalization method:",
                ["standard", "minmax", "robust", "log", "zscore"],
                help="Standard: StandardScaler | MinMax: [0,1] range | Robust: otporno na outliere | Log: log(x) | ZScore: Z-score"
            )
            
            columns_to_normalize = st.multiselect(
                "Select columns to normalize:", 
                [col for col in RELEVANT_COLUMNS if col in df.columns],
                key="normalize_cols"
            )
            
            if st.button("Normalize Data"):
                if columns_to_normalize:
                    try:
                        df_normalized = preprocessor.normalize_data(
                            df, columns_to_normalize, method=norm_method
                        )
                        st.success(f"✅ {norm_method.upper()} Normalization complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Original Data (sample)**")
                            st.dataframe(df[columns_to_normalize].head())
                            st.write("**Original Statistics:**")
                            st.dataframe(df[columns_to_normalize].describe())
                        
                        with col2:
                            st.write("**Normalized Data (sample)**")
                            st.dataframe(df_normalized[columns_to_normalize].head())
                            st.write("**Normalized Statistics:**")
                            st.dataframe(df_normalized[columns_to_normalize].describe())
                            
                    except Exception as e:
                        st.error(f"Error during normalization: {e}")
                else:
                    st.warning("Please select at least one column")
        
        with tab4:
            st.subheader("Anomaly Detection")
            selected_col = st.selectbox("Select column for anomaly detection:", 
                                       [col for col in RELEVANT_COLUMNS if col in df.columns],
                                       key="anomaly_col")
            window = st.slider("Rolling window size:", 3, 20, 5)
            threshold = st.slider("Anomaly threshold (std dev):", 1.0, 5.0, 2.0)
            
            if st.button("Detect Anomalies"):
                try:
                    anomalies = preprocessor.detect_anomalies(
                        df, selected_col, window=window, threshold=threshold
                    )
                    anomaly_count = anomalies.sum()
                    st.metric("Anomalies detected", anomaly_count)
                    st.metric("Anomaly percentage", f"{(anomaly_count/len(df)*100):.2f}%")
                    
                    if anomaly_count > 0:
                        anomaly_indices = df[anomalies].index
                        st.dataframe(df.loc[anomaly_indices, [selected_col, "Time"]].head(10))
                    else:
                        st.success("✅ No anomalies detected!")
                except Exception as e:
                    st.error(f"Error during anomaly detection: {e}")

def display_hypothesis_h1(df):
    """H1: Verification of Control Panel Elements"""
    st.markdown("## ✅ H1: Verification of Control Panel Elements")
    
    with st.expander("Control Panel Validation", expanded=True):
        validator = ControlPanelValidator()
        
        tab1, tab2, tab3 = st.tabs(["Data Range", "Consistency", "Sensor Health"])
        
        with tab1:
            st.subheader("Data Range Validation")
            selected_col = st.selectbox("Select column to validate:", 
                                       [col for col in RELEVANT_COLUMNS if col in df.columns],
                                       key="range_col")
            
            col1, col2 = st.columns(2)
            with col1:
                min_val = st.number_input("Minimum acceptable value:", value=0.0)
            with col2:
                max_val = st.number_input("Maximum acceptable value:", value=100.0)
            
            if st.button("Validate Range"):
                try:
                    result = validator.validate_data_range(df, selected_col, min_val, max_val)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Validity Score", f"{result['validity_score']:.2f}%")
                    with col2:
                        st.metric("Invalid Rows", result['invalid_count'])
                    with col3:
                        st.metric("Status", "✅ Valid" if result['valid'] else "❌ Invalid")
                    
                    if result['invalid_count'] > 0:
                        st.warning(f"Found {result['invalid_count']} out-of-range values")
                        st.dataframe(result['out_of_range_rows'].head())
                except Exception as e:
                    st.error(f"Error during validation: {e}")
        
        with tab2:
            st.subheader("Data Consistency Check")
            st.info("Validates logical relationships between columns")
            
            if st.button("Check Consistency"):
                try:
                    # Example consistency checks
                    consistency_checks = [
                        ("Voltage", "Current", lambda v, c: (v > 0) & (c >= 0)),
                        ("Active_Power", "Apperent_Power", lambda ap, app: ap <= app)
                    ]
                    
                    inconsistencies = validator.validate_data_consistency(
                        df, consistency_checks
                    )
                    
                    if inconsistencies:
                        st.warning("Found inconsistencies:")
                        for inc in inconsistencies:
                            st.write(f"- {inc['columns']}: {inc['inconsistent_count']} rows")
                    else:
                        st.success("✅ All consistency checks passed!")
                except Exception as e:
                    st.error(f"Error during consistency check: {e}")
        
        with tab3:
            st.subheader("Sensor Health Status")
            if st.button("Check Sensor Health"):
                try:
                    health = validator.validate_sensor_health(
                        df, RELEVANT_COLUMNS
                    )
                    
                    health_data = []
                    for sensor, metrics in health.items():
                        health_data.append({
                            "Sensor": sensor,
                            "Total Readings": metrics['total_readings'],
                            "Missing": metrics['missing_readings'],
                            "Availability %": f"{metrics['data_availability']:.2f}%"
                        })
                    
                    st.dataframe(pd.DataFrame(health_data))
                    
                    # Overall sensor status
                    avg_availability = np.mean([h['data_availability'] for h in health.values()])
                    st.metric("Average Sensor Availability", f"{avg_availability:.2f}%")
                except Exception as e:
                    st.error(f"Error during sensor health check: {e}")

def display_hypothesis_h2(df):
    """H2: Quality Assessment of Control Panel Corrections"""
    st.markdown("## 📊 H2: Quality Assessment of Control Panel Corrections")
    
    with st.expander("Quality Assessment", expanded=True):
        quality = QualityAssessment()
        
        tab1, tab2 = st.tabs(["Quality Score", "Correction Impact"])
        
        with tab1:
            st.subheader("Data Quality Assessment")
            if st.button("Calculate Quality Score"):
                try:
                    overall_score, scores = quality.assess_data_quality_score(
                        df, RELEVANT_COLUMNS
                    )
                    
                    report = quality.generate_quality_report(overall_score)
                    
                    # Display main score
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.metric("Overall Quality Score", f"{overall_score:.2f}/100")
                    with col2:
                        st.write(report['status'])
                    
                    st.info(report['recommendation'])
                    
                    # Breakdown of scores
                    st.subheader("Quality Components:")
                    for component, score, weight in scores:
                        st.write(f"**{component}** (Weight: {weight*100:.0f}%): {score:.2f}/100")
                        st.progress(min(score/100, 1.0))
                except Exception as e:
                    st.error(f"Error calculating quality score: {e}")
        
        with tab2:
            st.subheader("Correction Impact Analysis")
            st.info("Compare original vs corrected data to measure improvement")
            
            # Simulate correction
            if st.button("Analyze Correction Impact"):
                try:
                    # For demo, we create a slightly modified version
                    df_corrected = df.copy()
                    
                    # Apply some corrections (remove outliers, interpolate)
                    preprocessor = DataPreprocessor()
                    df_corrected = preprocessor.interpolate_missing_values(
                        df_corrected, RELEVANT_COLUMNS
                    )
                    
                    impact = quality.calculate_correction_impact(
                        df, df_corrected, RELEVANT_COLUMNS
                    )
                    
                    impact_data = []
                    for col, metrics in impact.items():
                        impact_data.append({
                            "Column": col,
                            "Rows Changed": metrics['rows_changed'],
                            "% Changed": f"{metrics['percent_changed']:.2f}%",
                            "MAE": f"{metrics['mean_absolute_error']:.4f}"
                        })
                    
                    st.dataframe(pd.DataFrame(impact_data))
                except Exception as e:
                    st.error(f"Error analyzing correction impact: {e}")

# ==================== MAIN APP ====================

def main():
    st.set_page_config(page_title='Smart Dashboard with AI', layout="wide")
    st.title("🎯 Smart Dashboard with AI Integration")
    
    # Load data
    df = load_data(DATA_FILE)
    if df is None:
        st.stop()
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select section:", 
        ["Dashboard Overview", "H: Data Preprocessing", "H1: Element Verification", "H2: Quality Assessment"]
    )
    
    if page == "Dashboard Overview":
        st.markdown("""
        ### Welcome to the Smart Dashboard with AI Integration
        
        This dashboard integrates AI capabilities based on three main hypotheses:
        
        **H - Data Preprocessing**: AI-powered data cleaning and preparation
        - Outlier detection (IQR & Z-score methods)
        - Missing value interpolation
        - Data normalization (5 different methods)
        - Anomaly detection
        
        **H1 - Element Verification**: Verification of control panel elements
        - Data range validation
        - Consistency checking
        - Sensor health monitoring
        
        **H2 - Quality Assessment**: Quality assessment of corrections
        - Data quality scoring
        - Correction impact analysis
        
        Select a section from the sidebar to get started!
        """)
    
    elif page == "H: Data Preprocessing":
        display_hypothesis_h(df)
    
    elif page == "H1: Element Verification":
        display_hypothesis_h1(df)
    
    elif page == "H2: Quality Assessment":
        display_hypothesis_h2(df)

if __name__ == "__main__":
    main()
