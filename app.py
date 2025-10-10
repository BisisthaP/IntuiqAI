import streamlit as st
import pandas as pd
import numpy as np
from processing import analyze_dataset, DataProcessor
import json
import io

def init_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'processor' not in st.session_state:
        st.session_state.processor = DataProcessor()
    if 'cleaning_recommendations' not in st.session_state:
        st.session_state.cleaning_recommendations = []
    if 'cleaning_applied' not in st.session_state:
        st.session_state.cleaning_applied = False
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'changes_made' not in st.session_state:
        st.session_state.changes_made = []
    if 'dataset_id' not in st.session_state:
        st.session_state.dataset_id = None

def main():
    st.set_page_config(page_title="Data Dashboard Prep", page_icon="📊", layout="wide")
    st.title("📊 Data Dashboard Preparation Tool")
    
    init_session_state()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a step",
        ["Upload Data", "Data Analysis", "Data Cleaning", "Preview Results"]
    )
    
    if app_mode == "Upload Data":
        show_upload_section()
    elif app_mode == "Data Analysis":
        show_analysis_section()
    elif app_mode == "Data Cleaning":
        show_cleaning_section()
    elif app_mode == "Preview Results":
        show_results_section()

def show_upload_section():
    st.header("📁 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset for analysis and cleaning"
    )
    
    if uploaded_file is not None:
        try:
            # Read file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            if df.empty:
                st.error("The uploaded file is empty.")
                return
            
            st.session_state.df = df
            st.session_state.analysis = None
            st.session_state.cleaning_applied = False
            st.session_state.processed_df = None
            st.session_state.changes_made = []
            
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Show data preview ONLY - removed column information table
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def show_analysis_section():
    st.header("🔍 Data Analysis")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    if st.button("Run Data Analysis") or st.session_state.analysis is not None:
        with st.spinner("Analyzing dataset..."):
            if st.session_state.analysis is None:
                st.session_state.analysis = analyze_dataset(st.session_state.df)
            
            # Display analysis results
            st.subheader("Dataset Analysis Summary")
            
            # Overall metrics
            total_rows = len(st.session_state.df)
            total_missing = sum(info['missing'] for info in st.session_state.analysis.values())
            total_custom_missing = sum(info['custom_missing'] for info in st.session_state.analysis.values())
            total_outliers = sum(info['outliers'] for info in st.session_state.analysis.values() if info['outliers'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", total_rows)
            with col2:
                st.metric("Missing Values", total_missing)
            with col3:
                st.metric("Custom Missing", total_custom_missing)
            with col4:
                st.metric("Outliers Found", total_outliers if total_outliers else 0)
            
            # Detailed column analysis - TABLE RESTORED
            st.subheader("Column-by-Column Analysis")
            
            analysis_data = []
            for col, info in st.session_state.analysis.items():
                total_missing_col = info['missing'] + info['custom_missing']
                missing_pct = (total_missing_col / info['total_count']) * 100
                
                analysis_data.append({
                    'Column': col,
                    'Type': info['type'],
                    'Unique Values': info['nunique'],
                    'Missing Values': info['missing'],
                    'Custom Missing': info['custom_missing'],
                    'Total Missing %': f"{missing_pct:.1f}%",
                    'Outliers': info['outliers'] if info['outliers'] is not None else 'N/A'
                })
            
            analysis_df = pd.DataFrame(analysis_data)
            st.dataframe(analysis_df, use_container_width=True)
            
            # Data quality issues summary
            st.subheader("Data Quality Issues Found")
            
            issues = []
            if total_missing > 0:
                issues.append(f"❌ {total_missing} standard missing values")
            if total_custom_missing > 0:
                issues.append(f"❌ {total_custom_missing} custom missing values (Not Given, Unknown, etc.)")
            if total_outliers:
                issues.append(f"⚠️ {total_outliers} outliers in numerical columns")
            if len(st.session_state.df) != len(st.session_state.df.drop_duplicates()):
                duplicates = len(st.session_state.df) - len(st.session_state.df.drop_duplicates())
                issues.append(f"❌ {duplicates} duplicate rows")
            
            if issues:
                for issue in issues:
                    st.write(issue)
            else:
                st.success("✅ No major data quality issues found!")
            
            # Generate cleaning recommendations
            st.session_state.cleaning_recommendations = st.session_state.processor.get_cleaning_recommendations(
                st.session_state.df, st.session_state.analysis
            )

def show_cleaning_section():
    st.header("🧹 Data Cleaning")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    if st.session_state.analysis is None:
        st.warning("Please run data analysis first.")
        return
    
    st.subheader("Recommended Cleaning Operations")
    
    if not st.session_state.cleaning_recommendations:
        st.info("No cleaning recommendations available. Your data appears to be clean!")
        return
    
    # Display recommendations with checkboxes
    selected_functions = []
    
    for rec in st.session_state.cleaning_recommendations:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            st.write(priority_color[rec['priority']])
        with col2:
            if st.checkbox(rec['reason'], key=rec['function']):
                selected_functions.append(rec['function'])
        with col3:
            st.write(rec['priority'].title())

def show_results_section():
    st.header("📈 Preview Results")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    st.info("Data cleaning functionality is available. Use the Data Cleaning section to apply cleaning operations.")

if __name__ == "__main__":
    main()