# app.py
import streamlit as st
import pandas as pd
import numpy as np
from processing import analyze_dataset, analysis_to_jsonl, DataProcessor
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
            
            # REMOVED: Column Information table and any other displays
            
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
            
            # Detailed column analysis
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
    
    # Additional options
    st.subheader("Additional Options")
    
    col1, col2 = st.columns(2)
    with col1:
        remove_columns_threshold = st.slider(
            "Remove columns with missing data > %",
            min_value=0, max_value=100, value=50,
            help="Columns with higher percentage of missing values will be removed"
        )
    
    with col2:
        outlier_handling = st.selectbox(
            "Outlier Handling",
            ["Cap outliers (recommended)", "Remove outliers", "Keep as is"],
            help="How to handle outliers in numerical data"
        )
    
    # Apply cleaning button
    if st.button("Apply Selected Cleaning", type="primary"):
        if not selected_functions:
            st.warning("Please select at least one cleaning operation.")
            return
        
        # Add remove_unnecessary_columns if threshold is set
        if remove_columns_threshold > 0:
            selected_functions.append('remove_unnecessary_columns')
        
        with st.spinner("Applying cleaning operations..."):
            try:
                processed_df, changes_made, dataset_id = st.session_state.processor.apply_cleaning_functions(
                    st.session_state.df, 
                    st.session_state.analysis, 
                    selected_functions
                )
                
                st.session_state.processed_df = processed_df
                st.session_state.changes_made = changes_made
                st.session_state.dataset_id = dataset_id
                st.session_state.cleaning_applied = True
                
                st.success("✅ Cleaning applied successfully!")
                
            except Exception as e:
                st.error(f"Error during cleaning: {str(e)}")
    
    # Show changes if cleaning was applied
    if st.session_state.cleaning_applied and st.session_state.changes_made:
        st.subheader("Changes Applied")
        for change in st.session_state.changes_made:
            st.write(f"• {change}")

def show_results_section():
    st.header("📈 Preview Results")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    
    if not st.session_state.cleaning_applied:
        st.info("No cleaning operations applied yet. Go to the Data Cleaning section to clean your data.")
        
        # Show original data stats
        st.subheader("Original Dataset")
        show_dataset_stats(st.session_state.df, "Original")
        return
    
    # Show comparison between original and processed
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Dataset")
        show_dataset_stats(st.session_state.df, "Original")
    
    with col2:
        st.subheader("Cleaned Dataset")
        show_dataset_stats(st.session_state.processed_df, "Cleaned")
    
    # Show changes summary
    st.subheader("Cleaning Summary")
    if st.session_state.changes_made:
        st.success(f"✅ Applied {len(st.session_state.changes_made)} cleaning operations")
        for change in st.session_state.changes_made:
            st.write(f"• {change}")
    else:
        st.info("No changes were made to the dataset.")
    
    # Show cleaned data preview
    st.subheader("Cleaned Data Preview")
    st.dataframe(st.session_state.processed_df.head(10), use_container_width=True)
    
    # Download options
    st.subheader("Download Cleaned Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = convert_df_to_csv(st.session_state.processed_df)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )
    
    with col2:
        excel = convert_df_to_excel(st.session_state.processed_df)
        st.download_button(
            label="Download as Excel",
            data=excel,
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Restore original data option
    if st.button("Restore Original Data"):
        st.session_state.cleaning_applied = False
        st.session_state.processed_df = None
        st.session_state.changes_made = []
        st.rerun()

def show_dataset_stats(df, title):
    """Display dataset statistics"""
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])
    st.metric("Total Missing", df.isna().sum().sum())
    st.metric("Duplicates", len(df) - len(df.drop_duplicates()))
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    st.metric("Memory Usage", f"{memory_mb:.2f} MB")

def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    """Convert DataFrame to Excel for download"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
    return output.getvalue()

if __name__ == "__main__":
    main()