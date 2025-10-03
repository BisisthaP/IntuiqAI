import streamlit as st 
import pandas as pd 
from data import datatype_detection, detect_missing_values, detect_outliers_iqr, prepare_analysis_summary
from langchain_community.llms import Ollama

st.title('IntuiqAI')

def call_llama_verification_langchain(analysis_summary):
    """Call Ollama Llama model for analysis verification using LangChain"""
    
    # Build detailed missing indicators string
    missing_indicators_info = "None"
    if analysis_summary['missing_values']['missing_indicators_found']:
        indicators_list = []
        for col, indicators in analysis_summary['missing_values']['missing_indicators_found'].items():
            for indicator, count in indicators.items():
                indicators_list.append(f"'{indicator}': {count} in column '{col}'")
        missing_indicators_info = "; ".join(indicators_list)
    
    prompt = f"""
    Verify if this data analysis is correct and provide any suggestions for improvements:
    
    Dataset: {analysis_summary['dataset_info']['shape'][0]} rows, {analysis_summary['dataset_info']['shape'][1]} columns
    
    Data Types:
    - Numerical: {len(analysis_summary['datatypes']['numerical_columns'])} columns
    - Categorical: {len(analysis_summary['datatypes']['categorical_columns'])} columns  
    - Boolean: {len(analysis_summary['datatypes']['boolean_columns'])} columns
    - Other: {len(analysis_summary['datatypes']['other_columns'])} columns
    
    Missing Values:
    - Total nulls/NaN: {analysis_summary['missing_values']['total_nulls']}
    - Total missing value indicators (na, unknown, etc.): {analysis_summary['missing_values']['total_missing_indicators']}
    - Total all missing values: {analysis_summary['missing_values']['total_all_missing']}
    - Missing indicators found: {missing_indicators_info}
    
    Outliers:
    - Total outliers: {analysis_summary['outliers']['total_outliers']}
    - Outlier percentage: {analysis_summary['outliers']['outlier_percentage']:.2f}%
    - Columns with outliers: {analysis_summary['outliers']['columns_with_outliers']}
    
    Please verify:
    1. Are the data type classifications appropriate?
    2. Is the missing value analysis comprehensive (including both nulls and text indicators)?
    3. Are outlier detections reasonable?
    4. Suggest any improvements or additional analyses needed.
    
    Provide your response in a clear, structured format.
    """
    
    try:
        # Initialize Ollama with LangChain
        llm = Ollama(model="llama3.2:1b")
        
        # Call the model
        response = llm.invoke(prompt)
        return response
        
    except Exception as e:
        return f"Failed to connect to Ollama: {str(e)}. Make sure Ollama is running and the model is installed."

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Dataframe created!")

        st.subheader("Data Preview")
        st.dataframe(df)
        
        # Data type analysis
        st.subheader("Data Type Analysis")
        result = datatype_detection(df)
        st.write(result)
        
        # Missing value analysis
        st.subheader("Missing Value Analysis")
        missing_data = detect_missing_values(df)

        # Display null counts
        st.write("**Null/NaN Values per Column:**")
        st.write(missing_data['null_counts'])
        st.write(f"**Total Null Values:** {missing_data['total_nulls']}")

        # Display missing value indicators
        if missing_data['missing_indicators']:
            st.write("**Missing Value Indicators Found:**")
            for column, indicators in missing_data['missing_indicators'].items():
                st.write(f"**{column}**:")
                for indicator, count in indicators.items():
                    st.write(f"  - '{indicator}': {count} occurrences")
            st.write(f"**Total Missing Indicators:** {missing_data['total_missing_indicators']}")
        else:
            st.write("No common missing value indicators found (na, unknown, error, etc.)")
        
        # Outlier detection
        st.subheader("Outlier Detection (IQR Method)")
        outlier_data = detect_outliers_iqr(df, result["numerical"])
        
        # Display outlier summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Outliers", outlier_data['total_outliers'])
        with col2:
            st.metric("Outlier Percentage", f"{outlier_data['total_outlier_percentage']:.2f}%")
        with col3:
            st.metric("Columns with Outliers", len(outlier_data['columns_with_outliers']))
        
        # Prepare comprehensive analysis summary
        analysis_summary = prepare_analysis_summary(df, result, missing_data, outlier_data)

        # Display Analysis Summary
        st.subheader("📊 Comprehensive Analysis Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Columns", analysis_summary["dataset_info"]["total_columns"])
        with col2:
            st.metric("Total Rows", analysis_summary["dataset_info"]["total_rows"])
        with col3:
            st.metric("All Missing Values", analysis_summary["missing_values"]["total_all_missing"])
        with col4:
            st.metric("Outliers", analysis_summary["outliers"]["total_outliers"])

        # Additional missing value metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Null/NaN Values", analysis_summary["missing_values"]["total_nulls"])
        with col2:
            st.metric("Missing Indicators", analysis_summary["missing_values"]["total_missing_indicators"])

        # Display detailed analysis tables
        st.subheader("Detailed Analysis")

        with st.expander("Data Types"):
            st.table(pd.DataFrame({
                'Type': ['Numerical', 'Categorical', 'Boolean', 'Other'],
                'Count': [
                    len(analysis_summary["datatypes"]["numerical_columns"]),
                    len(analysis_summary["datatypes"]["categorical_columns"]),
                    len(analysis_summary["datatypes"]["boolean_columns"]),
                    len(analysis_summary["datatypes"]["other_columns"])
                ]
            }))

        with st.expander("Missing Values - Complete Analysis"):
            # Show null counts
            st.write("**Null/NaN Values:**")
            null_df = pd.DataFrame(list(analysis_summary["missing_values"]["nulls_per_column"].items()), 
                                 columns=['Column', 'Null Count'])
            st.table(null_df[null_df['Null Count'] > 0])
            
            # Show missing indicators
            if analysis_summary["missing_values"]["missing_indicators_found"]:
                st.write("**Missing Value Indicators:**")
                indicators_data = []
                for col, indicators in analysis_summary["missing_values"]["missing_indicators_found"].items():
                    for indicator, count in indicators.items():
                        indicators_data.append({'Column': col, 'Indicator': indicator, 'Count': count})
                st.table(pd.DataFrame(indicators_data))
            else:
                st.write("No missing value indicators found")

        with st.expander("Outliers"):
            outlier_df = pd.DataFrame([
                {'Column': col, 'Outlier Count': outlier_data['outliers_per_column'][col]['outlier_count']}
                for col in analysis_summary["outliers"]["columns_with_outliers"]
            ])
            st.table(outlier_df)

        # LLM Verification
        st.subheader("🤖 AI Verification")
        if st.button("Verify Analysis with AI"):
            with st.spinner("Consulting AI for verification..."):
                verification_result = call_llama_verification_langchain(analysis_summary)
                
            st.success("AI Verification Complete!")
            st.write("**AI Review:**")
            st.write(verification_result)
        
    except Exception as e:
        st.error(f"Error loading/reading the CSV file: {e}")
else:
    st.info("Please upload a CSV file to get started.")