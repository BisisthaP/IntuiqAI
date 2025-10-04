import streamlit as st 
import pandas as pd 
from data import datatype_detection, detect_missing_values, detect_outliers_iqr, prepare_analysis_summary
from langchain_community.llms import Ollama
import chromadb
from chromadb.config import Settings

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
    
    # Get column names for better context
    numerical_cols = analysis_summary['datatypes']['numerical_columns']
    categorical_cols = analysis_summary['datatypes']['categorical_columns']
    datetime_cols = analysis_summary['datatypes']['datetime_columns']
    
    prompt = f"""
CRITICAL ANALYSIS REQUEST - BE SPECIFIC AND DATA-DRIVEN

You are a senior data analyst reviewing a data quality assessment. Provide CRITICAL, SPECIFIC feedback based on the actual data patterns.

DATASET OVERVIEW:
- Shape: {analysis_summary['dataset_info']['shape'][0]} rows × {analysis_summary['datatypes']['total_columns']} columns
- Numerical columns ({len(numerical_cols)}): {numerical_cols}
- Categorical columns ({len(categorical_cols)}): {categorical_cols}
- Datetime columns ({len(datetime_cols)}): {datetime_cols}
- Boolean columns: {len(analysis_summary['datatypes']['boolean_columns'])}
- Other columns: {len(analysis_summary['datatypes']['other_columns'])}

DATA QUALITY ASSESSMENT:

MISSING VALUES ANALYSIS:
- Total nulls/NaN: {analysis_summary['missing_values']['total_nulls']}
- Total missing indicators: {analysis_summary['missing_values']['total_missing_indicators']}
- Combined missing: {analysis_summary['missing_values']['total_all_missing']} ({analysis_summary['missing_values']['total_all_missing']/analysis_summary['dataset_info']['total_rows']*100:.1f}% of data)
- Specific indicators: {missing_indicators_info}

OUTLIER ANALYSIS:
- Total outliers: {analysis_summary['outliers']['total_outliers']}
- Outlier percentage: {analysis_summary['outliers']['outlier_percentage']:.2f}%
- Columns with outliers: {analysis_summary['outliers']['columns_with_outliers']}

CRITICAL VERIFICATION REQUIRED - BE SPECIFIC:

1. DATA TYPE VALIDATION:
   - Check if numerical columns ({numerical_cols}) truly contain quantitative data
   - Verify categorical columns ({categorical_cols}) - are there any that should be numerical?
   - Validate datetime columns ({datetime_cols}) - are these actual dates or just date-like strings?
   - Flag any suspicious type assignments with SPECIFIC column names

2. MISSING DATA ASSESSMENT:
   - Are missing value patterns concerning? Which columns are most affected?
   - Are the missing indicators ('unknown', 'Not Given') properly handled?
   - Suggest SPECIFIC imputation strategies for EACH problematic column type

3. OUTLIER ANALYSIS CRITIQUE:
   - For outlier columns {analysis_summary['outliers']['columns_with_outliers']}, are these true outliers or data errors?
   - Given the data types, does outlier detection make sense?

4. ACTIONABLE RECOMMENDATIONS:
   - Provide 3-5 MOST IMPORTANT next steps
   - Be SPECIFIC about which columns need attention
   - Suggest concrete data cleaning steps
   - Recommend appropriate visualizations for THIS dataset

RESPONSE FORMAT:
- Start with "DATA QUALITY SCORE: X/10" based on overall data health
- Use bullet points for specific issues found
- Provide column-specific recommendations
- End with "TOP 3 PRIORITY ACTIONS"

Avoid generic advice - focus on THIS dataset's specific characteristics.
"""
    
    try:
        # Initialize Ollama with LangChain
        llm = Ollama(model="llama3.2:1b")
        
        # Call the model
        response = llm.invoke(prompt)
        
        # Store response in ChromaDB for context memory
        store_analysis_context(analysis_summary, response)
        
        return response
        
    except Exception as e:
        return f"Failed to connect to Ollama: {str(e)}. Make sure Ollama is running and the model is installed."


def store_analysis_context(analysis_summary, response):
    """Store analysis context in ChromaDB for memory"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Initialize ChromaDB
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_storage"
        ))
        
        # Create or get collection
        collection = client.get_or_create_collection(name="analysis_context")
        
        # Create a unique ID based on dataset characteristics
        dataset_id = f"dataset_{analysis_summary['dataset_info']['shape'][0]}_{analysis_summary['dataset_info']['shape'][1]}"
        
        # Store the context
        collection.add(
            documents=[f"Analysis: {analysis_summary}\n\nLLM Response: {response}"],
            metadatas=[{
                "dataset_shape": str(analysis_summary['dataset_info']['shape']),
                "timestamp": str(pd.Timestamp.now()),
                "numerical_cols": str(analysis_summary['datatypes']['numerical_columns']),
                "categorical_cols": str(analysis_summary['datatypes']['categorical_columns'])
            }],
            ids=[dataset_id]
        )
        
    except Exception as e:
        print(f"ChromaDB storage failed: {e}")
        # Continue without storage - non-critical feature


def get_previous_analysis(dataset_shape):
    """Retrieve previous analysis from ChromaDB"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_storage"
        ))
        
        collection = client.get_collection(name="analysis_context")
        dataset_id = f"dataset_{dataset_shape[0]}_{dataset_shape[1]}"
        
        results = collection.get(ids=[dataset_id])
        if results['documents']:
            return results['documents'][0]
        return None
        
    except Exception as e:
        print(f"ChromaDB retrieval failed: {e}")
        return None

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
        
        st.metric("Datetime Columns", len(analysis_summary["datatypes"]["datetime_columns"]))
        # Display detailed analysis tables
        st.subheader("Detailed Analysis")

        with st.expander("Data Types"):
            st.table(pd.DataFrame({
                'Type': ['Numerical', 'Categorical', 'Boolean', 'Other','Datetime'],
                'Count': [
                    len(analysis_summary["datatypes"]["numerical_columns"]),
                    len(analysis_summary["datatypes"]["categorical_columns"]),
                    len(analysis_summary["datatypes"]["boolean_columns"]),
                    len(analysis_summary["datatypes"]["other_columns"]),
                    len(analysis_summary["datatypes"]["datetime_columns"])
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
            # Check if we have previous analysis
            previous_analysis = get_previous_analysis(df.shape)
            
            if previous_analysis:
                st.info("Found previous analysis for this dataset shape")
                if st.button("Use Previous Analysis"):
                    st.success("Using cached analysis!")
                    st.write("**AI Review:**")
                    st.write(previous_analysis.split("LLM Response: ")[-1])
            
            with st.spinner("Consulting AI for verification..."):
                verification_result = call_llama_verification_langchain(analysis_summary)
                
            st.success("AI Verification Complete!")
            st.write("**AI Review:**")
            st.write(verification_result)
        
    except Exception as e:
        st.error(f"Error loading/reading the CSV file: {e}")
else:
    st.info("Please upload a CSV file to get started.")