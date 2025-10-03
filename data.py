import pandas as pd 
import numpy as np 
import requests
import json

def datatype_detection(df):
    dtypes = df.dtypes #identify all the datatypes 
    num_cols = dtypes[ (dtypes == 'int64') | (dtypes == 'float64') | (dtypes == 'int32') | (dtypes == 'float32') ].index.tolist() #numerical columns
    cat_cols = dtypes[ (dtypes == 'object') | (dtypes == 'category') ].index.tolist() #categorical columns
    bool_cols = dtypes[ (dtypes == 'bool') ].index.tolist() #boolean columns

    other_cols = [col for col in df.columns if col not in num_cols + cat_cols + bool_cols]

    # Detect datetime-related columns
    datetime_keywords = ['year', 'month', 'date', 'day', 'yr', 'mon', 'dt', 'time', 
                        'week', 'quarter', 'season', 'period']
    
    datetime_cols = []
    
    # Check column names for datetime keywords
    for col in num_cols + cat_cols + other_cols:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in datetime_keywords):
            datetime_cols.append(col)
    
    # Remove datetime columns from their original categories and add to categorical
    for col in datetime_cols:
        if col in num_cols:
            num_cols.remove(col)
        if col in other_cols:
            other_cols.remove(col)
        if col not in cat_cols:
            cat_cols.append(col)
    
    category_counts = {col: df[col].nunique() for col in cat_cols}
    #unique_categories = {col: df[col].unique().tolist() for col in cat_cols}
    
    return {
        "numerical":num_cols,
        "categorical":cat_cols,
        "boolean":bool_cols,
        "other":other_cols,
        "category_counts": category_counts
        #"unique_categories": unique_categories
    }

def detect_missing_values(df):
    """Detect missing values including nulls and common missing value indicators"""
    
    # Count null/NaN values
    null_counts = df.isnull().sum()
    
    # Common missing value indicators to check for
    missing_indicators = ['', 'na', 'NA', 'Na', 'n/a', 'N/A', 'null', 'NULL', 
                         'unknown', 'missing',
                         'error', 'ERROR', 'Error', 'none', 'NONE', 'None',
                         '-', '--', '---', 'NaN', 'nan', 'NAN', 'Not Given']
    
    # Dictionary to store results
    missing_data = {
        'null_counts': null_counts.to_dict(),
        'total_nulls': null_counts.sum(),
        'missing_indicators': {},
        'total_missing_indicators': 0
    }
    
    # Check for missing value indicators in each column
    for column in df.columns:
        if df[column].dtype == 'object':  # Only check string columns
            indicator_counts = {}
            for indicator in missing_indicators:
                count = (df[column].astype(str).str.lower() == indicator.lower()).sum()
                if count > 0:
                    indicator_counts[indicator] = count
            
            if indicator_counts:
                missing_data['missing_indicators'][column] = indicator_counts
                missing_data['total_missing_indicators'] += sum(indicator_counts.values())
    
    return missing_data

def detect_outliers_iqr(df, numerical_columns=None):
    """Detect outliers in numerical columns using IQR method"""
    
    if numerical_columns is None:
        # Auto-detect numerical columns if not provided
        numerical_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    outliers_data = {}
    
    for col in numerical_columns:
        # Calculate Q1, Q3, and IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        outliers_data[col] = {
            'q1': Q1,
            'q3': Q3,
            'iqr': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(df)) * 100,
            'outlier_values': outliers.tolist(),
            'min_value': df[col].min(),
            'max_value': df[col].max()
        }
    
    # Calculate total outliers
    total_outliers = sum([data['outlier_count'] for data in outliers_data.values()])
    total_outlier_percentage = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
    
    return {
        'outliers_per_column': outliers_data,
        'total_outliers': total_outliers,
        'total_outlier_percentage': total_outlier_percentage,
        'columns_with_outliers': [col for col, data in outliers_data.items() if data['outlier_count'] > 0]
    } 

def prepare_analysis_summary(df, datatype_result, missing_result, outlier_result):
    """Prepare comprehensive analysis results for LLM verification"""
    
    # Calculate total missing values (nulls + indicators)
    total_missing_with_indicators = missing_result["total_nulls"] + missing_result["total_missing_indicators"]
    
    analysis_summary = {
        "dataset_info": {
            "shape": df.shape,
            "total_columns": len(df.columns),
            "total_rows": len(df)
        },
        "datatypes": {
            "numerical_columns": datatype_result["numerical"],
            "categorical_columns": datatype_result["categorical"],
            "boolean_columns": datatype_result["boolean"],
            "other_columns": datatype_result["other"]
        },
        "missing_values": {
            "total_nulls": missing_result["total_nulls"],
            "total_missing_indicators": missing_result["total_missing_indicators"],
            "total_all_missing": total_missing_with_indicators,
            "nulls_per_column": missing_result["null_counts"],
            "missing_indicators_found": missing_result["missing_indicators"]  # This includes the actual indicators
        },
        "outliers": {
            "total_outliers": outlier_result["total_outliers"],
            "outlier_percentage": outlier_result["total_outlier_percentage"],
            "columns_with_outliers": outlier_result["columns_with_outliers"]
        }
    }
    
    return analysis_summary

