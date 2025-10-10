import pandas as pd
import numpy as np
import json
import sqlite3
from typing import Dict, Any, List, Tuple
import hashlib
from datetime import datetime

class DatasetAnalyzer:
    def __init__(self):
        self.common_date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
            '%Y-%m-%d %H:%M:%S', '%m-%d-%Y', '%d-%m-%Y',
            '%b %d %Y', '%d %b %Y', '%Y %b %d', '%B %d, %Y',
            '%d %B %Y'
        ]
        self.missing_indicators = ['not given', 'unknown', 'na', 'n/a', 'none', 'null', 'missing', '']
        self.categorical_threshold = 0.05

    def detect_column_type(self, series: pd.Series) -> str:
        dtype = series.dtype
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return 'unknown'
        
        if self._is_datetime_column(series):
            return 'datetime'
        
        if self._is_boolean_column(series):
            return 'boolean'
        
        if self._is_duration_column(series):
            return 'duration'
        
        if pd.api.types.is_numeric_dtype(dtype):
            return self._classify_numeric_type(series)
        
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            return self._classify_object_type(series)
        
        return 'categorical'

    def _is_datetime_column(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        if pd.api.types.is_object_dtype(series.dtype):
            for fmt in self.common_date_formats:
                try:
                    parsed = pd.to_datetime(series, format=fmt, errors='coerce')
                    if parsed.notna().mean() > 0.8:
                        return True
                except (ValueError, TypeError):
                    continue
        return False

    def _is_boolean_column(self, series: pd.Series) -> bool:
        if pd.api.types.is_bool_dtype(series):
            return True
        
        if pd.api.types.is_object_dtype(series.dtype):
            unique_vals = series.dropna().astype(str).str.lower().unique()
            boolean_vals = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
            if all(str(val) in boolean_vals for val in unique_vals if pd.notna(val)):
                return True
        return False

    def _is_duration_column(self, series: pd.Series) -> bool:
        if pd.api.types.is_timedelta64_dtype(series):
            return True
        
        if pd.api.types.is_object_dtype(series.dtype):
            sample_values = series.dropna().astype(str).head(100)
            duration_patterns = ['min', 'hour', 'hr', 'sec', 'day', 'week', 'month', 'year', 'duration', 'time']
            pattern_matches = sum(any(pattern in str(val).lower() for pattern in duration_patterns) 
                                for val in sample_values)
            if pattern_matches / len(sample_values) > 0.7:
                return True
        return False

    def _classify_numeric_type(self, series: pd.Series) -> str:
        non_null = series.dropna()
        
        if (non_null == non_null.astype(int)).all() and non_null.nunique() / len(non_null) < self.categorical_threshold:
            return 'categorical'
        
        if ((non_null >= 1900) & (non_null <= 2100)).all() and non_null.nunique() < 150:
            return 'categorical'
        
        if non_null.nunique() / len(non_null) < self.categorical_threshold:
            return 'categorical'
        
        return 'numerical'

    def _classify_object_type(self, series: pd.Series) -> str:
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return 'categorical'
        
        avg_length = non_null.astype(str).str.len().mean()
        unique_ratio = non_null.nunique() / len(non_null)
        
        if unique_ratio > 0.9 and avg_length > 50:
            return 'text'
        elif unique_ratio < self.categorical_threshold:
            return 'categorical'
        else:
            if series.name and any(keyword in series.name.lower() for keyword in ['id', 'code', 'key', 'num']):
                return 'identifier'
            return 'categorical'

    def detect_custom_missing(self, series: pd.Series, col_type: str) -> int:
        if col_type in ['text', 'identifier']:
            return 0
        
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            return series.astype(str).str.lower().isin(self.missing_indicators).sum()
        return 0

    def detect_outliers(self, series: pd.Series, col_type: str) -> int:
        if col_type != 'numerical':
            return None
        
        non_null = series.dropna()
        if len(non_null) < 4:
            return 0
        
        Q1 = non_null.quantile(0.25)
        Q3 = non_null.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            std = non_null.std()
            if std == 0:
                return 0
            outliers = ((non_null < (non_null.mean() - 3 * std)) | 
                       (non_null > (non_null.mean() + 3 * std))).sum()
        else:
            outliers = ((non_null < (Q1 - 1.5 * IQR)) | 
                       (non_null > (Q3 + 1.5 * IQR))).sum()
        
        return outliers

class DataProcessor:
    def __init__(self):
        self.original_df = None
        self.processed_df = None
        self.changes_log = []
        self.dataset_id = None

    def get_cleaning_recommendations(self, df, analysis):
        """Generate cleaning recommendations based on analysis"""
        recommendations = []
        
        # Check for duplicates
        if len(df) != len(df.drop_duplicates()):
            recommendations.append({
                'function': 'remove_duplicates',
                'reason': f"Dataset has {len(df) - len(df.drop_duplicates())} duplicate rows",
                'priority': 'high'
            })
        
        # Check for custom missing values
        total_custom_missing = sum(info['custom_missing'] for info in analysis.values())
        if total_custom_missing > 0:
            recommendations.append({
                'function': 'fill_missing_custom_keyword',
                'reason': f"Found {total_custom_missing} custom missing values (Not Given, Unknown, etc.)",
                'priority': 'high'
            })
        
        # Check for standard missing values
        total_missing = sum(info['missing'] + info['custom_missing'] for info in analysis.values())
        if total_missing > 0:
            recommendations.append({
                'function': 'fill_missing_mean_median_mode', 
                'reason': f"Found {total_missing} total missing values needing imputation",
                'priority': 'medium'
            })
        
        # Check for whitespace issues
        text_columns = df.select_dtypes(include=['object']).columns
        whitespace_issues = any(df[col].astype(str).str.strip().ne(df[col].astype(str)).any() for col in text_columns)
        if whitespace_issues:
            recommendations.append({
                'function': 'remove_whitespace_and_trim',
                'reason': "Found leading/trailing whitespace in text columns",
                'priority': 'medium'
            })
        
        # Check for data type issues
        type_issues = any(
            (info['type'] == 'datetime' and not pd.api.types.is_datetime64_any_dtype(df[col])) or
            (info['type'] == 'numerical' and not pd.api.types.is_numeric_dtype(df[col]))
            for col, info in analysis.items()
        )
        if type_issues:
            recommendations.append({
                'function': 'convert_dtype',
                'reason': "Found columns with incorrect data types",
                'priority': 'high'
            })
        
        # Check for outliers
        total_outliers = sum(info['outliers'] for info in analysis.values() if info['outliers'])
        if total_outliers > 0:
            recommendations.append({
                'function': 'remove_outliers',
                'reason': f"Found {total_outliers} outliers in numerical columns",
                'priority': 'low'
            })
        
        # Check for formatting issues
        formatting_issues = any(info['type'] in ['categorical', 'text'] for info in analysis.values())
        if formatting_issues:
            recommendations.append({
                'function': 'standardize_data_formats',
                'reason': "Text columns need standardization for better filtering",
                'priority': 'low'
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    analyzer = DatasetAnalyzer()
    analysis = {}
    
    for col in df.columns:
        col_type = analyzer.detect_column_type(df[col])
        missing = df[col].isnull().sum()
        custom_missing = analyzer.detect_custom_missing(df[col], col_type)
        outliers = analyzer.detect_outliers(df[col], col_type)
        
        analysis[col] = {
            'type': col_type,
            'missing': int(missing),
            'custom_missing': int(custom_missing),
            'outliers': int(outliers) if outliers is not None else None,
            'nunique': df[col].nunique(),
            'total_count': len(df[col])
        }
    
    return analysis