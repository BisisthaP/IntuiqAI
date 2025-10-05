# processing.py
import pandas as pd
import numpy as np
from langchain_community.llms import Ollama
import json
from typing import Dict, Any, List

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
        """Detect the most appropriate data type for a column"""
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

def analysis_to_jsonl(df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
    dataset_info = {
        "dataset_summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
    }
    
    jsonl_lines = [json.dumps(dataset_info)]
    
    for col, info in analysis.items():
        total_count = info['total_count']
        missing_pct = (info['missing'] / total_count) * 100 if total_count > 0 else 0
        custom_missing_pct = (info['custom_missing'] / total_count) * 100 if total_count > 0 else 0
        outliers_pct = (info['outliers'] / total_count) * 100 if info['outliers'] is not None and total_count > 0 else None
        
        sample_values = df[col].dropna().head(3)
        sample_list = []
        for val in sample_values:
            if hasattr(val, 'item'):
                sample_list.append(val.item())
            elif pd.isna(val):
                continue
            else:
                sample_list.append(str(val))
        
        column_data = {
            "column_name": col,
            "detected_type": info['type'],
            "statistics": {
                "total_count": total_count,
                "unique_count": info['nunique'],
                "missing_count": info['missing'],
                "custom_missing_count": info['custom_missing'],
                "outliers_count": info['outliers'],
                "missing_percentage": round(missing_pct, 2),
                "custom_missing_percentage": round(custom_missing_pct, 2),
                "outliers_percentage": round(outliers_pct, 2) if outliers_pct is not None else None,
                "completeness_score": round(100 - missing_pct - custom_missing_pct, 2)
            },
            "sample_values": sample_list
        }
        
        jsonl_lines.append(json.dumps(column_data))
    
    return "\n".join(jsonl_lines)

def verify_with_llm(analysis_jsonl: str) -> str:
    llm = Ollama(model='llama3.2:1b')
    
    prompt = f"""
You are a data cleaning specialist for dashboard preparation. Provide SPECIFIC cleaning steps for each column.

STRUCTURED DATASET ANALYSIS (JSONL):
{analysis_jsonl}

CRITICAL: Provide EXACT cleaning code and steps for dashboard preparation. Be specific and actionable.

DASHBOARD CLEANING REQUIREMENTS:
1. Handle missing values with dashboard-friendly imputation
2. Standardize formats for filtering and grouping
3. Ensure data types work with common visualization libraries
4. Create derived columns for common analytics

OUTPUT FORMAT (JSON):
{{
  "column_assessments": [
    {{
      "column_name": "string",
      "verified_type": "string",
      "dashboard_issues": ["specific issues affecting dashboards"],
      "cleaning_steps": [
        {{
          "step": "string",
          "code": "pandas code snippet",
          "reason": "dashboard benefit"
        }}
      ],
      "derived_columns": ["suggested new columns for analytics"]
    }}
  ],
  "overall_cleaning_pipeline": [
    {{
      "step": "string",
      "code": "pandas code snippet", 
      "priority": "high|medium|low"
    }}
  ],
  "dashboard_optimizations": [
    "specific optimizations for better dashboard performance"
  ]
}}

CLEANING PATTERNS BY DATA TYPE:

IDENTIFIER COLUMNS:
- Issues: Duplicates, format inconsistencies
- Cleaning: Validate uniqueness, standardize formats
- Code: `df['col'] = df['col'].astype(str).str.strip()`

CATEGORICAL COLUMNS:
- Issues: Too many categories, inconsistent casing
- Cleaning: Group rare categories, standardize text
- Code: `df['col'] = df['col'].str.title().replace({{'old_val': 'new_val'}})`

NUMERICAL COLUMNS:  
- Issues: Outliers, wrong data types
- Cleaning: Handle outliers, convert types
- Code: `df['col'] = pd.to_numeric(df['col'], errors='coerce')`

DATETIME COLUMNS:
- Issues: Multiple formats, timezone issues
- Cleaning: Standardize format, extract features
- Code: `df['col'] = pd.to_datetime(df['col'], errors='coerce')`

DURATION COLUMNS:
- Issues: Mixed units, inconsistent formats
- Cleaning: Convert to standard units
- Code: `df['duration_min'] = df['col'].apply(convert_to_minutes)`

HIGH-CARDINALITY COLUMNS (>1000 unique):
- Issues: Poor filter performance, slow visualizations
- Cleaning: Create grouped versions, use top-N categories

MISSING DATA STRATEGIES:
- <5% missing: Simple imputation
- 5-30% missing: Advanced imputation with validation
- >30% missing: Consider exclusion or flagging

Provide ready-to-use pandas code for each cleaning step.
"""
    
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return json.dumps({
            "error": f"LLM processing failed: {str(e)}",
            "column_assessments": [],
            "overall_cleaning_pipeline": []
        })

def analysis_to_string(df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
    return analysis_to_jsonl(df, analysis)