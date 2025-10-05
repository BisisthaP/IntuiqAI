# processing.py
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
        
    def generate_dataset_id(self, df):
        """Generate unique ID for dataset"""
        content_hash = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dataset_{timestamp}_{content_hash[:8]}"
    
    def store_datasets(self, original_df, processed_df, dataset_id):
        """Store both datasets in SQLite"""
        self.dataset_id = dataset_id
        conn = sqlite3.connect('datasets.db')
        
        # Store original
        original_df.to_sql(f'{dataset_id}_original', conn, if_exists='replace', index=False)
        
        # Store processed
        processed_df.to_sql(f'{dataset_id}_processed', conn, if_exists='replace', index=False)
        
        # Store changes log
        changes_df = pd.DataFrame(self.changes_log)
        changes_df.to_sql(f'{dataset_id}_changes', conn, if_exists='replace', index=False)
        
        conn.close()
    
    def get_stored_dataset(self, dataset_id, version='processed'):
        """Retrieve stored dataset"""
        conn = sqlite3.connect('datasets.db')
        try:
            df = pd.read_sql(f'SELECT * FROM {dataset_id}_{version}', conn)
            return df
        except:
            return None
        finally:
            conn.close()

    def remove_whitespace_and_trim(self, df):
        """Remove leading/trailing whitespace from all string columns"""
        changes = []
        for col in df.select_dtypes(include=['object']).columns:
            before_count = df[col].astype(str).str.strip().ne(df[col].astype(str)).sum()
            if before_count > 0:
                df[col] = df[col].astype(str).str.strip()
                changes.append(f"Trimmed whitespace from {col}: {before_count} cells")
        
        if changes:
            self.changes_log.extend(changes)
        return df, changes

    def fill_missing_custom_keyword(self, df, analysis):
        """Replace custom missing indicators with proper nulls"""
        changes = []
        for col, info in analysis.items():
            if info['custom_missing'] > 0:
                before_count = info['custom_missing']
                # Replace custom missing with NaN first
                df[col] = df[col].replace(self.missing_indicators, np.nan)
                changes.append(f"Replaced custom missing in {col}: {before_count} values")
        
        if changes:
            self.changes_log.extend(changes)
        return df, changes

    def fill_missing_mean_median_mode(self, df, analysis):
        """Impute missing values based on column type"""
        changes = []
        for col, info in analysis.items():
            total_missing = info['missing'] + info['custom_missing']
            if total_missing > 0:
                if info['type'] == 'numerical':
                    # Use median for numerical to handle outliers
                    impute_value = df[col].median()
                    df[col].fillna(impute_value, inplace=True)
                    changes.append(f"Imputed {col} with median: {impute_value:.2f}")
                elif info['type'] in ['categorical', 'boolean']:
                    # Use mode for categorical
                    impute_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(impute_value, inplace=True)
                    changes.append(f"Imputed {col} with mode: {impute_value}")
                elif info['type'] == 'datetime':
                    # Forward fill for datetime
                    df[col].fillna(method='ffill', inplace=True)
                    changes.append(f"Forward-filled missing dates in {col}")
        
        if changes:
            self.changes_log.extend(changes)
        return df, changes

    def remove_duplicates(self, df):
        """Remove duplicate rows across entire dataset"""
        before_rows = len(df)
        df = df.drop_duplicates()
        after_rows = len(df)
        duplicates_removed = before_rows - after_rows
        
        changes = []
        if duplicates_removed > 0:
            changes.append(f"Removed {duplicates_removed} duplicate rows")
            self.changes_log.extend(changes)
        
        return df, changes

    def convert_dtype(self, df, analysis):
        """Convert columns to proper data types"""
        changes = []
        for col, info in analysis.items():
            current_dtype = str(df[col].dtype)
            
            if info['type'] == 'datetime' and not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    changes.append(f"Converted {col} to datetime")
                except:
                    pass
                    
            elif info['type'] == 'numerical' and not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    changes.append(f"Converted {col} to numerical")
                except:
                    pass
            
            elif info['type'] == 'boolean' and not pd.api.types.is_bool_dtype(df[col]):
                try:
                    df[col] = df[col].astype(bool)
                    changes.append(f"Converted {col} to boolean")
                except:
                    pass
        
        if changes:
            self.changes_log.extend(changes)
        return df, changes

    def remove_outliers(self, df, analysis):
        """Cap outliers using IQR method for numerical columns"""
        changes = []
        for col, info in analysis.items():
            if info['type'] == 'numerical' and info['outliers'] and info['outliers'] > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing
                outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                
                changes.append(f"Capped outliers in {col}: {outliers_before} values")
        
        if changes:
            self.changes_log.extend(changes)
        return df, changes

    def standardize_data_formats(self, df, analysis):
        """Standardize text formats and date formats"""
        changes = []
        for col, info in analysis.items():
            if info['type'] in ['categorical', 'text']:
                # Standardize text casing
                df[col] = df[col].astype(str).str.title()
                changes.append(f"Standardized text casing in {col}")
            
            elif info['type'] == 'datetime':
                # Ensure consistent datetime format
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                changes.append(f"Standardized date format in {col}")
        
        if changes:
            self.changes_log.extend(changes)
        return df, changes

    def remove_unnecessary_columns(self, df, analysis, threshold=50):
        """Remove columns with high missing values or low variance"""
        changes = []
        columns_to_drop = []
        
        for col, info in analysis.items():
            missing_percentage = (info['missing'] + info['custom_missing']) / info['total_count'] * 100
            
            # Drop if missing > threshold% or very low variance
            if missing_percentage > threshold:
                columns_to_drop.append(col)
                changes.append(f"Removed {col}: {missing_percentage:.1f}% missing")
            elif info['nunique'] == 1:
                columns_to_drop.append(col)
                changes.append(f"Removed {col}: constant value")
        
        df = df.drop(columns=columns_to_drop)
        
        if changes:
            self.changes_log.extend(changes)
        return df, changes

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

    def apply_cleaning_functions(self, df, analysis, selected_functions):
        """Apply selected cleaning functions in optimal order"""
        self.original_df = df.copy()
        self.processed_df = df.copy()
        self.changes_log = []
        
        # Define optimal execution order
        function_order = [
            'remove_whitespace_and_trim',
            'fill_missing_custom_keyword', 
            'convert_dtype',
            'remove_duplicates',
            'fill_missing_mean_median_mode',
            'remove_outliers',
            'standardize_data_formats',
            'remove_unnecessary_columns'
        ]
        
        # Filter and order selected functions
        ordered_functions = [f for f in function_order if f in selected_functions]
        
        all_changes = []
        for func_name in ordered_functions:
            func = getattr(self, func_name)
            if func_name in ['remove_unnecessary_columns']:
                self.processed_df, changes = func(self.processed_df, analysis)
            else:
                self.processed_df, changes = func(self.processed_df, analysis)
            all_changes.extend(changes)
        
        # Store datasets
        dataset_id = self.generate_dataset_id(self.original_df)
        self.store_datasets(self.original_df, self.processed_df, dataset_id)
        
        return self.processed_df, all_changes, dataset_id

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

# Remove the verify_with_llm function since we're no longer using AI verification