from loaders.base_loader import BaseDataLoader, register_loader
import pandas as pd
from merlion.utils import TimeSeries


@register_loader("single_csv")
class SingleCSVDatasetLoader(BaseDataLoader):
    """Loader for datasets with a single CSV file"""

    @staticmethod
    def _to_binary_labels(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return (series.fillna(0).astype(float) > 0).astype(float)

        lowered = series.astype(str).str.strip().str.lower()
        positive = {"1", "true", "t", "yes", "y", "anomaly", "abnormal"}
        return lowered.isin(positive).astype(float)
    
    def load(self):
        file_path = self.config.get('file_path')
        target_columns = self.config.get('target_columns', None) 
        index_column = self.config.get('index_column', None)
        datetime_columns = self.config.get('datetime_columns', None) 
        datetime_format = self.config.get('datetime_format', None)
        na_values = self.config.get('na_values', None)
        label_column = self.config.get('label_column', None)
        
        # Read CSV
        df = pd.read_csv(file_path, na_values=na_values)
        
        # Handle datetime index creation if needed
        if datetime_columns:
            if len(datetime_columns) > 1:
                # Combine multiple columns (e.g., Date + Time)
                df['Datetime'] = pd.to_datetime(
                    df[datetime_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1),
                    format=datetime_format
                )
            else:
                df['Datetime'] = pd.to_datetime(df[datetime_columns[0]], format=datetime_format)
            df = df.set_index('Datetime')
            df = df.drop(columns=datetime_columns, errors='ignore')
        elif index_column:
            df = df.set_index(index_column)
            df.index = pd.to_datetime(df.index)

        label_series = None
        if label_column is not None:
            if label_column not in df.columns:
                raise ValueError(f"Configured label_column '{label_column}' not found in CSV columns")
            label_series = self._to_binary_labels(df[label_column])
            df = df.drop(columns=[label_column])
        
        # Select target columns
        if target_columns:
            df = df[target_columns]
        
        df = df.astype(float)
        df = df.sort_index()
        
        if self.test_mode:
            df = df.head(100)

        # Split data
        train_split_idx = int(len(df) * (1 - self.test_split_ratio - self.validation_split_ratio))
        test_split_idx = int(len(df) * (1 - self.test_split_ratio))

        train_df = df.iloc[:train_split_idx]
        val_df = df.iloc[train_split_idx:test_split_idx]
        test_df = df.iloc[test_split_idx:]

        train_data = TimeSeries.from_pd(train_df)
        val_data = TimeSeries.from_pd(val_df)
        test_data = TimeSeries.from_pd(test_df)

        if label_series is None:
            return train_data, val_data, test_data

        label_df = label_series.to_frame(name="label")
        train_labels = TimeSeries.from_pd(label_df.loc[train_df.index])
        val_labels = TimeSeries.from_pd(label_df.loc[val_df.index])
        test_labels = TimeSeries.from_pd(label_df.loc[test_df.index])

        return (train_data, train_labels), (test_data, test_labels), (val_data, val_labels)