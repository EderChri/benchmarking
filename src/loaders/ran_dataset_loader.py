from .base_loader import BaseDataLoader, register_loader
import pandas as pd
from merlion.utils import TimeSeries
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@register_loader("ran")
class RANDatasetLoader(BaseDataLoader):
    """Loader for ran_2025 with optional label column"""

    def _build_usecols(self, keep_columns, datetime_column, label_column, sector_column):
        if keep_columns:
            required = {datetime_column, sector_column}
            if label_column:
                required.add(label_column)
            return list(set(keep_columns) | required)
        return None

    def _load_by_sectors(self, file_path, use_cols, na_values, target_sectors, sector_column, max_per_sector, chunksize):
        collected = {}
        for chunk in pd.read_csv(file_path, na_values=na_values, usecols=use_cols, chunksize=chunksize):
            chunk = chunk[chunk[sector_column].isin(target_sectors)]
            for sector, group in chunk.groupby(sector_column):
                if sector not in collected:
                    collected[sector] = []
                existing = sum(len(g) for g in collected[sector])
                remaining = max_per_sector - existing
                if remaining > 0:
                    collected[sector].append(group.iloc[:remaining])

            if all(
                sum(len(g) for g in collected.get(s, [])) >= max_per_sector
                for s in target_sectors
            ):
                logger.info("All target sectors reached max samples, stopping early.")
                break

        if not collected:
            raise ValueError(f"No data found for target sectors: {target_sectors}")

        return pd.concat([pd.concat(frames) for frames in collected.values()])

    @staticmethod
    def _safe_stratify(y):
        return y if min(Counter(y).values()) >= 2 else None

    def load(self):
        file_path       = self.config.get("file_path")
        keep_columns    = self.config.get("keep_columns", None)
        label_column    = self.config.get("label_column", None)
        datetime_column = self.config.get("datetime_column", "period_start_time")
        datetime_format = self.config.get("datetime_format", None)
        na_values       = self.config.get("na_values", None)
        sector_column   = self.config.get("sector_column", "sector")
        target_sectors  = self.config.get("target_sectors", None)
        max_per_sector  = self.config.get("max_per_sector", 1000)
        chunksize       = self.config.get("chunksize", 50_000)

        use_cols = self._build_usecols(keep_columns, datetime_column, label_column, sector_column)

        if target_sectors:
            logger.info(f"Loading sectors {target_sectors} (max {max_per_sector} each)...")
            df = self._load_by_sectors(file_path, use_cols, na_values, target_sectors, sector_column, max_per_sector, chunksize)
        else:
            nrows = 1000 if self.test_mode else None
            logger.info(f"Loading data from {file_path} (nrows={nrows})...")
            df = pd.read_csv(file_path, na_values=na_values, nrows=nrows, usecols=use_cols)

        # Set datetime index
        df[datetime_column] = pd.to_datetime(df[datetime_column], format=datetime_format)
        df = df.set_index(datetime_column).sort_index()
        
        # Remove timezone info if present to avoid issues with some models
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_convert(None)


        # Drop sector column if not a feature
        if sector_column in df.columns and (not keep_columns or sector_column not in keep_columns):
            df = df.drop(columns=[sector_column])

        # Extract labels
        labels = None
        if label_column:
            labels = df[[label_column]]
            df = df.drop(columns=[label_column])

        if keep_columns:
            df = df[[c for c in keep_columns if c in df.columns]]

        df = df.astype(float)

        # Stratified split when labels are present
        if labels is not None:
            labels = labels[label_column].astype("category")
            labels = labels.cat.set_categories(sorted(labels.cat.categories))
            logger.info(f"Label mapping: {dict(zip(labels.cat.categories, range(len(labels.cat.categories))))}")

            label_codes = labels.cat.codes.astype(float)
            indices = df.index.to_numpy()
            y = label_codes.to_numpy()

            train_val_idx, test_idx = train_test_split(
                indices, test_size=self.test_split_ratio,
                stratify=self._safe_stratify(y), random_state=42
            )
            y_train_val = y[[df.index.get_loc(i) for i in train_val_idx]]
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=self.validation_split_ratio / (1 - self.test_split_ratio),
                stratify=self._safe_stratify(y_train_val), random_state=42
            )

            def get_pos(idx): return [df.index.get_loc(i) for i in idx]

            train_data   = TimeSeries.from_pd(df.loc[train_idx])
            val_data     = TimeSeries.from_pd(df.loc[val_idx])
            test_data    = TimeSeries.from_pd(df.loc[test_idx])
            train_labels = TimeSeries.from_pd(label_codes.iloc[get_pos(train_idx)])
            val_labels   = TimeSeries.from_pd(label_codes.iloc[get_pos(val_idx)])
            test_labels  = TimeSeries.from_pd(label_codes.iloc[get_pos(test_idx)])

            logger.info(f"Split sizes — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
            return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

        # Time-based split (no labels)
        train_end   = int(len(df) * (1 - self.test_split_ratio - self.validation_split_ratio))
        test_start  = int(len(df) * (1 - self.test_split_ratio))
        train_data  = TimeSeries.from_pd(df.iloc[:train_end])
        val_data    = TimeSeries.from_pd(df.iloc[train_end:test_start])
        test_data   = TimeSeries.from_pd(df.iloc[test_start:])
        return train_data, val_data, test_data
