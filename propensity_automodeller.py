"""
PropensityAutomodeller: Python Conversion of SAS SuperFred/Automodeller
=======================================================================

FIXED VERSION - Addresses the following issues:
1. Target handling standardization (binary 0/1 internally)
2. Weighted quantile binning
3. min_bin_pct enforcement with bin merging
4. Missing value handling for categorical variables
5. Consistent score scaling and sign conventions
6. Proper calibration with cross-validation (no leakage)
7. Time-based train/test split option
8. Monotonic WOE binning option
9. Configurable terminology (event_rate vs bad_rate)
10. Production additions (PSI monitoring, stability checks)

Author: Converted from SAS by Claude
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime
import json
import pickle
from pathlib import Path
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScoreDirection(Enum):
    """
    Defines score direction convention.
    
    HIGHER_IS_BETTER: Higher score = higher probability of positive outcome (e.g., payment)
    HIGHER_IS_SAFER: Higher score = lower probability of negative outcome (traditional credit score)
    """
    HIGHER_IS_BETTER = "higher_is_better"  # Higher score = higher P(event)
    HIGHER_IS_SAFER = "higher_is_safer"    # Higher score = lower P(event) (credit score convention)


@dataclass
class VariableSummary:
    """
    Summary statistics for a variable - equivalent to SAS _UNI_ETL_VAR_SUMMARY
    """
    name: str
    dtype: str  # 'numeric' or 'categorical'
    n_distinct: int = 0
    sum_wt: float = 0.0
    sum_wt_event: float = 0.0      # Renamed from sum_wt_B
    sum_wt_non_event: float = 0.0  # Renamed from sum_wt_G
    event_rate: float = 0.0        # Renamed from mean_bad_rate
    missing_freq: int = 0
    missing_wt: float = 0.0
    missing_wt_event: float = 0.0
    missing_wt_non_event: float = 0.0
    missing_event_rate: float = 0.0
    zero_freq: int = 0
    zero_wt: float = 0.0
    zero_wt_event: float = 0.0
    zero_wt_non_event: float = 0.0
    nonmissing_sum_wt: float = 0.0
    nonmissing_sum_wt_event: float = 0.0
    nonmissing_sum_wt_non_event: float = 0.0
    nonmissing_event_rate: float = 0.0
    iv: float = None


@dataclass
class WOEBin:
    """
    Represents a single WOE bin.
    """
    bin_id: int
    bin_label: str
    lower_bound: float = None
    upper_bound: float = None
    lower_inclusive: bool = False
    upper_inclusive: bool = True
    values: List[str] = None  # For categorical variables
    count: float = 0.0
    count_non_event: float = 0.0  # Renamed from count_good
    count_event: float = 0.0      # Renamed from count_bad
    pct_non_event: float = 0.0
    pct_event: float = 0.0
    woe: float = 0.0
    iv_contribution: float = 0.0
    event_rate: float = 0.0


@dataclass
class WOEResult:
    """
    Complete WOE transformation result for a variable.
    """
    variable_name: str
    variable_type: str
    bins: List[WOEBin] = field(default_factory=list)
    iv: float = 0.0
    total_non_event: float = 0.0
    total_event: float = 0.0
    is_monotonic: bool = False


def weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray,
    sample_weight: np.ndarray
) -> np.ndarray:
    """
    Calculate weighted quantiles.
    
    FIX #2: Proper weighted quantile implementation.
    
    Parameters:
    -----------
    values : np.ndarray
        Values to compute quantiles for
    quantiles : np.ndarray
        Quantile levels (0-1)
    sample_weight : np.ndarray
        Weights for each value
        
    Returns:
    --------
    np.ndarray
        Quantile values
    """
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    
    # Handle edge cases
    if len(values) == 0:
        return np.array([])
    
    # Remove NaN values
    mask = ~np.isnan(values)
    values = values[mask]
    sample_weight = sample_weight[mask]
    
    if len(values) == 0:
        return np.array([])
    
    # Sort by values
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    
    # Compute cumulative weights (ensure float for division)
    cum_weights = np.cumsum(sample_weight).astype(float)
    cum_weights = cum_weights / cum_weights[-1]
    
    # Interpolate to find quantiles
    return np.interp(quantiles, cum_weights, values)


class DataExtractor:
    """
    Equivalent to SAS macro UNI_ETL.
    
    FIX #1: Now accepts binary target (0/1) directly, no G/B confusion.
    """
    
    def __init__(
        self,
        target_col: str = 'target',
        weight_col: str = None,
        id_cols: List[str] = None,
        event_label: str = 'event'  # FIX #9: Configurable terminology
    ):
        """
        Initialize DataExtractor.
        
        Parameters:
        -----------
        target_col : str
            Name of the binary target column (must be 0/1)
        weight_col : str, optional
            Name of the weight column
        id_cols : List[str], optional
            ID columns to keep
        event_label : str
            Label for the positive event (for logging/display)
        """
        self.target_col = target_col
        self.weight_col = weight_col
        self.id_cols = id_cols or []
        self.event_label = event_label
        
        # Results storage
        self.variable_summaries: Dict[str, VariableSummary] = {}
        self.mart_summary: Dict[str, float] = {}
        self.extracted_data: pd.DataFrame = None
        
    def extract(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        time_col: str = None
    ) -> pd.DataFrame:
        """
        Extract and profile data.
        
        FIX #1: Expects target_col to already be binary (0/1).
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset with binary target (0/1)
        feature_cols : List[str]
            List of feature columns to analyze
        time_col : str, optional
            Time column to preserve for time-based splits
            
        Returns:
        --------
        pd.DataFrame
            Filtered and prepared dataset
        """
        logger.info(f"DataExtractor.extract() started at {datetime.now()}")
        logger.info(f"Input data shape: {data.shape}")
        
        # Validate feature columns exist
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        
        # Validate target column
        if self.target_col not in data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
        
        # FIX #1: Validate target is binary
        unique_targets = data[self.target_col].dropna().unique()
        if not set(unique_targets).issubset({0, 1}):
            raise ValueError(
                f"Target column '{self.target_col}' must be binary (0/1). "
                f"Found values: {unique_targets}. "
                "Please preprocess your target before calling extract()."
            )
        
        # Filter to valid records (non-null target)
        mask = data[self.target_col].notna()
        
        if self.weight_col and self.weight_col in data.columns:
            mask = mask & (data[self.weight_col] > 0)
        
        filtered_data = data.loc[mask].copy()
        logger.info(f"Filtered data shape: {filtered_data.shape}")
        
        # Create weight column if not specified
        if self.weight_col and self.weight_col in filtered_data.columns:
            filtered_data['weight'] = filtered_data[self.weight_col]
        else:
            filtered_data['weight'] = 1.0
        
        # Target is already binary
        filtered_data['target_binary'] = filtered_data[self.target_col].astype(int)
        
        # Calculate mart-level summary
        self._calculate_mart_summary(filtered_data)
        
        # Calculate variable summaries
        self._calculate_variable_summaries(filtered_data, feature_cols)
        
        # Store extracted data - include time_col if specified
        keep_cols = self.id_cols + feature_cols + [self.target_col, 'weight', 'target_binary']
        if time_col and time_col in filtered_data.columns:
            keep_cols.append(time_col)
        keep_cols = [c for c in keep_cols if c in filtered_data.columns]
        self.extracted_data = filtered_data[keep_cols].copy()
        
        logger.info(f"DataExtractor.extract() completed at {datetime.now()}")
        return self.extracted_data
    
    def _calculate_mart_summary(self, data: pd.DataFrame) -> None:
        """Calculate overall mart statistics."""
        event_mask = data['target_binary'] == 1
        non_event_mask = data['target_binary'] == 0
        
        self.mart_summary = {
            'mart_sum_wt_non_event': data.loc[non_event_mask, 'weight'].sum(),
            'mart_sum_wt_event': data.loc[event_mask, 'weight'].sum(),
            'mart_sum_wt': data['weight'].sum(),
            'mart_n_records': len(data),
            'mart_n_non_event': non_event_mask.sum(),
            'mart_n_event': event_mask.sum()
        }
        self.mart_summary['mart_event_rate'] = (
            self.mart_summary['mart_sum_wt_event'] / self.mart_summary['mart_sum_wt']
        )
        
        logger.info(f"Mart Summary: {self.mart_summary['mart_n_records']} records, "
                   f"Event Rate: {self.mart_summary['mart_event_rate']:.4f}")
    
    def _calculate_variable_summaries(
        self,
        data: pd.DataFrame,
        feature_cols: List[str]
    ) -> None:
        """Calculate summary statistics for each variable."""
        for col in feature_cols:
            dtype = 'numeric' if pd.api.types.is_numeric_dtype(data[col]) else 'categorical'
            
            summary = VariableSummary(name=col, dtype=dtype)
            
            # Overall statistics
            summary.n_distinct = data[col].nunique(dropna=False)
            summary.sum_wt = data['weight'].sum()
            
            event_mask = data['target_binary'] == 1
            non_event_mask = data['target_binary'] == 0
            
            summary.sum_wt_non_event = data.loc[non_event_mask, 'weight'].sum()
            summary.sum_wt_event = data.loc[event_mask, 'weight'].sum()
            summary.event_rate = summary.sum_wt_event / summary.sum_wt if summary.sum_wt > 0 else 0
            
            # Missing value statistics
            missing_mask = data[col].isna()
            summary.missing_freq = missing_mask.sum()
            summary.missing_wt = data.loc[missing_mask, 'weight'].sum()
            summary.missing_wt_non_event = data.loc[missing_mask & non_event_mask, 'weight'].sum()
            summary.missing_wt_event = data.loc[missing_mask & event_mask, 'weight'].sum()
            if summary.missing_wt > 0:
                summary.missing_event_rate = summary.missing_wt_event / summary.missing_wt
            
            # Non-missing statistics
            nonmissing_mask = ~missing_mask
            summary.nonmissing_sum_wt = data.loc[nonmissing_mask, 'weight'].sum()
            summary.nonmissing_sum_wt_non_event = data.loc[nonmissing_mask & non_event_mask, 'weight'].sum()
            summary.nonmissing_sum_wt_event = data.loc[nonmissing_mask & event_mask, 'weight'].sum()
            if summary.nonmissing_sum_wt > 0:
                summary.nonmissing_event_rate = summary.nonmissing_sum_wt_event / summary.nonmissing_sum_wt
            
            # Zero value statistics (for numeric only)
            if dtype == 'numeric':
                zero_mask = data[col] == 0
                summary.zero_freq = zero_mask.sum()
                summary.zero_wt = data.loc[zero_mask, 'weight'].sum()
                summary.zero_wt_non_event = data.loc[zero_mask & non_event_mask, 'weight'].sum()
                summary.zero_wt_event = data.loc[zero_mask & event_mask, 'weight'].sum()
            
            self.variable_summaries[col] = summary
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Return variable summaries as a DataFrame."""
        records = []
        for name, summary in self.variable_summaries.items():
            records.append({
                'name': summary.name,
                'dtype': summary.dtype,
                'n_distinct': summary.n_distinct,
                'sum_wt': summary.sum_wt,
                'sum_wt_event': summary.sum_wt_event,
                'sum_wt_non_event': summary.sum_wt_non_event,
                'event_rate': summary.event_rate,
                'missing_freq': summary.missing_freq,
                'missing_wt': summary.missing_wt,
                'missing_event_rate': summary.missing_event_rate,
                'nonmissing_sum_wt': summary.nonmissing_sum_wt,
                'nonmissing_event_rate': summary.nonmissing_event_rate,
                'iv': summary.iv
            })
        return pd.DataFrame(records)


class WOETransformer:
    """
    Equivalent to SAS macros UNI_WOE and UNI_WOE_NUM.
    
    FIXES:
    - #2: Weighted quantile binning
    - #3: min_bin_pct enforcement with merging
    - #4: Proper missing handling for categorical
    - #8: Monotonic binning option
    """
    
    EPSILON = 1e-10
    
    def __init__(
        self,
        n_quantiles: int = 10,
        max_char_levels: int = 15,
        min_bin_pct: float = 0.05,
        min_bin_count: int = 50,
        handle_missing: str = 'separate_bin',
        enforce_monotonic: bool = False,
        monotonic_direction: str = 'auto'  # 'auto', 'increasing', 'decreasing'
    ):
        """
        Initialize WOETransformer.
        
        Parameters:
        -----------
        n_quantiles : int
            Number of quantile bins for numeric variables
        max_char_levels : int
            Maximum levels for categorical variables
        min_bin_pct : float
            FIX #3: Minimum percentage of total weight in a bin
        min_bin_count : int
            FIX #3: Minimum count of records in a bin
        handle_missing : str
            How to handle missing values: 'separate_bin' or 'most_frequent'
        enforce_monotonic : bool
            FIX #8: Whether to enforce monotonic WOE for numeric variables
        monotonic_direction : str
            FIX #8: Direction for monotonicity ('auto', 'increasing', 'decreasing')
        """
        self.n_quantiles = n_quantiles
        self.max_char_levels = max_char_levels
        self.min_bin_pct = min_bin_pct
        self.min_bin_count = min_bin_count
        self.handle_missing = handle_missing
        self.enforce_monotonic = enforce_monotonic
        self.monotonic_direction = monotonic_direction
        
        # Results storage
        self.woe_results: Dict[str, WOEResult] = {}
        self.iv_summary: pd.DataFrame = None
        
    def fit(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        weight_col: str = None
    ) -> 'WOETransformer':
        """
        Fit WOE transformation for all specified features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        feature_cols : List[str]
            List of features to transform
        target_col : str
            Binary target column (0/1)
        weight_col : str, optional
            Weight column
            
        Returns:
        --------
        self
        """
        logger.info(f"WOETransformer.fit() started at {datetime.now()}")
        
        weights = data[weight_col] if weight_col and weight_col in data.columns else pd.Series(1.0, index=data.index)
        
        # Calculate totals
        total_non_event = (weights * (data[target_col] == 0)).sum()
        total_event = (weights * (data[target_col] == 1)).sum()
        total_weight = weights.sum()
        
        iv_records = []
        
        for col in feature_cols:
            dtype = 'numeric' if pd.api.types.is_numeric_dtype(data[col]) else 'categorical'
            
            logger.info(f"Processing variable: {col} (type: {dtype})")
            
            if dtype == 'numeric':
                woe_result = self._fit_numeric(
                    data[col], data[target_col], weights, col, 
                    total_non_event, total_event, total_weight
                )
            else:
                woe_result = self._fit_categorical(
                    data[col], data[target_col], weights, col, 
                    total_non_event, total_event, total_weight
                )
            
            self.woe_results[col] = woe_result
            iv_records.append({
                'variable': col,
                'type': dtype,
                'iv': woe_result.iv,
                'n_bins': len(woe_result.bins),
                'is_monotonic': woe_result.is_monotonic
            })
            
            logger.info(f"  -> IV = {woe_result.iv:.4f}, Bins = {len(woe_result.bins)}, "
                       f"Monotonic = {woe_result.is_monotonic}")
        
        self.iv_summary = pd.DataFrame(iv_records).sort_values('iv', ascending=False)
        logger.info(f"WOETransformer.fit() completed at {datetime.now()}")
        
        return self
    
    def _fit_numeric(
        self,
        series: pd.Series,
        target: pd.Series,
        weights: pd.Series,
        var_name: str,
        total_non_event: float,
        total_event: float,
        total_weight: float
    ) -> WOEResult:
        """
        Fit WOE for numeric variable with weighted quantile binning.
        
        FIX #2: Uses proper weighted quantiles.
        FIX #3: Enforces min_bin_pct through merging.
        FIX #8: Optional monotonic binning.
        """
        result = WOEResult(
            variable_name=var_name,
            variable_type='numeric',
            total_non_event=total_non_event,
            total_event=total_event
        )
        
        # Separate missing values
        missing_mask = series.isna()
        non_missing_idx = ~missing_mask
        non_missing_series = series[non_missing_idx].values
        non_missing_target = target[non_missing_idx].values
        non_missing_weights = weights[non_missing_idx].values
        
        # FIX #2: Determine bin boundaries using WEIGHTED quantiles
        n_distinct = len(np.unique(non_missing_series))
        
        if n_distinct <= self.n_quantiles:
            # Use distinct values as boundaries
            boundaries = sorted(np.unique(non_missing_series))[:-1]  # Exclude max
        else:
            # Use weighted quantile-based boundaries
            quantiles = np.linspace(0, 1, self.n_quantiles + 1)[1:-1]
            boundaries = list(np.unique(weighted_quantile(
                non_missing_series, quantiles, non_missing_weights
            )))
        
        # Create initial bins
        bins = []
        bin_id = 0
        
        # Handle missing values first (if any)
        if missing_mask.sum() > 0:
            missing_non_event = (weights[missing_mask] * (target[missing_mask] == 0)).sum()
            missing_event = (weights[missing_mask] * (target[missing_mask] == 1)).sum()
            
            woe_bin = WOEBin(
                bin_id=bin_id,
                bin_label='Missing',
                count=weights[missing_mask].sum(),
                count_non_event=missing_non_event,
                count_event=missing_event
            )
            bins.append(woe_bin)
            bin_id += 1
        
        # Create numeric range bins
        prev_bound = float('-inf')
        for i, bound in enumerate(list(boundaries) + [float('inf')]):
            if bound == float('inf'):
                mask = non_missing_series > prev_bound
                label = f">{prev_bound:.6g}" if prev_bound != float('-inf') else "All"
            elif i == 0:
                mask = non_missing_series <= bound
                label = f"<={bound:.6g}"
            else:
                mask = (non_missing_series > prev_bound) & (non_missing_series <= bound)
                label = f"({prev_bound:.6g}, {bound:.6g}]"
            
            if mask.sum() > 0:
                bin_non_event = (non_missing_weights[mask] * (non_missing_target[mask] == 0)).sum()
                bin_event = (non_missing_weights[mask] * (non_missing_target[mask] == 1)).sum()
                
                woe_bin = WOEBin(
                    bin_id=bin_id,
                    bin_label=label,
                    lower_bound=prev_bound if prev_bound != float('-inf') else None,
                    upper_bound=bound if bound != float('inf') else None,
                    lower_inclusive=False,
                    upper_inclusive=True,
                    count=non_missing_weights[mask].sum(),
                    count_non_event=bin_non_event,
                    count_event=bin_event
                )
                bins.append(woe_bin)
                bin_id += 1
            
            prev_bound = bound
        
        # FIX #3: Merge sparse bins
        bins = self._merge_sparse_bins(bins, total_weight, is_numeric=True)
        
        # Calculate WOE and IV
        bins = self._calculate_woe(bins, total_non_event, total_event)
        
        # FIX #8: Enforce monotonicity if requested
        if self.enforce_monotonic:
            bins = self._enforce_monotonicity(bins, total_non_event, total_event)
        
        # Check if WOE is monotonic
        result.is_monotonic = self._check_monotonicity(bins)
        
        # Re-assign bin IDs after merging
        for i, bin in enumerate(bins):
            bin.bin_id = i
        
        result.bins = bins
        result.iv = sum(b.iv_contribution for b in bins)
        
        return result
    
    def _fit_categorical(
        self,
        series: pd.Series,
        target: pd.Series,
        weights: pd.Series,
        var_name: str,
        total_non_event: float,
        total_event: float,
        total_weight: float
    ) -> WOEResult:
        """
        Fit WOE for categorical variable.
        
        FIX #4: Always preserves missing as separate bin when present.
        FIX #3: Enforces min_bin_pct.
        """
        result = WOEResult(
            variable_name=var_name,
            variable_type='categorical',
            total_non_event=total_non_event,
            total_event=total_event
        )
        
        bins = []
        bin_id = 0
        
        # Check for missing values
        has_missing = series.isna().any()
        
        # Get value counts (excluding missing)
        non_missing_series = series.fillna('__MISSING__')
        value_counts = non_missing_series.value_counts()
        
        # FIX #4: Determine levels to keep, always preserving missing if present
        all_levels = value_counts.index.tolist()
        
        if len(all_levels) > self.max_char_levels:
            # Keep top levels, but always preserve missing
            top_levels = value_counts.head(self.max_char_levels - 1).index.tolist()
            
            # FIX #4: Force missing into bins if it exists
            if has_missing and '__MISSING__' not in top_levels:
                # Remove the last top level and add missing
                top_levels = top_levels[:-1] + ['__MISSING__']
            
            other_levels = [l for l in all_levels if l not in top_levels and l != '__MISSING__']
        else:
            top_levels = all_levels
            other_levels = []
        
        # Create bins for each level (excluding missing and other for now)
        for level in sorted([l for l in top_levels if l not in ['__MISSING__', '__OTHER__']]):
            mask = series == level
            if mask.sum() > 0:
                bin_non_event = (weights[mask] * (target[mask] == 0)).sum()
                bin_event = (weights[mask] * (target[mask] == 1)).sum()
                
                woe_bin = WOEBin(
                    bin_id=bin_id,
                    bin_label=str(level),
                    values=[level],
                    count=weights[mask].sum(),
                    count_non_event=bin_non_event,
                    count_event=bin_event
                )
                bins.append(woe_bin)
                bin_id += 1
        
        # Handle missing values (FIX #4: Always create separate bin if missing exists)
        if has_missing:
            mask = series.isna()
            if mask.sum() > 0:
                bin_non_event = (weights[mask] * (target[mask] == 0)).sum()
                bin_event = (weights[mask] * (target[mask] == 1)).sum()
                
                woe_bin = WOEBin(
                    bin_id=bin_id,
                    bin_label='Missing',
                    values=[None],
                    count=weights[mask].sum(),
                    count_non_event=bin_non_event,
                    count_event=bin_event
                )
                bins.append(woe_bin)
                bin_id += 1
        
        # Handle "Other" group
        if other_levels:
            mask = series.isin(other_levels)
            if mask.sum() > 0:
                bin_non_event = (weights[mask] * (target[mask] == 0)).sum()
                bin_event = (weights[mask] * (target[mask] == 1)).sum()
                
                woe_bin = WOEBin(
                    bin_id=bin_id,
                    bin_label='Other',
                    values=other_levels,
                    count=weights[mask].sum(),
                    count_non_event=bin_non_event,
                    count_event=bin_event
                )
                bins.append(woe_bin)
                bin_id += 1
        
        # FIX #3: Merge sparse bins (for categorical, fold into "Other")
        bins = self._merge_sparse_bins(bins, total_weight, is_numeric=False)
        
        # Calculate WOE and IV
        bins = self._calculate_woe(bins, total_non_event, total_event)
        
        # Re-assign bin IDs
        for i, bin in enumerate(bins):
            bin.bin_id = i
        
        result.bins = bins
        result.iv = sum(b.iv_contribution for b in bins)
        result.is_monotonic = True  # N/A for categorical
        
        return result
    
    def _merge_sparse_bins(
        self,
        bins: List[WOEBin],
        total_weight: float,
        is_numeric: bool
    ) -> List[WOEBin]:
        """
        FIX #3: Merge bins that don't meet minimum thresholds.
        
        For numeric: merge with adjacent bin
        For categorical: merge into "Other"
        """
        if len(bins) <= 2:
            return bins
        
        min_weight = total_weight * self.min_bin_pct
        
        if is_numeric:
            # Merge with adjacent bins
            merged = True
            while merged and len(bins) > 2:
                merged = False
                new_bins = []
                i = 0
                
                while i < len(bins):
                    current_bin = bins[i]
                    
                    # Skip missing bin from merging consideration
                    if current_bin.bin_label == 'Missing':
                        new_bins.append(current_bin)
                        i += 1
                        continue
                    
                    # Check if current bin is too small
                    if current_bin.count < min_weight or current_bin.count < self.min_bin_count:
                        # Find best neighbor to merge with
                        if i == 0 or (i > 0 and bins[i-1].bin_label == 'Missing'):
                            # Merge with next
                            if i + 1 < len(bins) and bins[i+1].bin_label != 'Missing':
                                merged_bin = self._merge_two_bins(current_bin, bins[i+1])
                                new_bins.append(merged_bin)
                                i += 2
                                merged = True
                                continue
                        else:
                            # Merge with previous (which is already in new_bins)
                            if new_bins and new_bins[-1].bin_label != 'Missing':
                                prev_bin = new_bins.pop()
                                merged_bin = self._merge_two_bins(prev_bin, current_bin)
                                new_bins.append(merged_bin)
                                i += 1
                                merged = True
                                continue
                    
                    new_bins.append(current_bin)
                    i += 1
                
                bins = new_bins
        else:
            # For categorical, merge small bins into "Other"
            other_bin = None
            kept_bins = []
            
            for bin in bins:
                if bin.bin_label in ['Missing', 'Other']:
                    if bin.bin_label == 'Other':
                        other_bin = bin
                    else:
                        kept_bins.append(bin)
                elif bin.count < min_weight or bin.count < self.min_bin_count:
                    # Merge into Other
                    if other_bin is None:
                        other_bin = WOEBin(
                            bin_id=len(kept_bins),
                            bin_label='Other',
                            values=[],
                            count=0,
                            count_non_event=0,
                            count_event=0
                        )
                    other_bin.count += bin.count
                    other_bin.count_non_event += bin.count_non_event
                    other_bin.count_event += bin.count_event
                    other_bin.values = (other_bin.values or []) + (bin.values or [bin.bin_label])
                else:
                    kept_bins.append(bin)
            
            if other_bin is not None and other_bin.count > 0:
                kept_bins.append(other_bin)
            
            bins = kept_bins
        
        return bins
    
    def _merge_two_bins(self, bin1: WOEBin, bin2: WOEBin) -> WOEBin:
        """Merge two adjacent numeric bins."""
        # Determine which is lower/upper
        if bin1.upper_bound is None or (bin2.lower_bound is not None and bin1.upper_bound <= bin2.lower_bound):
            lower_bin, upper_bin = bin1, bin2
        else:
            lower_bin, upper_bin = bin2, bin1
        
        # Create label handling None bounds
        lower_str = f"{lower_bin.lower_bound:.6g}" if lower_bin.lower_bound is not None else "-inf"
        upper_str = f"{upper_bin.upper_bound:.6g}" if upper_bin.upper_bound is not None else "inf"
        
        return WOEBin(
            bin_id=lower_bin.bin_id,
            bin_label=f"({lower_str}, {upper_str}]",
            lower_bound=lower_bin.lower_bound,
            upper_bound=upper_bin.upper_bound,
            lower_inclusive=lower_bin.lower_inclusive,
            upper_inclusive=upper_bin.upper_inclusive,
            count=lower_bin.count + upper_bin.count,
            count_non_event=lower_bin.count_non_event + upper_bin.count_non_event,
            count_event=lower_bin.count_event + upper_bin.count_event
        )
    
    def _calculate_woe(
        self,
        bins: List[WOEBin],
        total_non_event: float,
        total_event: float
    ) -> List[WOEBin]:
        """Calculate WOE and IV for bins."""
        for bin in bins:
            # Avoid division by zero
            pct_non_event = (bin.count_non_event + self.EPSILON) / (total_non_event + self.EPSILON)
            pct_event = (bin.count_event + self.EPSILON) / (total_event + self.EPSILON)
            
            bin.pct_non_event = pct_non_event
            bin.pct_event = pct_event
            
            # WOE formula: ln(Dist_NonEvent / Dist_Event)
            bin.woe = np.log(pct_non_event / pct_event)
            
            # IV contribution
            bin.iv_contribution = (pct_non_event - pct_event) * bin.woe
            
            # Event rate
            total_bin = bin.count_non_event + bin.count_event
            bin.event_rate = bin.count_event / total_bin if total_bin > 0 else 0
        
        return bins
    
    def _check_monotonicity(self, bins: List[WOEBin]) -> bool:
        """Check if WOE values are monotonic (excluding missing)."""
        numeric_bins = [b for b in bins if b.bin_label != 'Missing']
        if len(numeric_bins) <= 1:
            return True
        
        woe_values = [b.woe for b in numeric_bins]
        
        # Check increasing or decreasing
        is_increasing = all(woe_values[i] <= woe_values[i+1] for i in range(len(woe_values)-1))
        is_decreasing = all(woe_values[i] >= woe_values[i+1] for i in range(len(woe_values)-1))
        
        return is_increasing or is_decreasing
    
    def _enforce_monotonicity(
        self,
        bins: List[WOEBin],
        total_non_event: float,
        total_event: float
    ) -> List[WOEBin]:
        """
        FIX #8: Enforce monotonic WOE by merging adjacent bins.
        """
        # Separate missing bin
        missing_bin = None
        numeric_bins = []
        for b in bins:
            if b.bin_label == 'Missing':
                missing_bin = b
            else:
                numeric_bins.append(b)
        
        if len(numeric_bins) <= 2:
            return bins
        
        # Determine target direction
        if self.monotonic_direction == 'auto':
            # Use correlation between bin order and WOE
            woe_values = [b.woe for b in numeric_bins]
            correlation = np.corrcoef(range(len(woe_values)), woe_values)[0, 1]
            target_increasing = correlation >= 0
        else:
            target_increasing = self.monotonic_direction == 'increasing'
        
        # Iteratively merge bins that violate monotonicity
        max_iterations = len(numeric_bins) * 2
        iteration = 0
        
        while iteration < max_iterations and len(numeric_bins) > 2:
            violation_found = False
            
            for i in range(len(numeric_bins) - 1):
                current_woe = numeric_bins[i].woe
                next_woe = numeric_bins[i + 1].woe
                
                if target_increasing and current_woe > next_woe:
                    violation_found = True
                elif not target_increasing and current_woe < next_woe:
                    violation_found = True
                
                if violation_found:
                    # Merge these two bins
                    merged = self._merge_two_bins(numeric_bins[i], numeric_bins[i + 1])
                    numeric_bins = numeric_bins[:i] + [merged] + numeric_bins[i+2:]
                    # Recalculate WOE for merged bin
                    numeric_bins = self._calculate_woe(numeric_bins, total_non_event, total_event)
                    break
            
            if not violation_found:
                break
            
            iteration += 1
        
        # Reconstruct bins list
        result = []
        if missing_bin:
            result.append(missing_bin)
        result.extend(numeric_bins)
        
        return result
    
    def transform(
        self,
        data: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> pd.DataFrame:
        """Transform features to WOE values."""
        if feature_cols is None:
            feature_cols = list(self.woe_results.keys())
        
        result = data.copy()
        
        for col in feature_cols:
            if col not in self.woe_results:
                logger.warning(f"Variable {col} not fitted, skipping")
                continue
            
            woe_result = self.woe_results[col]
            woe_col = f"{col}_WOE"
            
            if woe_result.variable_type == 'numeric':
                result[woe_col] = self._transform_numeric(data[col], woe_result)
            else:
                result[woe_col] = self._transform_categorical(data[col], woe_result)
        
        return result
    
    def _transform_numeric(self, series: pd.Series, woe_result: WOEResult) -> pd.Series:
        """Transform numeric variable using fitted WOE bins."""
        result = pd.Series(index=series.index, dtype=float)
        
        for bin in woe_result.bins:
            if bin.bin_label == 'Missing':
                mask = series.isna()
            elif bin.lower_bound is None and bin.upper_bound is not None:
                mask = series <= bin.upper_bound
            elif bin.lower_bound is not None and bin.upper_bound is None:
                mask = series > bin.lower_bound
            elif bin.lower_bound is not None and bin.upper_bound is not None:
                mask = (series > bin.lower_bound) & (series <= bin.upper_bound)
            else:
                continue
            
            result.loc[mask] = bin.woe
        
        # Handle any unmapped values with overall mean WOE
        unmapped = result.isna() & ~series.isna()
        if unmapped.sum() > 0:
            mean_woe = np.mean([b.woe for b in woe_result.bins])
            result.loc[unmapped] = mean_woe
        
        return result
    
    def _transform_categorical(self, series: pd.Series, woe_result: WOEResult) -> pd.Series:
        """Transform categorical variable using fitted WOE bins."""
        woe_lookup = {}
        other_woe = 0.0
        missing_woe = 0.0
        
        for bin in woe_result.bins:
            if bin.bin_label == 'Missing':
                missing_woe = bin.woe
            elif bin.bin_label == 'Other':
                other_woe = bin.woe
            else:
                for val in (bin.values or [bin.bin_label]):
                    if val is not None:
                        woe_lookup[val] = bin.woe
        
        def map_woe(val):
            if pd.isna(val):
                return missing_woe
            if val in woe_lookup:
                return woe_lookup[val]
            return other_woe
        
        return series.apply(map_woe)
    
    def get_iv_summary(self) -> pd.DataFrame:
        """Return IV summary for all variables."""
        return self.iv_summary.copy()
    
    def get_woe_table(self, variable: str) -> pd.DataFrame:
        """Get WOE table for a specific variable."""
        if variable not in self.woe_results:
            raise ValueError(f"Variable {variable} not found")
        
        result = self.woe_results[variable]
        records = []
        for bin in result.bins:
            records.append({
                'bin_id': bin.bin_id,
                'bin_label': bin.bin_label,
                'count': bin.count,
                'count_non_event': bin.count_non_event,
                'count_event': bin.count_event,
                'pct_non_event': bin.pct_non_event,
                'pct_event': bin.pct_event,
                'woe': bin.woe,
                'iv_contribution': bin.iv_contribution,
                'event_rate': bin.event_rate
            })
        
        df = pd.DataFrame(records)
        df['variable'] = variable
        return df
    
    def calculate_psi(
        self,
        baseline_data: pd.DataFrame,
        comparison_data: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        FIX #10: Calculate Population Stability Index (PSI) for WOE bins.
        
        PSI < 0.1: No significant shift
        0.1 <= PSI < 0.25: Moderate shift, investigate
        PSI >= 0.25: Major shift, model may need retraining
        """
        if feature_cols is None:
            feature_cols = list(self.woe_results.keys())
        
        psi_records = []
        
        for col in feature_cols:
            if col not in self.woe_results:
                continue
            
            woe_result = self.woe_results[col]
            
            # Transform both datasets
            baseline_woe = self._transform_numeric(baseline_data[col], woe_result) if woe_result.variable_type == 'numeric' else self._transform_categorical(baseline_data[col], woe_result)
            comparison_woe = self._transform_numeric(comparison_data[col], woe_result) if woe_result.variable_type == 'numeric' else self._transform_categorical(comparison_data[col], woe_result)
            
            # Calculate distribution in each bin
            total_psi = 0.0
            for bin in woe_result.bins:
                baseline_pct = (baseline_woe == bin.woe).mean() + self.EPSILON
                comparison_pct = (comparison_woe == bin.woe).mean() + self.EPSILON
                
                bin_psi = (comparison_pct - baseline_pct) * np.log(comparison_pct / baseline_pct)
                total_psi += bin_psi
            
            psi_records.append({
                'variable': col,
                'psi': total_psi,
                'status': 'OK' if total_psi < 0.1 else ('Warning' if total_psi < 0.25 else 'Critical')
            })
        
        return pd.DataFrame(psi_records).sort_values('psi', ascending=False)


class PropensityModel:
    """
    Propensity scoring model using logistic regression.
    
    FIX #5: Consistent score scaling and sign conventions.
    FIX #6: Proper calibration with cross-validation.
    """
    
    def __init__(
        self,
        iv_min: float = 0.02,
        regularization: str = 'l2',
        C: float = 1.0,
        max_iter: int = 1000,
        calibrate: bool = False,
        calibration_method: str = 'sigmoid',
        calibration_cv: int = 5,
        points_for_1to1: int = 600,
        points_to_double_odds: int = 20,
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_BETTER
    ):
        """
        Initialize PropensityModel.
        
        Parameters:
        -----------
        iv_min : float
            Minimum IV threshold for variable selection
        regularization : str
            Regularization type ('l1', 'l2', or 'none')
        C : float
            Inverse regularization strength
        max_iter : int
            Maximum iterations for solver
        calibrate : bool
            Whether to calibrate probabilities
        calibration_method : str
            Calibration method ('sigmoid' or 'isotonic')
        calibration_cv : int
            FIX #6: Number of CV folds for calibration
        points_for_1to1 : int
            Score at 1:1 odds
        points_to_double_odds : int
            Points to double odds
        score_direction : ScoreDirection
            FIX #5: Direction convention for scores
        """
        self.iv_min = iv_min
        self.regularization = regularization
        self.C = C
        self.max_iter = max_iter
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.calibration_cv = calibration_cv
        self.points_for_1to1 = points_for_1to1
        self.points_to_double_odds = points_to_double_odds
        self.score_direction = score_direction
        
        # Model components
        self.model = None
        self.calibrated_model = None
        self.selected_features: List[str] = []
        self.coefficients: Dict[str, float] = {}
        self.intercept: float = 0.0
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series = None,
        iv_summary: pd.DataFrame = None
    ) -> 'PropensityModel':
        """
        Fit propensity model.
        
        FIX #6: Uses cross-validation for calibration to avoid leakage.
        """
        logger.info(f"PropensityModel.fit() started at {datetime.now()}")
        
        # Feature selection based on IV threshold
        if iv_summary is not None:
            high_iv_vars = iv_summary[iv_summary['iv'] >= self.iv_min]['variable'].tolist()
            self.selected_features = [f"{v}_WOE" for v in high_iv_vars if f"{v}_WOE" in X.columns]
        else:
            self.selected_features = [c for c in X.columns if c.endswith('_WOE')]
        
        logger.info(f"Selected {len(self.selected_features)} features with IV >= {self.iv_min}")
        
        if len(self.selected_features) == 0:
            raise ValueError("No features selected. Lower iv_min threshold or check data.")
        
        X_selected = X[self.selected_features].copy().fillna(0)
        
        # Configure logistic regression
        penalty = None if self.regularization == 'none' else self.regularization
        solver = 'lbfgs' if penalty in [None, 'l2'] else 'saga'
        
        if self.calibrate:
            # FIX #6: Use cross-validation for calibration
            base_model = LogisticRegression(
                penalty=penalty,
                C=self.C,
                max_iter=self.max_iter,
                solver=solver,
                random_state=42
            )
            
            self.calibrated_model = CalibratedClassifierCV(
                base_model,
                method=self.calibration_method,
                cv=StratifiedKFold(n_splits=self.calibration_cv, shuffle=True, random_state=42)
            )
            
            if sample_weight is not None:
                self.calibrated_model.fit(X_selected, y, sample_weight=sample_weight)
            else:
                self.calibrated_model.fit(X_selected, y)
            
            # Get coefficients from base estimators (average across CV folds)
            # Access the base estimator within each _CalibratedClassifier
            coef_list = []
            intercept_list = []
            for calibrated_classifier in self.calibrated_model.calibrated_classifiers_:
                # The base estimator is stored in .estimator attribute
                base_est = calibrated_classifier.estimator
                coef_list.append(base_est.coef_[0])
                intercept_list.append(base_est.intercept_[0])
            
            coefs = np.mean(coef_list, axis=0)
            intercepts = np.mean(intercept_list)
            
            self.intercept = intercepts
            self.coefficients = dict(zip(self.selected_features, coefs))
            
            # Also fit a regular model for coefficient access and decision_function
            self.model = LogisticRegression(
                penalty=penalty,
                C=self.C,
                max_iter=self.max_iter,
                solver=solver,
                random_state=42
            )
            if sample_weight is not None:
                self.model.fit(X_selected, y, sample_weight=sample_weight)
            else:
                self.model.fit(X_selected, y)
        else:
            self.model = LogisticRegression(
                penalty=penalty,
                C=self.C,
                max_iter=self.max_iter,
                solver=solver,
                random_state=42
            )
            
            if sample_weight is not None:
                self.model.fit(X_selected, y, sample_weight=sample_weight)
            else:
                self.model.fit(X_selected, y)
            
            self.intercept = self.model.intercept_[0]
            self.coefficients = dict(zip(self.selected_features, self.model.coef_[0]))
        
        logger.info(f"PropensityModel.fit() completed at {datetime.now()}")
        logger.info(f"Intercept: {self.intercept:.4f}")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict propensity scores (probabilities).
        
        Returns probability of positive class (event).
        """
        X_selected = X[self.selected_features].copy().fillna(0)
        
        if self.calibrate and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_selected)[:, 1]
        else:
            return self.model.predict_proba(X_selected)[:, 1]
    
    def predict_logit(self, X: pd.DataFrame) -> np.ndarray:
        """Predict raw logit (log-odds) values."""
        X_selected = X[self.selected_features].copy().fillna(0)
        return self.model.decision_function(X_selected)
    
    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict scaled score.
        
        FIX #5: Consistent sign convention based on score_direction.
        
        HIGHER_IS_BETTER: Higher score = higher P(event)
        HIGHER_IS_SAFER: Higher score = lower P(event) (traditional credit score)
        """
        logit = self.predict_logit(X)
        
        if self.score_direction == ScoreDirection.HIGHER_IS_BETTER:
            # Higher score = higher probability of event
            score = self.points_for_1to1 + self.points_to_double_odds * logit / np.log(2)
        else:
            # Higher score = lower probability of event (credit score convention)
            score = self.points_for_1to1 + self.points_to_double_odds * (-logit) / np.log(2)
        
        return score
    
    def score_to_probability(self, score: np.ndarray) -> np.ndarray:
        """
        Convert score back to probability.
        
        FIX #5: Matches the sign convention used in predict_score().
        """
        if self.score_direction == ScoreDirection.HIGHER_IS_BETTER:
            logit = (score - self.points_for_1to1) * np.log(2) / self.points_to_double_odds
        else:
            logit = -(score - self.points_for_1to1) * np.log(2) / self.points_to_double_odds
        
        probability = 1 / (1 + np.exp(-logit))
        return probability
    
    def get_coefficients_df(self) -> pd.DataFrame:
        """Get model coefficients as DataFrame."""
        records = [{'variable': 'Intercept', 'coefficient': self.intercept}]
        for var, coef in self.coefficients.items():
            records.append({'variable': var, 'coefficient': coef})
        return pd.DataFrame(records)


class ModelEvaluator:
    """
    Model evaluation metrics and visualizations.
    
    FIX #9: Uses configurable terminology.
    """
    
    def __init__(self, event_label: str = 'event'):
        self.event_label = event_label
        self.metrics: Dict[str, float] = {}
        self.curves: Dict[str, Any] = {}
        
    def evaluate(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        sample_weight: pd.Series = None,
        score: np.ndarray = None
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        logger.info("Evaluating model performance...")
        
        # AUC / Gini
        auc = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
        gini = 2 * auc - 1
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob, sample_weight=sample_weight)
        
        # KS statistic
        ks_statistic = np.max(tpr - fpr)
        ks_threshold = roc_thresholds[np.argmax(tpr - fpr)]
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob, sample_weight=sample_weight)
        
        self.metrics = {
            'auc': auc,
            'gini': gini,
            'ks_statistic': ks_statistic,
            'ks_threshold': ks_threshold
        }
        
        self.curves = {
            'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
        }
        
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  Gini: {gini:.4f}")
        logger.info(f"  KS: {ks_statistic:.4f} at threshold {ks_threshold:.4f}")
        
        return self.metrics
    
    def get_score_bands(
        self,
        y_true: pd.Series,
        score: np.ndarray,
        sample_weight: pd.Series = None,
        n_bands: int = 10
    ) -> pd.DataFrame:
        """Analyze event rates by score bands."""
        df = pd.DataFrame({
            'score': score,
            'actual': y_true,
            'weight': sample_weight if sample_weight is not None else 1.0
        })
        
        df['score_band'] = pd.qcut(df['score'], q=n_bands, labels=False, duplicates='drop')
        
        band_stats = df.groupby('score_band').agg({
            'score': ['min', 'max', 'mean'],
            'actual': ['sum', 'count'],
            'weight': 'sum'
        }).round(4)
        
        band_stats.columns = ['score_min', 'score_max', 'score_mean', 
                              f'n_{self.event_label}', 'n_total', 'weight_sum']
        band_stats[f'{self.event_label}_rate'] = band_stats[f'n_{self.event_label}'] / band_stats['n_total']
        band_stats['pct_total'] = band_stats['n_total'] / band_stats['n_total'].sum()
        band_stats[f'pct_{self.event_label}'] = band_stats[f'n_{self.event_label}'] / band_stats[f'n_{self.event_label}'].sum()
        
        return band_stats.reset_index()
    
    def get_lift_table(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        sample_weight: pd.Series = None,
        n_deciles: int = 10
    ) -> pd.DataFrame:
        """Generate lift table by probability deciles."""
        df = pd.DataFrame({
            'prob': y_prob,
            'actual': y_true,
            'weight': sample_weight if sample_weight is not None else 1.0
        })
        
        df = df.sort_values('prob', ascending=False)
        df['decile'] = pd.qcut(df['prob'].rank(method='first'), q=n_deciles, labels=False)
        
        overall_event_rate = df['actual'].mean()
        
        lift = df.groupby('decile').agg({
            'prob': ['min', 'max', 'mean'],
            'actual': ['sum', 'count', 'mean']
        })
        
        lift.columns = ['prob_min', 'prob_max', 'prob_mean', 
                        f'n_{self.event_label}', 'n_total', f'{self.event_label}_rate']
        lift['lift'] = lift[f'{self.event_label}_rate'] / overall_event_rate
        lift['cum_n_total'] = lift['n_total'].cumsum()
        lift[f'cum_n_{self.event_label}'] = lift[f'n_{self.event_label}'].cumsum()
        lift[f'cum_{self.event_label}_rate'] = lift[f'cum_n_{self.event_label}'] / lift['cum_n_total']
        lift['cum_lift'] = lift[f'cum_{self.event_label}_rate'] / overall_event_rate
        lift[f'cum_pct_{self.event_label}'] = lift[f'cum_n_{self.event_label}'] / lift[f'n_{self.event_label}'].sum()
        lift['cum_pct_total'] = lift['cum_n_total'] / lift['n_total'].sum()
        
        return lift.reset_index()


class ScorecardBuilder:
    """
    Build scorecard points from model coefficients.
    
    FIX #5: Consistent with score_direction convention.
    """
    
    def __init__(
        self,
        points_for_1to1: int = 600,
        points_to_double_odds: int = 20,
        round_points: bool = True,
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_BETTER
    ):
        self.points_for_1to1 = points_for_1to1
        self.points_to_double_odds = points_to_double_odds
        self.round_points = round_points
        self.score_direction = score_direction
        
        self.scorecard: pd.DataFrame = None
        self.base_score: float = 0.0
        
    def build(
        self,
        model: PropensityModel,
        woe_transformer: WOETransformer
    ) -> pd.DataFrame:
        """Build scorecard from fitted model and WOE transformer."""
        logger.info("Building scorecard...")
        
        records = []
        factor = self.points_to_double_odds / np.log(2)
        
        # FIX #5: Sign adjustment based on direction
        sign = 1 if self.score_direction == ScoreDirection.HIGHER_IS_BETTER else -1
        
        for woe_var, coef in model.coefficients.items():
            orig_var = woe_var.replace('_WOE', '')
            
            if orig_var not in woe_transformer.woe_results:
                continue
            
            woe_result = woe_transformer.woe_results[orig_var]
            
            bin_log_odds = []
            for bin in woe_result.bins:
                lo_raw = sign * bin.woe * coef
                bin_log_odds.append({
                    'variable': orig_var,
                    'bin_id': bin.bin_id,
                    'bin_label': bin.bin_label,
                    'woe': bin.woe,
                    'coefficient': coef,
                    'lo_raw': lo_raw
                })
            
            lo_min = min(b['lo_raw'] for b in bin_log_odds)
            
            for bin_data in bin_log_odds:
                points = factor * (bin_data['lo_raw'] - lo_min)
                if self.round_points:
                    points = round(points)
                
                records.append({
                    'variable': bin_data['variable'],
                    'bin_id': bin_data['bin_id'],
                    'bin_label': bin_data['bin_label'],
                    'woe': bin_data['woe'],
                    'coefficient': bin_data['coefficient'],
                    'lo_raw': bin_data['lo_raw'],
                    'points': points
                })
        
        self.scorecard = pd.DataFrame(records)
        
        total_lo_min = sum(
            min(self.scorecard[self.scorecard['variable'] == var]['lo_raw'])
            for var in self.scorecard['variable'].unique()
        )
        
        self.base_score = self.points_for_1to1 + factor * (sign * (-model.intercept) + total_lo_min)
        if self.round_points:
            self.base_score = round(self.base_score)
        
        logger.info(f"Scorecard built with {len(self.scorecard)} bins")
        logger.info(f"Base score: {self.base_score}")
        
        return self.scorecard
    
    def get_scorecard_summary(self) -> pd.DataFrame:
        """Get summarized scorecard by variable."""
        if self.scorecard is None:
            raise ValueError("Scorecard not built yet. Call build() first.")
        
        summary = self.scorecard.groupby('variable').agg({
            'bin_id': 'count',
            'points': ['min', 'max']
        })
        summary.columns = ['n_bins', 'min_points', 'max_points']
        summary['point_range'] = summary['max_points'] - summary['min_points']
        
        return summary.reset_index()
    
    def generate_scoring_code(self, language: str = 'python') -> str:
        """Generate scoring code for deployment."""
        if self.scorecard is None:
            raise ValueError("Scorecard not built yet. Call build() first.")
        
        if language == 'python':
            return self._generate_python_code()
        elif language == 'sql':
            return self._generate_sql_code()
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def _generate_python_code(self) -> str:
        """Generate Python scoring code."""
        direction_comment = (
            "Higher score = higher probability of event" 
            if self.score_direction == ScoreDirection.HIGHER_IS_BETTER 
            else "Higher score = lower probability of event (credit score convention)"
        )
        
        lines = [
            '"""',
            'Auto-generated Propensity Scorecard Scoring Function',
            f'Base Score: {self.base_score}',
            f'Points for 1:1 odds: {self.points_for_1to1}',
            f'Points to double odds: {self.points_to_double_odds}',
            f'Score Direction: {direction_comment}',
            '"""',
            '',
            'import numpy as np',
            'import pandas as pd',
            '',
            '',
            'def calculate_score(row: dict) -> float:',
            '    """Calculate propensity score for a single record."""',
            f'    score = {self.base_score}  # Base score',
            ''
        ]
        
        for var in self.scorecard['variable'].unique():
            var_data = self.scorecard[self.scorecard['variable'] == var].copy()
            
            lines.append(f'    # Variable: {var}')
            lines.append(f'    val = row.get("{var}")')
            
            # Determine if numeric or categorical based on bin labels
            has_numeric_labels = any(
                '<=' in str(b) or '>' in str(b) or '(' in str(b) 
                for b in var_data['bin_label'] if b != 'Missing'
            )
            
            lines.append('    if pd.isna(val):')
            missing_row = var_data[var_data['bin_label'] == 'Missing']
            if not missing_row.empty:
                lines.append(f'        score += {int(missing_row.iloc[0]["points"])}')
            else:
                lines.append('        pass  # No missing bin defined')
            
            if has_numeric_labels:
                for _, bin_row in var_data.iterrows():
                    if bin_row['bin_label'] == 'Missing':
                        continue
                    label = str(bin_row['bin_label'])
                    points = int(bin_row['points'])
                    
                    if '<=' in label and '(' not in label:
                        bound = label.replace('<=', '').strip()
                        lines.append(f'    elif val <= {bound}:')
                        lines.append(f'        score += {points}')
                    elif label.startswith('>') and ',' not in label:
                        bound = label.replace('>', '').strip()
                        lines.append(f'    elif val > {bound}:')
                        lines.append(f'        score += {points}')
                    elif '(' in label and ']' in label:
                        # Range like (a, b]
                        parts = label.replace('(', '').replace(']', '').split(',')
                        if len(parts) == 2:
                            lower = parts[0].strip()
                            upper = parts[1].strip()
                            lines.append(f'    elif {lower} < val <= {upper}:')
                            lines.append(f'        score += {points}')
            else:
                for _, bin_row in var_data.iterrows():
                    if bin_row['bin_label'] in ['Missing', 'Other']:
                        continue
                    lines.append(f'    elif val == "{bin_row["bin_label"]}":')
                    lines.append(f'        score += {int(bin_row["points"])}')
                
                other_row = var_data[var_data['bin_label'] == 'Other']
                if not other_row.empty:
                    lines.append('    else:')
                    lines.append(f'        score += {int(other_row.iloc[0]["points"])}')
            
            lines.append('')
        
        sign = '' if self.score_direction == ScoreDirection.HIGHER_IS_BETTER else '-'
        
        lines.extend([
            '    return score',
            '',
            '',
            'def score_to_probability(score: float) -> float:',
            '    """Convert score to probability (propensity)."""',
            f'    log_odds = {sign}(score - {self.points_for_1to1}) * np.log(2) / {self.points_to_double_odds}',
            '    probability = 1 / (1 + np.exp(-log_odds))',
            '    return probability',
            '',
            '',
            'def batch_score(df: pd.DataFrame) -> pd.DataFrame:',
            '    """Score a batch of records."""',
            '    result = df.copy()',
            '    result["score"] = df.apply(lambda row: calculate_score(row.to_dict()), axis=1)',
            '    result["propensity"] = result["score"].apply(score_to_probability)',
            '    return result'
        ])
        
        return '\n'.join(lines)
    
    def _generate_sql_code(self) -> str:
        """Generate SQL scoring code."""
        lines = [
            '-- Auto-generated Propensity Scorecard Scoring SQL',
            f'-- Base Score: {self.base_score}',
            '',
            'SELECT',
            '    *,',
            f'    {self.base_score}'
        ]
        
        for var in self.scorecard['variable'].unique():
            var_data = self.scorecard[self.scorecard['variable'] == var]
            
            lines.append(f'    + CASE')
            for _, bin_row in var_data.iterrows():
                label = bin_row['bin_label']
                points = int(bin_row['points'])
                
                if label == 'Missing':
                    lines.append(f'        WHEN {var} IS NULL THEN {points}')
                else:
                    lines.append(f'        -- {label}: {points}')
            
            lines.append('        ELSE 0')
            lines.append(f'      END -- {var}')
        
        lines.append('    AS score')
        lines.append('FROM your_table;')
        
        return '\n'.join(lines)


class PropensityAutomodeller:
    """
    Main orchestrator class - equivalent to SAS %SuperFred macro.
    
    FIXES:
    - #1: Standardized target handling (binary 0/1)
    - #5: Configurable score direction
    - #7: Time-based split option
    - #9: Configurable terminology
    - #10: PSI monitoring support
    """
    
    def __init__(
        self,
        target_col: str = 'target',
        weight_col: str = None,
        id_cols: List[str] = None,
        positive_label: Union[str, int] = 1,
        negative_label: Union[str, int] = 0,
        n_quantiles: int = 10,
        max_char_levels: int = 15,
        min_bin_pct: float = 0.05,
        min_bin_count: int = 50,
        iv_min: float = 0.02,
        test_size: float = 0.25,
        random_state: int = 42,
        time_col: str = None,  # FIX #7: Time-based split
        time_split_date: str = None,  # FIX #7: Cutoff date for time split
        points_for_1to1: int = 600,
        points_to_double_odds: int = 20,
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_BETTER,
        calibrate_probabilities: bool = False,
        calibration_cv: int = 5,
        enforce_monotonic: bool = False,
        event_label: str = 'event'  # FIX #9: Configurable terminology
    ):
        """
        Initialize PropensityAutomodeller.
        
        Parameters:
        -----------
        target_col : str
            Name of target column
        weight_col : str, optional
            Name of weight column
        id_cols : List[str], optional
            ID columns to preserve
        positive_label : Union[str, int]
            Label for positive outcome (event of interest, e.g., payment, default)
        negative_label : Union[str, int]
            Label for negative outcome
        n_quantiles : int
            Number of quantile bins for numeric variables
        max_char_levels : int
            Maximum levels for categorical variables
        min_bin_pct : float
            FIX #3: Minimum percentage of weight in a bin
        min_bin_count : int
            FIX #3: Minimum count in a bin
        iv_min : float
            Minimum IV for feature selection
        test_size : float
            Proportion of data for validation (ignored if time_col specified)
        random_state : int
            Random seed
        time_col : str, optional
            FIX #7: Column for time-based split
        time_split_date : str, optional
            FIX #7: Cutoff date for time split (format: 'YYYY-MM-DD')
        points_for_1to1 : int
            Scorecard base points
        points_to_double_odds : int
            Scorecard scaling factor
        score_direction : ScoreDirection
            FIX #5: Score direction convention
        calibrate_probabilities : bool
            Whether to calibrate output probabilities
        calibration_cv : int
            FIX #6: CV folds for calibration
        enforce_monotonic : bool
            FIX #8: Whether to enforce monotonic WOE
        event_label : str
            FIX #9: Label for the event (for display/logging)
        """
        self.target_col = target_col
        self.weight_col = weight_col
        self.id_cols = id_cols or []
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.n_quantiles = n_quantiles
        self.max_char_levels = max_char_levels
        self.min_bin_pct = min_bin_pct
        self.min_bin_count = min_bin_count
        self.iv_min = iv_min
        self.test_size = test_size
        self.random_state = random_state
        self.time_col = time_col
        self.time_split_date = time_split_date
        self.points_for_1to1 = points_for_1to1
        self.points_to_double_odds = points_to_double_odds
        self.score_direction = score_direction
        self.calibrate_probabilities = calibrate_probabilities
        self.calibration_cv = calibration_cv
        self.enforce_monotonic = enforce_monotonic
        self.event_label = event_label
        
        # Component instances
        self.data_extractor = None
        self.woe_transformer = None
        self.model = None
        self.evaluator = None
        self.scorecard_builder = None
        
        # Data storage
        self.train_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        self.train_woe: pd.DataFrame = None
        self.test_woe: pd.DataFrame = None
        
        # Results
        self.iv_summary: pd.DataFrame = None
        self.selected_features: List[str] = []
        self.train_metrics: Dict[str, float] = {}
        self.test_metrics: Dict[str, float] = {}
        self.scorecard: pd.DataFrame = None
        
    def fit(
        self,
        data: pd.DataFrame,
        feature_cols: List[str]
    ) -> 'PropensityAutomodeller':
        """
        Run complete propensity modeling pipeline.
        
        FIX #1: Standardizes target to binary (0/1) before processing.
        FIX #7: Uses time-based split if time_col is specified.
        """
        logger.info("=" * 60)
        logger.info("PropensityAutomodeller Pipeline Started")
        logger.info("=" * 60)
        
        # FIX #1: Standardize target to binary (0/1) BEFORE any processing
        data = data.copy()
        
        # Convert target to binary
        if data[self.target_col].dtype == object or data[self.target_col].dtype.name == 'category':
            # String/categorical target
            data[self.target_col] = data[self.target_col].map({
                str(self.positive_label): 1,
                str(self.negative_label): 0,
                self.positive_label: 1,
                self.negative_label: 0
            })
        else:
            # Numeric target
            data[self.target_col] = data[self.target_col].map({
                self.positive_label: 1,
                self.negative_label: 0
            })
        
        # Validate conversion
        unique_vals = data[self.target_col].dropna().unique()
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(
                f"Failed to convert target to binary. Unique values after mapping: {unique_vals}. "
                f"Please verify positive_label={self.positive_label} and negative_label={self.negative_label}"
            )
        
        logger.info(f"Target converted to binary: {data[self.target_col].value_counts().to_dict()}")
        
        # Step 1: Data Extraction
        logger.info("\nStep 1: Data Extraction")
        logger.info("-" * 40)
        
        self.data_extractor = DataExtractor(
            target_col=self.target_col,
            weight_col=self.weight_col,
            id_cols=self.id_cols,
            event_label=self.event_label
        )
        
        extracted = self.data_extractor.extract(data, feature_cols, time_col=self.time_col)
        
        # Step 2: Train/Test Split
        logger.info("\nStep 2: Train/Validation Split")
        logger.info("-" * 40)
        
        # FIX #7: Time-based split if specified
        if self.time_col and self.time_split_date:
            logger.info(f"Using TIME-BASED split on column '{self.time_col}'")
            logger.info(f"Cutoff date: {self.time_split_date}")
            
            # Ensure time column is datetime
            if self.time_col not in extracted.columns:
                raise ValueError(f"Time column '{self.time_col}' not found in data")
            
            time_series = pd.to_datetime(extracted[self.time_col])
            cutoff = pd.to_datetime(self.time_split_date)
            
            train_mask = time_series < cutoff
            test_mask = time_series >= cutoff
            
            self.train_data = extracted[train_mask].copy()
            self.test_data = extracted[test_mask].copy()
            
            logger.info(f"Training set (before {self.time_split_date}): {len(self.train_data)} records")
            logger.info(f"Validation set (>= {self.time_split_date}): {len(self.test_data)} records")
        else:
            # Random stratified split
            logger.info("Using RANDOM stratified split")
            self.train_data, self.test_data = train_test_split(
                extracted,
                test_size=self.test_size,
                stratify=extracted['target_binary'],
                random_state=self.random_state
            )
            
            logger.info(f"Training set: {len(self.train_data)} records")
            logger.info(f"Validation set: {len(self.test_data)} records")
        
        # Step 3: WOE Transformation
        logger.info("\nStep 3: WOE Transformation")
        logger.info("-" * 40)
        
        self.woe_transformer = WOETransformer(
            n_quantiles=self.n_quantiles,
            max_char_levels=self.max_char_levels,
            min_bin_pct=self.min_bin_pct,
            min_bin_count=self.min_bin_count,
            enforce_monotonic=self.enforce_monotonic
        )
        
        weight_col = 'weight' if 'weight' in self.train_data.columns else None
        
        self.woe_transformer.fit(
            self.train_data,
            feature_cols,
            'target_binary',
            weight_col
        )
        
        self.iv_summary = self.woe_transformer.get_iv_summary()
        
        logger.info("\nInformation Value Summary (Top 20):")
        logger.info(self.iv_summary.head(20).to_string())
        
        # Step 4: Transform data to WOE
        logger.info("\nStep 4: Creating WOE-Transformed Data")
        logger.info("-" * 40)
        
        self.train_woe = self.woe_transformer.transform(self.train_data, feature_cols)
        self.test_woe = self.woe_transformer.transform(self.test_data, feature_cols)
        
        # Step 5: Fit Model
        logger.info("\nStep 5: Fitting Propensity Model")
        logger.info("-" * 40)
        
        self.model = PropensityModel(
            iv_min=self.iv_min,
            points_for_1to1=self.points_for_1to1,
            points_to_double_odds=self.points_to_double_odds,
            score_direction=self.score_direction,
            calibrate=self.calibrate_probabilities,
            calibration_cv=self.calibration_cv
        )
        
        train_weights = self.train_woe['weight'] if 'weight' in self.train_woe.columns else None
        
        self.model.fit(
            self.train_woe,
            self.train_woe['target_binary'],
            sample_weight=train_weights,
            iv_summary=self.iv_summary
        )
        
        self.selected_features = self.model.selected_features
        
        logger.info("\nModel Coefficients:")
        logger.info(self.model.get_coefficients_df().to_string())
        
        # Step 6: Evaluate Model
        logger.info("\nStep 6: Model Evaluation")
        logger.info("-" * 40)
        
        self.evaluator = ModelEvaluator(event_label=self.event_label)
        
        # Training evaluation
        train_proba = self.model.predict_proba(self.train_woe)
        self.train_metrics = self.evaluator.evaluate(
            self.train_woe['target_binary'],
            train_proba,
            sample_weight=train_weights
        )
        logger.info(f"\nTraining Metrics:")
        logger.info(f"  Gini: {self.train_metrics['gini']:.4f}")
        logger.info(f"  AUC: {self.train_metrics['auc']:.4f}")
        
        # Validation evaluation
        test_proba = self.model.predict_proba(self.test_woe)
        test_weights = self.test_woe['weight'] if 'weight' in self.test_woe.columns else None
        self.test_metrics = self.evaluator.evaluate(
            self.test_woe['target_binary'],
            test_proba,
            sample_weight=test_weights
        )
        logger.info(f"\nValidation Metrics:")
        logger.info(f"  Gini: {self.test_metrics['gini']:.4f}")
        logger.info(f"  AUC: {self.test_metrics['auc']:.4f}")
        
        # Step 7: Build Scorecard
        logger.info("\nStep 7: Building Scorecard")
        logger.info("-" * 40)
        
        self.scorecard_builder = ScorecardBuilder(
            points_for_1to1=self.points_for_1to1,
            points_to_double_odds=self.points_to_double_odds,
            score_direction=self.score_direction
        )
        
        self.scorecard = self.scorecard_builder.build(self.model, self.woe_transformer)
        
        logger.info("\nScorecard Summary:")
        logger.info(self.scorecard_builder.get_scorecard_summary().to_string())
        
        logger.info("\n" + "=" * 60)
        logger.info("PropensityAutomodeller Pipeline Completed")
        logger.info("=" * 60)
        
        return self
    
    def predict(
        self,
        data: pd.DataFrame,
        return_score: bool = True,
        return_propensity: bool = True
    ) -> pd.DataFrame:
        """Score new data with fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get feature columns
        feature_cols = [f.replace('_WOE', '') for f in self.selected_features]
        
        # Transform to WOE
        data_woe = self.woe_transformer.transform(data, feature_cols)
        
        # Predict
        result = data.copy()
        
        if return_propensity:
            result['propensity'] = self.model.predict_proba(data_woe)
        
        if return_score:
            result['score'] = self.model.predict_score(data_woe)
        
        return result
    
    def calculate_psi(self, comparison_data: pd.DataFrame) -> pd.DataFrame:
        """
        FIX #10: Calculate PSI between training data and new data.
        """
        if self.woe_transformer is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        feature_cols = [f.replace('_WOE', '') for f in self.selected_features]
        return self.woe_transformer.calculate_psi(
            self.train_data,
            comparison_data,
            feature_cols
        )
    
    def get_lift_table(self, dataset: str = 'test') -> pd.DataFrame:
        """Get lift table for specified dataset."""
        if dataset == 'train':
            data = self.train_woe
            proba = self.model.predict_proba(self.train_woe)
        else:
            data = self.test_woe
            proba = self.model.predict_proba(self.test_woe)
        
        weights = data['weight'] if 'weight' in data.columns else None
        
        return self.evaluator.get_lift_table(
            data['target_binary'],
            proba,
            sample_weight=weights
        )
    
    def get_woe_table(self, variable: str) -> pd.DataFrame:
        """Get WOE table for a specific variable."""
        return self.woe_transformer.get_woe_table(variable)
    
    def get_scorecard(self) -> pd.DataFrame:
        """Get the full scorecard."""
        return self.scorecard.copy()
    
    def generate_scoring_code(self, language: str = 'python') -> str:
        """Generate scoring code for deployment."""
        return self.scorecard_builder.generate_scoring_code(language)
    
    def save(self, filepath: str) -> None:
        """Save fitted model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'woe_transformer': self.woe_transformer,
                'model': self.model,
                'scorecard_builder': self.scorecard_builder,
                'iv_summary': self.iv_summary,
                'selected_features': self.selected_features,
                'train_metrics': self.train_metrics,
                'test_metrics': self.test_metrics,
                'scorecard': self.scorecard,
                'config': {
                    'target_col': self.target_col,
                    'weight_col': self.weight_col,
                    'id_cols': self.id_cols,
                    'n_quantiles': self.n_quantiles,
                    'max_char_levels': self.max_char_levels,
                    'min_bin_pct': self.min_bin_pct,
                    'min_bin_count': self.min_bin_count,
                    'iv_min': self.iv_min,
                    'points_for_1to1': self.points_for_1to1,
                    'points_to_double_odds': self.points_to_double_odds,
                    'score_direction': self.score_direction,
                    'event_label': self.event_label
                }
            }, f)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PropensityAutomodeller':
        """Load fitted model from file."""
        with open(filepath, 'rb') as f:
            saved = pickle.load(f)
        
        config = saved['config']
        instance = cls(
            target_col=config['target_col'],
            weight_col=config['weight_col'],
            id_cols=config['id_cols'],
            n_quantiles=config['n_quantiles'],
            max_char_levels=config['max_char_levels'],
            min_bin_pct=config.get('min_bin_pct', 0.05),
            min_bin_count=config.get('min_bin_count', 50),
            iv_min=config['iv_min'],
            points_for_1to1=config['points_for_1to1'],
            points_to_double_odds=config['points_to_double_odds'],
            score_direction=config.get('score_direction', ScoreDirection.HIGHER_IS_BETTER),
            event_label=config.get('event_label', 'event')
        )
        
        instance.woe_transformer = saved['woe_transformer']
        instance.model = saved['model']
        instance.scorecard_builder = saved['scorecard_builder']
        instance.iv_summary = saved['iv_summary']
        instance.selected_features = saved['selected_features']
        instance.train_metrics = saved['train_metrics']
        instance.test_metrics = saved['test_metrics']
        instance.scorecard = saved['scorecard']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def export_results(self, output_dir: str) -> None:
        """Export all results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # IV Summary
        self.iv_summary.to_csv(output_path / 'iv_summary.csv', index=False)
        
        # WOE Tables
        for var in self.woe_transformer.woe_results.keys():
            woe_table = self.get_woe_table(var)
            woe_table.to_csv(output_path / f'woe_{var}.csv', index=False)
        
        # Coefficients
        self.model.get_coefficients_df().to_csv(output_path / 'coefficients.csv', index=False)
        
        # Scorecard
        self.scorecard.to_csv(output_path / 'scorecard.csv', index=False)
        
        # Lift tables
        self.get_lift_table('train').to_csv(output_path / 'lift_train.csv', index=False)
        self.get_lift_table('test').to_csv(output_path / 'lift_test.csv', index=False)
        
        # Metrics
        metrics = {
            'train': self.train_metrics,
            'test': self.test_metrics
        }
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Scoring code
        with open(output_path / 'scoring_code.py', 'w') as f:
            f.write(self.generate_scoring_code('python'))
        
        logger.info(f"Results exported to {output_dir}")