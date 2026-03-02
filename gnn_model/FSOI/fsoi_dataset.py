"""
FSOI Dataset - Sequential paired dataset for FSOI computation.

This dataset yields (prev_batch, curr_batch) pairs where:
- prev_batch: Input window k-1 (for background forecast)
- curr_batch: Input window k (analysis)

Sequential ordering is CRITICAL for FSOI because the background
must be built from the previous window.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import pandas as pd


class FSOIDataset(Dataset):
    """
    Sequential dataset that yields (previous_window, current_window) pairs.

    Each pair represents:
    - previous_window: observations from time k-1 (used to forecast background)
    - current_window: observations from time k (analysis input)

    The model will:
    1. Use previous_window to predict what the current_window observations should be (background)
    2. Compare actual current_window observations (analysis) vs background
    3. Compute gradients and FSOI
    """

    def __init__(
        self,
        base_dataset,
        bin_names: List[str],
    ):
        """
        Args:
            base_dataset: The underlying BinDataset that can fetch individual bins
            bin_names: Ordered list of bin names (sorted chronologically)
        """
        self.base_dataset = base_dataset
        self.bin_names = sorted(bin_names)  # Ensure chronological order

        if len(self.bin_names) < 2:
            raise ValueError("FSOI requires at least 2 time windows (previous + current)")

        print(f"[FSOI Dataset] Created with {len(self)} sequential pairs")
        print(f"[FSOI Dataset] First bin: {self.bin_names[0]}")
        print(f"[FSOI Dataset] Last bin: {self.bin_names[-1]}")

    def __len__(self):
        # Number of valid (prev, curr) pairs
        # If we have N bins, we can form N-1 pairs: (0,1), (1,2), ..., (N-2, N-1)
        return len(self.bin_names) - 1

    def __getitem__(self, idx) -> Tuple:
        """
        Returns (prev_batch, curr_batch) where:
        - prev_batch: data from bin_names[idx] (window k-1)
        - curr_batch: data from bin_names[idx+1] (window k)
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for {len(self)} pairs")

        prev_bin_name = self.bin_names[idx]
        curr_bin_name = self.bin_names[idx + 1]

        # Temporarily override the base dataset to fetch specific bins
        prev_idx = self.base_dataset.bin_names.index(prev_bin_name)
        curr_idx = self.base_dataset.bin_names.index(curr_bin_name)

        prev_batch = self.base_dataset[prev_idx]
        curr_batch = self.base_dataset[curr_idx]

        # Add metadata for tracking
        prev_batch.fsoi_bin_name = prev_bin_name
        curr_batch.fsoi_bin_name = curr_bin_name
        prev_batch.fsoi_pair_idx = idx
        curr_batch.fsoi_pair_idx = idx

        return prev_batch, curr_batch


class FSOIDatasetSingleBin(Dataset):
    """
    Alternative: Single-bin dataset that still provides access to previous bin.

    This is useful if you want to iterate over individual bins but still
    need access to the previous window for background computation.
    """

    def __init__(
        self,
        base_dataset,
        bin_names: List[str],
        include_previous: bool = True,
    ):
        """
        Args:
            base_dataset: The underlying BinDataset
            bin_names: Ordered list of bin names
            include_previous: If True, returns (prev, curr); if False, returns curr only
        """
        self.base_dataset = base_dataset
        self.bin_names = sorted(bin_names)
        self.include_previous = include_previous

        # Skip first bin if we need previous context
        self.start_idx = 1 if include_previous else 0

    def __len__(self):
        return len(self.bin_names) - self.start_idx

    def __getitem__(self, idx):
        actual_idx = idx + self.start_idx
        curr_bin_name = self.bin_names[actual_idx]

        # Get index in base dataset
        curr_base_idx = self.base_dataset.bin_names.index(curr_bin_name)
        curr_batch = self.base_dataset[curr_base_idx]

        if not self.include_previous:
            return curr_batch

        # Get previous bin
        prev_bin_name = self.bin_names[actual_idx - 1]
        prev_base_idx = self.base_dataset.bin_names.index(prev_bin_name)
        prev_batch = self.base_dataset[prev_base_idx]

        return prev_batch, curr_batch


def create_fsoi_bin_list(
    data_summary: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> List[str]:
    """
    Create a chronologically ordered list of bin names for FSOI.

    Args:
        data_summary: DataFrame with bin information
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        Sorted list of bin names in chronological order
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Filter bins within date range
    # Assumes data_summary has 'bin_name' and extractable timestamps
    valid_bins = []

    for bin_name in data_summary.keys():
        # Extract timestamp from bin_name (format: "YYYY-MM-DD_HHMM")
        try:
            # Common format: "2024-01-01_0000"
            date_part = bin_name.split('_')[0]
            bin_time = pd.to_datetime(date_part)

            if start <= bin_time <= end:
                valid_bins.append(bin_name)
        except Exception as e:
            print(f"[WARNING] Could not parse bin_name: {bin_name}, error: {e}")
            continue

    # Sort chronologically
    valid_bins = sorted(valid_bins)

    print(f"[FSOI] Found {len(valid_bins)} bins between {start_date} and {end_date}")

    return valid_bins


def verify_sequential_consistency(
    bin_names: List[str],
    expected_interval_hours: int = 12,
) -> bool:
    """
    Verify that bins are evenly spaced in time (important for FSOI).

    Args:
        bin_names: List of bin names
        expected_interval_hours: Expected time between bins (e.g., 12 hours)

    Returns:
        True if consistent, False otherwise
    """
    if len(bin_names) < 2:
        return True

    intervals = []

    for i in range(len(bin_names) - 1):
        try:
            t1 = pd.to_datetime(bin_names[i].split('_')[0])
            t2 = pd.to_datetime(bin_names[i+1].split('_')[0])
            interval_hours = (t2 - t1).total_seconds() / 3600
            intervals.append(interval_hours)
        except Exception as e:
            print(f"[WARNING] Could not parse time interval for {bin_names[i]} -> {bin_names[i+1]}")
            return False

    # Check if all intervals match expected
    consistent = all(abs(h - expected_interval_hours) < 0.1 for h in intervals)

    if not consistent:
        print(f"[WARNING] Inconsistent time intervals found:")
        print(f"  Expected: {expected_interval_hours} hours")
        print(f"  Found: {intervals}")
    else:
        print(f"[FSOI] Sequential consistency verified: {expected_interval_hours}h intervals")

    return consistent
