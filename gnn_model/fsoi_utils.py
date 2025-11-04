import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

class FSOICalculator:
    """
    Compute Forecast Sensitivity to Observation Impact (FSOI).
    
    FSOI measures how much each observation impacts the forecast error.
    FSOI = -∇J · d, where:
    - ∇J is the gradient of forecast loss w.r.t. observations
    - d is the innovation (obs - background)
    """
    
    def __init__(self, observation_config, feature_stats=None):
        self.observation_config = observation_config
        self.feature_stats = feature_stats
        self.fsoi_accumulator = defaultdict(list)
        
    def compute_fsoi(
        self,
        gradients: torch.Tensor,
        innovations: torch.Tensor,
        metadata: Dict,
        node_type: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute FSOI for a batch of observations.
        
        Args:
            gradients: (N, C) gradient of loss w.r.t. observations
            innovations: (N, C) observation - background
            metadata: Dict with 'lat', 'lon', 'channel_ids', 'instrument_ids', etc.
            node_type: e.g., 'atms_input', 'amsua_input'
            
        Returns:
            Dict with FSOI values and aggregated statistics
        """
        # Element-wise FSOI: -gradient * innovation
        fsoi = -gradients * innovations  # (N, C)
        
        results = {
            'fsoi_per_obs': fsoi,
            'gradients': gradients,
            'innovations': innovations,
            'metadata': metadata,
            'node_type': node_type,
        }
        
        # Aggregate by channel
        results['fsoi_by_channel'] = fsoi.mean(dim=0)
        results['fsoi_std_by_channel'] = fsoi.std(dim=0)
        
        # Aggregate by instrument if available
        if 'instrument_ids' in metadata and metadata['instrument_ids'] is not None:
            results['fsoi_by_instrument'] = self._aggregate_by_instrument(
                fsoi, metadata['instrument_ids']
            )
        
        # Aggregate by region if lat/lon available
        if 'lat' in metadata and 'lon' in metadata:
            results['fsoi_by_region'] = self._aggregate_by_region(
                fsoi, metadata['lat'], metadata['lon']
            )
        
        return results
    
    def _aggregate_by_instrument(
        self, 
        fsoi: torch.Tensor, 
        instrument_ids: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Aggregate FSOI by instrument ID."""
        unique_ids = torch.unique(instrument_ids)
        by_instrument = {}
        
        for inst_id in unique_ids:
            mask = instrument_ids == inst_id
            by_instrument[int(inst_id.item())] = fsoi[mask].mean(dim=0)
        
        return by_instrument
    
    def _aggregate_by_region(
        self,
        fsoi: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        lat_bins: int = 6,
        lon_bins: int = 12
    ) -> Dict[str, torch.Tensor]:
        """Aggregate FSOI by geographic region (30° bands by default)."""
        # Convert to degrees if in radians
        if lat.abs().max() < 10:  # likely radians
            lat = torch.rad2deg(lat)
            lon = torch.rad2deg(lon)
        
        lat_edges = torch.linspace(-90, 90, lat_bins + 1)
        lon_edges = torch.linspace(-180, 180, lon_bins + 1)
        
        by_region = {}
        
        for i in range(lat_bins):
            for j in range(lon_bins):
                lat_mask = (lat >= lat_edges[i]) & (lat < lat_edges[i+1])
                lon_mask = (lon >= lon_edges[j]) & (lon < lon_edges[j+1])
                mask = lat_mask & lon_mask
                
                if mask.sum() > 0:
                    region_key = f"lat_{lat_edges[i]:.0f}_{lat_edges[i+1]:.0f}_lon_{lon_edges[j]:.0f}_{lon_edges[j+1]:.0f}"
                    by_region[region_key] = fsoi[mask].mean(dim=0)
        
        return by_region
    
    def accumulate(self, fsoi_results: Dict):
        """Accumulate FSOI results across batches."""
        node_type = fsoi_results['node_type']
        self.fsoi_accumulator[node_type].append(fsoi_results)
    
    def save_fsoi_to_csv(self, output_dir: str, epoch: int):
        """Save FSOI results to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for node_type, results_list in self.fsoi_accumulator.items():
            if not results_list:
                continue
            
            # Summary statistics
            all_fsoi = [r['fsoi_per_obs'] for r in results_list]
            all_fsoi = torch.cat(all_fsoi, dim=0)
            
            summary = {
                'channel': list(range(all_fsoi.shape[1])),
                'mean_fsoi': all_fsoi.mean(dim=0).cpu().numpy(),
                'std_fsoi': all_fsoi.std(dim=0).cpu().numpy(),
                'min_fsoi': all_fsoi.min(dim=0)[0].cpu().numpy(),
                'max_fsoi': all_fsoi.max(dim=0)[0].cpu().numpy(),
                'median_fsoi': all_fsoi.median(dim=0)[0].cpu().numpy(),
            }
            
            summary_df = pd.DataFrame(summary)
            summary_file = f"{output_dir}/fsoi_summary_{node_type}_epoch{epoch}.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"Saved FSOI summary: {summary_file}")
    
    def reset(self):
        """Reset accumulator for new epoch."""
        self.fsoi_accumulator.clear()