# Phase 2: Programmatic Exploratory Data Analysis Engine (CRISP-DM Standard)
# Skills engaged: exploratory-data-analysis, programmatic-eda, visualization-builder

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any

class ProgrammaticEDAEngine:
    """
    Automated EDA engine extracting statistical profiles, correlation heatmaps,
    feature distributions, and origin-destination mobility matrix dynamics.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute mean, std, quantiles, skewness, and kurtosis for continuous columns."""
        num_cols = [
            "trip_distance_km", "haversine_distance_km", "trip_duration_min",
            "total_fare_usd", "temperature_c", "precipitation_mm", "wind_speed_kmh",
            "carbon_emissions_kg"
        ]
        
        summary = {}
        for col in num_cols:
            s = self.df[col]
            summary[col] = {
                "count": int(len(s)),
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std()), 3),
                "min": round(float(s.min()), 3),
                "p25": round(float(s.quantile(0.25)), 3),
                "median": round(float(s.median()), 3),
                "p75": round(float(s.quantile(0.75)), 3),
                "p95": round(float(s.quantile(0.95)), 3),
                "max": round(float(s.max()), 3),
                "skewness": round(float(stats.skew(s)), 3),
                "kurtosis": round(float(stats.kurtosis(s)), 3)
            }
        return summary

    def get_correlation_matrix(self) -> Dict[str, Any]:
        """Compute Pearson and Spearman correlation matrices across continuous variables."""
        num_cols = [
            "trip_distance_km", "haversine_distance_km", "trip_duration_min",
            "temperature_c", "precipitation_mm", "wind_speed_kmh",
            "congestion_surcharge", "total_fare_usd", "high_tip_indicator"
        ]
        
        corr = self.df[num_cols].corr()
        return {
            "columns": num_cols,
            "matrix": [[round(float(val), 3) for val in row] for row in corr.values]
        }

    def get_target_distributions(self) -> Dict[str, Any]:
        """Calculate bin frequencies for regression target and class balance for classification."""
        fare_hist, fare_bins = np.histogram(self.df["total_fare_usd"], bins=15)
        dist_hist, dist_bins = np.histogram(self.df["trip_distance_km"], bins=15)
        
        tip_counts = self.df["high_tip_indicator"].value_counts().to_dict()
        
        return {
            "fare_distribution": {
                "counts": [int(x) for x in fare_hist],
                "bin_edges": [round(float(x), 2) for x in fare_bins]
            },
            "distance_distribution": {
                "counts": [int(x) for x in dist_hist],
                "bin_edges": [round(float(x), 2) for x in dist_bins]
            },
            "high_tip_balance": {
                "regular_tip_count (0)": int(tip_counts.get(0, 0)),
                "high_tip_count (1)": int(tip_counts.get(1, 0)),
                "high_tip_percentage": round((tip_counts.get(1, 0) / len(self.df)) * 100.0, 2)
            }
        }

    def get_origin_destination_flow(self) -> Dict[str, Any]:
        """Compute top spatial mobility zone pairs."""
        flow = self.df.groupby(["pickup_zone", "dropoff_zone"]).size().reset_index(name="trip_volume")
        top_flows = flow.sort_values(by="trip_volume", ascending=False).head(10).to_dict(orient="records")
        return {
            "top_mobility_corridors": top_flows
        }

    def generate_full_eda_dossier(self) -> Dict[str, Any]:
        return {
            "summary_statistics": self.get_summary_statistics(),
            "correlation_analysis": self.get_correlation_matrix(),
            "target_distributions": self.get_target_distributions(),
            "mobility_flows": self.get_origin_destination_flow()
        }
