"""Raw checkpoint validation and canonical OD-leg construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "data",
    "trip_creation_time",
    "route_schedule_uuid",
    "route_type",
    "trip_uuid",
    "source_center",
    "source_name",
    "destination_center",
    "destination_name",
    "od_start_time",
    "od_end_time",
    "start_scan_to_end_scan",
    "actual_distance_to_destination",
    "actual_time",
    "osrm_time",
    "osrm_distance",
    "segment_actual_time",
    "segment_osrm_time",
    "segment_osrm_distance",
}

DATETIME_COLUMNS = [
    "trip_creation_time",
    "od_start_time",
    "od_end_time",
    "cutoff_timestamp",
]

LEG_KEY = [
    "data",
    "trip_uuid",
    "source_center",
    "destination_center",
    "od_start_time",
]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the supplied CSV and validate the fields needed by the project."""
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for column in DATETIME_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    if frame["trip_uuid"].isna().any():
        raise ValueError("trip_uuid contains missing values")
    if frame[["source_center", "destination_center"]].isna().any().any():
        raise ValueError("Facility IDs must not be missing")
    return frame


def time_of_day(hour: pd.Series) -> pd.Series:
    """Convert departure hour into stable operational buckets."""
    return pd.cut(
        hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype("string")


def build_leg_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse cumulative checkpoint rows to one leakage-safe OD-leg record.

    Cumulative columns use their maximum because segment sums contain occasional
    corrections and negative values. Scan-to-scan elapsed time is the primary ETA
    target. Diagnostic movement and dwell components are retained separately.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame = raw.copy()
    for column in DATETIME_COLUMNS:
        if column in frame.columns and not pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    legs = (
        frame.groupby(LEG_KEY, dropna=False, sort=False)
        .agg(
            trip_creation_time=("trip_creation_time", "min"),
            route_schedule_uuid=("route_schedule_uuid", "first"),
            route_type=("route_type", "first"),
            source_name=("source_name", "first"),
            destination_name=("destination_name", "first"),
            od_end_time=("od_end_time", "max"),
            actual_elapsed_min=("start_scan_to_end_scan", "max"),
            actual_distance_km=("actual_distance_to_destination", "max"),
            actual_movement_min=("actual_time", "max"),
            osrm_time_min=("osrm_time", "max"),
            osrm_distance_km=("osrm_distance", "max"),
            checkpoint_count=("segment_actual_time", "size"),
            minimum_segment_actual_min=("segment_actual_time", "min"),
        )
        .reset_index()
    )

    numeric = [
        "actual_elapsed_min",
        "actual_distance_km",
        "actual_movement_min",
        "osrm_time_min",
        "osrm_distance_km",
    ]
    for column in numeric:
        legs[column] = pd.to_numeric(legs[column], errors="coerce")

    legs["dwell_time_min"] = (
        legs["actual_elapsed_min"] - legs["actual_movement_min"]
    ).clip(lower=0)
    denominator = legs["osrm_time_min"].where(legs["osrm_time_min"] > 0)
    legs["movement_delay_ratio"] = legs["actual_movement_min"] / denominator
    legs["elapsed_delay_ratio"] = legs["actual_elapsed_min"] / denominator
    legs["movement_excess_min"] = (
        legs["actual_movement_min"] - legs["osrm_time_min"]
    ).clip(lower=0)
    legs["departure_hour"] = legs["od_start_time"].dt.hour
    legs["departure_weekday"] = legs["od_start_time"].dt.dayofweek
    legs["is_weekend"] = legs["departure_weekday"].isin([5, 6]).astype("int8")
    legs["time_bucket"] = time_of_day(legs["departure_hour"])
    legs["has_negative_segment"] = (legs["minimum_segment_actual_min"] < 0).astype("int8")

    legs = legs.replace([np.inf, -np.inf], np.nan)
    essential = [
        "actual_elapsed_min",
        "actual_movement_min",
        "osrm_time_min",
        "osrm_distance_km",
        "od_start_time",
    ]
    legs["is_model_eligible"] = (
        legs[essential].notna().all(axis=1)
        & legs["actual_elapsed_min"].gt(0)
        & legs["actual_movement_min"].gt(0)
        & legs["osrm_time_min"].gt(0)
        & legs["osrm_distance_km"].gt(0)
    )
    return legs.sort_values(["trip_creation_time", "trip_uuid", "od_start_time"]).reset_index(drop=True)


def data_quality_report(raw: pd.DataFrame, legs: pd.DataFrame) -> dict[str, int | float]:
    """Return compact, JSON-serializable quality diagnostics."""
    return {
        "raw_rows": int(len(raw)),
        "leg_rows": int(len(legs)),
        "unique_trips": int(legs["trip_uuid"].nunique()),
        "unique_facilities": int(
            pd.unique(pd.concat([legs["source_center"], legs["destination_center"]])).size
        ),
        "unique_directed_corridors": int(
            legs[["source_center", "destination_center"]].drop_duplicates().shape[0]
        ),
        "model_eligible_legs": int(legs["is_model_eligible"].sum()),
        "negative_segment_legs": int(legs["has_negative_segment"].sum()),
        "missing_source_names": int(legs["source_name"].isna().sum()),
        "missing_destination_names": int(legs["destination_name"].isna().sum()),
    }
