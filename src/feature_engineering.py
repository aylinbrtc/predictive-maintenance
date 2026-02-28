"""
feature_engineering.py
-----------------------
Functions to transform and merge the 5 raw tables into a single feature matrix
suitable for machine learning classification.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Telemetry aggregation
# ---------------------------------------------------------------------------

def aggregate_telemetry(telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly telemetry readings to daily statistics per machine.

    For each sensor (volt, rotate, pressure, vibration) computes:
    - Daily mean, std, min, max
    - 3-day and 7-day rolling mean and std (based on daily aggregates)

    Parameters
    ----------
    telemetry : raw telemetry DataFrame (datetime, machineID, volt, rotate, pressure, vibration)

    Returns
    -------
    DataFrame indexed by (machineID, date) with aggregated sensor features.
    """
    df = telemetry.copy()
    df['date'] = df['datetime'].dt.normalize()

    sensors = ['volt', 'rotate', 'pressure', 'vibration']
    agg_dict = {s: ['mean', 'std', 'min', 'max'] for s in sensors}

    daily = df.groupby(['machineID', 'date']).agg(agg_dict)
    daily.columns = ['_'.join(col) for col in daily.columns]
    daily = daily.reset_index()

    # Rolling windows (3-day and 7-day) per machine
    rolling_features = []
    for machine_id, group in daily.groupby('machineID'):
        group = group.sort_values('date').copy()
        for s in sensors:
            for window in [3, 7]:
                col_mean = f'{s}_mean'
                group[f'{s}_rolling{window}_mean'] = (
                    group[col_mean].rolling(window, min_periods=1).mean()
                )
                group[f'{s}_rolling{window}_std'] = (
                    group[col_mean].rolling(window, min_periods=1).std().fillna(0)
                )
        rolling_features.append(group)

    return pd.concat(rolling_features, ignore_index=True)


# ---------------------------------------------------------------------------
# Error features
# ---------------------------------------------------------------------------

def compute_error_features(errors: pd.DataFrame, telemetry_dates: pd.DataFrame) -> pd.DataFrame:
    """
    Count errors per machine in the past 24 h and past 7 days at each telemetry timestamp.

    Parameters
    ----------
    errors           : raw errors DataFrame
    telemetry_dates  : DataFrame with columns ['machineID', 'date'] (one row per machine-day)

    Returns
    -------
    DataFrame with error count features merged onto telemetry_dates.
    """
    error_ids = errors['errorID'].unique()
    records = []

    for _, row in telemetry_dates.iterrows():
        machine = row['machineID']
        date = row['date']
        machine_errors = errors[errors['machineID'] == machine]

        window_24h = machine_errors[
            (machine_errors['datetime'] >= date - pd.Timedelta(hours=24)) &
            (machine_errors['datetime'] < date)
        ]
        window_7d = machine_errors[
            (machine_errors['datetime'] >= date - pd.Timedelta(days=7)) &
            (machine_errors['datetime'] < date)
        ]

        record = {'machineID': machine, 'date': date}
        for eid in error_ids:
            record[f'{eid}_count_24h'] = (window_24h['errorID'] == eid).sum()
            record[f'{eid}_count_7d'] = (window_7d['errorID'] == eid).sum()
        record['total_errors_24h'] = len(window_24h)
        record['total_errors_7d'] = len(window_7d)
        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Maintenance features
# ---------------------------------------------------------------------------

def compute_days_since_maintenance(maintenance: pd.DataFrame, telemetry_dates: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days since last maintenance for each component at each machine-day.

    Parameters
    ----------
    maintenance     : raw maintenance DataFrame
    telemetry_dates : DataFrame with columns ['machineID', 'date']

    Returns
    -------
    DataFrame with days_since_comp{1..4}_maint columns merged onto telemetry_dates.
    """
    components = maintenance['comp'].unique()
    records = []

    for _, row in telemetry_dates.iterrows():
        machine = row['machineID']
        date = row['date']
        machine_maint = maintenance[maintenance['machineID'] == machine]

        record = {'machineID': machine, 'date': date}
        for comp in components:
            comp_history = machine_maint[
                (machine_maint['comp'] == comp) &
                (machine_maint['datetime'] < date)
            ]
            if len(comp_history) == 0:
                record[f'days_since_{comp}_maint'] = np.nan
            else:
                last_maint = comp_history['datetime'].max()
                record[f'days_since_{comp}_maint'] = (date - last_maint).days
        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def create_target_variable(telemetry_dates: pd.DataFrame, failures: pd.DataFrame,
                           horizon_hours: int = 24) -> pd.DataFrame:
    """
    Create binary target: will a failure occur within the next `horizon_hours`?

    Parameters
    ----------
    telemetry_dates : DataFrame with columns ['machineID', 'date']
    failures        : raw failures DataFrame
    horizon_hours   : prediction window in hours (default 24)

    Returns
    -------
    telemetry_dates with additional column 'failure_within_horizon' (0 or 1)
    and 'failure_component' (which component, or 'none').
    """
    df = telemetry_dates.copy()
    df['failure_within_horizon'] = 0
    df['failure_component'] = 'none'

    for idx, row in df.iterrows():
        machine = row['machineID']
        date = row['date']
        future_window = failures[
            (failures['machineID'] == machine) &
            (failures['datetime'] >= date) &
            (failures['datetime'] < date + pd.Timedelta(hours=horizon_hours))
        ]
        if len(future_window) > 0:
            df.at[idx, 'failure_within_horizon'] = 1
            df.at[idx, 'failure_component'] = future_window.iloc[0]['failure']

    return df


# ---------------------------------------------------------------------------
# Machine metadata
# ---------------------------------------------------------------------------

def encode_machine_metadata(machines: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'model' column; keep 'age' as numeric.

    Returns
    -------
    DataFrame with machineID, age, and model_* dummy columns.
    """
    return pd.get_dummies(machines, columns=['model'], drop_first=False)
