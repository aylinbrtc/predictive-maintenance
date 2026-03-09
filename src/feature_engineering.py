"""
feature_engineering.py
-----------------------
Vectorised pipeline: transforms 5 raw tables into a single ML-ready feature matrix.

Steps
-----
1. aggregate_telemetry   — hourly → daily stats + rolling windows
2. build_error_features  — daily & 7-day cumulative error counts
3. build_maint_features  — days since last maintenance per component
4. create_target         — binary label: failure within the next 24 h
5. encode_machines       — one-hot model type, keep age
6. merge_all             — assemble everything into one DataFrame
"""

import numpy as np
import pandas as pd


SENSORS = ['volt', 'rotate', 'pressure', 'vibration']


def aggregate_telemetry(telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly telemetry to daily statistics per machine.

    For each sensor computes daily mean, std, min, max.
    Adds 3-day and 7-day rolling mean and std of the daily mean.

    Parameters
    ----------
    telemetry : raw PdM_telemetry DataFrame (datetime, machineID, sensors...)

    Returns
    -------
    DataFrame with one row per (machineID, date), shape ~(36600, 34).
    """
    df = telemetry.copy()
    df['date'] = df['datetime'].dt.normalize()

    agg_dict = {s: ['mean', 'std', 'min', 'max'] for s in SENSORS}
    daily = df.groupby(['machineID', 'date']).agg(agg_dict)
    daily.columns = ['_'.join(col) for col in daily.columns]
    daily = daily.reset_index().sort_values(['machineID', 'date'])

    mean_cols = [f'{s}_mean' for s in SENSORS]
    for window in [3, 7]:
        rolled = (daily.groupby('machineID')[mean_cols]
                  .transform(lambda x: x.rolling(window, min_periods=1).mean()))
        rolled_std = (daily.groupby('machineID')[mean_cols]
                      .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0)))
        for col in mean_cols:
            daily[f'{col}_rolling{window}']     = rolled[col]
            daily[f'{col}_rolling{window}_std'] = rolled_std[col]

    return daily


def build_error_features(errors: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily and 7-day cumulative error count features.

    Parameters
    ----------
    errors : raw PdM_errors DataFrame (datetime, machineID, errorID)

    Returns
    -------
    DataFrame with columns: machineID, date, error{1-5}_count,
    total_errors, and their _7d rolling-sum counterparts.
    Only rows with at least one error event are present — merge with
    the telemetry base using a left join and fill NaN with 0.
    """
    df = errors.copy()
    df['date'] = df['datetime'].dt.normalize()

    err_daily = (df.groupby(['machineID', 'date', 'errorID'])
                 .size().unstack(fill_value=0))
    err_daily.columns = [f'{c}_count' for c in err_daily.columns]
    err_daily = err_daily.reset_index()
    err_daily['total_errors'] = err_daily[[c for c in err_daily.columns if '_count' in c]].sum(axis=1)

    err_daily = err_daily.sort_values(['machineID', 'date'])
    base_cols = [c for c in err_daily.columns if '_count' in c or c == 'total_errors']
    for col in base_cols:
        err_daily[f'{col}_7d'] = (err_daily.groupby('machineID')[col]
                                  .transform(lambda x: x.rolling(7, min_periods=1).sum()))
    return err_daily


def build_maint_features(maintenance: pd.DataFrame, machines: pd.DataFrame,
                         date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute days since last maintenance per component for every machine-day.

    Days before the first recorded maintenance event for a component are
    filled with -1 (sentinel: 'no prior maintenance').

    Parameters
    ----------
    maintenance : raw PdM_maint DataFrame (datetime, machineID, comp)
    machines    : raw PdM_machines DataFrame (machineID, ...)
    date_range  : DatetimeIndex covering the full observation period

    Returns
    -------
    DataFrame with columns: machineID, date, days_since_comp{1..4}.
    Shape: len(machines) × len(date_range).
    """
    maint = maintenance.copy()
    maint['date'] = maint['datetime'].dt.normalize()

    grid = pd.MultiIndex.from_product(
        [machines['machineID'].unique(), date_range],
        names=['machineID', 'date']
    )
    grid_df = pd.DataFrame(index=grid).reset_index()

    for comp in sorted(maint['comp'].unique()):
        comp_dates = (maint[maint['comp'] == comp][['machineID', 'date']]
                      .drop_duplicates().assign(maint_flag=1))
        grid_df = grid_df.merge(comp_dates, on=['machineID', 'date'], how='left')
        grid_df = grid_df.sort_values(['machineID', 'date'])
        grid_df['_last_maint'] = grid_df['date'].where(grid_df['maint_flag'] == 1)
        grid_df['_last_maint'] = grid_df.groupby('machineID')['_last_maint'].ffill()
        grid_df[f'days_since_{comp}'] = (grid_df['date'] - grid_df['_last_maint']).dt.days
        grid_df = grid_df.drop(columns=['maint_flag', '_last_maint'])

    maint_cols = [c for c in grid_df.columns if 'days_since' in c]
    grid_df[maint_cols] = grid_df[maint_cols].fillna(-1)
    return grid_df


def create_target(failures: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """
    Create a binary target variable: will a failure occur within `horizon_days`?

    The label is attached to the day *before* the failure, giving a
    ~24-hour prediction window when horizon_days=1.

    Parameters
    ----------
    failures      : raw PdM_failures DataFrame (datetime, machineID, failure)
    horizon_days  : number of days to look ahead (default 1)

    Returns
    -------
    DataFrame with columns: machineID, date, failure_within_24h (=1),
    failure_component (first component to fail in the window).
    Only positive examples are returned; merge with left join and fill 0.
    """
    df = failures.copy()
    df['date'] = df['datetime'].dt.normalize()
    df['prediction_date'] = df['date'] - pd.Timedelta(days=horizon_days)

    target = (df.groupby(['machineID', 'prediction_date'])
              .agg(failure_within_24h=('failure', 'count'),
                   failure_component=('failure', 'first'))
              .reset_index()
              .rename(columns={'prediction_date': 'date'}))
    target['failure_within_24h'] = 1
    return target


def encode_machines(machines: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'model' column; keep 'age' as numeric.

    Returns
    -------
    DataFrame with machineID, age, and model_* binary indicator columns.
    """
    enc = pd.get_dummies(machines, columns=['model'], drop_first=False)
    bool_cols = enc.select_dtypes(bool).columns
    enc[bool_cols] = enc[bool_cols].astype(int)
    return enc


def merge_all(telemetry: pd.DataFrame, errors: pd.DataFrame,
              failures: pd.DataFrame, maintenance: pd.DataFrame,
              machines: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline and return the final dataset.

    Parameters
    ----------
    All five raw DataFrames (see data_loader.py).

    Returns
    -------
    Merged DataFrame ready for train/test splitting.
    Columns: machineID, date, [55 features], failure_within_24h, failure_component.
    """
    daily = aggregate_telemetry(telemetry)

    err_feats = build_error_features(errors)
    error_fill_cols = [c for c in err_feats.columns if c not in ['machineID', 'date']]

    date_range = pd.date_range(
        telemetry['datetime'].min().normalize(),
        telemetry['datetime'].max().normalize(), freq='D'
    )
    maint_feats = build_maint_features(maintenance, machines, date_range)

    target = create_target(failures)
    mach_enc = encode_machines(machines)

    df = daily.copy()
    df = df.merge(err_feats, on=['machineID', 'date'], how='left')
    df[error_fill_cols] = df[error_fill_cols].fillna(0)
    df = df.merge(maint_feats, on=['machineID', 'date'], how='left')
    df = df.merge(mach_enc, on='machineID', how='left')
    df = df.merge(
        target[['machineID', 'date', 'failure_within_24h', 'failure_component']],
        on=['machineID', 'date'], how='left'
    )
    df['failure_within_24h'] = df['failure_within_24h'].fillna(0).astype(int)
    df['failure_component']  = df['failure_component'].fillna('none')
    return df
