"""
data_loader.py
--------------
Functions to load and merge the 5 Microsoft Azure Predictive Maintenance tables.
"""

import os
import pandas as pd


DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')


def load_telemetry(data_dir: str = DATA_RAW_DIR) -> pd.DataFrame:
    """Load PdM_telemetry.csv and parse datetime column."""
    path = os.path.join(data_dir, 'PdM_telemetry.csv')
    df = pd.read_csv(path, parse_dates=['datetime'])
    return df


def load_errors(data_dir: str = DATA_RAW_DIR) -> pd.DataFrame:
    """Load PdM_errors.csv and parse datetime column."""
    path = os.path.join(data_dir, 'PdM_errors.csv')
    df = pd.read_csv(path, parse_dates=['datetime'])
    return df


def load_failures(data_dir: str = DATA_RAW_DIR) -> pd.DataFrame:
    """Load PdM_failures.csv and parse datetime column."""
    path = os.path.join(data_dir, 'PdM_failures.csv')
    df = pd.read_csv(path, parse_dates=['datetime'])
    return df


def load_maintenance(data_dir: str = DATA_RAW_DIR) -> pd.DataFrame:
    """Load PdM_maint.csv and parse datetime column."""
    path = os.path.join(data_dir, 'PdM_maint.csv')
    df = pd.read_csv(path, parse_dates=['datetime'])
    return df


def load_machines(data_dir: str = DATA_RAW_DIR) -> pd.DataFrame:
    """Load PdM_machines.csv (machine metadata — 100 rows)."""
    path = os.path.join(data_dir, 'PdM_machines.csv')
    df = pd.read_csv(path)
    return df


def load_all(data_dir: str = DATA_RAW_DIR) -> dict:
    """
    Load all 5 tables and return them as a dict.

    Returns
    -------
    dict with keys: 'telemetry', 'errors', 'failures', 'maintenance', 'machines'
    """
    return {
        'telemetry': load_telemetry(data_dir),
        'errors': load_errors(data_dir),
        'failures': load_failures(data_dir),
        'maintenance': load_maintenance(data_dir),
        'machines': load_machines(data_dir),
    }
