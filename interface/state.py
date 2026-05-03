"""
Session State Management.

Centralized management of Streamlit session state variables.
"""

import streamlit as st
from typing import Optional, Any
import pandas as pd


def initialize_state():
    """Initialize all session state variables."""
    defaults = {
        "uploaded_file_path": None,
        "df_preview": None,
        "last_uploaded_file": None,
        "target_col": None,
        "task_type": "classification",
        "auto_task_type": None,
        "auto_detect_reason": "",
        "pipeline_report": None,
        "best_model": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_file(file, filepath: str, df: pd.DataFrame):
    """Save file info to state."""
    st.session_state.last_uploaded_file = file
    st.session_state.uploaded_file_path = filepath
    st.session_state.df_preview = df


def clear_file():
    """Clear file-related state."""
    st.session_state.uploaded_file_path = None
    st.session_state.df_preview = None
    st.session_state.last_uploaded_file = None
    st.session_state.pipeline_report = None
    st.session_state.best_model = None


def set_task_type(task_type: str, auto_type: str, reason: str):
    """Save task type info to state."""
    st.session_state.task_type = task_type
    st.session_state.auto_task_type = auto_type
    st.session_state.auto_detect_reason = reason


def set_report(report: dict, model=None):
    """Save pipeline results to state."""
    st.session_state.pipeline_report = report
    st.session_state.best_model = model