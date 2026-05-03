"""
Custom CSS Styles.

Centralized styling for the interface.
"""


def get_custom_css() -> str:
    """Return custom CSS styles."""
    return """
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .success-box {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 10px 0;
        }
        .info-box {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
    """