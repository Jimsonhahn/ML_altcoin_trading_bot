"""
Data Sources Package.
Stellt Zugriff auf verschiedene Datenquellen für den Trading Bot bereit.
"""

# Base classes only - DataManager should be imported directly to avoid circular imports
from data_sources.base import DataSourceBase, DataSourceException

# Import DataManager only when needed to avoid circular import issues
def get_data_manager():
    """Lazy import of DataManager to avoid circular imports"""
    from data_sources.data_manager import DataManager
    return DataManager