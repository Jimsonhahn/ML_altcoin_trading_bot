"""
Trading Bot API Package
======================

Flask-based REST API for the Altcoin Trading Bot.
Provides endpoints for trading control, monitoring, and configuration.
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Team"

from flask import Flask

def create_app():
    """
    Application factory pattern for creating Flask app
    """
    from .app import create_app as _create_app
    return _create_app()

__all__ = ['create_app']