#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logger-Modul für den Trading Bot.
Konfiguriert und liefert Logger für verschiedene Komponenten.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Richtet einen Logger mit formatierter Konsolen- und optionaler Dateiausgabe ein.

    Args:
        level: Log-Level (INFO, DEBUG, etc.)
        log_file: Pfad zur Log-Datei (optional)

    Returns:
        Konfigurierter Logger
    """
    # Root-Logger konfigurieren
    logger = logging.getLogger()
    logger.setLevel(level)

    # Bestehende Handler entfernen, um Duplikate zu vermeiden
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Konsolen-Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Formatierung
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)

    # Handler zum Logger hinzufügen
    logger.addHandler(console_handler)

    # Datei-Handler, falls angegeben
    if log_file:
        # Verzeichnis erstellen, falls nicht vorhanden
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)

        logger.addHandler(file_handler)

    return logger


def get_dated_log_file(base_dir: str = 'logs') -> str:
    """
    Erstellt einen Dateinamen für eine Log-Datei mit aktuellem Datum und Uhrzeit.

    Args:
        base_dir: Basisverzeichnis für Log-Dateien

    Returns:
        Pfad zur Log-Datei
    """
    # Verzeichnis erstellen, falls nicht vorhanden
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    # Dateiname mit Datum und Uhrzeit
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"trading_bot_{timestamp}.log"

    return os.path.join(base_dir, filename)


def get_console_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Erstellt einen einfachen Logger nur mit Konsolenausgabe.

    Args:
        name: Name des Loggers
        level: Log-Level

    Returns:
        Konfigurierter Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Bestehende Handler entfernen, um Duplikate zu vermeiden
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Konsolen-Handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Formatierung
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
