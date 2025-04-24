#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript zum Aufräumen redundanter Dateien und Verzeichnisse im Projekt.
"""

import os
import shutil
import json
import argparse
import glob
from datetime import datetime


def is_directory_empty(directory):
    """Prüft, ob ein Verzeichnis leer ist."""
    return len(os.listdir(directory)) == 0


def find_nested_data_dirs(start_dir="data"):
    """Findet verschachtelte data-Verzeichnisse."""
    nested_dirs = []

    for root, dirs, _ in os.walk(start_dir):
        for d in dirs:
            if d == 'data':
                nested_path = os.path.join(root, d)
                nested_dirs.append(nested_path)

    return nested_dirs


def find_duplicate_registries(main_registry_path="data/backtest_registry.json"):
    """Findet doppelte Registry-Dateien."""
    dup_registries = []

    for root, _, files in os.walk('.'):
        for f in files:
            if f == 'backtest_registry.json':
                file_path = os.path.join(root, f)
                if os.path.abspath(file_path) != os.path.abspath(main_registry_path):
                    dup_registries.append(file_path)

    return dup_registries


def find_old_backtest_dirs(main_results_dir="data/backtest_results"):
    """Findet alte Backtest-Ergebnisverzeichnisse."""
    old_dirs = []

    if not os.path.exists(main_results_dir):
        return old_dirs

    # Hauptergebnisverzeichnis und sein Inhalt
    main_result_contents = set(os.listdir(main_results_dir)) if os.path.exists(main_results_dir) else set()

    # Finde alle anderen backtest_results-Verzeichnisse
    for root, dirs, _ in os.walk('.'):
        for d in dirs:
            if d == 'backtest_results' or d == 'results':
                dir_path = os.path.join(root, d)
                if os.path.abspath(dir_path) != os.path.abspath(main_results_dir):
                    old_dirs.append(dir_path)

    return old_dirs


def find_pycache_dirs():
    """Findet alle __pycache__-Verzeichnisse."""
    pycache_dirs = []

    for root, dirs, _ in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                pycache_dirs.append(os.path.join(root, d))

    return pycache_dirs


def find_duplicate_code_files(main_data_sources_dir="core/data_sources"):
    """Findet doppelte Python-Dateien (z.B. in data_sources und core/data_sources)."""
    dup_files = []

    if not os.path.exists(main_data_sources_dir):
        return dup_files

    # Dateinamen im Hauptverzeichnis
    main_files = set(os.listdir(main_data_sources_dir)) if os.path.exists(main_data_sources_dir) else set()
    main_files = {f for f in main_files if f.endswith('.py')}

    # Ursprüngliches data_sources-Verzeichnis
    old_dir = 'data_sources'
    if os.path.exists(old_dir) and os.path.isdir(old_dir):
        for file in os.listdir(old_dir):
            if file.endswith('.py') and file in main_files:
                dup_files.append(os.path.join(old_dir, file))

    return dup_files


def cleanup_project(args):
    """Führt die Bereinigung des Projekts durch."""
    total_removed = 0
    total_saved = 0

    log_lines = [f"=== Projektbereinigung gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"]

    # 1. Verschachtelte data-Verzeichnisse
    nested_data_dirs = find_nested_data_dirs()
    log_lines.append(f"\n--- Verschachtelte Data-Verzeichnisse: {len(nested_data_dirs)} gefunden ---")

    for dir_path in nested_data_dirs:
        contents = os.listdir(dir_path)
        space_saved = 0

        # Größe berechnen
        for item in contents:
            item_path = os.path.join(dir_path, item)
            try:
                if os.path.isfile(item_path):
                    space_saved += os.path.getsize(item_path)
                elif os.path.isdir(item_path):
                    for dirpath, _, filenames in os.walk(item_path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            space_saved += os.path.getsize(fp)
            except Exception as e:
                log_lines.append(f"  Fehler bei der Größenberechnung für {item_path}: {e}")

        total_saved += space_saved

        action = 'GELÖSCHT' if args.execute else 'WÜRDE LÖSCHEN'
        log_lines.append(f"  {action}: {dir_path} ({space_saved / 1024:.1f} KB)")

        if args.execute:
            try:
                shutil.rmtree(dir_path)
                total_removed += 1
            except Exception as e:
                log_lines.append(f"  FEHLER beim Löschen von {dir_path}: {e}")

    # 2. Doppelte Registry-Dateien
    dup_registries = find_duplicate_registries()
    log_lines.append(f"\n--- Doppelte Registry-Dateien: {len(dup_registries)} gefunden ---")

    for file_path in dup_registries:
        try:
            file_size = os.path.getsize(file_path)
            total_saved += file_size

            action = 'GELÖSCHT' if args.execute else 'WÜRDE LÖSCHEN'
            log_lines.append(f"  {action}: {file_path} ({file_size / 1024:.1f} KB)")

            if args.execute:
                os.remove(file_path)
                total_removed += 1
        except Exception as e:
            log_lines.append(f"  FEHLER beim Löschen von {file_path}: {e}")

    # 3. Alte Backtest-Verzeichnisse
    old_dirs = find_old_backtest_dirs()
    log_lines.append(f"\n--- Alte Backtest-Verzeichnisse: {len(old_dirs)} gefunden ---")

    for dir_path in old_dirs:
        space_saved = 0

        # Größe berechnen
        try:
            for dirpath, _, filenames in os.walk(dir_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    space_saved += os.path.getsize(fp)
        except Exception as e:
            log_lines.append(f"  Fehler bei der Größenberechnung für {dir_path}: {e}")

        total_saved += space_saved

        action = 'GELÖSCHT' if args.execute else 'WÜRDE LÖSCHEN'
        log_lines.append(f"  {action}: {dir_path} ({space_saved / 1024:.1f} KB)")

        if args.execute:
            try:
                shutil.rmtree(dir_path)
                total_removed += 1
            except Exception as e:
                log_lines.append(f"  FEHLER beim Löschen von {dir_path}: {e}")

    # 4. __pycache__ Verzeichnisse
    if args.pycache:
        pycache_dirs = find_pycache_dirs()
        log_lines.append(f"\n--- __pycache__ Verzeichnisse: {len(pycache_dirs)} gefunden ---")

        for dir_path in pycache_dirs:
            space_saved = 0

            # Größe berechnen
            try:
                for f in os.listdir(dir_path):
                    fp = os.path.join(dir_path, f)
                    if os.path.isfile(fp):
                        space_saved += os.path.getsize(fp)
            except Exception as e:
                log_lines.append(f"  Fehler bei der Größenberechnung für {dir_path}: {e}")

            total_saved += space_saved

            action = 'GELÖSCHT' if args.execute else 'WÜRDE LÖSCHEN'
            log_lines.append(f"  {action}: {dir_path} ({space_saved / 1024:.1f} KB)")

            if args.execute:
                try:
                    shutil.rmtree(dir_path)
                    total_removed += 1
                except Exception as e:
                    log_lines.append(f"  FEHLER beim Löschen von {dir_path}: {e}")

    # 5. Doppelte Python-Dateien in data_sources
    if args.duplicate_code:
        dup_files = find_duplicate_code_files()
        log_lines.append(f"\n--- Doppelte Python-Dateien: {len(dup_files)} gefunden ---")

        for file_path in dup_files:
            try:
                file_size = os.path.getsize(file_path)
                total_saved += file_size

                action = 'GELÖSCHT' if args.execute else 'WÜRDE LÖSCHEN'
                log_lines.append(f"  {action}: {file_path} ({file_size / 1024:.1f} KB)")

                if args.execute:
                    os.remove(file_path)
                    total_removed += 1
            except Exception as e:
                log_lines.append(f"  FEHLER beim Löschen von {file_path}: {e}")

    # Zusammenfassung
    log_lines.append(f"\n=== Zusammenfassung ===")
    action = "Gelöscht" if args.execute else "Würde löschen"
    log_lines.append(f"{action}: {total_removed} Verzeichnisse/Dateien")
    log_lines.append(f"Gespeicherter Speicherplatz: {total_saved / 1024:.1f} KB ({total_saved / (1024 * 1024):.2f} MB)")
    log_lines.append(f"\n=== Bereinigung abgeschlossen: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # Log-Datei schreiben
    logfile = f"cleanup_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(logfile, 'w') as f:
        f.write('\n'.join(log_lines))

    print(f"Log gespeichert in: {logfile}")

    # Log auch auf der Konsole ausgeben
    for line in log_lines:
        print(line)


def main():
    parser = argparse.ArgumentParser(description='Projekt aufräumen und überflüssige Dateien entfernen')
    parser.add_argument('--execute', action='store_true', help='Änderungen wirklich durchführen (sonst nur Vorschau)')
    parser.add_argument('--pycache', action='store_true', help='__pycache__ Verzeichnisse entfernen')
    parser.add_argument('--duplicate-code', action='store_true',
                        help='Doppelte Python-Dateien in data_sources entfernen')

    args = parser.parse_args()

    if not args.execute:
        print("HINWEIS: Dies ist nur eine Vorschau. Verwende --execute, um die Änderungen wirklich durchzuführen.")

    cleanup_project(args)


if __name__ == "__main__":
    main()