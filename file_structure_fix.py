#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dieses Skript überprüft und korrigiert die Dateistruktur für das data_sources Modul.
"""

import os
import shutil
import sys


def check_and_fix_structure():
    """Überprüft und korrigiert die Dateistruktur."""

    # Der Pfad vom aktuellen Verzeichnis zum Projekt-Root
    current_dir = os.getcwd()
    print(f"Aktuelles Verzeichnis: {current_dir}")

    # Prüfen, ob wir im Projekt-Root sind oder in einem Unterverzeichnis
    if os.path.basename(current_dir) == 'altcoin_trading_bot':
        project_root = current_dir
    else:
        # Wir müssen den Projekt-Root finden
        potential_root = current_dir
        while potential_root and not os.path.exists(os.path.join(potential_root, 'core')) and not os.path.exists(
                os.path.join(potential_root, 'data_sources')):
            parent = os.path.dirname(potential_root)
            if parent == potential_root:  # Wir sind am Filesystem-Root angekommen
                break
            potential_root = parent

        if os.path.exists(os.path.join(potential_root, 'core')) and os.path.exists(
                os.path.join(potential_root, 'data_sources')):
            project_root = potential_root
        else:
            print("FEHLER: Konnte den Projekt-Root nicht finden.")
            return False

    print(f"Projekt-Root: {project_root}")

    # Prüfen, ob die erforderlichen Verzeichnisse existieren
    data_sources_dir = os.path.join(project_root, 'data_sources')
    if not os.path.exists(data_sources_dir):
        print(f"FEHLER: Verzeichnis {data_sources_dir} existiert nicht.")
        return False

    # Prüfen, ob eine verschachtelte Struktur existiert
    nested_data_sources_dir = os.path.join(data_sources_dir, 'data_sources')
    has_nested_structure = os.path.exists(nested_data_sources_dir)

    # Prüfen, ob die erforderlichen Dateien existieren
    required_files = ['base.py', 'data_manager.py', 'binance_source.py', 'coingecko_source.py']

    # Zunächst prüfen wir, ob die Dateien im Hauptverzeichnis existieren
    files_in_main = [f for f in required_files if os.path.exists(os.path.join(data_sources_dir, f))]
    print(f"Dateien im Hauptverzeichnis data_sources: {files_in_main}")

    # Wenn es ein verschachteltes Verzeichnis gibt, prüfen wir dort auch
    files_in_nested = []
    if has_nested_structure:
        files_in_nested = [f for f in required_files if os.path.exists(os.path.join(nested_data_sources_dir, f))]
        print(f"Dateien im verschachtelten Verzeichnis data_sources/data_sources: {files_in_nested}")

    # Wir müssen alle Dateien aus dem verschachtelten Verzeichnis ins Hauptverzeichnis kopieren
    if has_nested_structure and files_in_nested:
        print("Kopiere Dateien aus dem verschachtelten Verzeichnis ins Hauptverzeichnis...")

        for file in files_in_nested:
            source = os.path.join(nested_data_sources_dir, file)
            destination = os.path.join(data_sources_dir, file)

            # Wenn die Datei bereits im Hauptverzeichnis existiert, sie umbenennen
            if os.path.exists(destination):
                backup = f"{destination}.bak"
                print(f"Die Datei {file} existiert bereits im Hauptverzeichnis. Sichere als {backup}.")
                shutil.copy2(destination, backup)

            # Datei kopieren
            print(f"Kopiere {source} nach {destination}")
            shutil.copy2(source, destination)
    else:
        print("Keine Dateien zum Kopieren gefunden oder kein verschachteltes Verzeichnis.")

    # Jetzt müssen wir die __init__.py-Datei überprüfen und ggf. korrigieren
    init_file = os.path.join(data_sources_dir, '__init__.py')
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            content = f.read()

        # Prüfen und korrigieren der Import-Anweisungen
        if 'from data_sources.data_sources.' in content:
            print("Korrigiere Import-Anweisungen in __init__.py...")
            fixed_content = content.replace('from data_sources.data_sources.', 'from data_sources.')

            # Sicherheitskopie erstellen
            shutil.copy2(init_file, f"{init_file}.bak")

            # Korrigierte Datei schreiben
            with open(init_file, 'w') as f:
                f.write(fixed_content)

            print("__init__.py wurde korrigiert.")
        else:
            print("Die Import-Anweisungen in __init__.py scheinen bereits korrekt zu sein.")
    else:
        print(f"FEHLER: Die Datei {init_file} existiert nicht.")
        return False

    print("\nDie Dateistruktur wurde überprüft und korrigiert.")
    print("Bitte führen Sie Ihren Code erneut aus, um zu sehen, ob die Fehler behoben wurden.")

    return True


if __name__ == "__main__":
    check_and_fix_structure()