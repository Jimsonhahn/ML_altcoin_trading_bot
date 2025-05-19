#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Import-Validator für den Trading Bot.
Prüft, ob alle Module die korrekten Importstrukturen verwenden.
"""

import os
import sys
import re
from typing import List, Dict, Any, Tuple

# Farbcodes für die Ausgabe
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check_file(filepath: str) -> Tuple[bool, List[str], List[str]]:
    """
    Überprüft die Importstruktur einer Datei.

    Args:
        filepath: Pfad zur Datei

    Returns:
        Tuple aus: (Alles korrekt, Liste korrekte Importe, Liste problematische Importe)
    """
    correct_imports = []
    problematic_imports = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Nach problematischen data_sources Importen suchen
        data_sources_imports = re.findall(r'from\s+data_sources\.data_sources', content)
        for imp in data_sources_imports:
            problematic_imports.append(imp)

        # Nach korrekten data_sources Importen suchen
        correct_ds_imports = re.findall(r'from\s+data_sources\s+import', content)
        for imp in correct_ds_imports:
            correct_imports.append(imp)

        # Nach direkten Importen der data_sources Module suchen
        direct_imports = re.findall(r'from\s+data_sources\.[a-zA-Z_]+\s+import', content)
        for imp in direct_imports:
            correct_imports.append(imp)

        # Überprüfen, ob alles OK ist
        all_ok = len(problematic_imports) == 0

        return all_ok, correct_imports, problematic_imports

    except Exception as e:
        print(f"Fehler beim Lesen von {filepath}: {e}")
        return False, [], [f"Fehler: {e}"]


def scan_directory(directory: str) -> Dict[str, Any]:
    """
    Scannt ein Verzeichnis rekursiv nach Python-Dateien.

    Args:
        directory: Zu scannendes Verzeichnis

    Returns:
        Dictionary mit Ergebnissen
    """
    results = {
        "all_ok": True,
        "files_checked": 0,
        "problems_found": 0,
        "problem_files": []
    }

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)

                # Datei prüfen
                all_ok, correct, problematic = check_file(filepath)
                results["files_checked"] += 1

                if not all_ok:
                    results["all_ok"] = False
                    results["problems_found"] += 1
                    results["problem_files"].append({
                        "file": filepath,
                        "problematic_imports": problematic
                    })

    return results


def print_results(results: Dict[str, Any]) -> None:
    """
    Gibt die Ergebnisse der Überprüfung aus.

    Args:
        results: Ergebnisse der Überprüfung
    """
    print("\n===== IMPORT VALIDATION RESULTS =====")
    print(f"Files checked: {results['files_checked']}")

    if results["all_ok"]:
        print(f"{GREEN}✓ All imports are correct!{RESET}")
    else:
        print(f"{RED}✗ Found {results['problems_found']} files with problematic imports!{RESET}")

        for problem in results["problem_files"]:
            print(f"\n{YELLOW}File: {problem['file']}{RESET}")
            print("Problematic imports:")
            for imp in problem["problematic_imports"]:
                print(f"  {RED}- {imp}{RESET}")

            print("Recommended fix:")
            for imp in problem["problematic_imports"]:
                # Korrekten Import empfehlen
                fixed = imp.replace('data_sources.data_sources', 'data_sources')
                print(f"  {GREEN}+ {fixed}{RESET}")


def main():
    """
    Hauptfunktion.
    """
    # Aktuelles Verzeichnis
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Übergeordnetes Verzeichnis (Projekt-Root)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    print(f"Scanning project directory: {project_root}")

    # Verzeichnisse scannen
    directories_to_scan = [
        os.path.join(project_root, "core"),
        os.path.join(project_root, "data_sources"),
        os.path.join(project_root, "ml_components"),
        os.path.join(project_root, "strategies")
    ]

    all_results = {
        "all_ok": True,
        "files_checked": 0,
        "problems_found": 0,
        "problem_files": []
    }

    for directory in directories_to_scan:
        if os.path.exists(directory):
            print(f"Scanning {directory}...")
            results = scan_directory(directory)

            # Ergebnisse aggregieren
            all_results["files_checked"] += results["files_checked"]
            all_results["problems_found"] += results["problems_found"]
            all_results["problem_files"].extend(results["problem_files"])

            if not results["all_ok"]:
                all_results["all_ok"] = False
        else:
            print(f"Directory not found: {directory}")

    # Ergebnisse ausgeben
    print_results(all_results)

    # Exitcode setzen
    sys.exit(0 if all_results["all_ok"] else 1)


if __name__ == "__main__":
    main()