#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug-Skript zur Überprüfung von Importpfaden und Modulstruktur.
"""

import os
import sys
import importlib
import logging
import traceback

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("import_debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("import_debug")


def print_separator(title=None):
    """Gibt einen Separator mit optionalem Titel aus"""
    width = 80
    if title:
        side_width = (width - len(title) - 2) // 2
        print("=" * side_width + f" {title} " + "=" * (side_width if len(title) % 2 == 0 else side_width + 1))
    else:
        print("=" * width)


def check_python_environment():
    """Überprüft die Python-Umgebung"""
    print_separator("PYTHON-UMGEBUNG")

    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {sys.platform}")

    if hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix:
        print("Virtuelle Umgebung: Ja")
        print(f"Virtuelle Umgebung Pfad: {sys.prefix}")
    else:
        print("Virtuelle Umgebung: Nein")


def check_sys_path():
    """Überprüft sys.path"""
    print_separator("PYTHON-PFADE")

    print("sys.path (Suchpfade für Module):")
    for i, path in enumerate(sys.path):
        exists = os.path.exists(path)
        print(f"  {i}: {path} {'(existiert)' if exists else '(existiert NICHT)'}")

    # Aktuelles Arbeitsverzeichnis
    print(f"\nAktuelles Arbeitsverzeichnis: {os.getcwd()}")

    # Prüfen, ob das Projektverzeichnis im Pfad ist
    project_path = os.path.dirname(os.path.abspath(__file__))
    if project_path in sys.path:
        print(f"Projektverzeichnis ist im PYTHONPATH: {project_path}")
    else:
        print(f"Projektverzeichnis ist NICHT im PYTHONPATH: {project_path}")
        print("Dies könnte Importprobleme verursachen!")


def check_project_structure():
    """Untersucht die Projektstruktur"""
    print_separator("PROJEKTSTRUKTUR")

    project_path = os.path.dirname(os.path.abspath(__file__))
    print(f"Projektverzeichnis: {project_path}")

    # Wichtige Verzeichnisse überprüfen
    important_dirs = [
        "config", "core", "data_sources", "ml_components", "strategies",
        "data", "data/market_data", "data/ml_models"
    ]

    print("\nÜberprüfung wichtiger Verzeichnisse:")
    for dir_name in important_dirs:
        dir_path = os.path.join(project_path, dir_name)
        exists = os.path.exists(dir_path)

        if exists:
            if dir_name in ["config", "core", "data_sources", "ml_components", "strategies"]:
                init_file = os.path.join(dir_path, "__init__.py")
                has_init = os.path.exists(init_file)
                print(f"  - {dir_name}: Existiert, __init__.py: {'Vorhanden' if has_init else 'FEHLT'}")
            else:
                print(f"  - {dir_name}: Existiert")
        else:
            print(f"  - {dir_name}: FEHLT")


def test_imports():
    """Testet die Importe kritischer Module"""
    print_separator("IMPORT-TESTS")

    modules_to_test = [
        # Basis-Module
        "config.settings",
        "core.enhanced_backtesting",
        "core.position",
        "data_sources",

        # Strategie-Module
        "strategies.strategy_base",
        "strategies.momentum",

        # ML-Module
        "ml_components.market_regime",
        "ml_components.asset_clusters"
    ]

    results = {}

    for module_name in modules_to_test:
        print(f"\nTeste Import: {module_name}")
        try:
            module = importlib.import_module(module_name)
            file_path = getattr(module, '__file__', 'Unbekannt')
            print(f"  ✓ Import erfolgreich")
            print(f"  Dateipfad: {file_path}")
            results[module_name] = True
        except ImportError as e:
            print(f"  ✗ Import fehlgeschlagen: {e}")

            # Pfad überprüfen
            module_path = module_name.replace('.', '/')
            project_path = os.path.dirname(os.path.abspath(__file__))
            expected_path = os.path.join(project_path, module_path + '.py')

            if os.path.exists(expected_path):
                print(f"  Datei existiert: {expected_path}, aber kann nicht importiert werden")
            else:
                print(f"  Erwartete Datei nicht gefunden: {expected_path}")

            results[module_name] = False

    # Zusammenfassung
    print_separator("IMPORT-ZUSAMMENFASSUNG")

    successful = [m for m, r in results.items() if r]
    failed = [m for m, r in results.items() if not r]

    print(f"Erfolgreiche Importe: {len(successful)}/{len(modules_to_test)}")
    print(f"Fehlgeschlagene Importe: {len(failed)}/{len(modules_to_test)}")

    if failed:
        print("\nFehlgeschlagene Module:")
        for module in failed:
            print(f"  - {module}")

    return results


def fix_common_issues():
    """Versucht, häufige Importprobleme zu beheben"""
    print_separator("PROBLEMBEHEBUNG")

    project_path = os.path.dirname(os.path.abspath(__file__))

    # 1. Überprüfen und erstellen fehlender Verzeichnisse und __init__.py Dateien
    important_dirs = [
        "config", "core", "data_sources", "ml_components", "strategies",
        "data", "data/market_data", "data/ml_models"
    ]

    fixes_applied = []

    print("Überprüfe und erstelle fehlende Verzeichnisse und __init__.py Dateien...")
    for dir_name in important_dirs:
        dir_path = os.path.join(project_path, dir_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"  Verzeichnis erstellt: {dir_path}")
            fixes_applied.append(f"Verzeichnis erstellt: {dir_path}")

        if dir_name in ["config", "core", "data_sources", "ml_components", "strategies"]:
            init_file = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("# Automatisch erstellt von debug_imports.py\n")
                print(f"  __init__.py erstellt: {init_file}")
                fixes_applied.append(f"__init__.py erstellt: {init_file}")

    # 2. Projekt-Root zum PYTHONPATH hinzufügen
    if project_path not in sys.path:
        sys.path.append(project_path)
        print(f"Projektverzeichnis zum PYTHONPATH hinzugefügt: {project_path}")
        fixes_applied.append(f"Projektverzeichnis zum PYTHONPATH hinzugefügt: {project_path}")

    # 3. Fix-Imports-Datei erstellen
    fix_imports_path = os.path.join(project_path, "fix_imports.py")
    with open(fix_imports_path, 'w') as f:
        f.write('''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Import-Fix für das Trading-Bot-Projekt.
"""

import os
import sys

# Projektverzeichnis zum Pfad hinzufügen
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)
    print(f"Projektverzeichnis zum PYTHONPATH hinzugefügt: {project_path}")
''')

    print(f"Fix-Imports-Datei erstellt: {fix_imports_path}")
    fixes_applied.append(f"Fix-Imports-Datei erstellt: {fix_imports_path}")

    if fixes_applied:
        print("\nAngewendete Fixes:")
        for fix in fixes_applied:
            print(f"  - {fix}")
    else:
        print("Keine Fixes angewendet.")

    return fixes_applied


def main():
    """Hauptfunktion"""
    # Fügen Sie das Projektverzeichnis zum Pfad hinzu
    project_path = os.path.dirname(os.path.abspath(__file__))
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
        print(f"Projektverzeichnis zum PYTHONPATH hinzugefügt: {project_path}")

    print("===== IMPORT-PFADE UND MODULSTRUKTUR DEBUG-TOOL =====")

    try:
        # Python-Umgebung überprüfen
        check_python_environment()

        # sys.path überprüfen
        check_sys_path()

        # Projektstruktur überprüfen
        check_project_structure()

        # Importe testen
        import_results = test_imports()

        # Probleme beheben
        if not all(import_results.values()):
            print("\nEs wurden Importprobleme gefunden. Sollen automatische Fixes angewendet werden? (j/n)")
            choice = input().lower()

            if choice == 'j':
                fixes_applied = fix_common_issues()
            else:
                print("Keine automatischen Fixes angewendet.")

        print_separator("FERTIG")
        print("Debug-Vorgang abgeschlossen. Überprüfen Sie die Ausgabe und die Logdatei 'import_debug.log'.")

    except Exception as e:
        logger.error(f"Fehler bei der Ausführung des Debug-Tools: {e}")
        logger.error(traceback.format_exc())
        print(f"\nFEHLER: {e}")
        print("Siehe Logdatei 'import_debug.log' für Details.")


if __name__ == "__main__":
    main()