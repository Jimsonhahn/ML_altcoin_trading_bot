#!/usr/bin/env python3
"""
Project Connection Analyzer
Analysiert Verbindungen zwischen Modulen und findet Probleme
"""

import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re


class ProjectConnectionAnalyzer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            "import_graph": defaultdict(set),
            "missing_imports": [],
            "circular_imports": [],
            "undefined_references": [],
            "unused_files": [],
            "config_usage": defaultdict(set),
            "class_inheritance": defaultdict(set),
            "function_calls": defaultdict(set),
            "critical_files": set(),
            "problem_summary": defaultdict(list)
        }

        # Module, die wir ignorieren
        self.ignore_dirs = {'.venv', 'venv', '__pycache__', '.git', 'node_modules',
                            '.idea', '.vscode', 'env', 'build', 'dist', '.trash_*'}

    def analyze(self):
        """Hauptanalyse-Funktion"""
        print("🔍 Starte Projekt-Verbindungsanalyse...")

        # 1. Sammle alle Python-Dateien
        python_files = self._collect_python_files()
        print(f"📁 Gefunden: {len(python_files)} Python-Dateien")

        # 2. Analysiere jede Datei
        file_data = {}
        for py_file in python_files:
            data = self._analyze_file(py_file)
            if data:
                file_data[str(py_file)] = data

        # 3. Finde Probleme
        self._find_missing_imports(file_data)
        self._find_circular_imports(file_data)
        self._find_undefined_references(file_data)
        self._find_unused_files(file_data, python_files)
        self._analyze_config_usage(file_data)
        self._identify_critical_files(file_data)

        # 4. Erstelle Zusammenfassung
        self._create_summary()

        return self.results

    def _collect_python_files(self) -> List[Path]:
        """Sammle alle Python-Dateien"""
        python_files = []

        for py_file in self.project_root.rglob("*.py"):
            # Ignoriere bestimmte Verzeichnisse
            if any(ignore_dir in py_file.parts for ignore_dir in self.ignore_dirs):
                continue

            python_files.append(py_file.relative_to(self.project_root))

        return sorted(python_files)

    def _analyze_file(self, file_path: Path) -> Dict:
        """Analysiere eine einzelne Datei"""
        try:
            with open(self.project_root / file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            data = {
                "imports": set(),
                "from_imports": defaultdict(set),
                "classes": {},
                "functions": set(),
                "calls": set(),
                "config_refs": set(),
                "has_main": False
            }

            # Durchlaufe AST
            for node in ast.walk(tree):
                # Imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        data["imports"].add(alias.name)

                # From Imports
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module
                        # Konvertiere relative Imports
                        if node.level > 0:
                            module = self._resolve_relative_import(file_path, node.module, node.level)
                        for alias in node.names:
                            data["from_imports"][module].add(alias.name)

                # Klassen
                elif isinstance(node, ast.ClassDef):
                    bases = [self._get_name(base) for base in node.bases]
                    data["classes"][node.name] = {
                        "bases": bases,
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    }

                # Funktionen
                elif isinstance(node, ast.FunctionDef):
                    data["functions"].add(node.name)
                    if node.name == "main":
                        data["has_main"] = True

                # Funktionsaufrufe
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        data["calls"].add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        data["calls"].add(f"{self._get_name(node.func.value)}.{node.func.attr}")

                # Config-Referenzen (suche nach config, settings, etc.)
                elif isinstance(node, ast.Attribute):
                    attr_str = self._get_name(node)
                    if any(conf in attr_str.lower() for conf in ['config', 'settings', 'env']):
                        data["config_refs"].add(attr_str)

            return data

        except Exception as e:
            print(f"⚠️  Fehler beim Analysieren von {file_path}: {e}")
            return None

    def _resolve_relative_import(self, file_path: Path, module: str, level: int) -> str:
        """Löse relative Imports auf"""
        parts = list(file_path.parts[:-1])  # Ohne Dateiname

        # Gehe 'level' Ebenen nach oben
        if level > len(parts):
            return module or ""

        parts = parts[:-level] if level > 0 else parts

        if module:
            parts.append(module)

        return ".".join(parts)

    def _get_name(self, node) -> str:
        """Extrahiere Namen aus AST-Knoten"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, str):
            return node
        return ""

    def _find_missing_imports(self, file_data: Dict):
        """Finde fehlende Imports"""
        print("🔍 Suche nach fehlenden Imports...")

        # Erstelle Mapping von verfügbaren Modulen
        available_modules = set()
        for file_path in file_data.keys():
            # Konvertiere Dateipfad zu Modul-Name
            module_path = Path(file_path).with_suffix('')
            module_name = str(module_path).replace(os.sep, '.')
            available_modules.add(module_name)

            # Füge auch Parent-Module hinzu
            parts = module_name.split('.')
            for i in range(1, len(parts)):
                available_modules.add('.'.join(parts[:i]))

        # Prüfe Imports
        for file_path, data in file_data.items():
            if not data:
                continue

            # Prüfe normale Imports
            for imp in data["imports"]:
                if imp.startswith('.'):
                    continue  # Relative Imports separat prüfen

                # Prüfe ob es ein internes Modul ist
                if any(imp.startswith(proj) for proj in
                       ['core', 'strategies', 'ml_components', 'utils', 'analysis', 'data_sources']):
                    if imp not in available_modules and not any(am.startswith(imp + '.') for am in available_modules):
                        self.results["missing_imports"].append({
                            "file": file_path,
                            "import": imp,
                            "type": "module"
                        })

            # Prüfe From-Imports
            for module, names in data["from_imports"].items():
                if not module or module.startswith('.'):
                    continue

                # Prüfe ob es ein internes Modul ist
                if any(module.startswith(proj) for proj in
                       ['core', 'strategies', 'ml_components', 'utils', 'analysis', 'data_sources']):
                    if module not in available_modules:
                        self.results["missing_imports"].append({
                            "file": file_path,
                            "import": f"from {module} import {', '.join(names)}",
                            "type": "from_import",
                            "module": module,
                            "names": list(names)
                        })

    def _find_circular_imports(self, file_data: Dict):
        """Finde zirkuläre Imports"""
        print("🔍 Suche nach zirkulären Imports...")

        # Baue Import-Graph
        import_graph = defaultdict(set)

        for file_path, data in file_data.items():
            if not data:
                continue

            module_name = str(Path(file_path).with_suffix('')).replace(os.sep, '.')

            # Füge alle Imports hinzu
            for imp in data["imports"]:
                if any(imp.startswith(proj) for proj in ['core', 'strategies', 'ml_components']):
                    import_graph[module_name].add(imp)

            for from_imp in data["from_imports"].keys():
                if any(from_imp.startswith(proj) for proj in ['core', 'strategies', 'ml_components']):
                    import_graph[module_name].add(from_imp)

        # Finde Zyklen mit DFS
        def find_cycles(node, path, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in import_graph.get(node, []):
                if neighbor not in visited:
                    if find_cycles(neighbor, path, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    # Zyklus gefunden
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    self.results["circular_imports"].append(cycle)
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        visited = set()
        for node in import_graph:
            if node not in visited:
                find_cycles(node, [], visited, set())

    def _find_undefined_references(self, file_data: Dict):
        """Finde undefinierte Referenzen"""
        print("🔍 Suche nach undefinierten Referenzen...")

        # Sammle alle verfügbaren Klassen und Funktionen
        available_items = defaultdict(set)

        for file_path, data in file_data.items():
            if not data:
                continue

            module_name = str(Path(file_path).with_suffix('')).replace(os.sep, '.')

            # Klassen
            for class_name in data["classes"].keys():
                available_items[module_name].add(class_name)

            # Funktionen
            for func_name in data["functions"]:
                available_items[module_name].add(func_name)

        # Prüfe Referenzen
        for file_path, data in file_data.items():
            if not data:
                continue

            # Was wird in dieser Datei importiert?
            imported_items = set()

            # Direkte Imports
            for imp in data["imports"]:
                imported_items.add(imp)

            # From Imports
            for module, names in data["from_imports"].items():
                for name in names:
                    if name != "*":
                        imported_items.add(name)
                    else:
                        # Bei * Import alles aus dem Modul verfügbar
                        if module in available_items:
                            imported_items.update(available_items[module])

            # Prüfe Calls
            for call in data["calls"]:
                # Ignoriere Built-ins und bekannte externe Libraries
                if call in ['print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set',
                            'open', 'range', 'enumerate', 'zip', 'map', 'filter']:
                    continue

                # Prüfe ob es importiert oder lokal definiert wurde
                if '.' not in call:  # Einfacher Funktionsaufruf
                    if (call not in imported_items and
                            call not in data["functions"] and
                            call not in data["classes"]):
                        self.results["undefined_references"].append({
                            "file": file_path,
                            "reference": call,
                            "type": "function_or_class"
                        })

    def _find_unused_files(self, file_data: Dict, all_files: List[Path]):
        """Finde ungenutzte Dateien"""
        print("🔍 Suche nach ungenutzten Dateien...")

        # Dateien, die importiert werden
        imported_files = set()

        for file_path, data in file_data.items():
            if not data:
                continue

            for imp in data["imports"]:
                if any(imp.startswith(proj) for proj in ['core', 'strategies', 'ml_components']):
                    # Konvertiere Modul zu Dateipfad
                    file_possibilities = [
                        Path(imp.replace('.', os.sep) + '.py'),
                        Path(imp.replace('.', os.sep)) / '__init__.py'
                    ]
                    for poss in file_possibilities:
                        if poss in all_files:
                            imported_files.add(str(poss))

            for from_imp in data["from_imports"].keys():
                if any(from_imp.startswith(proj) for proj in ['core', 'strategies', 'ml_components']):
                    file_possibilities = [
                        Path(from_imp.replace('.', os.sep) + '.py'),
                        Path(from_imp.replace('.', os.sep)) / '__init__.py'
                    ]
                    for poss in file_possibilities:
                        if poss in all_files:
                            imported_files.add(str(poss))

        # Dateien mit main() oder die main.py sind Einstiegspunkte
        entry_points = set()
        for file_path, data in file_data.items():
            if data and (data.get("has_main") or Path(file_path).name == "main.py"):
                entry_points.add(file_path)

        # Finde ungenutzte Dateien
        for file_path in all_files:
            str_path = str(file_path)
            if (str_path not in imported_files and
                    str_path not in entry_points and
                    not str_path.endswith('__init__.py')):
                self.results["unused_files"].append(str_path)

    def _analyze_config_usage(self, file_data: Dict):
        """Analysiere Config-Nutzung"""
        print("🔍 Analysiere Config-Nutzung...")

        for file_path, data in file_data.items():
            if not data:
                continue

            for config_ref in data["config_refs"]:
                self.results["config_usage"][config_ref].add(file_path)

    def _identify_critical_files(self, file_data: Dict):
        """Identifiziere kritische Dateien"""
        print("🔍 Identifiziere kritische Dateien...")

        # Zähle wie oft Dateien importiert werden
        import_count = defaultdict(int)

        for file_path, data in file_data.items():
            if not data:
                continue

            for imp in data["imports"]:
                import_count[imp] += 1

            for from_imp in data["from_imports"].keys():
                import_count[from_imp] += 1

        # Kritisch = oft importiert oder zentrale Dateien
        critical_patterns = ['main.py', 'trading_bot.py', 'exchange.py', 'strategy_base.py']

        for file_path in file_data.keys():
            # Oft importiert?
            module_name = str(Path(file_path).with_suffix('')).replace(os.sep, '.')
            if import_count[module_name] > 3:
                self.results["critical_files"].add(file_path)

            # Zentrale Datei?
            if any(pattern in file_path for pattern in critical_patterns):
                self.results["critical_files"].add(file_path)

    def _create_summary(self):
        """Erstelle Zusammenfassung"""
        print("\n" + "=" * 60)
        print("📊 ANALYSE-ZUSAMMENFASSUNG")
        print("=" * 60)

        # Problem-Übersicht
        problems = []

        if self.results["missing_imports"]:
            problems.append(f"❌ {len(self.results['missing_imports'])} fehlende Imports")
            print(f"\n❌ Fehlende Imports ({len(self.results['missing_imports'])}):")
            for item in self.results["missing_imports"][:5]:
                print(f"   - {item['file']}: {item['import']}")
            if len(self.results["missing_imports"]) > 5:
                print(f"   ... und {len(self.results['missing_imports']) - 5} weitere")

        if self.results["circular_imports"]:
            problems.append(f"🔄 {len(self.results['circular_imports'])} zirkuläre Imports")
            print(f"\n🔄 Zirkuläre Imports ({len(self.results['circular_imports'])}):")
            for cycle in self.results["circular_imports"][:3]:
                print(f"   - {' -> '.join(cycle[:3])}...")

        if self.results["undefined_references"]:
            problems.append(f"❓ {len(self.results['undefined_references'])} undefinierte Referenzen")
            print(f"\n❓ Undefinierte Referenzen ({len(self.results['undefined_references'])}):")
            for item in self.results["undefined_references"][:5]:
                print(f"   - {item['file']}: {item['reference']}")

        if self.results["unused_files"]:
            problems.append(f"📄 {len(self.results['unused_files'])} ungenutzte Dateien")
            print(f"\n📄 Ungenutzte Dateien ({len(self.results['unused_files'])}):")
            for file in self.results["unused_files"][:5]:
                print(f"   - {file}")

        # Kritische Dateien
        print(f"\n🎯 Kritische Dateien ({len(self.results['critical_files'])}):")
        for file in sorted(self.results["critical_files"])[:10]:
            print(f"   - {file}")

        # Empfehlungen
        print("\n" + "=" * 60)
        print("📌 EMPFOHLENE DATEIEN ZUM ZEIGEN")
        print("=" * 60)

        files_to_show = set()

        # 1. Dateien mit fehlenden Imports
        if self.results["missing_imports"]:
            print("\n1️⃣ Dateien mit fehlenden Imports:")
            missing_files = set(item["file"] for item in self.results["missing_imports"])
            for f in list(missing_files)[:5]:
                print(f"   - {f}")
                files_to_show.add(f)

        # 2. Dateien in zirkulären Imports
        if self.results["circular_imports"]:
            print("\n2️⃣ Dateien mit zirkulären Imports:")
            circular_files = set()
            for cycle in self.results["circular_imports"]:
                for module in cycle:
                    file_path = module.replace('.', os.sep) + '.py'
                    if Path(file_path).exists():
                        circular_files.add(file_path)
            for f in list(circular_files)[:3]:
                print(f"   - {f}")
                files_to_show.add(f)

        # 3. Kritische Dateien mit Problemen
        print("\n3️⃣ Kritische Dateien:")
        critical_with_problems = self.results["critical_files"] & files_to_show
        if not critical_with_problems:
            critical_with_problems = list(self.results["critical_files"])[:3]
        for f in critical_with_problems:
            print(f"   - {f}")
            files_to_show.add(f)

        # Finale Liste
        print("\n" + "=" * 60)
        print("📋 BITTE ZEIGE MIR DIESE DATEIEN:")
        print("=" * 60)
        for i, f in enumerate(sorted(files_to_show)[:10], 1):
            print(f"{i}. {f}")

        # Speichere detaillierte Ergebnisse
        self._save_results()

    def _save_results(self):
        """Speichere Analyseergebnisse"""
        output_file = "connection_analysis.json"

        # Konvertiere Sets zu Listen für JSON
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, set):
                json_results[key] = list(value)
            elif isinstance(value, defaultdict):
                json_results[key] = {k: list(v) if isinstance(v, set) else v
                                     for k, v in value.items()}
            else:
                json_results[key] = value

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Detaillierte Ergebnisse gespeichert in: {output_file}")


def main():
    analyzer = ProjectConnectionAnalyzer()
    analyzer.analyze()

    print("\n✅ Analyse abgeschlossen!")
    print("📌 Bitte zeige mir die empfohlenen Dateien,")
    print("   damit wir die Probleme gemeinsam lösen können.")


if __name__ == "__main__":
    main()