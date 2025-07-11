#!/usr/bin/env python3
"""
Intelligente Projekt-Bereinigung für Trading Bot
"""
import os
import re
import hashlib
from pathlib import Path
from datetime import datetime
import json

class ProjectCleaner:
    def __init__(self):
        self.project_root = Path.cwd()
        self.report = {
            'backup_files': [],
            'duplicate_files': [],
            'unused_files': [],
            'old_configs': [],
            'temp_files': [],
            'obsolete_strategies': [],
            'test_artifacts': []
        }

        # Dateien die IMMER behalten werden sollen
        self.essential_files = {
            'main.py',
            'requirements.txt',
            '.env',
            '.gitignore',
            'README.md',
            'setup.py'
        }

        # Wichtige Verzeichnisse für den Trading Bot
        self.essential_dirs = {
            'strategies',
            'core',
            'config',
            'data',
            'utils',
            'analysis',
            'data_sources'
        }

        # Bekannte notwendige Strategien für den 6-Strategy Plan
        self.required_strategies = {
            'strategy_base.py',
            'base.py',
            '__init__.py',
            'autopilot.py',
            'momentum.py',
            'mean_reversion.py',
            'ml_strategy.py',
            'grid_trading.py',
            'arbitrage.py',
            'liquidation.py',
            'copy_trading.py',
            'defi_yield.py'
        }

    def analyze_project(self):
        """Analysiere das Projekt und finde zu bereinigende Dateien"""
        print("\n📊 Analysiere Projektstruktur...")

        # 1. Finde Backup-Dateien
        self._find_backup_files()

        # 2. Finde temporäre Dateien
        self._find_temp_files()

        # 3. Finde doppelte Dateien
        self._find_duplicate_files()

        # 4. Finde veraltete Configs
        self._find_old_configs()

        # 5. Finde ungenutzte Strategien
        self._find_obsolete_strategies()

        # 6. Finde Test-Artefakte
        self._find_test_artifacts()

        return self.report

    def _find_backup_files(self):
        """Finde alle Backup-Dateien"""
        backup_patterns = [
            r'.*\.backup$',
            r'.*\.bak$',
            r'.*\.old$',
            r'.*~$',
            r'.*\.swp$',
            r'backup_\d{8}_\d{6}',  # backup_20250711_144916
            r'.*\.py\.backup'
        ]

        for root, dirs, files in os.walk(self.project_root):
            # Skip .git und __pycache__
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv'}]

            for file in files:
                for pattern in backup_patterns:
                    if re.match(pattern, file):
                        filepath = os.path.join(root, file)
                        self.report['backup_files'].append(filepath)
                        break

    def _find_temp_files(self):
        """Finde temporäre Dateien"""
        temp_patterns = [
            r'.*\.tmp$',
            r'.*\.temp$',
            r'\.DS_Store$',
            r'Thumbs\.db$',
            r'.*\.pyc$',
            r'.*\.pyo$',
            r'.*\.log$',
            r'debug.*\.py$',
            r'test_.*\.py$',
            r'fix_.*\.py$',
            r'patch_.*\.py$',
            r'_mock_.*\.py$',
            r'instant_.*\.py$'
        ]

        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv'}]

            for file in files:
                for pattern in temp_patterns:
                    if re.match(pattern, file):
                        filepath = os.path.join(root, file)
                        # Nicht löschen wenn es eine wichtige Test-Datei ist
                        if not any(essential in file for essential in ['test_backtest.py', 'test_strategies.py']):
                            self.report['temp_files'].append(filepath)
                        break

    def _find_duplicate_files(self):
        """Finde doppelte Dateien basierend auf Inhalt"""
        file_hashes = {}

        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv', 'data'}]

            for file in files:
                if file.endswith(('.py', '.json', '.txt', '.md')):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()

                        if file_hash in file_hashes:
                            # Behalte die Datei im wichtigeren Verzeichnis
                            existing = file_hashes[file_hash]
                            if self._is_less_important(filepath, existing):
                                self.report['duplicate_files'].append({
                                    'file': filepath,
                                    'duplicate_of': existing
                                })
                            else:
                                self.report['duplicate_files'].append({
                                    'file': existing,
                                    'duplicate_of': filepath
                                })
                                file_hashes[file_hash] = filepath
                        else:
                            file_hashes[file_hash] = filepath
                    except:
                        pass

    def _is_less_important(self, file1, file2):
        """Bestimme welche Datei weniger wichtig ist"""
        # Priorität: essential_dirs > andere
        for essential_dir in self.essential_dirs:
            if essential_dir in file1 and essential_dir not in file2:
                return False
            elif essential_dir not in file1 and essential_dir in file2:
                return True

        # Wenn beide gleich wichtig, behalte die mit kürzerem Pfad
        return len(file1) > len(file2)

    def _find_old_configs(self):
        """Finde veraltete Konfigurationsdateien"""
        config_dir = self.project_root / 'config'
        if config_dir.exists():
            for root, dirs, files in os.walk(config_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    # Veraltete Config-Patterns
                    if any(pattern in file.lower() for pattern in ['test', 'old', 'backup', 'copy', 'beispiel']):
                        self.report['old_configs'].append(filepath)

    def _find_obsolete_strategies(self):
        """Finde nicht mehr benötigte Strategien"""
        strategies_dir = self.project_root / 'strategies'
        if strategies_dir.exists():
            for file in strategies_dir.glob('*.py'):
                if file.name not in self.required_strategies:
                    # Prüfe ob es eine experimentelle oder alte Strategie ist
                    with open(file, 'r') as f:
                        content = f.read()
                        if any(marker in content.lower() for marker in ['deprecated', 'old', 'test', 'experimental', 'unused']):
                            self.report['obsolete_strategies'].append(str(file))
                        # Oder wenn es keine richtige Strategie-Klasse hat
                        elif 'class' not in content or 'Strategy' not in content:
                            self.report['obsolete_strategies'].append(str(file))

    def _find_test_artifacts(self):
        """Finde Test-Artefakte und Debug-Dateien"""
        patterns = [
            'paper_trading_wrapper.py',
            'run_paper_trading.py',
            'timestamp_patch.py',
            'register_autopilot.py',
            'debug_imports.py',
            'test_bot.py',
            'start_autopilot.py',
            'run_all_strategies.sh',
            'fix_*.sh',
            'LAZY_MILLIONAIRE_STACK_SETUP.sh'
        ]

        for pattern in patterns:
            for file in self.project_root.glob(pattern):
                if file.exists():
                    self.report['test_artifacts'].append(str(file))

    def generate_cleanup_script(self):
        """Generiere ein Cleanup-Script basierend auf der Analyse"""
        script_content = '''#!/bin/bash
# Auto-generiertes Cleanup-Script
# Generiert am: ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''

echo "🧹 PROJEKT-BEREINIGUNG"
echo "===================="

# Farben
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

# Sicherheitsabfrage
echo -e "${YELLOW}⚠️  WARNUNG: Dieses Script wird Dateien löschen!${NC}"
echo -e "Bitte überprüfen Sie cleanup_review.txt für Details."
read -p "Fortfahren? (j/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Jj]$ ]]; then
    echo "Abgebrochen."
    exit 1
fi

# Erstelle Trash-Verzeichnis
TRASH_DIR=".trash_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TRASH_DIR"
echo -e "${GREEN}Trash-Verzeichnis erstellt: $TRASH_DIR${NC}"

'''

        # Füge Löschbefehle hinzu
        categories = [
            ('backup_files', 'Backup-Dateien'),
            ('temp_files', 'Temporäre Dateien'),
            ('test_artifacts', 'Test-Artefakte'),
            ('old_configs', 'Alte Configs'),
            ('obsolete_strategies', 'Veraltete Strategien')
        ]

        for category, label in categories:
            if self.report[category]:
                script_content += f'\n# {label}\necho -e "\\n${{YELLOW}}{label} verschieben...$${{NC}}"\n'
                for file in self.report[category]:
                    script_content += f'mv "{file}" "$TRASH_DIR/" 2>/dev/null && echo -e "${{GREEN}}✓ {os.path.basename(file)}${{NC}}" || echo -e "${{RED}}✗ {os.path.basename(file)}${{NC}}"\n'

        # Duplikate separat behandeln
        if self.report['duplicate_files']:
            script_content += '\n# Duplikate\necho -e "\\n${YELLOW}Duplikate verschieben...${NC}"\n'
            for dup in self.report['duplicate_files']:
                script_content += f'mv "{dup["file"]}" "$TRASH_DIR/" 2>/dev/null && echo -e "${{GREEN}}✓ {os.path.basename(dup["file"])} (Duplikat von {os.path.basename(dup["duplicate_of"])})${{NC}}"\n'

        # __pycache__ Verzeichnisse
        script_content += '''
# __pycache__ Verzeichnisse
echo -e "\\n${YELLOW}__pycache__ Verzeichnisse löschen...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo -e "${GREEN}✓ __pycache__ Verzeichnisse gelöscht${NC}"

# .pyc Dateien
find . -name "*.pyc" -delete 2>/dev/null

echo -e "\\n${GREEN}✅ Bereinigung abgeschlossen!${NC}"
echo -e "${YELLOW}Gelöschte Dateien befinden sich in: $TRASH_DIR${NC}"
echo -e "${YELLOW}Sie können das Verzeichnis mit 'rm -rf $TRASH_DIR' endgültig löschen.${NC}"
'''

        return script_content

    def generate_report(self):
        """Generiere einen detaillierten Bericht"""
        report = "DETAILLIERTER BEREINIGUNGSBERICHT\n"
        report += "=================================\n\n"

        total_files = sum(len(v) if isinstance(v, list) else len(v) for v in self.report.values())
        report += f"Gefundene zu bereinigende Dateien: {total_files}\n\n"

        # Backup-Dateien
        if self.report['backup_files']:
            report += f"BACKUP-DATEIEN ({len(self.report['backup_files'])})\n"
            report += "-" * 50 + "\n"
            for file in sorted(self.report['backup_files']):
                report += f"  - {file}\n"
            report += "\n"

        # Temporäre Dateien
        if self.report['temp_files']:
            report += f"TEMPORÄRE DATEIEN ({len(self.report['temp_files'])})\n"
            report += "-" * 50 + "\n"
            for file in sorted(self.report['temp_files']):
                report += f"  - {file}\n"
            report += "\n"

        # Duplikate
        if self.report['duplicate_files']:
            report += f"DUPLIKATE ({len(self.report['duplicate_files'])})\n"
            report += "-" * 50 + "\n"
            for dup in self.report['duplicate_files']:
                report += f"  - {dup['file']}\n    (Duplikat von: {dup['duplicate_of']})\n"
            report += "\n"

        # Veraltete Configs
        if self.report['old_configs']:
            report += f"VERALTETE CONFIGS ({len(self.report['old_configs'])})\n"
            report += "-" * 50 + "\n"
            for file in sorted(self.report['old_configs']):
                report += f"  - {file}\n"
            report += "\n"

        # Obsolete Strategien
        if self.report['obsolete_strategies']:
            report += f"VERALTETE STRATEGIEN ({len(self.report['obsolete_strategies'])})\n"
            report += "-" * 50 + "\n"
            for file in sorted(self.report['obsolete_strategies']):
                report += f"  - {file}\n"
            report += "\n"

        # Test-Artefakte
        if self.report['test_artifacts']:
            report += f"TEST-ARTEFAKTE ({len(self.report['test_artifacts'])})\n"
            report += "-" * 50 + "\n"
            for file in sorted(self.report['test_artifacts']):
                report += f"  - {file}\n"
            report += "\n"

        # Empfehlungen
        report += "\nEMPFEHLUNGEN\n"
        report += "=" * 50 + "\n"
        report += "1. Überprüfen Sie die Liste sorgfältig\n"
        report += "2. Führen Sie das generierte cleanup.sh Script aus\n"
        report += "3. Die Dateien werden in .trash_* verschoben (nicht gelöscht)\n"
        report += "4. Nach Überprüfung können Sie .trash_* endgültig löschen\n"

        return report


if __name__ == "__main__":
    cleaner = ProjectCleaner()

    print("🔍 Starte Projekt-Analyse...")
    cleaner.analyze_project()

    # Generiere Bericht
    report = cleaner.generate_report()
    with open('cleanup_review.txt', 'w') as f:
        f.write(report)
    print("✅ Bericht erstellt: cleanup_review.txt")

    # Generiere Cleanup-Script
    script = cleaner.generate_cleanup_script()
    with open('cleanup.sh', 'w') as f:
        f.write(script)
    os.chmod('cleanup.sh', 0o755)
    print("✅ Cleanup-Script erstellt: cleanup.sh")

    # Zusammenfassung
    total_files = sum(len(v) if isinstance(v, list) else len(v) for v in cleaner.report.values())
    print(f"\n📊 Zusammenfassung: {total_files} Dateien zur Bereinigung gefunden")
    print("📄 Bitte überprüfen Sie cleanup_review.txt für Details")
    print("🚀 Führen Sie ./cleanup.sh aus um die Bereinigung durchzuführen")
