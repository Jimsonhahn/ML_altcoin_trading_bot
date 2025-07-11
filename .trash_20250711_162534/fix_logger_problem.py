#!/usr/bin/env python3
"""
Fix Logger Type Error
=====================
Behebt das Problem mit log_level als Integer statt String
"""

import os
import re
import shutil
from datetime import datetime


def fix_logger_type():
    """Behebt das Logger Type Problem"""

    print("🔧 LOGGER TYPE FIX")
    print("=" * 50)

    # Backup erstellen
    backup_dir = 'backups'
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = os.path.join(backup_dir, f'main.py.backup.{timestamp}')

    print(f"📁 Erstelle Backup: {backup_file}")
    shutil.copy2('main.py', backup_file)

    # Datei lesen
    with open('main.py', 'r') as f:
        content = f.read()

    # Finde die problematische Stelle
    print("\n🔍 Suche nach log_level Definition...")

    # Pattern für log_level Definition
    # Wahrscheinlich sowas wie: log_level = logging.DEBUG if args.debug else logging.INFO
    pattern1 = r'log_level\s*=\s*logging\.(DEBUG|INFO)'

    if re.search(pattern1, content):
        print("✅ Gefunden: log_level wird als logging.LEVEL definiert")

        # Fix 1: Ändere die Definition zu Strings
        content = re.sub(
            r'log_level\s*=\s*logging\.DEBUG\s+if\s+args\.debug\s+else\s+logging\.INFO',
            "log_level = 'DEBUG' if args.debug else 'INFO'",
            content
        )

        # Alternative Pattern
        content = re.sub(
            r'log_level\s*=\s*logging\.DEBUG',
            "log_level = 'DEBUG'",
            content
        )
        content = re.sub(
            r'log_level\s*=\s*logging\.INFO',
            "log_level = 'INFO'",
            content
        )

        print("✅ Fix angewendet: log_level ist jetzt ein String")
    else:
        # Alternative: Fix beim setup_logger Aufruf
        print("⚠️  Alternative Lösung: Fix beim setup_logger Aufruf")

        # Finde setup_logger Aufruf
        old_call = r"logger = setup_logger\(name='trading_bot', level=log_level\)"
        new_call = "logger = setup_logger(name='trading_bot', level='DEBUG' if args.debug else 'INFO')"

        content = re.sub(old_call, new_call, content)
        print("✅ Alternative Fix angewendet")

    # Schreibe die korrigierte Datei
    with open('main.py', 'w') as f:
        f.write(content)

    # Test
    print("\n🧪 Teste den Fix...")
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, 'main.py', '--mode=paper', '--strategy=autopilot', '--config=ultimate_autopilot', '--dry-run'],
        capture_output=True, text=True
    )

    if result.returncode == 0 or "DRY RUN MODE" in result.stdout:
        print("✅ Logger funktioniert!")
        print("\n🚀 Bot ist bereit! Entferne --dry-run für echtes Paper Trading:")
        print("   python main.py --mode=paper --strategy=autopilot --config=ultimate_autopilot")
    else:
        print("⚠️  Ausgabe:")
        if result.stdout:
            print("STDOUT:", result.stdout[:200])
        if result.stderr:
            print("STDERR:", result.stderr[:200])


if __name__ == "__main__":
    fix_logger_type()