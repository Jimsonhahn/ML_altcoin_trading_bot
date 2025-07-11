# Erstellen Sie test_setup.py im Hauptverzeichnis
cat > test_setup.py << 'EOF'
# !/usr/bin/env python3
"""Test ob alle Komponenten installiert sind"""

import sys


def test_imports():
    """Testet alle wichtigen Imports"""

    packages = {
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'tensorflow': 'TensorFlow',
        'ccxt': 'CCXT',
        'fastapi': 'FastAPI',
        'websockets': 'WebSockets',
        'aiohttp': 'AioHTTP',
        'redis': 'Redis'
    }

    print("=== Testing Imports ===")

    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} importiert")
        except ImportError as e:
            print(f"✗ {name} fehlt: {e}")
            failed.append(name)

    if failed:
        print(f"\nFehlende Pakete: {', '.join(failed)}")
        return False
    else:
        print("\n✓ Alle Pakete erfolgreich importiert!")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
EOF

# Test ausführen
python
test_setup.py