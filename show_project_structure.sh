#!/bin/bash

# Pfad zum Projektverzeichnis
PROJECT_DIR="/Users/jnb/PycharmProjects/altcoin_trading_bot"

# Ausgabedatei
OUTPUT_FILE="project_structure.txt"

echo "Generating project structure for $PROJECT_DIR"
echo "Output will be saved to $OUTPUT_FILE"

# Erstellt eine detaillierte Verzeichnisstruktur und speichert sie in einer Datei
find "$PROJECT_DIR" -type d -not -path "*/\.*" -not -path "*/__pycache__*" -not -path "*/.venv*" | sort > "$OUTPUT_FILE"

# Fügt eine Leerzeile hinzu
echo "" >> "$OUTPUT_FILE"
echo "# Python Files:" >> "$OUTPUT_FILE"

# Listet alle Python-Dateien auf
find "$PROJECT_DIR" -name "*.py" -not -path "*/\.*" -not -path "*/__pycache__*" -not -path "*/.venv*" | sort >> "$OUTPUT_FILE"

# Fügt eine Leerzeile hinzu
echo "" >> "$OUTPUT_FILE"
echo "# JSON Files:" >> "$OUTPUT_FILE"

# Listet alle JSON-Dateien auf
find "$PROJECT_DIR" -name "*.json" -not -path "*/\.*" -not -path "*/__pycache__*" -not -path "*/.venv*" | sort >> "$OUTPUT_FILE"

# Fügt eine Leerzeile hinzu
echo "" >> "$OUTPUT_FILE"
echo "# Data Files (CSV, etc):" >> "$OUTPUT_FILE"

# Listet alle CSV-Dateien auf
find "$PROJECT_DIR" -name "*.csv" -not -path "*/\.*" -not -path "*/__pycache__*" -not -path "*/.venv*" | sort >> "$OUTPUT_FILE"

echo "Structure has been generated in $OUTPUT_FILE"

# Optional: Zeigt die Verzeichnisstruktur direkt im Terminal an
echo ""
echo "Directory structure overview:"
echo "============================"
find "$PROJECT_DIR" -type d -not -path "*/\.*" -not -path "*/__pycache__*" -not -path "*/.venv*" -maxdepth 2 | sort | sed -e "s|$PROJECT_DIR/||g" | sed -e "s|^$PROJECT_DIR$|./|g" | sed -e 's/^/  /'

echo ""
echo "For full details, check $OUTPUT_FILE"