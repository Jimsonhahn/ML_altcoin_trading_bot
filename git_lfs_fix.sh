#!/bin/bash

# Fehlgeschlagene Push-Probleme mit großen Dateien beheben

echo "🔍 Überprüfe Git LFS Installation..."
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS ist nicht installiert. Installation wird gestartet..."

    # Betriebssystem erkennen und Git LFS installieren
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install git-lfs
        else
            echo "❌ Homebrew ist nicht installiert. Bitte installiere Git LFS manuell."
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y git-lfs
        elif command -v yum &> /dev/null; then
            sudo yum install -y git-lfs
        else
            echo "❌ Paketmanager nicht erkannt. Bitte installiere Git LFS manuell."
            exit 1
        fi
    else
        echo "❌ Nicht unterstütztes Betriebssystem. Bitte installiere Git LFS manuell."
        exit 1
    fi
fi

echo "✅ Git LFS ist verfügbar."
echo "🔧 Konfiguriere Git LFS für große Dateien..."

# Git LFS initialisieren
git lfs install

# Große Dateien für LFS konfigurieren
git lfs track "*.log"
git lfs track "*.xlsx"
git add .gitattributes
git commit -m "Konfiguriere Git LFS für große Dateien"

# Große Dateien identifizieren
echo "🔍 Suche nach großen Dateien (>50MB)..."
GROSSE_DATEIEN=$(find . -type f -size +50M -not -path "./.git/*" | grep -v ".git")

if [ -z "$GROSSE_DATEIEN" ]; then
    echo "✅ Keine großen Dateien gefunden."
else
    echo "🔍 Folgende große Dateien wurden gefunden:"
    echo "$GROSSE_DATEIEN"

    echo "🔧 Entferne große Dateien aus Git-Tracking und füge sie zu LFS hinzu..."
    for datei in $GROSSE_DATEIEN; do
        echo "Bearbeite $datei..."
        git rm --cached "$datei"
        git add "$datei"  # Wird jetzt via LFS getrackt
    done

    git commit -m "Verschiebe große Dateien zu Git LFS"
fi

# ml_backtest.log speziell behandeln, da es im Log erwähnt wurde
if [ -f "ml_backtest.log" ]; then
    echo "🔧 Behandle ml_backtest.log speziell..."
    git rm --cached ml_backtest.log
    git add ml_backtest.log
    git commit -m "Verschiebe ml_backtest.log zu Git LFS"
fi

# Repository-Status anzeigen
echo "📊 Git Status nach Änderungen:"
git status

echo "🚀 Versuche jetzt zu pushen..."
git push origin main

echo "✅ Fertig! Wenn der Push erfolgreich war, ist dein Problem gelöst."
echo "Falls weiterhin Probleme auftreten, erwäge einen 'git push --force origin main'"
echo "WARNUNG: --force nur verwenden, wenn du weißt was du tust, da es den Remote-Verlauf überschreibt!"