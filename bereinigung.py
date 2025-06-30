#!/bin/bash
# Bereinigungsscript für Git Repository

echo "🧹 Bereinige Git Repository..."

# 1. Virtuelle Umgebung aus Git entfernen (falls versehentlich hinzugefügt)
echo "→ Entferne .venv aus Git..."
git rm -r --cached .venv/ 2>/dev/null || true
git rm -r --cached venv/ 2>/dev/null || true

# 2. Stelle sicher, dass .gitignore korrekt ist
echo "→ Erstelle korrekte .gitignore..."
cat > .gitignore << 'EOF'
# Python Virtual Environment - WICHTIG!
.venv/
venv/
env/
ENV/
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# ML Models und große Dateien
data/ml_models/*.pkl
data/ml_models/*.h5
data/ml_models/*.joblib
*.model

# Marktdaten
data/market_data/*
!data/market_data/.gitkeep

# Backtest Ergebnisse
data/backtest_results/*
!data/backtest_results/.gitkeep

# Trading Logs
logs/
*.log

# IDE
.vscode/
.idea/
.DS_Store

# Jupyter
.ipynb_checkpoints/

# Database
*.db
*.sqlite
EOF

# 3. Korrigiere .gitattributes
echo "→ Korrigiere Git LFS Konfiguration..."
cat > .gitattributes << 'EOF'
*.pkl filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text
*.xlsx filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
EOF

# 4. Erstelle .gitkeep Dateien
echo "→ Erstelle Verzeichnisstruktur..."
mkdir -p data/market_data data/ml_models data/backtest_results logs
touch data/market_data/.gitkeep
touch data/ml_models/.gitkeep
touch data/backtest_results/.gitkeep
touch logs/.gitkeep

# 5. Stage alle Änderungen
echo "→ Stage Änderungen..."
git add .gitignore
git add .gitattributes
git add data/*/.gitkeep
git add logs/.gitkeep

# 6. Status anzeigen
echo ""
echo "📊 Git Status:"
git status

echo ""
echo "✅ Bereinigung abgeschlossen!"
echo ""
echo "Nächste Schritte:"
echo "1. git commit -m 'Clean repository and fix gitignore'"
echo "2. SSH-Key Setup abschließen (falls noch nicht geschehen)"
echo "3. git remote set-url origin git@github.com:DEIN-USERNAME/ML_altcoin_trading_bot.git"
echo "4. git push -u origin main"