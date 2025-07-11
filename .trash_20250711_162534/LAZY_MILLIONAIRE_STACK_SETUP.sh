#!/bin/bash

# 🚀 LAZY MILLIONAIRE STACK - SETUP SCRIPT
# ========================================
# Dieses Script erstellt alle fehlenden Dateien für den kompletten Stack

echo "🚀 LAZY MILLIONAIRE STACK - SETUP"
echo "================================="
echo "Setting up the ultimate automated trading system..."
echo ""

# Farben für Output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# Funktion zum Erstellen von Dateien mit Backup
create_file() {
    local filepath=$1
    local description=$2

    if [ -f "$filepath" ]; then
        echo -e "${YELLOW}⚠️  $filepath already exists, creating backup...${NC}"
        cp "$filepath" "${filepath}.backup.$(date +%Y%m%d_%H%M%S)"
    fi

    echo -e "${GREEN}✅ Creating $filepath - $description${NC}"
}

# Phase 2: DeFi Yield Compounder
if [ ! -f "strategies/defi_yield.py" ]; then
    create_file "strategies/defi_yield.py" "DeFi Yield Compounder"
    echo "⚠️  MANUAL ACTION REQUIRED: Copy content from 'strategies/defi_yield.py - DeFi Yield Compounder' artifact"
fi

if [ ! -f "config/profiles/defi.json" ]; then
    create_file "config/profiles/defi.json" "DeFi Configuration"
    echo "⚠️  MANUAL ACTION REQUIRED: Copy content from 'config/profiles/defi.json' artifact"
fi

# Phase 3: Liquidation Bot
if [ ! -f "strategies/liquidation.py" ]; then
    create_file "strategies/liquidation.py" "Liquidation Bot"
    echo "⚠️  MANUAL ACTION REQUIRED: Copy content from 'strategies/liquidation.py - Liquidation Bot' artifact"
fi

if [ ! -f "config/profiles/liquidation.json" ]; then
    create_file "config/profiles/liquidation.json" "Liquidation Configuration"
    echo "⚠️  MANUAL ACTION REQUIRED: Copy content from 'config/profiles/liquidation.json' artifact"
fi

# Phase 3: Copy Trading Bot
if [ ! -f "strategies/copy_trading.py" ]; then
    create_file "strategies/copy_trading.py" "Copy Trading Bot"
    echo "⚠️  MANUAL ACTION REQUIRED: Copy content from 'strategies/copy_trading.py - Copy Trading Bot' artifact"
fi

if [ ! -f "config/profiles/copy_trading.json" ]; then
    create_file "config/profiles/copy_trading.json" "Copy Trading Configuration"
    echo "⚠️  MANUAL ACTION REQUIRED: Copy content from 'config/profiles/copy_trading.json' artifact"
fi

# Ultimate AutoPilot Configuration
if [ ! -f "config/profiles/ultimate_autopilot.json" ]; then
    create_file "config/profiles/ultimate_autopilot.json" "Ultimate AutoPilot Configuration"
    echo "⚠️  MANUAL ACTION REQUIRED: Copy content from 'config/profiles/ultimate_autopilot.json' artifact"
fi

# Update strategies/__init__.py
echo -e "${YELLOW}📝 Don't forget to update strategies/__init__.py with the new imports!${NC}"
echo "   Copy content from 'Erweiterte strategies/__init__.py' artifact"

echo ""
echo "================================="
echo "📋 SETUP CHECKLIST:"
echo "================================="
echo ""
echo "1. Create missing files by copying artifact contents:"
echo "   - strategies/defi_yield.py"
echo "   - strategies/liquidation.py"
echo "   - strategies/copy_trading.py"
echo "   - config/profiles/defi.json"
echo "   - config/profiles/liquidation.json"
echo "   - config/profiles/copy_trading.json"
echo "   - config/profiles/ultimate_autopilot.json"
echo ""
echo "2. Update strategies/__init__.py"
echo ""
echo "3. Test all imports:"
echo "   python -c 'from strategies import *'"
echo ""
echo "4. Run status check:"
echo "   python check_project_status.py"
echo ""
echo "================================="
echo "🚀 READY TO USE COMMANDS:"
echo "================================="
echo ""
echo "# Test individual strategies:"
echo "python main.py --mode=paper --strategy=defi_yield --config=defi"
echo "python main.py --mode=paper --strategy=liquidation --config=liquidation"
echo "python main.py --mode=paper --strategy=copy_trading --config=copy_trading"
echo ""
echo "# Run ULTIMATE AUTOPILOT (all 6 strategies):"
echo "python main.py --mode=paper --strategy=autopilot --config=ultimate_autopilot"
echo ""
echo "💰 Happy automated trading! 🚀"