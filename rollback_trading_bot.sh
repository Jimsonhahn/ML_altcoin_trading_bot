#!/bin/bash
# Notfall-Rollback falls etwas schief geht
if [ -f "core/trading_bot.py.syntax_backup" ]; then
    cp core/trading_bot.py.syntax_backup core/trading_bot.py
    echo "✅ trading_bot.py zurückgesetzt"
else
    echo "❌ Kein Backup gefunden"
fi
