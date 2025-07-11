#!/bin/bash
# Quick start script for the bot

echo "🚀 Starting Ultimate Trading Bot..."
echo "Choose mode:"
echo "1) Paper Trading (Test)"
echo "2) Live Trading (Real Money)"
echo "3) Backtest"
read -p "Enter choice (1-3): " choice

case $choice in
    1) mode="paper" ;;
    2) mode="live" ;;
    3) mode="backtest" ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo "Choose strategy:"
echo "1) AutoPilot (All 6 Strategies)"
echo "2) Momentum"
echo "3) Grid Trading"
echo "4) Arbitrage"
read -p "Enter choice (1-4): " strat

case $strat in
    1) strategy="autopilot" ;;
    2) strategy="momentum" ;;
    3) strategy="grid_trading" ;;
    4) strategy="arbitrage" ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo "Choose risk profile:"
echo "1) Conservative"
echo "2) Default"
echo "3) Aggressive"
read -p "Enter choice (1-3): " risk

case $risk in
    1) config="conservative" ;;
    2) config="default" ;;
    3) config="aggressive" ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo "Starting bot with: mode=$mode, strategy=$strategy, config=$config"
python main.py --mode=$mode --strategy=$strategy --config=$config
