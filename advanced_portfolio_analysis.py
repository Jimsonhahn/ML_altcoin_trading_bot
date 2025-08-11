#!/usr/bin/env python3
"""
Advanced Portfolio Strategy - Problem Analysis
==============================================

Analysiert warum die Advanced Portfolio Strategy unrentabel ist
"""

import json
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_advanced_portfolio_problems():
    """Analysiert die Probleme der Advanced Portfolio Strategy"""
    
    print("🔍 ADVANCED PORTFOLIO STRATEGY - PROBLEM ANALYSE")
    print("=" * 60)
    
    # Lade die Backtest-Ergebnisse
    try:
        with open('simple_defensive_backtest_20250724_151134.json', 'r') as f:
            results = json.load(f)
        
        adv_portfolio = results['advanced_portfolio']
        
        print(f"\n📊 AKTUELLE PERFORMANCE:")
        print(f"   • Start-Kapital: ${adv_portfolio['initial_capital']:,.2f}")
        print(f"   • End-Wert: ${adv_portfolio['final_value']:,.2f}")
        print(f"   • Total Return: {adv_portfolio['total_return']:.2%}")
        print(f"   • Max Drawdown: {adv_portfolio['max_drawdown']:.2%}")
        print(f"   • Volatilität: {adv_portfolio['volatility']:.2%}")
        print(f"   • Total Trades: {adv_portfolio['total_trades']}")
        print(f"   • Total Fees: ${adv_portfolio['total_fees']:.2f}")
        
        # Analysiere Portfolio-Historie
        portfolio_history = adv_portfolio['portfolio_history']
        
        print(f"\n📈 PORTFOLIO-ENTWICKLUNG (Letzte {len(portfolio_history)} Punkte):")
        for entry in portfolio_history:
            print(f"   {entry['date'][:10]}: ${entry['portfolio_value']:,.2f} "
                  f"(Crypto: {entry['crypto_weight']:.1%}, Vol: {entry['volatility']:.2f})")
        
        # Identifiziere Hauptprobleme
        print(f"\n🚨 IDENTIFIZIERTE PROBLEME:")
        
        # Problem 1: Extrem hohe Volatilität
        if adv_portfolio['volatility'] > 1.0:  # > 100%
            print(f"   1. ❌ EXTREM HOHE VOLATILITÄT: {adv_portfolio['volatility']:.1%}")
            print(f"      → Portfolio-Volatilität ist unrealistisch hoch")
            print(f"      → Deutet auf Berechnungsfehler hin")
        
        # Problem 2: Hohe Trading-Kosten
        cost_impact = adv_portfolio['total_fees'] / adv_portfolio['initial_capital']
        if cost_impact > 0.002:  # > 0.2%
            print(f"   2. ❌ HOHE TRADING-KOSTEN: {cost_impact:.3%} des Kapitals")
            print(f"      → ${adv_portfolio['total_fees']:.2f} Gebühren bei nur {adv_portfolio['total_trades']} Trades")
            print(f"      → Durchschnitt ${adv_portfolio['total_fees']/max(adv_portfolio['total_trades'],1):.2f} pro Trade")
        
        # Problem 3: Großer Drawdown
        if adv_portfolio['max_drawdown'] > 0.2:  # > 20%
            print(f"   3. ❌ HOHER MAXIMUM DRAWDOWN: {adv_portfolio['max_drawdown']:.1%}")
            print(f"      → Portfolio verliert über 30% vom Peak")
            print(f"      → Zeigt schlechtes Risiko-Management")
        
        # Problem 4: Wenige Trades für "aktive" Strategie
        if adv_portfolio['total_trades'] < 10:
            print(f"   4. ❌ ZU WENIGE TRADES: {adv_portfolio['total_trades']} in einem Jahr")
            print(f"      → Monatliches Rebalancing sollte ~12 Trades generieren")
            print(f"      → 10% Rebalancing-Schwelle ist zu hoch")
        
        return True
        
    except FileNotFoundError:
        print("❌ Backtest-Ergebnisse nicht gefunden")
        return False


def identify_root_causes():
    """Identifiziert die Grundursachen der Probleme"""
    
    print(f"\n🔬 ROOT CAUSE ANALYSIS:")
    print("-" * 40)
    
    print(f"\n1. 📊 VOLATILITÄTS-BERECHNUNGSFEHLER:")
    print(f"   • Problem: Portfolio-Volatilität von 203% ist unrealistisch")
    print(f"   • Ursache: Fehlerhafte Berechnung der Returns")
    print(f"   • Code-Issue: Zeile 278 'portfolio_value = (portfolio_value - position_size * current_price) + position_size * current_price'")
    print(f"   • Lösung: Korrekte Portfolio-Wert-Berechnung implementieren")
    
    print(f"\n2. 💰 CASH-MANAGEMENT-PROBLEME:")
    print(f"   • Problem: Keine explizite Cash-Position")
    print(f"   • Ursache: 'Stable' Position wird nicht richtig als Cash behandelt")
    print(f"   • Code-Issue: positions['stable'] -= trade_value (kann negativ werden)")
    print(f"   • Lösung: Explizite Cash-Verwaltung wie bei Smart Rebalancing")
    
    print(f"\n3. 🎯 REBALANCING-LOGIK-FEHLER:")
    print(f"   • Problem: 10% Rebalancing-Schwelle ist zu hoch")
    print(f"   • Ursache: Nur extreme Abweichungen triggern Trades")
    print(f"   • Impact: Verpasste Rebalancing-Chancen")
    print(f"   • Lösung: Schwelle auf 3-5% reduzieren")
    
    print(f"\n4. 📉 REGIME-DETECTION-PROBLEME:")
    print(f"   • Problem: Volatilitäts-basierte Allokation zu simpel")
    print(f"   • Ursache: Nur 3 Volatilitäts-Buckets")
    print(f"   • Impact: Schlechtes Timing bei Regime-Wechseln")
    print(f"   • Lösung: Sophisticated Regime-Detection")


def propose_fixes():
    """Schlägt konkrete Fixes vor"""
    
    print(f"\n🛠️  VORGESCHLAGENE FIXES:")
    print("-" * 40)
    
    print(f"\n✅ FIX 1: Portfolio-Wert-Berechnung korrigieren")
    print(f"   Aktuell: portfolio_value = (portfolio_value - crypto_value) + crypto_value")
    print(f"   Korrekt: portfolio_value = crypto_value + stable_value")
    
    print(f"\n✅ FIX 2: Explizite Cash-Verwaltung")
    print(f"   • Separate 'cash' Variable einführen")
    print(f"   • Cash kann nie negativ werden")
    print(f"   • Trades nur bei verfügbarem Cash/Assets")
    
    print(f"\n✅ FIX 3: Rebalancing-Schwelle senken")
    print(f"   • Von 10% auf 5% reduzieren")
    print(f"   • Mehr Rebalancing-Aktivität")
    print(f"   • Bessere Portfolio-Kontrolle")
    
    print(f"\n✅ FIX 4: Verbesserte Regime-Detection")
    print(f"   • Momentum-Komponente hinzufügen")
    print(f"   • Mehrere Zeitrahmen berücksichtigen")
    print(f"   • Trend + Volatilität kombinieren")
    
    print(f"\n✅ FIX 5: Transaktionskosten-Optimierung")
    print(f"   • Minimum Trade Size einführen")
    print(f"   • Batch-Rebalancing implementieren")
    print(f"   • Kosten-Nutzen-Analyse vor jedem Trade")


def compare_with_working_strategies():
    """Vergleicht mit funktionierenden Strategien"""
    
    print(f"\n🔄 VERGLEICH MIT FUNKTIONIERENDEN STRATEGIEN:")
    print("-" * 50)
    
    try:
        with open('simple_defensive_backtest_20250724_151134.json', 'r') as f:
            results = json.load(f)
        
        adv_portfolio = results['advanced_portfolio']
        defensive_vol = results['defensive_volatility']
        smart_rebal = results['smart_rebalancing']
        
        print(f"\n📊 PERFORMANCE-VERGLEICH:")
        print(f"   Strategy               Return    Volatility   Max DD    Trades")
        print(f"   Advanced Portfolio     {adv_portfolio['total_return']:>6.1%}    {adv_portfolio['volatility']:>8.1%}   {adv_portfolio['max_drawdown']:>6.1%}    {adv_portfolio['total_trades']:>6}")
        print(f"   Defensive Volatility   {defensive_vol['total_return']:>6.1%}    {defensive_vol['volatility']:>8.1%}   {defensive_vol['max_drawdown']:>6.1%}    {defensive_vol['total_trades']:>6}")
        print(f"   Smart Rebalancing      {smart_rebal['total_return']:>6.1%}    {smart_rebal['volatility']:>8.1%}   {smart_rebal['max_drawdown']:>6.1%}    {smart_rebal['total_trades']:>6}")
        
        print(f"\n🎯 WAS FUNKTIONIERT BEI DEN ANDEREN:")
        
        print(f"\n• Defensive Volatility (WINNER):")
        print(f"  ✅ Einfache, klare Logik")
        print(f"  ✅ Niedrige Volatilität durch defensives Positioning")
        print(f"  ✅ Wenige, aber profitable Trades")
        print(f"  ✅ Ausgezeichnetes Risiko-Management")
        
        print(f"\n• Smart Rebalancing (FUNCTIONAL):")
        print(f"  ✅ Explizite Cash-Verwaltung")
        print(f"  ✅ Initial Allocation korrekt implementiert")
        print(f"  ✅ Realistische Kosten-Nutzen-Analyse")
        print(f"  ✅ Momentum-Integration")
        
        print(f"\n❌ WAS BEI ADVANCED PORTFOLIO FEHLT:")
        print(f"  • Korrekte Portfolio-Wert-Berechnung")
        print(f"  • Cash-Management")
        print(f"  • Realistische Volatilitäts-Metriken")
        print(f"  • Kosten-Nutzen-Optimierung")
        
    except Exception as e:
        print(f"❌ Fehler beim Vergleich: {e}")


def main():
    """Hauptfunktion für die Analyse"""
    
    if analyze_advanced_portfolio_problems():
        identify_root_causes()
        propose_fixes()
        compare_with_working_strategies()
        
        print(f"\n💡 ZUSAMMENFASSUNG:")
        print(f"   Die Advanced Portfolio Strategy ist hauptsächlich wegen")
        print(f"   technischer Implementierungsfehler unrentabel, nicht wegen")
        print(f"   der grundlegenden Strategie-Idee.")
        print(f"   ")
        print(f"   Mit den vorgeschlagenen Fixes könnte sie profitabel werden.")


if __name__ == "__main__":
    main()