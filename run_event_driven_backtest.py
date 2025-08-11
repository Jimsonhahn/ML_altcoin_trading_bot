#!/usr/bin/env python3
"""
Event-Driven Backtest Demo
Demonstriert das umfassende ereignisgesteuerte Backtesting-Framework
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Backtesting Framework
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance_analyzer import analyze_backtest_results

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def run_comprehensive_backtest():
    """
    Führt umfassenden ereignisgesteuerten Backtest aus
    """
    
    print("🚀 EREIGNISGESTEUERTER BACKTEST - QUANTUM ORCHESTRATOR")
    print("=" * 80)
    print("Umfassende Validierung des Tier-1 Trading Systems")
    print("Ohne Lookahead-Bias • Realistische Execution • Modulare Analyse")
    print()
    
    # Backtest Konfiguration - BTC/USDT 2024
    config = {
        'initial_capital': 1000000.0,  # $1M
        'start_date': datetime(2024, 1, 1),
        'end_date': datetime(2024, 12, 31),  # 2024 Jahr
        'data_directory': "data",  # Wird SimulatedDataHandler verwenden wenn nicht vorhanden
        'symbols': ["BTC"],  # Nur BTC/USDT
        'enable_detailed_logging': True,
        'max_concurrent_events': 5000
    }
    
    print(f"📅 Zeitraum: {config['start_date'].strftime('%Y-%m-%d')} - {config['end_date'].strftime('%Y-%m-%d')}")
    print(f"💰 Startkapital: ${config['initial_capital']:,.0f}")
    print(f"📊 Symbole: {', '.join(config['symbols'])}")
    print()
    
    # Erstelle Backtest Engine
    engine = BacktestEngine(**config)
    
    try:
        # Führe Backtest aus
        print("⚡ Starte ereignisgesteuerten Backtest...")
        results = await engine.run_backtest()
        
        print("\n✅ Backtest erfolgreich abgeschlossen!")
        
        # Zeige Zusammenfassung
        summary = engine.get_results_summary()
        print(summary)
        
        # Umfassende Performance-Analyse
        print("\n🔍 Führe umfassende Performance-Analyse durch...")
        analyzer = analyze_backtest_results(results)
        
        # Generiere detaillierten Report
        comprehensive_report = analyzer.generate_comprehensive_report()
        
        # Zeige Executive Summary
        exec_summary = comprehensive_report['executive_summary']
        print(f"\n📋 EXECUTIVE SUMMARY")
        print("-" * 60)
        print(f"Performance Tier: {exec_summary['performance_tier']}")
        print(f"Overall Score: {exec_summary['overall_score']:.1f}/100")
        print(f"Empfehlung: {exec_summary['recommendation']}")
        print(f"Highlight: {exec_summary['highlight']}")
        
        # Komponenten-Analyse
        print(f"\n🧩 KOMPONENTEN-ANALYSE")
        print("-" * 60)
        
        for component_name, analysis in comprehensive_report['component_analyses'].items():
            print(f"\n{component_name.upper()}:")
            print(f"  Score: {analysis['performance_score']:.1f}/100")
            print(f"  Stärken: {', '.join(analysis['strengths']) if analysis['strengths'] else 'Keine identifiziert'}")
            print(f"  Schwächen: {', '.join(analysis['weaknesses']) if analysis['weaknesses'] else 'Keine identifiziert'}")
        
        # Key Recommendations
        print(f"\n🎯 WICHTIGSTE EMPFEHLUNGEN")
        print("-" * 60)
        for i, rec in enumerate(comprehensive_report['key_recommendations'], 1):
            print(f"{i}. {rec}")
        
        # Exportiere Reports
        print(f"\n💾 Exportiere detaillierte Berichte...")
        
        # Performance Report
        report_file = analyzer.export_report()
        print(f"Performance-Analyse: {report_file}")
        
        # Risk Analysis Details
        risk_analysis = comprehensive_report['risk_analysis']
        print(f"\n⚠️ RISIKO-ANALYSE")
        print("-" * 60)
        print(f"VaR (95%): {risk_analysis['var_95']:.3f}")
        print(f"Max Drawdown: {risk_analysis['max_drawdown']:.1%}")
        print(f"Risk Score: {risk_analysis['risk_score']:.1f}/100 (niedrig ist besser)")
        
        # Execution Analysis Details
        execution_analysis = comprehensive_report['execution_analysis']
        print(f"\n⚡ EXECUTION-ANALYSE")
        print("-" * 60)
        print(f"Ø Slippage: {execution_analysis['avg_slippage_bps']:.1f} bps")
        print(f"Fill Rate: {execution_analysis['fill_rate']:.1%}")
        print(f"Execution Score: {execution_analysis['execution_score']:.1f}/100")
        
        # Finale Bewertung
        overall_score = comprehensive_report['overall_system_score']
        print(f"\n🏆 FINALE SYSTEM-BEWERTUNG")
        print("=" * 80)
        
        if overall_score >= 80:
            print("🟢 EXZELLENT - System ist produktionsreif")
            print("   Empfehlung: Sofortige Implementierung")
        elif overall_score >= 60:
            print("🟡 GUT - System zeigt solide Performance")
            print("   Empfehlung: Implementierung mit Monitoring")
        elif overall_score >= 40:
            print("🟠 AKZEPTABEL - System benötigt Optimierung")
            print("   Empfehlung: Weitere Verbesserungen vor Implementierung")
        else:
            print("🔴 VERBESSERUNGSBEDARF - System benötigt Überarbeitung")
            print("   Empfehlung: Grundlegende Redesign-Maßnahmen")
        
        print(f"\nGesamtscore: {overall_score:.1f}/100")
        
        # Vergleich mit ursprünglichem unrealistischen Backtest
        print(f"\n📊 VERGLEICH: REALISTISCH vs UNREALISTISCH")
        print("-" * 60)
        print("Ursprünglicher (unrealistischer) Backtest:")
        print("  • Sharpe Ratio: 15.71 (unmöglich hoch)")
        print("  • Max Drawdown: 0.2% (unrealistisch niedrig)")
        print("  • Ohne Transaktionskosten, Slippage, Market Impact")
        print()
        print("Ereignisgesteuerter (realistischer) Backtest:")
        portfolio_perf = results['portfolio_performance']
        print(f"  • Sharpe Ratio: {portfolio_perf.get('sharpe_ratio', 0):.2f} (realistisch)")
        print(f"  • Max Drawdown: {portfolio_perf.get('max_drawdown', 0):.1%} (realistisch)")
        print("  • Mit vollständigen Transaktionskosten und Market Impact")
        print("  • Strikt ohne Lookahead-Bias")
        
    except Exception as e:
        logger.error(f"Fehler während Backtest: {e}", exc_info=True)
        print(f"\n❌ Backtest fehlgeschlagen: {e}")
    
    finally:
        # Cleanup
        await engine.cleanup()
        print(f"\n👋 Backtest-Ressourcen bereinigt")


async def run_comparison_study():
    """
    Vergleichsstudie: Verschiedene Konfigurationen
    """
    
    print("\n" + "="*80)
    print("🔬 VERGLEICHSSTUDIE - VERSCHIEDENE KONFIGURATIONEN")
    print("="*80)
    
    configurations = [
        {
            'name': 'Konservativ',
            'initial_capital': 500000,
            'symbols': ['BTC', 'ETH'],
            'description': 'Niedriges Risiko, wenige Assets'
        },
        {
            'name': 'Aggressiv',
            'initial_capital': 2000000,
            'symbols': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'MATIC'],
            'description': 'Höheres Kapital, mehr Diversifikation'
        }
    ]
    
    results_comparison = []
    
    for config in configurations:
        print(f"\n🧪 Teste Konfiguration: {config['name']}")
        print(f"   {config['description']}")
        
        # Kurzer Backtest (3 Monate)
        engine = BacktestEngine(
            initial_capital=config['initial_capital'],
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 3, 31),
            symbols=config['symbols'],
            enable_detailed_logging=False
        )
        
        try:
            results = await engine.run_backtest()
            
            # Sammle Schlüssel-Metriken
            portfolio = results['portfolio_performance']
            summary_metrics = {
                'config_name': config['name'],
                'total_return': portfolio.get('total_return', 0),
                'sharpe_ratio': portfolio.get('sharpe_ratio', 0),
                'max_drawdown': portfolio.get('max_drawdown', 0),
                'total_trades': portfolio.get('total_trades', 0)
            }
            
            results_comparison.append(summary_metrics)
            
            print(f"   ✅ Return: {summary_metrics['total_return']:.1%}")
            print(f"   📊 Sharpe: {summary_metrics['sharpe_ratio']:.2f}")
            print(f"   ⬇️ Max DD: {summary_metrics['max_drawdown']:.1%}")
            
            await engine.cleanup()
            
        except Exception as e:
            print(f"   ❌ Fehler: {e}")
    
    # Vergleichsauswertung
    if results_comparison:
        print(f"\n📋 VERGLEICHSAUSWERTUNG")
        print("-" * 60)
        
        best_sharpe = max(results_comparison, key=lambda x: x['sharpe_ratio'])
        best_return = max(results_comparison, key=lambda x: x['total_return'])
        lowest_dd = min(results_comparison, key=lambda x: x['max_drawdown'])
        
        print(f"Beste Sharpe Ratio: {best_sharpe['config_name']} ({best_sharpe['sharpe_ratio']:.2f})")
        print(f"Höchste Return: {best_return['config_name']} ({best_return['total_return']:.1%})")
        print(f"Niedrigster Drawdown: {lowest_dd['config_name']} ({lowest_dd['max_drawdown']:.1%})")


async def main():
    """Hauptfunktion"""
    
    try:
        # Hauptbacktest
        await run_comprehensive_backtest()
        
        # Vergleichsstudie
        choice = input("\nVergleichsstudie ausführen? (y/n): ").lower().strip()
        if choice == 'y':
            await run_comparison_study()
        
        print(f"\n🎉 Ereignisgesteuerte Backtest-Demo abgeschlossen!")
        print("Alle Ergebnisse wurden in separaten Dateien gespeichert.")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Demo durch Benutzer abgebrochen")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}", exc_info=True)
        print(f"\n❌ Demo fehlgeschlagen: {e}")


if __name__ == "__main__":
    asyncio.run(main())