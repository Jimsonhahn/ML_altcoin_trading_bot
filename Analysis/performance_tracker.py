python  # analysis/performance_tracker.py
"""
Performance Tracker für Trading Bot Reports
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


class PerformanceTracker:
    """
    Tracked und analysiert die Trading-Performance
    """

    def __init__(self, settings=None):
        self.settings = settings or {}
        self.trades = []
        self.daily_balances = []

    def add_trade(self, trade: Dict[str, Any]):
        """Fügt einen Trade zur Historie hinzu"""
        self.trades.append(trade)

    def generate_report(self, start_date: datetime, end_date: datetime,
                        output_format: str = 'html') -> str:
        """
        Generiert einen Performance-Report

        Args:
            start_date: Startdatum für den Report
            end_date: Enddatum für den Report
            output_format: Format des Reports ('html', 'pdf', 'csv', 'json')

        Returns:
            Pfad zur generierten Report-Datei
        """
        # Report-Verzeichnis erstellen
        report_dir = os.path.join('data', 'reports')
        os.makedirs(report_dir, exist_ok=True)

        # Dateiname generieren
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"trading_report_{timestamp}"

        # Performance-Metriken berechnen
        metrics = self._calculate_metrics(start_date, end_date)

        # Report basierend auf Format generieren
        if output_format == 'html':
            return self._generate_html_report(metrics, report_dir, base_filename)
        elif output_format == 'pdf':
            return self._generate_pdf_report(metrics, report_dir, base_filename)
        elif output_format == 'csv':
            return self._generate_csv_report(metrics, report_dir, base_filename)
        elif output_format == 'json':
            return self._generate_json_report(metrics, report_dir, base_filename)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _calculate_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Berechnet Performance-Metriken"""
        # Beispiel-Metriken (erweitern Sie nach Bedarf)
        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days
            },
            'trades': {
                'total': len(self.trades),
                'profitable': sum(1 for t in self.trades if t.get('profit', 0) > 0),
                'loss': sum(1 for t in self.trades if t.get('profit', 0) < 0)
            },
            'returns': {
                'total': sum(t.get('profit', 0) for t in self.trades),
                'average': np.mean([t.get('profit', 0) for t in self.trades]) if self.trades else 0,
                'best': max([t.get('profit', 0) for t in self.trades]) if self.trades else 0,
                'worst': min([t.get('profit', 0) for t in self.trades]) if self.trades else 0
            },
            'statistics': {
                'win_rate': (sum(1 for t in self.trades if t.get('profit', 0) > 0) /
                             len(self.trades) * 100) if self.trades else 0,
                'profit_factor': self._calculate_profit_factor(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'max_drawdown': self._calculate_max_drawdown()
            }
        }

    def _calculate_profit_factor(self) -> float:
        """Berechnet den Profit Factor"""
        gross_profit = sum(t.get('profit', 0) for t in self.trades if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t.get('profit', 0) for t in self.trades if t.get('profit', 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def _calculate_sharpe_ratio(self) -> float:
        """Berechnet die Sharpe Ratio"""
        if not self.trades:
            return 0.0
        returns = [t.get('profit_pct', 0) for t in self.trades]
        if len(returns) < 2:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0

    def _calculate_max_drawdown(self) -> float:
        """Berechnet den maximalen Drawdown"""
        if not self.daily_balances:
            return 0.0

        cumulative = pd.Series(self.daily_balances)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        return abs(drawdown.min())

    def _generate_html_report(self, metrics: Dict[str, Any], report_dir: str,
                              base_filename: str) -> str:
        """Generiert HTML-Report"""
        filepath = os.path.join(report_dir, f"{base_filename}.html")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Trading Performance Report</h1>
            <p>Period: {metrics['period']['start']} to {metrics['period']['end']} 
               ({metrics['period']['days']} days)</p>

            <h2>Trade Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Trades</td>
                    <td>{metrics['trades']['total']}</td>
                </tr>
                <tr>
                    <td>Profitable Trades</td>
                    <td class="positive">{metrics['trades']['profitable']}</td>
                </tr>
                <tr>
                    <td>Loss Trades</td>
                    <td class="negative">{metrics['trades']['loss']}</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{metrics['statistics']['win_rate']:.2f}%</td>
                </tr>
            </table>

            <h2>Returns</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td class="{'positive' if metrics['returns']['total'] > 0 else 'negative'}">
                        ${metrics['returns']['total']:.2f}
                    </td>
                </tr>
                <tr>
                    <td>Average Return</td>
                    <td>${metrics['returns']['average']:.2f}</td>
                </tr>
                <tr>
                    <td>Best Trade</td>
                    <td class="positive">${metrics['returns']['best']:.2f}</td>
                </tr>
                <tr>
                    <td>Worst Trade</td>
                    <td class="negative">${metrics['returns']['worst']:.2f}</td>
                </tr>
            </table>

            <h2>Risk Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Profit Factor</td>
                    <td>{metrics['statistics']['profit_factor']:.2f}</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{metrics['statistics']['sharpe_ratio']:.2f}</td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td class="negative">{metrics['statistics']['max_drawdown']:.2f}%</td>
                </tr>
            </table>

            <p><small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        </body>
        </html>
        """

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def _generate_pdf_report(self, metrics: Dict[str, Any], report_dir: str,
                             base_filename: str) -> str:
        """Generiert PDF-Report mit Matplotlib"""
        filepath = os.path.join(report_dir, f"{base_filename}.pdf")

        with PdfPages(filepath) as pdf:
            # Seite 1: Zusammenfassung
            fig, ax = plt.subplots(figsize=(8, 11))
            ax.axis('off')

            # Title
            ax.text(0.5, 0.95, 'Trading Performance Report',
                    ha='center', va='top', fontsize=20, weight='bold')

            # Metrics Text
            y_pos = 0.85
            for section, data in metrics.items():
                if isinstance(data, dict):
                    ax.text(0.1, y_pos, section.title(), fontsize=14, weight='bold')
                    y_pos -= 0.05
                    for key, value in data.items():
                        if isinstance(value, (int, float)):
                            ax.text(0.15, y_pos, f"{key}: {value:.2f}", fontsize=12)
                        else:
                            ax.text(0.15, y_pos, f"{key}: {value}", fontsize=12)
                        y_pos -= 0.04
                    y_pos -= 0.02

            pdf.savefig(fig)
            plt.close()

        return filepath

    def _generate_csv_report(self, metrics: Dict[str, Any], report_dir: str,
                             base_filename: str) -> str:
        """Generiert CSV-Report"""
        filepath = os.path.join(report_dir, f"{base_filename}.csv")

        # Trades als DataFrame
        if self.trades:
            df = pd.DataFrame(self.trades)
            df.to_csv(filepath, index=False)
        else:
            # Leere CSV mit Metriken
            metrics_flat = []
            for section, data in metrics.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        metrics_flat.append({
                            'section': section,
                            'metric': key,
                            'value': value
                        })
            pd.DataFrame(metrics_flat).to_csv(filepath, index=False)

        return filepath

    def _generate_json_report(self, metrics: Dict[str, Any], report_dir: str,
                              base_filename: str) -> str:
        """Generiert JSON-Report"""
        filepath = os.path.join(report_dir, f"{base_filename}.json")

        report_data = {
            'generated_at': datetime.now().isoformat(),
            'metrics': metrics,
            'trades': self.trades
        }

        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        return filepath