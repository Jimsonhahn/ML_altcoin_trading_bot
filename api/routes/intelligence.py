#!/usr/bin/env python3
"""
Intelligence API Routes für Enhanced Decision Logger
Neue Endpunkte für KI-Features und Dashboard Integration
"""

from flask import Blueprint, jsonify, request, Response, stream_with_context
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import asyncio
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Blueprint erstellen
intelligence_bp = Blueprint('intelligence', __name__, url_prefix='/api/v1/intelligence')

# Global variable für Enhanced Decision Logger (wird von main.py gesetzt)
enhanced_logger = None

def set_enhanced_logger(logger_instance):
    """Set the enhanced logger instance"""
    global enhanced_logger
    enhanced_logger = logger_instance
    logger.info("✅ Enhanced logger set for API routes")

@intelligence_bp.route('/health', methods=['GET'])
def intelligence_health():
    """Intelligence system health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'intelligence_enabled': enhanced_logger is not None,
        'features': {
            'learning': enhanced_logger.learning_enabled if enhanced_logger else False,
            'dashboard_updates': enhanced_logger.dashboard_updates if enhanced_logger else False,
            'export_path': str(enhanced_logger.export_path) if enhanced_logger else None
        }
    })

@intelligence_bp.route('/metrics', methods=['GET'])
@jwt_required(optional=True)
async def get_intelligence_metrics():
    """Get comprehensive intelligence metrics für Dashboard"""
    try:
        if not enhanced_logger:
            return jsonify({'error': 'Enhanced logging not available'}), 503
        
        # Async call to get metrics
        metrics = await enhanced_logger.get_dashboard_metrics()
        
        return jsonify({
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Failed to get intelligence metrics: {e}")
        return jsonify({'error': 'Failed to get metrics'}), 500

@intelligence_bp.route('/decisions/recent', methods=['GET'])
@jwt_required(optional=True)
async def get_recent_decisions():
    """Get recent trading decisions with intelligence context"""
    try:
        if not enhanced_logger:
            return jsonify({'error': 'Enhanced logging not available'}), 503
            
        limit = request.args.get('limit', 50, type=int)
        limit = min(limit, 1000)  # Safety limit
        
        decisions = await enhanced_logger.get_recent_decisions(limit)
        
        return jsonify({
            'success': True,
            'count': len(decisions),
            'decisions': decisions,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get recent decisions: {e}")
        return jsonify({'error': 'Failed to get decisions'}), 500

@intelligence_bp.route('/anomalies', methods=['GET'])
@jwt_required(optional=True)
async def get_trading_anomalies():
    """Detect and return current trading anomalies"""
    try:
        if not enhanced_logger:
            return jsonify({'error': 'Enhanced logging not available'}), 503
            
        anomalies = await enhanced_logger.detect_trading_anomalies()
        
        # Calculate severity level
        severity_levels = [a.get('severity', 0) for a in anomalies]
        max_severity = max(severity_levels) if severity_levels else 0
        
        return jsonify({
            'success': True,
            'anomaly_count': len(anomalies),
            'max_severity': max_severity,
            'anomalies': anomalies,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to detect anomalies: {e}")
        return jsonify({'error': 'Failed to detect anomalies'}), 500

@intelligence_bp.route('/learning-report', methods=['GET', 'POST'])
@jwt_required(optional=True)
async def generate_learning_report():
    """Generate or retrieve AI learning report"""
    try:
        if not enhanced_logger:
            return jsonify({'error': 'Enhanced logging not available'}), 503
            
        if request.method == 'POST':
            # Generate new report
            data = request.get_json()
            days = data.get('days', 7) if data else 7
            days = max(1, min(days, 365))  # Safety limits
            
            report = await enhanced_logger.generate_ai_learning_report(days)
            
            return jsonify({
                'success': True,
                'report_generated': True,
                'report': report,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            # Get existing reports
            export_path = enhanced_logger.export_path
            report_files = list(export_path.glob("learning_report_*.json"))
            
            reports = []
            for file_path in sorted(report_files, reverse=True)[:10]:  # Last 10 reports
                try:
                    with open(file_path, 'r') as f:
                        report_data = json.load(f)
                        reports.append({
                            'filename': file_path.name,
                            'generation_time': report_data.get('generation_time'),
                            'analysis_period_days': report_data.get('analysis_period_days'),
                            'size': file_path.stat().st_size
                        })
                except Exception as e:
                    logger.warning(f"Failed to read report {file_path}: {e}")
            
            return jsonify({
                'success': True,
                'available_reports': reports,
                'export_path': str(export_path)
            })
        
    except Exception as e:
        logger.error(f"Learning report error: {e}")
        return jsonify({'error': 'Learning report failed'}), 500

@intelligence_bp.route('/export/<export_type>', methods=['GET'])
@jwt_required()
def export_data(export_type):
    """Export data für Claude Code analysis"""
    try:
        if not enhanced_logger:
            return jsonify({'error': 'Enhanced logging not available'}), 503
            
        export_path = enhanced_logger.export_path
        
        # Supported export types
        export_files = {
            'decisions': 'structured_decisions.json',
            'latest': 'latest_decisions.jsonl',
            'patterns': 'patterns_last_7d.json',
            'anomalies': 'anomalies_last_7d.json'
        }
        
        if export_type not in export_files:
            return jsonify({'error': 'Invalid export type'}), 400
            
        file_path = export_path / export_files[export_type]
        
        if not file_path.exists():
            return jsonify({'error': 'Export file not found'}), 404
            
        # Stream file content
        def generate():
            with open(file_path, 'r') as f:
                for line in f:
                    yield line
        
        response = Response(
            stream_with_context(generate()),
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename={export_files[export_type]}',
                'Content-Type': 'application/json'
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return jsonify({'error': 'Export failed'}), 500

@intelligence_bp.route('/patterns', methods=['GET'])
@jwt_required(optional=True)
async def get_pattern_analysis():
    """Get pattern recognition results"""
    try:
        if not enhanced_logger:
            return jsonify({'error': 'Enhanced logging not available'}), 503
            
        # Get pattern cache
        patterns = enhanced_logger._pattern_cache
        
        # Convert to API-friendly format
        pattern_list = []
        for pattern_hash, pattern_data in patterns.items():
            pattern_list.append({
                'pattern_hash': pattern_hash,
                'count': pattern_data['count'],
                'first_seen': pattern_data['first_seen'].isoformat(),
                'last_seen': pattern_data['last_seen'].isoformat(),
                'success_rate': pattern_data['success_rate'],
                'pattern_summary': pattern_data['pattern']
            })
        
        # Sort by frequency
        pattern_list.sort(key=lambda x: x['count'], reverse=True)
        
        return jsonify({
            'success': True,
            'pattern_count': len(pattern_list),
            'patterns': pattern_list[:50],  # Top 50 patterns
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        return jsonify({'error': 'Pattern analysis failed'}), 500

@intelligence_bp.route('/insights/latest', methods=['GET'])
@jwt_required(optional=True)
async def get_latest_insights():
    """Get latest ML insights and learnings"""
    try:
        if not enhanced_logger:
            return jsonify({'error': 'Enhanced logging not available'}), 503
            
        insights = enhanced_logger._learning_insights[-50:]  # Last 50 insights
        
        return jsonify({
            'success': True,
            'insight_count': len(insights),
            'insights': insights,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get insights: {e}")
        return jsonify({'error': 'Failed to get insights'}), 500

@intelligence_bp.route('/log-decision', methods=['POST'])
@jwt_required()
async def log_trading_decision():
    """Manually log a trading decision with context"""
    try:
        if not enhanced_logger:
            return jsonify({'error': 'Enhanced logging not available'}), 503
            
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        decision_data = data.get('decision_data', {})
        market_context = data.get('market_context', {})
        strategy_reasoning = data.get('strategy_reasoning', '')
        confidence_level = data.get('confidence_level', 0.5)
        
        decision_id = await enhanced_logger.log_trading_decision_with_context(
            decision_data=decision_data,
            market_context=market_context,
            strategy_reasoning=strategy_reasoning,
            confidence_level=confidence_level
        )
        
        return jsonify({
            'success': True,
            'decision_id': decision_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to log decision: {e}")
        return jsonify({'error': 'Failed to log decision'}), 500

@intelligence_bp.route('/config', methods=['GET', 'POST'])
@jwt_required()
async def intelligence_config():
    """Get or update intelligence configuration"""
    try:
        if not enhanced_logger:
            return jsonify({'error': 'Enhanced logging not available'}), 503
            
        if request.method == 'POST':
            # Update configuration
            data = request.get_json()
            
            if 'learning_enabled' in data:
                enhanced_logger.learning_enabled = data['learning_enabled']
            if 'dashboard_updates' in data:
                enhanced_logger.dashboard_updates = data['dashboard_updates']
                
            return jsonify({
                'success': True,
                'message': 'Configuration updated',
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            # Get current configuration
            return jsonify({
                'success': True,
                'config': {
                    'learning_enabled': enhanced_logger.learning_enabled,
                    'dashboard_updates': enhanced_logger.dashboard_updates,
                    'export_path': str(enhanced_logger.export_path),
                    'session_id': enhanced_logger.session_id,
                    'batch_size': enhanced_logger.batch_size,
                    'flush_interval': enhanced_logger.flush_interval
                },
                'timestamp': datetime.utcnow().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Config operation failed: {e}")
        return jsonify({'error': 'Config operation failed'}), 500

# WebSocket events für real-time updates
@intelligence_bp.route('/websocket/connect', methods=['POST'])
@jwt_required(optional=True)
def websocket_connect():
    """Connect to intelligence WebSocket for real-time updates"""
    return jsonify({
        'websocket_url': '/socket.io',
        'namespace': '/intelligence',
        'events': [
            'new_decision',
            'anomaly_alert',
            'pattern_discovered',
            'learning_update'
        ]
    })

# Error handlers specific to intelligence
@intelligence_bp.errorhandler(503)
def service_unavailable(error):
    return jsonify({
        'error': 'Intelligence service unavailable',
        'message': 'Enhanced logging not initialized'
    }), 503

@intelligence_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Intelligence service error',
        'message': 'Internal processing error'
    }), 500