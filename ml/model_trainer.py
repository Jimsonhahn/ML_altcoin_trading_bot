"""
Model Trainer - Automatisches Re-Training für ML-Modelle
Überwacht Model-Performance und führt automatisches Re-Training durch
"""

import logging
import asyncio
import schedule
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import joblib
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

try:
    from .market_predictor import MarketPredictor
    from .alpha_finder import AlphaFinder
    HAS_ML_COMPONENTS = True
except ImportError:
    HAS_ML_COMPONENTS = False

try:
    from sklearn.metrics import accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class TrainingJob:
    """Datenklasse für Training Jobs"""
    job_id: str
    model_type: str
    scheduled_time: datetime
    priority: str  # 'low', 'medium', 'high'
    trigger_reason: str
    config: Dict[str, Any]
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None


@dataclass
class ModelPerformanceMetrics:
    """Datenklasse für Model Performance Metriken"""
    model_type: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_samples: int
    validation_samples: int
    feature_count: int
    training_time_seconds: float
    cv_score: float
    feature_importance: Dict[str, float]


class ModelTrainer:
    """
    Automatisches Model Training und Re-Training System
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Training configuration
        self.training_schedule = self.config.get('training_schedule', {
            'daily_retrain': True,
            'retrain_time': '02:00',  # 2 AM
            'performance_check_interval': 6,  # 6 hours
            'emergency_retrain_threshold': 0.1  # 10% accuracy drop
        })
        
        # Performance monitoring
        self.performance_history = deque(maxlen=100)
        self.performance_thresholds = self.config.get('performance_thresholds', {
            'min_accuracy': 0.6,
            'accuracy_drop_threshold': 0.1,
            'max_training_time': 3600,  # 1 hour
            'min_samples_required': 1000
        })
        
        # Model configurations
        self.model_configs = self.config.get('model_configs', {
            'market_predictor': {
                'retrain_frequency_hours': 24,
                'min_performance_drop': 0.05,
                'feature_importance_threshold': 0.01
            },
            'alpha_finder': {
                'retrain_frequency_hours': 168,  # Weekly
                'signal_accuracy_threshold': 0.3
            }
        })
        
        # Training queue and status
        self.training_queue = []
        self.active_jobs = {}
        self.job_history = deque(maxlen=50)
        
        # Data management
        self.data_manager = None
        self.data_cache = {}
        self.last_data_update = {}
        
        # Paths
        self.base_path = Path(self.config.get('model_path', 'data/ml_models'))
        self.metrics_path = self.base_path / 'training_metrics.json'
        self.job_history_path = self.base_path / 'job_history.json'
        
        # Threading
        self.scheduler_thread = None
        self.is_running = False
        
        # Performance tracking
        self.training_metrics = {}
        self.model_performance_tracker = defaultdict(list)
        
        self.logger.info("ModelTrainer initialized")
    
    def start_scheduler(self) -> None:
        """Startet den automatischen Training Scheduler"""
        try:
            if self.is_running:
                self.logger.warning("Scheduler already running")
                return
            
            self.is_running = True
            
            # Schedule daily retraining
            if self.training_schedule['daily_retrain']:
                retrain_time = self.training_schedule['retrain_time']
                schedule.every().day.at(retrain_time).do(self._schedule_daily_retrain)
            
            # Schedule performance checks
            check_interval = self.training_schedule['performance_check_interval']
            schedule.every(check_interval).hours.do(self._check_model_performance)
            
            # Start scheduler thread
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            self.logger.info(f"Training scheduler started with daily retrain at {retrain_time}")
            
        except Exception as e:
            self.logger.error(f"Error starting scheduler: {e}")
    
    def stop_scheduler(self) -> None:
        """Stoppt den Training Scheduler"""
        try:
            self.is_running = False
            schedule.clear()
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            self.logger.info("Training scheduler stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping scheduler: {e}")
    
    def _run_scheduler(self) -> None:
        """Haupt-Scheduler Loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in scheduler: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _schedule_daily_retrain(self) -> None:
        """Schedules daily retraining jobs"""
        try:
            # Schedule market predictor retraining
            self.schedule_training(
                model_type='market_predictor',
                priority='medium',
                trigger_reason='daily_scheduled'
            )
            
            # Schedule alpha finder retraining (weekly)
            current_day = datetime.now().weekday()
            if current_day == 0:  # Monday
                self.schedule_training(
                    model_type='alpha_finder',
                    priority='low',
                    trigger_reason='weekly_scheduled'
                )
            
            self.logger.info("Daily retraining jobs scheduled")
            
        except Exception as e:
            self.logger.error(f"Error scheduling daily retrain: {e}")
    
    def _check_model_performance(self) -> None:
        """Überprüft Model Performance und triggert Re-Training wenn nötig"""
        try:
            # Check market predictor performance
            mp_performance = self._evaluate_model_performance('market_predictor')
            if mp_performance and self._needs_retraining('market_predictor', mp_performance):
                self.schedule_training(
                    model_type='market_predictor',
                    priority='high',
                    trigger_reason='performance_degradation'
                )
            
            # Check alpha finder performance
            af_performance = self._evaluate_model_performance('alpha_finder')
            if af_performance and self._needs_retraining('alpha_finder', af_performance):
                self.schedule_training(
                    model_type='alpha_finder',
                    priority='medium',
                    trigger_reason='performance_degradation'
                )
            
            self.logger.info("Model performance check completed")
            
        except Exception as e:
            self.logger.error(f"Error checking model performance: {e}")
    
    def schedule_training(self, model_type: str, priority: str = 'medium',
                         trigger_reason: str = 'manual', config: Optional[Dict] = None) -> str:
        """Schedules a training job"""
        try:
            job_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine scheduling time based on priority
            if priority == 'high':
                scheduled_time = datetime.now() + timedelta(minutes=1)
            elif priority == 'medium':
                scheduled_time = datetime.now() + timedelta(minutes=10)
            else:  # low
                scheduled_time = datetime.now() + timedelta(hours=1)
            
            # Create training job
            job = TrainingJob(
                job_id=job_id,
                model_type=model_type,
                scheduled_time=scheduled_time,
                priority=priority,
                trigger_reason=trigger_reason,
                config=config or self.model_configs.get(model_type, {})
            )
            
            # Add to queue
            self.training_queue.append(job)
            self.training_queue.sort(key=lambda x: (x.priority != 'high', x.scheduled_time))
            
            self.logger.info(f"Training job scheduled: {job_id} ({model_type}, {priority})")
            return job_id
            
        except Exception as e:
            self.logger.error(f"Error scheduling training: {e}")
            return ""
    
    async def run_training_queue(self) -> None:
        """Führt Training Jobs aus der Queue aus"""
        try:
            while self.training_queue:
                job = self.training_queue[0]
                
                # Check if job is ready to run
                if datetime.now() >= job.scheduled_time:
                    self.training_queue.pop(0)
                    await self._execute_training_job(job)
                else:
                    break  # Wait for next scheduled job
            
        except Exception as e:
            self.logger.error(f"Error running training queue: {e}")
    
    async def _execute_training_job(self, job: TrainingJob) -> None:
        """Führt einen einzelnen Training Job aus"""
        try:
            self.logger.info(f"Starting training job: {job.job_id}")
            
            job.status = 'running'
            job.start_time = datetime.now()
            self.active_jobs[job.job_id] = job
            
            # Execute training based on model type
            if job.model_type == 'market_predictor':
                result = await self._train_market_predictor(job.config)
            elif job.model_type == 'alpha_finder':
                result = await self._train_alpha_finder(job.config)
            else:
                result = {'error': f'Unknown model type: {job.model_type}'}
            
            # Update job status
            job.end_time = datetime.now()
            job.result = result
            
            if 'error' in result:
                job.status = 'failed'
                self.logger.error(f"Training job failed: {job.job_id} - {result['error']}")
            else:
                job.status = 'completed'
                self.logger.info(f"Training job completed: {job.job_id}")
                
                # Store performance metrics
                await self._store_training_metrics(job, result)
            
            # Move to history
            self.job_history.append(job)
            del self.active_jobs[job.job_id]
            
            # Save job history
            self._save_job_history()
            
        except Exception as e:
            job.status = 'failed'
            job.end_time = datetime.now()
            job.result = {'error': str(e)}
            self.logger.error(f"Error executing training job {job.job_id}: {e}")
    
    async def _train_market_predictor(self, config: Dict) -> Dict[str, Any]:
        """Trainiert den Market Predictor"""
        try:
            if not HAS_ML_COMPONENTS:
                return {'error': 'ML components not available'}
            
            # Initialize market predictor
            predictor = MarketPredictor(config)
            
            # Get training data
            training_data = await self._get_training_data('market_predictor')
            if training_data.empty:
                return {'error': 'No training data available'}
            
            # Train model
            start_time = time.time()
            success = predictor.train_model(training_data, retrain=True)
            training_time = time.time() - start_time
            
            if not success:
                return {'error': 'Model training failed'}
            
            # Get model info and performance
            model_info = predictor.get_model_info()
            
            return {
                'success': True,
                'model_info': model_info,
                'training_time': training_time,
                'training_samples': len(training_data),
                'features_count': len(predictor.feature_names),
                'performance': predictor.model_performance
            }
            
        except Exception as e:
            self.logger.error(f"Error training market predictor: {e}")
            return {'error': str(e)}
    
    async def _train_alpha_finder(self, config: Dict) -> Dict[str, Any]:
        """Trainiert den Alpha Finder (aktualisiert Konfiguration)"""
        try:
            if not HAS_ML_COMPONENTS:
                return {'error': 'ML components not available'}
            
            # Alpha Finder doesn't have traditional training, but we can:
            # 1. Update signal thresholds based on historical performance
            # 2. Calibrate confidence scores
            # 3. Update API rate limits and configurations
            
            start_time = time.time()
            
            # Initialize alpha finder
            alpha_finder = AlphaFinder(config)
            
            # Collect recent alpha signals
            recent_signals = await alpha_finder.find_alpha_signals()
            
            # Analyze signal performance
            signal_analysis = self._analyze_alpha_performance(recent_signals)
            
            training_time = time.time() - start_time
            
            return {
                'success': True,
                'signals_analyzed': len(recent_signals),
                'signal_analysis': signal_analysis,
                'training_time': training_time,
                'updated_thresholds': self._optimize_alpha_thresholds(signal_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error training alpha finder: {e}")
            return {'error': str(e)}
    
    def _analyze_alpha_performance(self, signals: List) -> Dict[str, Any]:
        """Analysiert die Performance von Alpha Signalen"""
        try:
            if not signals:
                return {'error': 'No signals to analyze'}
            
            # Group signals by type
            signal_types = defaultdict(list)
            for signal in signals:
                signal_types[signal.signal_type].append(signal)
            
            analysis = {}
            for signal_type, type_signals in signal_types.items():
                strengths = [s.strength for s in type_signals]
                confidences = [s.confidence for s in type_signals]
                
                analysis[signal_type] = {
                    'count': len(type_signals),
                    'avg_strength': np.mean(strengths),
                    'avg_confidence': np.mean(confidences),
                    'strength_std': np.std(strengths),
                    'confidence_std': np.std(confidences)
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing alpha performance: {e}")
            return {'error': str(e)}
    
    def _optimize_alpha_thresholds(self, analysis: Dict) -> Dict[str, float]:
        """Optimiert Alpha Signal Thresholds basierend auf Performance"""
        try:
            optimized_thresholds = {}
            
            for signal_type, metrics in analysis.items():
                if isinstance(metrics, dict) and 'avg_confidence' in metrics:
                    # Adjust thresholds based on historical performance
                    current_threshold = 0.5  # Default
                    avg_confidence = metrics['avg_confidence']
                    
                    # If average confidence is high, we can raise threshold
                    if avg_confidence > 0.7:
                        optimized_thresholds[signal_type] = min(0.8, current_threshold + 0.1)
                    elif avg_confidence < 0.3:
                        optimized_thresholds[signal_type] = max(0.2, current_threshold - 0.1)
                    else:
                        optimized_thresholds[signal_type] = current_threshold
            
            return optimized_thresholds
            
        except Exception as e:
            self.logger.error(f"Error optimizing thresholds: {e}")
            return {}
    
    async def _get_training_data(self, model_type: str) -> pd.DataFrame:
        """Holt Training Daten für ein Model"""
        try:
            # This would typically fetch data from your data manager
            # For now, we'll return empty DataFrame
            # In a real implementation, you'd fetch historical market data
            
            if model_type == 'market_predictor':
                # Fetch OHLCV data, funding rates, etc.
                # Return DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                pass
            elif model_type == 'alpha_finder':
                # Fetch historical signal data and outcomes
                pass
            
            return pd.DataFrame()  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()
    
    def _evaluate_model_performance(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Evaluiert die aktuelle Model Performance"""
        try:
            if model_type == 'market_predictor':
                # Load model and evaluate on recent data
                predictor = MarketPredictor()
                if predictor.load_model():
                    return predictor.model_performance
            elif model_type == 'alpha_finder':
                # Evaluate alpha signal accuracy
                return self._evaluate_alpha_signals()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {e}")
            return None
    
    def _evaluate_alpha_signals(self) -> Dict[str, Any]:
        """Evaluiert Alpha Signal Performance"""
        try:
            # This would evaluate recent alpha signals against actual market movements
            # Return performance metrics
            return {
                'signal_accuracy': 0.6,  # Placeholder
                'avg_confidence': 0.5,
                'signal_count': 100
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating alpha signals: {e}")
            return {}
    
    def _needs_retraining(self, model_type: str, performance: Dict) -> bool:
        """Bestimmt ob ein Model Re-Training braucht"""
        try:
            config = self.model_configs.get(model_type, {})
            
            if model_type == 'market_predictor':
                current_accuracy = performance.get('accuracy', 0)
                min_accuracy = self.performance_thresholds['min_accuracy']
                
                # Check if accuracy dropped below threshold
                if current_accuracy < min_accuracy:
                    return True
                
                # Check accuracy drop from recent performance
                if self.model_performance_tracker[model_type]:
                    recent_performance = self.model_performance_tracker[model_type][-5:]
                    avg_recent_accuracy = np.mean([p.get('accuracy', 0) for p in recent_performance])
                    
                    accuracy_drop = avg_recent_accuracy - current_accuracy
                    if accuracy_drop > self.performance_thresholds['accuracy_drop_threshold']:
                        return True
            
            elif model_type == 'alpha_finder':
                signal_accuracy = performance.get('signal_accuracy', 0)
                threshold = config.get('signal_accuracy_threshold', 0.3)
                
                if signal_accuracy < threshold:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking retraining need: {e}")
            return False
    
    async def _store_training_metrics(self, job: TrainingJob, result: Dict) -> None:
        """Speichert Training Metriken"""
        try:
            if job.model_type not in self.training_metrics:
                self.training_metrics[job.model_type] = []
            
            metrics = {
                'job_id': job.job_id,
                'timestamp': job.end_time.isoformat(),
                'training_time': (job.end_time - job.start_time).total_seconds(),
                'trigger_reason': job.trigger_reason,
                'result': result
            }
            
            self.training_metrics[job.model_type].append(metrics)
            self.model_performance_tracker[job.model_type].append(result)
            
            # Save to file
            self._save_training_metrics()
            
        except Exception as e:
            self.logger.error(f"Error storing training metrics: {e}")
    
    def _save_training_metrics(self) -> None:
        """Speichert Training Metriken in Datei"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            with open(self.metrics_path, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving training metrics: {e}")
    
    def _save_job_history(self) -> None:
        """Speichert Job History in Datei"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Convert job history to dict
            history_data = [asdict(job) for job in self.job_history]
            
            # Convert datetime objects to strings
            for job_data in history_data:
                for key, value in job_data.items():
                    if isinstance(value, datetime):
                        job_data[key] = value.isoformat()
            
            with open(self.job_history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving job history: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Gibt aktuellen Training Status zurück"""
        try:
            return {
                'is_running': self.is_running,
                'queued_jobs': len(self.training_queue),
                'active_jobs': len(self.active_jobs),
                'completed_jobs': len([j for j in self.job_history if j.status == 'completed']),
                'failed_jobs': len([j for j in self.job_history if j.status == 'failed']),
                'next_scheduled': self.training_queue[0].scheduled_time.isoformat() if self.training_queue else None,
                'active_job_ids': list(self.active_jobs.keys()),
                'recent_jobs': [
                    {
                        'job_id': job.job_id,
                        'model_type': job.model_type,
                        'status': job.status,
                        'trigger_reason': job.trigger_reason
                    } for job in list(self.job_history)[-5:]
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting training status: {e}")
            return {'error': str(e)}
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Gibt Model Performance Zusammenfassung zurück"""
        try:
            summary = {}
            
            for model_type, performances in self.model_performance_tracker.items():
                if performances:
                    recent_performance = performances[-1]
                    summary[model_type] = {
                        'latest_performance': recent_performance,
                        'performance_trend': self._calculate_performance_trend(performances),
                        'total_trainings': len(performances),
                        'needs_retraining': self._needs_retraining(model_type, recent_performance)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_trend(self, performances: List[Dict]) -> str:
        """Berechnet Performance Trend"""
        try:
            if len(performances) < 2:
                return 'insufficient_data'
            
            # Simple trend calculation based on accuracy
            recent_accuracies = [p.get('accuracy', 0) for p in performances[-3:]]
            if len(recent_accuracies) < 2:
                return 'stable'
            
            trend = np.polyfit(range(len(recent_accuracies)), recent_accuracies, 1)[0]
            
            if trend > 0.01:
                return 'improving'
            elif trend < -0.01:
                return 'degrading'
            else:
                return 'stable'
                
        except Exception as e:
            self.logger.error(f"Error calculating performance trend: {e}")
            return 'unknown'
    
    async def force_retrain(self, model_type: str, priority: str = 'high') -> str:
        """Erzwingt sofortiges Re-Training"""
        try:
            job_id = self.schedule_training(
                model_type=model_type,
                priority=priority,
                trigger_reason='forced_manual'
            )
            
            # Run training queue immediately
            await self.run_training_queue()
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Error forcing retrain: {e}")
            return ""
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Bereinigt alte Training Daten"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean job history
            self.job_history = deque([
                job for job in self.job_history
                if job.end_time and job.end_time > cutoff_date
            ], maxlen=50)
            
            # Clean training metrics
            for model_type in self.training_metrics:
                self.training_metrics[model_type] = [
                    metric for metric in self.training_metrics[model_type]
                    if datetime.fromisoformat(metric['timestamp']) > cutoff_date
                ]
            
            # Save cleaned data
            self._save_training_metrics()
            self._save_job_history()
            
            self.logger.info(f"Cleaned training data older than {days_to_keep} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning old data: {e}")