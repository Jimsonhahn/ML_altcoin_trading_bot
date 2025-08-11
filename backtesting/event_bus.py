"""
Event Bus - Zentrales asynchrones Message Queue System
Herzstück des ereignisgesteuerten Backtesting-Frameworks
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass
import time

from .event_models import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class EventStats:
    """Statistiken für Event-Verarbeitung"""
    event_type: EventType
    count: int = 0
    total_processing_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    last_processed: Optional[datetime] = None


class EventBus:
    """
    Asynchroner Event Bus für Backtesting
    
    Features:
    - Priority Queue für Event-Reihenfolge
    - Event Filtering und Routing
    - Performance Monitoring
    - Backpressure Handling
    """
    
    def __init__(self, max_queue_size: int = 10000, enable_stats: bool = True):
        self.max_queue_size = max_queue_size
        self.enable_stats = enable_stats
        
        # Haupt-Event-Queue
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Event Handlers (Subscribers)
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Event Statistics
        self._event_stats: Dict[EventType, EventStats] = {}
        self._total_events_processed = 0
        self._start_time = time.time()
        
        # Event History (für Debugging/Analyse)
        self._event_history: deque = deque(maxlen=1000)
        
        # Active Subscribers
        self._subscribers: Set[str] = set()
        
        # Control Flags
        self._running = False
        self._paused = False
        
        logger.info(f"EventBus initialisiert (max_queue_size={max_queue_size})")
    
    async def publish(self, event: Event) -> bool:
        """
        Publiziert Event in die Queue
        
        Returns:
            bool: True wenn erfolgreich, False wenn Queue voll
        """
        try:
            # Blockiert nicht, wirft exception wenn voll
            self._event_queue.put_nowait(event)
            
            if self.enable_stats:
                self._update_publish_stats(event)
            
            return True
            
        except asyncio.QueueFull:
            logger.warning(f"Event Queue voll! Event verworfen: {event.event_type.value}")
            return False
    
    async def publish_batch(self, events: List[Event]) -> int:
        """
        Publiziert mehrere Events gleichzeitig
        
        Returns:
            int: Anzahl erfolgreich publizierter Events
        """
        published = 0
        for event in events:
            if await self.publish(event):
                published += 1
        
        return published
    
    def subscribe(self, event_type: EventType, handler: Callable, 
                  subscriber_id: Optional[str] = None) -> str:
        """
        Registriert Handler für Event-Typ
        
        Args:
            event_type: Typ der Events die gehandelt werden sollen
            handler: Async Callable das Event verarbeitet
            subscriber_id: Optionale ID für den Subscriber
            
        Returns:
            str: Subscriber ID
        """
        if subscriber_id is None:
            subscriber_id = f"{event_type.value}_{len(self._handlers[event_type])}"
        
        self._handlers[event_type].append(handler)
        self._subscribers.add(subscriber_id)
        
        # Initialisiere Stats für Event Type
        if event_type not in self._event_stats:
            self._event_stats[event_type] = EventStats(event_type)
        
        logger.debug(f"Subscriber registriert: {subscriber_id} für {event_type.value}")
        return subscriber_id
    
    def unsubscribe(self, event_type: EventType, handler: Callable) -> bool:
        """Entfernt Handler für Event-Typ"""
        try:
            self._handlers[event_type].remove(handler)
            return True
        except ValueError:
            return False
    
    async def process_events(self) -> None:
        """
        Hauptschleife für Event-Verarbeitung
        Läuft kontinuierlich und verteilt Events an Handler
        """
        self._running = True
        logger.info("EventBus processing gestartet")
        
        while self._running:
            try:
                # Pause handling
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue
                
                # Warte auf nächstes Event (mit Timeout für graceful shutdown)
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process Event
                await self._process_single_event(event)
                
            except Exception as e:
                logger.error(f"Fehler in Event-Processing: {e}", exc_info=True)
    
    async def _process_single_event(self, event: Event) -> None:
        """Verarbeitet einzelnes Event"""
        start_time = time.time()
        
        # Get handlers for this event type
        handlers = self._handlers.get(event.event_type, [])
        
        if not handlers:
            logger.debug(f"Keine Handler für Event-Typ: {event.event_type.value}")
            return
        
        # Call all handlers concurrently
        handler_tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._call_handler(handler, event))
            handler_tasks.append(task)
        
        # Wait for all handlers to complete
        results = await asyncio.gather(*handler_tasks, return_exceptions=True)
        
        # Check for exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Handler {handlers[i].__name__} failed: {result}")
        
        # Update statistics
        if self.enable_stats:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_processing_stats(event, processing_time_ms)
        
        # Add to history
        self._event_history.append(event)
        self._total_events_processed += 1
    
    async def _call_handler(self, handler: Callable, event: Event) -> Any:
        """Ruft Handler mit Error Handling auf"""
        try:
            return await handler(event)
        except Exception as e:
            logger.error(f"Error in handler {handler.__name__}: {e}", exc_info=True)
            raise
    
    def _update_publish_stats(self, event: Event) -> None:
        """Aktualisiert Publish-Statistiken"""
        if event.event_type not in self._event_stats:
            self._event_stats[event.event_type] = EventStats(event.event_type)
    
    def _update_processing_stats(self, event: Event, processing_time_ms: float) -> None:
        """Aktualisiert Processing-Statistiken"""
        stats = self._event_stats.get(event.event_type)
        if not stats:
            stats = EventStats(event.event_type)
            self._event_stats[event.event_type] = stats
        
        stats.count += 1
        stats.total_processing_time_ms += processing_time_ms
        stats.avg_processing_time_ms = stats.total_processing_time_ms / stats.count
        stats.max_processing_time_ms = max(stats.max_processing_time_ms, processing_time_ms)
        stats.last_processed = datetime.now()
    
    async def wait_until_empty(self, timeout: Optional[float] = None) -> bool:
        """
        Wartet bis Queue leer ist
        
        Returns:
            bool: True wenn Queue leer, False bei Timeout
        """
        start_time = time.time()
        
        while not self._event_queue.empty():
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            await asyncio.sleep(0.1)
        
        return True
    
    def pause(self) -> None:
        """Pausiert Event-Verarbeitung"""
        self._paused = True
        logger.info("EventBus pausiert")
    
    def resume(self) -> None:
        """Setzt Event-Verarbeitung fort"""
        self._paused = False
        logger.info("EventBus fortgesetzt")
    
    def stop(self) -> None:
        """Stoppt Event-Verarbeitung"""
        self._running = False
        logger.info("EventBus gestoppt")
    
    def get_stats(self) -> Dict[str, Any]:
        """Gibt aktuelle Statistiken zurück"""
        runtime_seconds = time.time() - self._start_time
        
        return {
            'total_events_processed': self._total_events_processed,
            'events_per_second': self._total_events_processed / runtime_seconds if runtime_seconds > 0 else 0,
            'queue_size': self._event_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'queue_utilization': self._event_queue.qsize() / self.max_queue_size,
            'runtime_seconds': runtime_seconds,
            'paused': self._paused,
            'event_type_stats': {
                event_type.value: {
                    'count': stats.count,
                    'avg_processing_ms': round(stats.avg_processing_time_ms, 2),
                    'max_processing_ms': round(stats.max_processing_time_ms, 2),
                    'last_processed': stats.last_processed.isoformat() if stats.last_processed else None
                }
                for event_type, stats in self._event_stats.items()
            },
            'active_subscribers': len(self._subscribers)
        }
    
    def get_event_history(self, limit: Optional[int] = None) -> List[Event]:
        """Gibt Event-History zurück"""
        if limit:
            return list(self._event_history)[-limit:]
        return list(self._event_history)
    
    def clear_history(self) -> None:
        """Löscht Event-History"""
        self._event_history.clear()
    
    def __repr__(self) -> str:
        return (f"EventBus(queue_size={self._event_queue.qsize()}, "
                f"processed={self._total_events_processed}, "
                f"subscribers={len(self._subscribers)})")


class PriorityEventBus(EventBus):
    """
    Event Bus mit Priority Queue
    Events werden nach Timestamp sortiert verarbeitet
    """
    
    def __init__(self, max_queue_size: int = 10000, enable_stats: bool = True):
        super().__init__(max_queue_size, enable_stats)
        
        # Ersetze normale Queue mit Priority Queue
        self._event_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._sequence_number = 0
    
    async def publish(self, event: Event) -> bool:
        """
        Publiziert Event mit Priority (Timestamp)
        """
        try:
            # Priority tuple: (timestamp, sequence_number, event)
            # Sequence number für Events mit gleichem Timestamp
            priority = (event.timestamp.timestamp(), self._sequence_number, event)
            self._sequence_number += 1
            
            self._event_queue.put_nowait(priority)
            
            if self.enable_stats:
                self._update_publish_stats(event)
            
            return True
            
        except asyncio.QueueFull:
            logger.warning(f"Priority Queue voll! Event verworfen: {event.event_type.value}")
            return False
    
    async def _get_next_event(self) -> Optional[Event]:
        """Holt nächstes Event aus Priority Queue"""
        try:
            priority_tuple = await asyncio.wait_for(
                self._event_queue.get(), 
                timeout=1.0
            )
            # Extract event from priority tuple
            _, _, event = priority_tuple
            return event
        except asyncio.TimeoutError:
            return None


# Factory Function
def create_event_bus(priority: bool = True, **kwargs) -> EventBus:
    """
    Factory für Event Bus Erstellung
    
    Args:
        priority: Wenn True, wird PriorityEventBus verwendet
        **kwargs: Weitere Argumente für EventBus
        
    Returns:
        EventBus: Konfigurierte EventBus Instanz
    """
    if priority:
        return PriorityEventBus(**kwargs)
    return EventBus(**kwargs)