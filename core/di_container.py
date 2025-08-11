"""
Dependency Injection Container
Vereinfacht die Verwaltung von Abhängigkeiten und verhindert zirkuläre Imports
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Generic
from threading import Lock

from config.settings import Settings
from config.environment import EnvironmentConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DIContainer:
    """
    Einfacher Dependency Injection Container für saubere Abhängigkeitsverwaltung
    """
    
    def __init__(self, settings: Settings, env_config: EnvironmentConfig):
        self.settings = settings
        self.env_config = env_config
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._lock = Lock()
        
        # Core-Services registrieren
        self.register('settings', settings)
        self.register('env_config', env_config)
        
        logger.info("DI Container initialisiert")
    
    def register(self, name: str, instance: Any) -> None:
        """
        Registriert eine Service-Instanz
        
        Args:
            name: Service-Name
            instance: Service-Instanz
        """
        with self._lock:
            self._services[name] = instance
            logger.debug(f"Service registriert: {name}")
    
    def register_factory(self, name: str, factory: callable, singleton: bool = True) -> None:
        """
        Registriert eine Factory-Funktion für lazy loading
        
        Args:
            name: Service-Name
            factory: Factory-Funktion die Service erstellt
            singleton: Ob Service als Singleton behandelt werden soll
        """
        with self._lock:
            self._factories[name] = factory
            if singleton:
                self._singletons[name] = None
            logger.debug(f"Factory registriert: {name} (singleton: {singleton})")
    
    def get(self, name: str) -> Optional[Any]:
        """
        Holt einen Service aus dem Container
        
        Args:
            name: Service-Name
            
        Returns:
            Service-Instanz oder None
        """
        with self._lock:
            # Direkt registrierte Services
            if name in self._services:
                return self._services[name]
            
            # Factory-basierte Services
            if name in self._factories:
                # Singleton-Check
                if name in self._singletons:
                    if self._singletons[name] is not None:
                        return self._singletons[name]
                    
                    # Singleton erstellen
                    instance = self._factories[name]()
                    self._singletons[name] = instance
                    return instance
                else:
                    # Neue Instanz bei jedem Aufruf
                    return self._factories[name]()
            
            logger.warning(f"Service nicht gefunden: {name}")
            return None
    
    def get_required(self, name: str) -> Any:
        """
        Holt einen Service und wirft Exception wenn nicht gefunden
        
        Args:
            name: Service-Name
            
        Returns:
            Service-Instanz
            
        Raises:
            ValueError: Wenn Service nicht gefunden
        """
        service = self.get(name)
        if service is None:
            raise ValueError(f"Required service not found: {name}")
        return service
    
    def has(self, name: str) -> bool:
        """
        Prüft ob Service verfügbar ist
        
        Args:
            name: Service-Name
            
        Returns:
            True wenn Service verfügbar
        """
        return name in self._services or name in self._factories
    
    def list_services(self) -> Dict[str, str]:
        """
        Listet alle verfügbaren Services auf
        
        Returns:
            Dictionary mit Service-Namen und Typen
        """
        services = {}
        
        for name, instance in self._services.items():
            services[name] = type(instance).__name__
        
        for name in self._factories.keys():
            services[name] = "Factory"
        
        return services
    
    def cleanup(self) -> None:
        """
        Bereinigt Container und ruft cleanup-Methoden auf
        """
        logger.info("DI Container wird bereinigt...")
        
        # Cleanup für Services mit cleanup-Methode
        for name, service in self._services.items():
            if hasattr(service, 'cleanup') and callable(service.cleanup):
                try:
                    service.cleanup()
                    logger.debug(f"Cleanup ausgeführt für: {name}")
                except Exception as e:
                    logger.error(f"Fehler beim Cleanup von {name}: {e}")
        
        # Singletons bereinigen
        for name, service in self._singletons.items():
            if service and hasattr(service, 'cleanup') and callable(service.cleanup):
                try:
                    service.cleanup()
                    logger.debug(f"Singleton-Cleanup ausgeführt für: {name}")
                except Exception as e:
                    logger.error(f"Fehler beim Singleton-Cleanup von {name}: {e}")
        
        # Container zurücksetzen
        self._services.clear()
        self._factories.clear() 
        self._singletons.clear()
        
        logger.info("DI Container bereinigt")


# Globaler Container (wird in main.py initialisiert)
_global_container: Optional[DIContainer] = None


def get_container() -> Optional[DIContainer]:
    """
    Holt den globalen DI Container
    
    Returns:
        DI Container oder None wenn nicht initialisiert
    """
    return _global_container


def set_container(container: DIContainer) -> None:
    """
    Setzt den globalen DI Container
    
    Args:
        container: DI Container Instanz
    """
    global _global_container
    _global_container = container


def inject(service_name: str) -> Any:
    """
    Decorator für Dependency Injection
    
    Args:
        service_name: Name des zu injizierenden Services
        
    Returns:
        Decorator-Funktion
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            container = get_container()
            if container:
                service = container.get(service_name)
                if service:
                    kwargs[service_name] = service
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience-Funktionen für häufig verwendete Services
def get_settings() -> Optional[Settings]:
    """Holt Settings aus dem Container"""
    container = get_container()
    return container.get('settings') if container else None


def get_env_config() -> Optional[EnvironmentConfig]:
    """Holt Environment Config aus dem Container"""
    container = get_container()
    return container.get('env_config') if container else None


def get_data_manager():
    """Holt Data Manager aus dem Container"""
    container = get_container()
    return container.get('data_manager') if container else None


def get_ml_components():
    """Holt ML Components aus dem Container"""
    container = get_container()
    return container.get('ml_components') if container else None


def get_notifier():
    """Holt Notifier aus dem Container"""
    container = get_container()
    return container.get('notifier') if container else None


def get_safety_manager():
    """Holt Safety Manager aus dem Container"""
    container = get_container()
    return container.get('safety_manager') if container else None


# Factory-Funktionen für lazy loading
def create_data_manager_factory(settings: Settings) -> callable:
    """Factory für Data Manager"""
    def factory():
        from data_sources.data_manager import DataManager
        return DataManager(settings)
    return factory


def create_ml_components_factory(settings: Settings) -> callable:
    """Factory für ML Components"""
    def factory():
        try:
            from ml_components.enhanced_ml_components import create_enhanced_ml_components
            return create_enhanced_ml_components(settings)
        except ImportError:
            logger.warning("Enhanced ML Components nicht verfügbar")
            return None
    return factory


def create_notifier_factory(settings: Settings) -> callable:
    """Factory für Notifier"""
    def factory():
        from utils.notifier import Notifier
        return Notifier(settings)
    return factory


def create_safety_manager_factory(settings: Settings) -> callable:
    """Factory für Safety Manager"""
    def factory():
        from core.safety_manager import SafetyManager
        return SafetyManager(settings)
    return factory


def setup_default_container(settings: Settings, env_config: EnvironmentConfig) -> DIContainer:
    """
    Erstellt einen DI Container mit Standard-Services
    
    Args:
        settings: Settings-Instanz
        env_config: Environment Config
        
    Returns:
        Konfigurierter DI Container
    """
    container = DIContainer(settings, env_config)
    
    # Factories registrieren für lazy loading
    container.register_factory('data_manager', create_data_manager_factory(settings))
    container.register_factory('ml_components', create_ml_components_factory(settings))
    container.register_factory('notifier', create_notifier_factory(settings))
    container.register_factory('safety_manager', create_safety_manager_factory(settings))
    
    logger.info("Standard DI Container konfiguriert")
    return container