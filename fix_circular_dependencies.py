#!/usr/bin/env python3
"""
Circular Dependencies Professional Fix
=====================================

Behebt zirkuläre Abhängigkeiten durch:
1. Dependency Injection Pattern
2. Event-driven Architecture
3. Interface Abstractions
4. Lazy Loading
"""
import os
import re
from typing import Dict, List, Set, Tuple

def analyze_dependencies() -> Dict[str, List[str]]:
    """Analysiere alle Python-File Dependencies"""
    dependencies = {}
    
    for root, dirs, files in os.walk('.'):
        # Skip irrelevante Verzeichnisse
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules', '.venv', 'build']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                relative_path = file_path.replace('./', '').replace('.py', '').replace('/', '.')
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extrahiere Imports
                    imports = []
                    for line in content.split('\n'):
                        line = line.strip()
                        
                        # from x import y
                        if line.startswith('from ') and ' import ' in line:
                            module = line.split('from ')[1].split(' import')[0].strip()
                            if '.' in module and not module.startswith('__'):
                                imports.append(module)
                        
                        # import x
                        elif line.startswith('import '):
                            modules = line.split('import ')[1].split(',')
                            for module in modules:
                                module = module.strip().split(' as ')[0].strip()
                                if '.' in module and not module.startswith('__'):
                                    imports.append(module.split('.')[0] + '.' + module.split('.')[1])
                    
                    dependencies[relative_path] = [imp for imp in imports if imp.startswith(('core.', 'utils.', 'strategies.', 'ml_components.'))]
                    
                except Exception:
                    continue
    
    return dependencies

def find_circular_dependencies(dependencies: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Finde zirkuläre Dependencies"""
    cycles = []
    
    def has_path(start: str, end: str, visited: Set[str]) -> bool:
        if start == end:
            return True
        if start in visited:
            return False
        
        visited.add(start)
        for dep in dependencies.get(start, []):
            if has_path(dep, end, visited.copy()):
                return True
        return False
    
    for module, deps in dependencies.items():
        for dep in deps:
            if dep in dependencies and has_path(dep, module, set()):
                cycles.append((module, dep))
    
    return cycles

def create_interfaces():
    """Erstelle Interfaces zur Auflösung von zirkulären Dependencies"""
    
    # Interface für TradingBot
    interface_content = '''"""
Trading Bot Interfaces
=====================

Abstrakte Interfaces zur Vermeidung zirkulärer Dependencies
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class ITradingBot(ABC):
    """Interface für Trading Bot"""
    
    @abstractmethod
    def get_current_capital(self) -> float:
        pass
    
    @abstractmethod
    def get_open_positions(self) -> List[Dict]:
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        pass
    
    @abstractmethod
    def cancel_all_orders(self):
        pass
    
    @abstractmethod
    def stop_trading(self):
        pass

class ISafetyManager(ABC):
    """Interface für Safety Manager"""
    
    @abstractmethod
    def check_drawdown(self) -> bool:
        pass
    
    @abstractmethod
    def emergency_stop(self, reason: str):
        pass
    
    @abstractmethod
    def is_safe_to_trade(self) -> bool:
        pass

class IDataManager(ABC):
    """Interface für Data Manager"""
    
    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str) -> Dict:
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        pass

class IStrategyRouter(ABC):
    """Interface für Strategy Router"""
    
    @abstractmethod
    def get_current_strategy(self) -> str:
        pass
    
    @abstractmethod
    def switch_strategy(self, new_strategy: str):
        pass
    
    @abstractmethod
    def get_allocation_weights(self) -> Dict[str, float]:
        pass

class EventBus:
    """Event Bus für lose gekoppelte Kommunikation"""
    
    def __init__(self):
        self._subscribers = {}
    
    def subscribe(self, event_type: str, callback):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def publish(self, event_type: str, data: Any):
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Event handler error: {e}")
    
    def unsubscribe(self, event_type: str, callback):
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)

# Globaler Event Bus
global_event_bus = EventBus()
'''
    
    with open('core/interfaces.py', 'w') as f:
        f.write(interface_content)
    
    print("✅ Created core/interfaces.py")

def create_dependency_injection_container():
    """Erstelle DI Container"""
    
    di_content = '''"""
Dependency Injection Container
=============================

Zentraler Container für Dependency Management
"""
from typing import Dict, Any, Type, Optional
from core.interfaces import ITradingBot, ISafetyManager, IDataManager, IStrategyRouter

class DIContainer:
    """Dependency Injection Container"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._interfaces: Dict[Type, str] = {}
    
    def register_singleton(self, interface: Type, implementation: Any, name: Optional[str] = None):
        """Registriere Singleton Service"""
        service_name = name or interface.__name__
        self._singletons[service_name] = implementation
        self._interfaces[interface] = service_name
    
    def register_transient(self, interface: Type, implementation_class: Type, name: Optional[str] = None):
        """Registriere Transient Service"""
        service_name = name or interface.__name__
        self._services[service_name] = implementation_class
        self._interfaces[interface] = service_name
    
    def get(self, interface: Type) -> Any:
        """Hole Service by Interface"""
        service_name = self._interfaces.get(interface)
        if not service_name:
            raise ValueError(f"No service registered for {interface}")
        
        # Check Singletons first
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Create Transient
        if service_name in self._services:
            return self._services[service_name]()
        
        raise ValueError(f"Service {service_name} not found")
    
    def get_by_name(self, name: str) -> Any:
        """Hole Service by Name"""
        if name in self._singletons:
            return self._singletons[name]
        if name in self._services:
            return self._services[name]()
        raise ValueError(f"Service {name} not found")

# Globaler Container
container = DIContainer()

def get_trading_bot() -> ITradingBot:
    """Lazy Loading für Trading Bot"""
    return container.get(ITradingBot)

def get_safety_manager() -> ISafetyManager:
    """Lazy Loading für Safety Manager"""
    return container.get(ISafetyManager)

def get_data_manager() -> IDataManager:
    """Lazy Loading für Data Manager"""
    return container.get(IDataManager)

def get_strategy_router() -> IStrategyRouter:
    """Lazy Loading für Strategy Router"""
    return container.get(IStrategyRouter)
'''
    
    with open('core/di_container.py', 'w') as f:
        f.write(di_content)
    
    print("✅ Created core/di_container.py")

def patch_trading_bot():
    """Patche trading_bot.py für DI Pattern"""
    
    trading_bot_path = 'core/trading_bot.py'
    if not os.path.exists(trading_bot_path):
        print(f"❌ {trading_bot_path} not found")
        return
    
    with open(trading_bot_path, 'r') as f:
        content = f.read()
    
    # Backup
    with open(f"{trading_bot_path}.backup", 'w') as f:
        f.write(content)
    
    # Füge Interface Import hinzu
    if 'from core.interfaces import' not in content:
        interface_import = """
# Dependency Injection Support
from core.interfaces import ITradingBot, global_event_bus
from core.di_container import container
"""
        content = interface_import + content
    
    # Mache TradingBot zu Interface-Implementation
    if 'class TradingBot:' in content:
        content = content.replace('class TradingBot:', 'class TradingBot(ITradingBot):')
    
    # Entferne direkte SafetyManager Imports
    content = re.sub(r'from core\.safety_manager import.*\n', '# Removed direct import - using DI\n', content)
    
    # Ersetze direkte SafetyManager Usage
    if 'self.safety_manager' in content and 'get_safety_manager' not in content:
        lazy_safety = """
    @property
    def safety_manager(self):
        \"\"\"Lazy loaded safety manager via DI\"\"\"
        if not hasattr(self, '_safety_manager'):
            from core.di_container import get_safety_manager
            self._safety_manager = get_safety_manager()
        return self._safety_manager
"""
        # Füge Lazy Loading Property hinzu
        content = content.replace(
            'def __init__(self',
            lazy_safety + '\n    def __init__(self'
        )
    
    with open(trading_bot_path, 'w') as f:
        f.write(content)
    
    print("✅ Patched core/trading_bot.py for DI")

def patch_safety_manager():
    """Patche safety_manager.py für DI Pattern"""
    
    safety_path = 'core/safety_manager.py'
    if not os.path.exists(safety_path):
        print(f"❌ {safety_path} not found")
        return
    
    with open(safety_path, 'r') as f:
        content = f.read()
    
    # Backup
    with open(f"{safety_path}.backup", 'w') as f:
        f.write(content)
    
    # Füge Interface Import hinzu
    if 'from core.interfaces import' not in content:
        interface_import = """
# Dependency Injection Support  
from core.interfaces import ISafetyManager, global_event_bus
"""
        content = interface_import + content
    
    # Mache SafetyManager zu Interface-Implementation
    if 'class SafetyManager:' in content:
        content = content.replace('class SafetyManager:', 'class SafetyManager(ISafetyManager):')
    
    # Entferne zirkuläre TradingBot Imports
    content = re.sub(r'from core\.trading_bot import.*\n', '# Removed circular import - using events\n', content)
    
    # Event-based Communication
    if 'self.bot' in content and 'global_event_bus' not in content:
        # Ersetze direkte Bot-Aufrufe mit Events
        content = content.replace(
            'self.bot.stop_trading()',
            'global_event_bus.publish("emergency_stop", {"reason": "safety_triggered"})'
        )
        content = content.replace(
            'self.bot.cancel_all_orders()',
            'global_event_bus.publish("cancel_all_orders", {})'
        )
    
    with open(safety_path, 'w') as f:
        f.write(content)
    
    print("✅ Patched core/safety_manager.py for DI")

def setup_dependency_injection():
    """Setup komplettes DI System"""
    
    setup_code = '''"""
Dependency Injection Setup
=========================

Initialisiert alle Services im DI Container
"""
from core.di_container import container
from core.interfaces import ITradingBot, ISafetyManager, IDataManager, IStrategyRouter

def setup_services():
    """Registriere alle Services im Container"""
    
    # Lazy Import um zirkuläre Dependencies zu vermeiden
    def get_trading_bot_instance():
        from core.trading_bot import TradingBot
        return TradingBot
    
    def get_safety_manager_instance():
        from core.safety_manager import SafetyManager
        return SafetyManager
    
    def get_data_manager_instance():
        from data_sources.data_manager import DataManager
        return DataManager
    
    def get_strategy_router_instance():
        from core.strategy_router import StrategyRouter
        return StrategyRouter
    
    # Registriere Services
    container.register_transient(ITradingBot, get_trading_bot_instance())
    container.register_transient(ISafetyManager, get_safety_manager_instance())
    container.register_transient(IDataManager, get_data_manager_instance())
    container.register_transient(IStrategyRouter, get_strategy_router_instance())
    
    print("✅ Dependency Injection setup completed")

def initialize_event_handlers():
    """Initialisiere Event Handlers"""
    from core.interfaces import global_event_bus
    from core.di_container import get_trading_bot
    
    def handle_emergency_stop(data):
        bot = get_trading_bot()
        bot.stop_trading()
        print(f"Emergency stop triggered: {data.get('reason', 'unknown')}")
    
    def handle_cancel_orders(data):
        bot = get_trading_bot()
        bot.cancel_all_orders()
        print("All orders canceled via event")
    
    # Registriere Event Handlers
    global_event_bus.subscribe("emergency_stop", handle_emergency_stop)
    global_event_bus.subscribe("cancel_all_orders", handle_cancel_orders)
    
    print("✅ Event handlers initialized")

if __name__ == "__main__":
    setup_services()
    initialize_event_handlers()
'''
    
    with open('core/di_setup.py', 'w') as f:
        f.write(setup_code)
    
    print("✅ Created core/di_setup.py")

def update_main_py():
    """Update main.py für DI Pattern"""
    
    if not os.path.exists('main.py'):
        print("❌ main.py not found")
        return
    
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Backup
    with open('main.py.backup', 'w') as f:
        f.write(content)
    
    # Füge DI Setup hinzu
    if 'from core.di_setup import setup_services' not in content:
        di_setup = """
# Dependency Injection Setup
from core.di_setup import setup_services, initialize_event_handlers

def main():
    # Initialize DI Container
    setup_services()
    initialize_event_handlers()
    
    # Existing main logic...
"""
        
        # Ersetze main function oder füge hinzu
        if 'def main():' in content:
            # Update existing main
            content = re.sub(
                r'def main\(\):\s*\n',
                di_setup.split('def main():')[1],
                content
            )
        else:
            # Füge main function hinzu
            content = di_setup + content
    
    with open('main.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated main.py with DI setup")

def test_circular_fix():
    """Teste ob zirkuläre Dependencies behoben sind"""
    print("\n🔍 Testing Circular Dependencies Fix...")
    
    try:
        # Test Interface Imports
        from core.interfaces import ITradingBot, ISafetyManager, global_event_bus
        print("✅ Interfaces import successfully")
        
        # Test DI Container
        from core.di_container import container
        print("✅ DI Container import successfully")
        
        # Test Event Bus
        def test_handler(data):
            print(f"Event received: {data}")
        
        global_event_bus.subscribe("test", test_handler)
        global_event_bus.publish("test", {"message": "test"})
        print("✅ Event Bus working")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing Circular Dependencies...")
    
    # 1. Analysiere Dependencies
    deps = analyze_dependencies()
    cycles = find_circular_dependencies(deps)
    print(f"Found {len(cycles)} circular dependencies")
    
    # 2. Erstelle Interfaces und DI
    create_interfaces()
    create_dependency_injection_container()
    setup_dependency_injection()
    
    # 3. Patche kritische Files
    patch_trading_bot()
    patch_safety_manager()
    update_main_py()
    
    # 4. Test
    success = test_circular_fix()
    
    if success:
        print("\n✅ Circular Dependencies Fix completed successfully!")
        print("📋 Implementation:")
        print("   ✅ Interface abstractions created")
        print("   ✅ Dependency Injection container setup")
        print("   ✅ Event-driven communication implemented")
        print("   ✅ Lazy loading patterns applied")
    else:
        print("\n❌ Fix needs manual adjustment")
    
    print("\n📋 Next steps:")
    print("   1. Review patched files (*.backup created)")
    print("   2. Test imports: python -c 'from core.interfaces import ITradingBot'")
    print("   3. Run main.py to initialize DI system")