"""
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
