#!/usr/bin/env python3
"""
Nachhaltige Korrektur der Strategy-Loading in trading_bot.py
"""
import re
import os

def fix_strategy_loading():
    """Fix strategy loading to use STRATEGIES registry properly"""

    print("Fixing trading_bot.py strategy loading...")

    # Read current file
    with open('core/trading_bot.py', 'r') as f:
        content = f.read()

    # Backup
    with open('core/trading_bot.py.backup', 'w') as f:
        f.write(content)

    # Ensure proper imports
    if 'from strategies import STRATEGIES' not in content:
        # Add after other strategy imports
        content = re.sub(
            r'(from strategies.*?\n)',
            r'\1from strategies import STRATEGIES\n',
            content,
            count=1
        )

    # Find the __init__ method
    init_match = re.search(
        r'(def __init__.*?)(\n    def)',
        content,
        re.DOTALL
    )

    if init_match:
        init_content = init_match.group(1)

        # Find strategy initialization section
        if 'strategy_name' in init_content:
            # Create proper strategy loading code
            new_strategy_loading = '''
        # Load strategy from registry
        self.strategy = None
        strategy_name_lower = strategy_name.lower()

        if strategy_name_lower in STRATEGIES:
            try:
                strategy_class = STRATEGIES[strategy_name_lower]
                strategy_params = self.config.get('strategy_params', {})
                self.strategy = strategy_class(strategy_params)
                logger.info(f"Successfully loaded strategy: {strategy_name} ({strategy_class.__name__})")
            except Exception as e:
                logger.error(f"Failed to instantiate strategy {strategy_name}: {e}")
                self._load_fallback_strategy()
        else:
            available_strategies = list(STRATEGIES.keys())
            logger.warning(f"Strategy '{strategy_name}' not found. Available strategies: {available_strategies}")
            self._load_fallback_strategy()
'''

            # Replace the current strategy loading
            # Look for the warning about unknown strategy
            pattern = r'logger\.warning.*?Unknown strategy.*?(?:self\.strategy = .*?\n)+'

            if re.search(pattern, init_content, re.DOTALL):
                new_init = re.sub(pattern, new_strategy_loading.strip() + '\n', init_content, flags=re.DOTALL)
            else:
                # If pattern not found, look for self.strategy assignment
                pattern2 = r'self\.strategy = .*?(?=\n\s*[^\\s]|\Z)'
                if re.search(pattern2, init_content, re.DOTALL):
                    new_init = re.sub(pattern2, new_strategy_loading.strip(), init_content, flags=re.DOTALL)
                else:
                    # Add after strategy_name is set
                    new_init = init_content + '\n' + new_strategy_loading

            # Replace in content
            content = content.replace(init_content, new_init)

    # Add fallback method if not exists
    if '_load_fallback_strategy' not in content:
        # Find a good place to add it (after __init__)
        class_match = re.search(r'(class TradingBot.*?)(def __init__.*?\n\n)', content, re.DOTALL)
        if class_match:
            # Add after __init__
            fallback_method = '''
    def _load_fallback_strategy(self):
        """Load fallback strategy when requested strategy is not available"""
        logger.info("Loading fallback strategy: momentum")
        try:
            from strategies.momentum import MomentumStrategy
            self.strategy = MomentumStrategy(self.config.get('strategy_params', {}))
        except Exception as e:
            logger.error(f"Failed to load fallback strategy: {e}")
            raise RuntimeError("No strategy could be loaded")
'''

            # Insert after __init__
            init_end = content.find('\n    def', content.find('def __init__') + 1)
            if init_end > 0:
                content = content[:init_end] + fallback_method + content[init_end:]

    # Save fixed file
    with open('core/trading_bot.py', 'w') as f:
        f.write(content)

    print("✅ trading_bot.py strategy loading fixed")
    return True

if __name__ == "__main__":
    fix_strategy_loading()
