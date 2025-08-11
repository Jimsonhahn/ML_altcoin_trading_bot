#!/usr/bin/env python3
"""
Enhanced Configuration System
============================

Advanced configuration management system for the trading bot:
- Environment-based configuration (dev/staging/production)
- Dynamic config updates and hot reloading
- Configuration validation and type checking
- Encrypted sensitive data storage
- Configuration templates and profiles
- Real-time configuration monitoring
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jsonschema
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import copy

logger = logging.getLogger(__name__)

@dataclass
class ConfigProfile:
    """Configuration profile definition"""
    name: str
    description: str
    risk_level: str  # 'conservative', 'moderate', 'aggressive', 'extreme'
    environment: str  # 'development', 'staging', 'production'
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    
    # Core trading parameters
    daily_budget: float = 30.0
    max_positions: int = 3
    position_timeout_hours: int = 6
    min_confidence_threshold: float = 0.7
    
    # Risk management
    max_drawdown_percent: float = 25.0
    stop_loss_percent: float = 15.0
    take_profit_percent: float = 50.0
    daily_loss_limit: float = 30.0
    
    # Signal sources and weights
    volume_spike_enabled: bool = True
    volume_spike_weight: float = 0.3
    social_sentiment_enabled: bool = True
    social_sentiment_weight: float = 0.25
    technical_analysis_enabled: bool = True
    technical_analysis_weight: float = 0.2
    ml_prediction_enabled: bool = True
    ml_prediction_weight: float = 0.15
    news_analysis_enabled: bool = True
    news_analysis_weight: float = 0.1
    
    # Exchange settings
    primary_exchange: str = 'binance'
    backup_exchanges: List[str] = field(default_factory=lambda: ['coinbase', 'kraken'])
    arbitrage_enabled: bool = False
    arbitrage_min_profit: float = 0.8
    
    # Monitoring and alerts
    telegram_alerts: bool = True
    email_alerts: bool = False
    discord_alerts: bool = False
    performance_logging: bool = True
    debug_mode: bool = False

@dataclass
class APICredentials:
    """API credentials with encryption support"""
    service: str
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    sandbox: bool = True
    encrypted: bool = False
    
    def encrypt(self, encryption_key: bytes):
        """Encrypt sensitive credentials"""
        if not self.encrypted:
            fernet = Fernet(encryption_key)
            if self.api_key:
                self.api_key = fernet.encrypt(self.api_key.encode()).decode()
            if self.api_secret:
                self.api_secret = fernet.encrypt(self.api_secret.encode()).decode()
            if self.passphrase:
                self.passphrase = fernet.encrypt(self.passphrase.encode()).decode()
            self.encrypted = True
    
    def decrypt(self, encryption_key: bytes):
        """Decrypt sensitive credentials"""
        if self.encrypted:
            fernet = Fernet(encryption_key)
            if self.api_key:
                self.api_key = fernet.decrypt(self.api_key.encode()).decode()
            if self.api_secret:
                self.api_secret = fernet.decrypt(self.api_secret.encode()).decode()
            if self.passphrase:
                self.passphrase = fernet.decrypt(self.passphrase.encode()).decode()
            self.encrypted = False

class ConfigValidationError(Exception):
    """Configuration validation error"""
    pass

class ConfigEncryption:
    """Configuration encryption utilities"""
    
    @staticmethod
    def generate_key_from_password(password: str, salt: bytes = None) -> bytes:
        """Generate encryption key from password"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    @staticmethod
    def generate_random_key() -> bytes:
        """Generate random encryption key"""
        return Fernet.generate_key()

class ConfigWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.last_modified = {}
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and event.src_path.endswith(('.json', '.yaml', '.yml')):
            # Debounce rapid changes
            current_time = datetime.now().timestamp()
            if (event.src_path not in self.last_modified or 
                current_time - self.last_modified[event.src_path] > 1.0):
                
                self.last_modified[event.src_path] = current_time
                
                logger.info(f"🔄 Configuration file changed: {event.src_path}")
                
                try:
                    self.config_manager.reload_configuration()
                    logger.info("✅ Configuration reloaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to reload configuration: {e}")

class EnhancedConfigManager:
    """
    Enhanced configuration management system
    
    Provides comprehensive configuration management with encryption,
    validation, hot reloading, and environment support
    """
    
    def __init__(self, config_dir: str = "config", encryption_password: str = None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize encryption
        self.encryption_key = None
        if encryption_password:
            salt_file = self.config_dir / "salt.key"
            if salt_file.exists():
                salt = salt_file.read_bytes()
            else:
                salt = os.urandom(16)
                salt_file.write_bytes(salt)
            
            self.encryption_key = ConfigEncryption.generate_key_from_password(
                encryption_password, salt
            )
        
        # Configuration storage
        self.current_profile: Optional[ConfigProfile] = None
        self.available_profiles: Dict[str, ConfigProfile] = {}
        self.api_credentials: Dict[str, APICredentials] = {}
        self.config_schema = self._load_config_schema()
        
        # File watching
        self.observer = None
        self.watching = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing configurations
        self._load_existing_configurations()
        
        logger.info("⚙️ Enhanced Configuration Manager initialized")
    
    def _load_config_schema(self) -> Dict[str, Any]:
        """Load configuration validation schema"""
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "risk_level": {
                    "type": "string",
                    "enum": ["conservative", "moderate", "aggressive", "extreme"]
                },
                "environment": {
                    "type": "string", 
                    "enum": ["development", "staging", "production"]
                },
                "daily_budget": {"type": "number", "minimum": 1.0, "maximum": 10000.0},
                "max_positions": {"type": "integer", "minimum": 1, "maximum": 10},
                "position_timeout_hours": {"type": "integer", "minimum": 1, "maximum": 24},
                "min_confidence_threshold": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                "max_drawdown_percent": {"type": "number", "minimum": 5.0, "maximum": 50.0},
                "stop_loss_percent": {"type": "number", "minimum": 5.0, "maximum": 30.0},
                "take_profit_percent": {"type": "number", "minimum": 10.0, "maximum": 200.0},
                "daily_loss_limit": {"type": "number", "minimum": 1.0, "maximum": 10000.0}
            },
            "required": ["name", "risk_level", "environment"],
            "additionalProperties": True
        }
        
        return schema
    
    def _load_existing_configurations(self):
        """Load existing configuration profiles"""
        
        # Load profiles
        profiles_dir = self.config_dir / "profiles"
        if profiles_dir.exists():
            for profile_file in profiles_dir.glob("*.json"):
                try:
                    profile = self.load_profile_from_file(profile_file)
                    self.available_profiles[profile.name] = profile
                    logger.info(f"📋 Loaded profile: {profile.name}")
                except Exception as e:
                    logger.error(f"❌ Failed to load profile {profile_file}: {e}")
        
        # Load API credentials
        creds_file = self.config_dir / "credentials.json"
        if creds_file.exists():
            try:
                self._load_api_credentials()
                logger.info("🔐 Loaded API credentials")
            except Exception as e:
                logger.error(f"❌ Failed to load credentials: {e}")
        
        # Set default profile if available
        if self.available_profiles:
            default_name = list(self.available_profiles.keys())[0]
            self.set_active_profile(default_name)
    
    def create_profile(self, profile: ConfigProfile) -> ConfigProfile:
        """Create new configuration profile"""
        
        with self._lock:
            # Validate profile
            self.validate_profile(profile)
            
            # Update timestamps
            profile.created_at = datetime.now()
            profile.last_modified = datetime.now()
            
            # Store profile
            self.available_profiles[profile.name] = profile
            
            # Save to file
            self._save_profile_to_file(profile)
            
            logger.info(f"✅ Created profile: {profile.name}")
            
            return profile
    
    def update_profile(self, profile_name: str, **kwargs) -> ConfigProfile:
        """Update existing configuration profile"""
        
        with self._lock:
            if profile_name not in self.available_profiles:
                raise ValueError(f"Profile '{profile_name}' not found")
            
            profile = copy.deepcopy(self.available_profiles[profile_name])
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
                else:
                    logger.warning(f"Unknown profile field: {key}")
            
            # Update timestamp
            profile.last_modified = datetime.now()
            
            # Validate updated profile
            self.validate_profile(profile)
            
            # Store updated profile
            self.available_profiles[profile_name] = profile
            
            # Save to file
            self._save_profile_to_file(profile)
            
            # Update current profile if it's the active one
            if self.current_profile and self.current_profile.name == profile_name:
                self.current_profile = profile
            
            logger.info(f"✅ Updated profile: {profile_name}")
            
            return profile
    
    def set_active_profile(self, profile_name: str):
        """Set active configuration profile"""
        
        with self._lock:
            if profile_name not in self.available_profiles:
                raise ValueError(f"Profile '{profile_name}' not found")
            
            self.current_profile = self.available_profiles[profile_name]
            
            # Save active profile reference
            active_file = self.config_dir / "active_profile.txt"
            active_file.write_text(profile_name)
            
            logger.info(f"🎯 Set active profile: {profile_name}")
    
    def get_active_profile(self) -> Optional[ConfigProfile]:
        """Get currently active configuration profile"""
        return self.current_profile
    
    def list_profiles(self) -> List[str]:
        """List available configuration profiles"""
        return list(self.available_profiles.keys())
    
    def get_profile(self, profile_name: str) -> Optional[ConfigProfile]:
        """Get specific configuration profile"""
        return self.available_profiles.get(profile_name)
    
    def delete_profile(self, profile_name: str):
        """Delete configuration profile"""
        
        with self._lock:
            if profile_name not in self.available_profiles:
                raise ValueError(f"Profile '{profile_name}' not found")
            
            # Don't delete if it's the active profile
            if self.current_profile and self.current_profile.name == profile_name:
                raise ValueError("Cannot delete active profile")
            
            # Remove from memory
            del self.available_profiles[profile_name]
            
            # Remove file
            profile_file = self.config_dir / "profiles" / f"{profile_name}.json"
            if profile_file.exists():
                profile_file.unlink()
            
            logger.info(f"🗑️ Deleted profile: {profile_name}")
    
    def validate_profile(self, profile: ConfigProfile):
        """Validate configuration profile"""
        
        try:
            # Convert to dictionary for schema validation
            profile_dict = asdict(profile)
            
            # Remove datetime fields for validation (they're not in schema)
            profile_dict.pop('created_at', None)
            profile_dict.pop('last_modified', None)
            
            # Validate against schema
            jsonschema.validate(profile_dict, self.config_schema)
            
            # Additional business logic validation
            if profile.daily_budget < profile.daily_loss_limit:
                raise ConfigValidationError(
                    "Daily budget cannot be less than daily loss limit"
                )
            
            if profile.stop_loss_percent >= profile.take_profit_percent:
                logger.warning(
                    "Stop loss percent is >= take profit percent - this may limit profitability"
                )
            
            # Validate weight sum for signal sources
            total_weight = (
                (profile.volume_spike_weight if profile.volume_spike_enabled else 0) +
                (profile.social_sentiment_weight if profile.social_sentiment_enabled else 0) +
                (profile.technical_analysis_weight if profile.technical_analysis_enabled else 0) +
                (profile.ml_prediction_weight if profile.ml_prediction_enabled else 0) +
                (profile.news_analysis_weight if profile.news_analysis_enabled else 0)
            )
            
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                logger.warning(f"Signal source weights sum to {total_weight:.3f}, not 1.0")
            
        except jsonschema.ValidationError as e:
            raise ConfigValidationError(f"Configuration validation failed: {e.message}")
    
    def add_api_credentials(self, credentials: APICredentials):
        """Add API credentials"""
        
        with self._lock:
            # Encrypt if encryption is enabled
            if self.encryption_key:
                creds_copy = copy.deepcopy(credentials)
                creds_copy.encrypt(self.encryption_key)
                self.api_credentials[credentials.service] = creds_copy
            else:
                self.api_credentials[credentials.service] = credentials
            
            # Save to file
            self._save_api_credentials()
            
            logger.info(f"🔐 Added API credentials for: {credentials.service}")
    
    def get_api_credentials(self, service: str) -> Optional[APICredentials]:
        """Get API credentials for service"""
        
        if service not in self.api_credentials:
            return None
        
        credentials = copy.deepcopy(self.api_credentials[service])
        
        # Decrypt if needed
        if self.encryption_key and credentials.encrypted:
            credentials.decrypt(self.encryption_key)
        
        return credentials
    
    def remove_api_credentials(self, service: str):
        """Remove API credentials"""
        
        with self._lock:
            if service in self.api_credentials:
                del self.api_credentials[service]
                self._save_api_credentials()
                logger.info(f"🗑️ Removed API credentials for: {service}")
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value from active profile"""
        
        if not self.current_profile:
            return default
        
        return getattr(self.current_profile, key, default)
    
    def update_config_value(self, key: str, value: Any):
        """Update configuration value in active profile"""
        
        if not self.current_profile:
            raise ValueError("No active profile set")
        
        self.update_profile(self.current_profile.name, **{key: value})
    
    def create_profile_template(self, name: str, risk_level: str, environment: str) -> ConfigProfile:
        """Create configuration profile from template"""
        
        # Define templates based on risk level
        templates = {
            'conservative': {
                'daily_budget': 10.0,
                'max_positions': 1,
                'position_timeout_hours': 2,
                'min_confidence_threshold': 0.8,
                'max_drawdown_percent': 10.0,
                'stop_loss_percent': 10.0,
                'take_profit_percent': 20.0,
                'volume_spike_weight': 0.4,
                'social_sentiment_weight': 0.3,
                'technical_analysis_weight': 0.3,
                'ml_prediction_enabled': False,
                'news_analysis_enabled': False,
                'arbitrage_enabled': False
            },
            'moderate': {
                'daily_budget': 30.0,
                'max_positions': 2,
                'position_timeout_hours': 4,
                'min_confidence_threshold': 0.7,
                'max_drawdown_percent': 20.0,
                'stop_loss_percent': 15.0,
                'take_profit_percent': 40.0,
                'volume_spike_weight': 0.3,
                'social_sentiment_weight': 0.25,
                'technical_analysis_weight': 0.25,
                'ml_prediction_weight': 0.15,
                'news_analysis_weight': 0.05,
                'arbitrage_enabled': False
            },
            'aggressive': {
                'daily_budget': 50.0,
                'max_positions': 3,
                'position_timeout_hours': 6,
                'min_confidence_threshold': 0.6,
                'max_drawdown_percent': 30.0,
                'stop_loss_percent': 20.0,
                'take_profit_percent': 75.0,
                'volume_spike_weight': 0.25,
                'social_sentiment_weight': 0.25,
                'technical_analysis_weight': 0.2,
                'ml_prediction_weight': 0.2,
                'news_analysis_weight': 0.1,
                'arbitrage_enabled': True,
                'arbitrage_min_profit': 1.0
            },
            'extreme': {
                'daily_budget': 100.0,
                'max_positions': 5,
                'position_timeout_hours': 8,
                'min_confidence_threshold': 0.5,
                'max_drawdown_percent': 40.0,
                'stop_loss_percent': 25.0,
                'take_profit_percent': 100.0,
                'volume_spike_weight': 0.2,
                'social_sentiment_weight': 0.2,
                'technical_analysis_weight': 0.2,
                'ml_prediction_weight': 0.25,
                'news_analysis_weight': 0.15,
                'arbitrage_enabled': True,
                'arbitrage_min_profit': 0.5
            }
        }
        
        if risk_level not in templates:
            raise ValueError(f"Unknown risk level: {risk_level}")
        
        template = templates[risk_level]
        
        # Create profile from template
        profile = ConfigProfile(
            name=name,
            description=f"{risk_level.title()} risk profile for {environment}",
            risk_level=risk_level,
            environment=environment,
            **template
        )
        
        return self.create_profile(profile)
    
    def export_profile(self, profile_name: str, file_path: str):
        """Export profile to file"""
        
        if profile_name not in self.available_profiles:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        profile = self.available_profiles[profile_name]
        profile_dict = asdict(profile)
        
        # Convert datetime objects to ISO format
        profile_dict['created_at'] = profile.created_at.isoformat()
        profile_dict['last_modified'] = profile.last_modified.isoformat()
        
        export_path = Path(file_path)
        
        if export_path.suffix.lower() == '.yaml':
            with open(export_path, 'w') as f:
                yaml.dump(profile_dict, f, default_flow_style=False, indent=2)
        else:
            with open(export_path, 'w') as f:
                json.dump(profile_dict, f, indent=2)
        
        logger.info(f"📤 Exported profile {profile_name} to {file_path}")
    
    def import_profile(self, file_path: str) -> ConfigProfile:
        """Import profile from file"""
        
        import_path = Path(file_path)
        
        if not import_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load data
        if import_path.suffix.lower() in ['.yaml', '.yml']:
            with open(import_path, 'r') as f:
                profile_dict = yaml.safe_load(f)
        else:
            with open(import_path, 'r') as f:
                profile_dict = json.load(f)
        
        # Convert datetime strings back to datetime objects
        if 'created_at' in profile_dict:
            profile_dict['created_at'] = datetime.fromisoformat(profile_dict['created_at'])
        if 'last_modified' in profile_dict:
            profile_dict['last_modified'] = datetime.fromisoformat(profile_dict['last_modified'])
        
        # Create profile object
        profile = ConfigProfile(**profile_dict)
        
        # Create and return
        return self.create_profile(profile)
    
    def start_watching(self):
        """Start watching configuration files for changes"""
        
        if self.watching:
            return
        
        self.observer = Observer()
        event_handler = ConfigWatcher(self)
        self.observer.schedule(event_handler, str(self.config_dir), recursive=True)
        self.observer.start()
        self.watching = True
        
        logger.info("👁️ Started configuration file watching")
    
    def stop_watching(self):
        """Stop watching configuration files"""
        
        if self.observer and self.watching:
            self.observer.stop()
            self.observer.join()
            self.watching = False
            
            logger.info("👁️ Stopped configuration file watching")
    
    def reload_configuration(self):
        """Reload configuration from files"""
        
        with self._lock:
            old_profiles = copy.deepcopy(self.available_profiles)
            
            # Clear and reload
            self.available_profiles.clear()
            self._load_existing_configurations()
            
            # Check for changes
            changes = []
            
            for name, profile in self.available_profiles.items():
                if name in old_profiles:
                    if profile.last_modified != old_profiles[name].last_modified:
                        changes.append(f"Updated: {name}")
                else:
                    changes.append(f"Added: {name}")
            
            for name in old_profiles:
                if name not in self.available_profiles:
                    changes.append(f"Removed: {name}")
            
            if changes:
                logger.info(f"🔄 Configuration changes: {', '.join(changes)}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        
        active_profile = self.get_active_profile()
        
        return {
            'active_profile': active_profile.name if active_profile else None,
            'available_profiles': len(self.available_profiles),
            'profile_list': list(self.available_profiles.keys()),
            'api_credentials': list(self.api_credentials.keys()),
            'encryption_enabled': self.encryption_key is not None,
            'watching_enabled': self.watching,
            'config_directory': str(self.config_dir)
        }
    
    def _save_profile_to_file(self, profile: ConfigProfile):
        """Save profile to JSON file"""
        
        profiles_dir = self.config_dir / "profiles"
        profiles_dir.mkdir(exist_ok=True)
        
        profile_file = profiles_dir / f"{profile.name}.json"
        profile_dict = asdict(profile)
        
        # Convert datetime objects to ISO format
        profile_dict['created_at'] = profile.created_at.isoformat()
        profile_dict['last_modified'] = profile.last_modified.isoformat()
        
        with open(profile_file, 'w') as f:
            json.dump(profile_dict, f, indent=2)
    
    def load_profile_from_file(self, file_path: Path) -> ConfigProfile:
        """Load profile from JSON file"""
        
        with open(file_path, 'r') as f:
            profile_dict = json.load(f)
        
        # Convert ISO format strings back to datetime objects
        if 'created_at' in profile_dict:
            profile_dict['created_at'] = datetime.fromisoformat(profile_dict['created_at'])
        if 'last_modified' in profile_dict:
            profile_dict['last_modified'] = datetime.fromisoformat(profile_dict['last_modified'])
        
        return ConfigProfile(**profile_dict)
    
    def _save_api_credentials(self):
        """Save API credentials to file"""
        
        creds_file = self.config_dir / "credentials.json"
        creds_data = {}
        
        for service, credentials in self.api_credentials.items():
            creds_data[service] = asdict(credentials)
        
        with open(creds_file, 'w') as f:
            json.dump(creds_data, f, indent=2)
    
    def _load_api_credentials(self):
        """Load API credentials from file"""
        
        creds_file = self.config_dir / "credentials.json"
        
        if not creds_file.exists():
            return
        
        with open(creds_file, 'r') as f:
            creds_data = json.load(f)
        
        for service, cred_dict in creds_data.items():
            credentials = APICredentials(**cred_dict)
            self.api_credentials[service] = credentials
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_watching()

# Factory function
def create_config_manager(config_dir: str = "config", 
                         encryption_password: str = None) -> EnhancedConfigManager:
    """Create enhanced configuration manager"""
    return EnhancedConfigManager(config_dir, encryption_password)

# Test function
def test_config_system():
    """Test enhanced configuration system"""
    
    print("⚙️ Testing Enhanced Configuration System...")
    
    try:
        # Create config manager
        with create_config_manager("test_config") as config_manager:
            
            # Test profile creation
            print("\n📋 Testing profile creation...")
            
            test_profile = config_manager.create_profile_template(
                name="test_aggressive",
                risk_level="aggressive", 
                environment="development"
            )
            
            print(f"✅ Created profile: {test_profile.name}")
            print(f"   Risk Level: {test_profile.risk_level}")
            print(f"   Daily Budget: {test_profile.daily_budget}€")
            print(f"   Max Positions: {test_profile.max_positions}")
            
            # Test profile updates
            print("\n🔄 Testing profile updates...")
            
            updated_profile = config_manager.update_profile(
                "test_aggressive",
                daily_budget=75.0,
                max_positions=4
            )
            
            print(f"✅ Updated profile: {updated_profile.name}")
            print(f"   New Daily Budget: {updated_profile.daily_budget}€")
            print(f"   New Max Positions: {updated_profile.max_positions}")
            
            # Test API credentials
            print("\n🔐 Testing API credentials...")
            
            test_credentials = APICredentials(
                service="binance",
                api_key="test_key_123",
                api_secret="test_secret_456",
                sandbox=True
            )
            
            config_manager.add_api_credentials(test_credentials)
            
            retrieved_creds = config_manager.get_api_credentials("binance")
            print(f"✅ Added and retrieved credentials for: {retrieved_creds.service}")
            print(f"   API Key: {retrieved_creds.api_key[:10]}...")
            print(f"   Sandbox: {retrieved_creds.sandbox}")
            
            # Test profile validation
            print("\n✅ Testing profile validation...")
            
            try:
                invalid_profile = ConfigProfile(
                    name="invalid",
                    description="Invalid profile",
                    risk_level="invalid_risk",  # Invalid risk level
                    environment="development"
                )
                config_manager.validate_profile(invalid_profile)
                print("❌ Validation should have failed!")
            except ConfigValidationError:
                print("✅ Validation correctly rejected invalid profile")
            
            # Test export/import
            print("\n📤📥 Testing export/import...")
            
            export_file = "test_profile_export.json"
            config_manager.export_profile("test_aggressive", export_file)
            print(f"✅ Exported profile to: {export_file}")
            
            # Clean up test profile and re-import
            config_manager.delete_profile("test_aggressive")
            imported_profile = config_manager.import_profile(export_file)
            print(f"✅ Imported profile: {imported_profile.name}")
            
            # Test configuration summary
            print("\n📊 Configuration Summary:")
            summary = config_manager.get_config_summary()
            
            for key, value in summary.items():
                print(f"   {key}: {value}")
            
            # Clean up
            import shutil
            shutil.rmtree("test_config", ignore_errors=True)
            Path(export_file).unlink(missing_ok=True)
            
        print("\n🎉 All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run test
    test_config_system()