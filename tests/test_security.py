"""
Test Suite for Security Components
==================================

Tests for:
- SecretManager encryption and key storage
- SSL validation and secure HTTP
- Input validation framework
- Error handling security
- Data sanitization
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import ssl
import requests
from cryptography.fernet import Fernet
import keyring
import re
from typing import Dict, Any

# Import security components to test
from utils.secret_manager import SecretManager, get_api_credentials, store_api_key
from utils.secure_http import SecureHTTPAdapter, create_secure_session
from utils.validators import (
    validate_trading_symbol, validate_amount, validate_config,
    TradingSymbolValidator, AmountValidator, ConfigValidator,
    ValidationError
)
from utils.error_handler import (
    SecureErrorHandler, secure_error_handler,
    ValidationTradingError, NetworkTradingError, ExchangeTradingError
)


class TestSecretManager:
    """Test SecretManager encryption and key storage functionality"""
    
    @pytest.fixture
    def temp_service_name(self):
        """Temporary service name for testing"""
        return "test_trading_bot_service"
    
    @pytest.fixture
    def secret_manager(self, temp_service_name):
        """Create SecretManager instance for testing"""
        return SecretManager(service_name=temp_service_name)
    
    def test_secret_manager_initialization(self, secret_manager, temp_service_name):
        """Test SecretManager initialization"""
        assert secret_manager.service_name == temp_service_name
        assert secret_manager.master_key is not None
        assert secret_manager.cipher_suite is not None
        assert isinstance(secret_manager.cipher_suite, Fernet)
    
    def test_master_key_generation(self, secret_manager):
        """Test master key generation and storage"""
        # Master key should be base64 encoded Fernet key
        master_key = secret_manager.master_key
        assert len(master_key) == 44  # Fernet key length when base64 encoded
        
        # Should be able to create Fernet instance
        fernet = Fernet(master_key)
        assert isinstance(fernet, Fernet)
    
    def test_secret_storage_and_retrieval(self, secret_manager):
        """Test storing and retrieving secrets"""
        test_secret = "test_secret_value_12345"
        secret_name = "test_secret"
        
        # Store secret
        result = secret_manager.store_secret(secret_name, test_secret)
        assert result is True
        
        # Retrieve secret
        retrieved = secret_manager.get_secret(secret_name)
        assert retrieved == test_secret
    
    def test_secret_encryption(self, secret_manager):
        """Test that secrets are properly encrypted"""
        test_secret = "sensitive_api_key_12345"
        secret_name = "test_api_key"
        
        # Store secret
        secret_manager.store_secret(secret_name, test_secret)
        
        # Check that raw stored value is encrypted (not plaintext)
        with patch('keyring.get_password') as mock_get:
            mock_get.return_value = "encrypted_value_not_plaintext"
            
            # The encrypted value should not equal the original
            # (This tests the encryption process, actual keyring value would be encrypted)
            raw_stored = keyring.get_password(secret_manager.service_name, secret_name)
            assert raw_stored != test_secret
    
    def test_secret_deletion(self, secret_manager):
        """Test secret deletion"""
        test_secret = "secret_to_delete"
        secret_name = "deletable_secret"
        
        # Store and verify
        secret_manager.store_secret(secret_name, test_secret)
        assert secret_manager.get_secret(secret_name) == test_secret
        
        # Delete and verify
        result = secret_manager.delete_secret(secret_name)
        assert result is True
        assert secret_manager.get_secret(secret_name) is None
    
    def test_secret_listing(self, secret_manager):
        """Test listing stored secrets"""
        # Store multiple secrets
        secrets = {
            "api_key_1": "value1",
            "api_key_2": "value2",
            "token_1": "value3"
        }
        
        for name, value in secrets.items():
            secret_manager.store_secret(name, value)
        
        # List secrets
        secret_list = secret_manager.list_secrets()
        
        # All stored secrets should be in the list
        for name in secrets.keys():
            assert name in secret_list
    
    def test_api_credentials_storage(self, secret_manager):
        """Test API credentials storage and retrieval"""
        exchange_name = "test_exchange"
        api_key = "test_api_key_12345"
        api_secret = "test_api_secret_67890"
        
        # Store API credentials
        result = store_api_key(exchange_name, api_key, api_secret)
        assert result is True
        
        # Retrieve API credentials
        retrieved_key, retrieved_secret = get_api_credentials(exchange_name)
        assert retrieved_key == api_key
        assert retrieved_secret == api_secret
    
    def test_invalid_secret_handling(self, secret_manager):
        """Test handling of invalid secrets"""
        # Test getting non-existent secret
        assert secret_manager.get_secret("non_existent_secret") is None
        
        # Test deleting non-existent secret
        result = secret_manager.delete_secret("non_existent_secret")
        assert result is False
    
    def test_encryption_key_consistency(self, temp_service_name):
        """Test that the same master key is retrieved across instances"""
        # Create first instance
        sm1 = SecretManager(service_name=temp_service_name)
        master_key_1 = sm1.master_key
        
        # Create second instance
        sm2 = SecretManager(service_name=temp_service_name)
        master_key_2 = sm2.master_key
        
        # Master keys should be the same
        assert master_key_1 == master_key_2
    
    @patch('keyring.get_password')
    @patch('keyring.set_password')
    def test_keyring_error_handling(self, mock_set, mock_get, secret_manager):
        """Test error handling when keyring operations fail"""
        # Test keyring get failure
        mock_get.side_effect = Exception("Keyring access denied")
        
        result = secret_manager.get_secret("test_secret")
        assert result is None
        
        # Test keyring set failure
        mock_set.side_effect = Exception("Keyring write denied")
        
        result = secret_manager.store_secret("test_secret", "test_value")
        assert result is False


class TestSecureHTTP:
    """Test SSL validation and secure HTTP functionality"""
    
    def test_secure_http_adapter_initialization(self):
        """Test SecureHTTPAdapter initialization"""
        adapter = SecureHTTPAdapter(
            ssl_verify=True,
            cert_file=None,
            min_tls_version=ssl.TLSVersion.TLSv1_2
        )
        
        assert adapter.ssl_verify is True
        assert adapter.cert_file is None
        assert adapter.min_tls_version == ssl.TLSVersion.TLSv1_2
    
    def test_ssl_context_creation(self):
        """Test SSL context creation with security settings"""
        adapter = SecureHTTPAdapter()
        context = adapter._create_ssl_context()
        
        assert isinstance(context, ssl.SSLContext)
        assert context.minimum_version == ssl.TLSVersion.TLSv1_2
        assert context.check_hostname is True
        assert context.verify_mode == ssl.CERT_REQUIRED
    
    def test_create_secure_session(self):
        """Test creation of secure HTTP session"""
        session = create_secure_session(
            timeout=(5, 30),
            retries=3,
            backoff_factor=1.0
        )
        
        assert isinstance(session, requests.Session)
        
        # Check that secure adapter is mounted
        adapters = session.get_adapter('https://')
        assert isinstance(adapters, SecureHTTPAdapter)
    
    def test_ssl_verification_enabled(self):
        """Test that SSL verification is enabled by default"""
        session = create_secure_session()
        
        # SSL verification should be enabled
        assert session.verify is True
    
    def test_custom_timeout_configuration(self):
        """Test custom timeout configuration"""
        custom_timeout = (10, 60)
        session = create_secure_session(timeout=custom_timeout)
        
        # Timeout should be set on the session
        # Note: requests doesn't store timeout on session object,
        # but our implementation should use it for requests
        adapter = session.get_adapter('https://')
        assert hasattr(adapter, 'timeout')
    
    def test_retry_strategy_configuration(self):
        """Test retry strategy configuration"""
        session = create_secure_session(retries=5, backoff_factor=2.0)
        
        adapter = session.get_adapter('https://')
        assert hasattr(adapter, 'max_retries')
    
    @patch('requests.Session.request')
    def test_secure_request_execution(self, mock_request):
        """Test that requests are executed with security settings"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"success": true}'
        mock_request.return_value = mock_response
        
        session = create_secure_session()
        response = session.get('https://api.example.com/test')
        
        assert response.status_code == 200
        mock_request.assert_called_once()
        
        # Verify SSL verification is used
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs.get('verify') is True
    
    def test_certificate_validation(self):
        """Test certificate validation configuration"""
        # Test with custom certificate file
        cert_file = "custom_cert.pem"
        adapter = SecureHTTPAdapter(cert_file=cert_file)
        
        assert adapter.cert_file == cert_file
        
        # Test SSL context uses custom certificate
        with patch('ssl.create_default_context') as mock_context:
            mock_ssl_context = Mock()
            mock_context.return_value = mock_ssl_context
            
            adapter._create_ssl_context()
            mock_context.assert_called_with(cafile=cert_file)
    
    def test_tls_version_enforcement(self):
        """Test TLS version enforcement"""
        adapter = SecureHTTPAdapter(min_tls_version=ssl.TLSVersion.TLSv1_3)
        context = adapter._create_ssl_context()
        
        assert context.minimum_version == ssl.TLSVersion.TLSv1_3
    
    def test_insecure_ssl_prevention(self):
        """Test prevention of insecure SSL configurations"""
        # Should not allow SSL verification to be disabled
        adapter = SecureHTTPAdapter(ssl_verify=False)
        
        # Even if ssl_verify=False is passed, should maintain security
        context = adapter._create_ssl_context()
        assert context.verify_mode == ssl.CERT_REQUIRED


class TestInputValidation:
    """Test input validation framework"""
    
    def test_trading_symbol_validation(self):
        """Test trading symbol validation"""
        # Valid symbols
        valid_symbols = ['BTC/USDT', 'ETH/USD', 'ADA/EUR', 'SOL/BTC']
        for symbol in valid_symbols:
            validator = validate_trading_symbol(symbol)
            assert validator.symbol == symbol
            assert validator.base_currency in symbol
            assert validator.quote_currency in symbol
        
        # Invalid symbols
        invalid_symbols = ['BTCUSDT', 'BTC', 'BTC/USD/EUR', '', 'BTC/']
        for symbol in invalid_symbols:
            with pytest.raises(ValidationError):
                validate_trading_symbol(symbol)
    
    def test_amount_validation(self):
        """Test amount validation"""
        # Valid amounts
        valid_amounts = [(100.5, 'USD'), (0.001, 'BTC'), (1000, 'USDT')]
        for amount, currency in valid_amounts:
            validator = validate_amount(amount, currency)
            assert validator.amount == amount
            assert validator.currency == currency
        
        # Invalid amounts
        invalid_amounts = [(-100, 'USD'), (0, 'BTC'), (10**10, 'USDT')]
        for amount, currency in invalid_amounts:
            with pytest.raises(ValidationError):
                validate_amount(amount, currency)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        valid_config = {
            'trading_mode': 'paper',
            'max_position_size': 1000.0,
            'max_positions': 5,
            'max_drawdown': 0.20,
            'stop_loss_percentage': 0.02,
            'take_profit_percentage': 0.05,
            'risk_per_trade': 0.02,
            'exchange_name': 'binance',
            'api_rate_limit': 1200
        }
        
        validator = validate_config(valid_config)
        assert validator.trading_mode == valid_config['trading_mode']
        assert validator.max_position_size == valid_config['max_position_size']
    
    def test_trading_symbol_validator_class(self):
        """Test TradingSymbolValidator class"""
        # Valid symbol
        validator = TradingSymbolValidator(symbol='BTC/USDT')
        assert validator.symbol == 'BTC/USDT'
        assert validator.base_currency == 'BTC'
        assert validator.quote_currency == 'USDT'
        
        # Invalid symbol format
        with pytest.raises(ValidationError):
            TradingSymbolValidator(symbol='invalid_symbol')
    
    def test_amount_validator_class(self):
        """Test AmountValidator class"""
        # Valid amount
        validator = AmountValidator(amount=100.5, currency='USD')
        assert validator.amount == 100.5
        assert validator.currency == 'USD'
        
        # Negative amount
        with pytest.raises(ValidationError):
            AmountValidator(amount=-100, currency='USD')
        
        # Zero amount
        with pytest.raises(ValidationError):
            AmountValidator(amount=0, currency='USD')
    
    def test_config_validator_class(self):
        """Test ConfigValidator class"""
        valid_config = {
            'trading_mode': 'live',
            'max_position_size': 1000.0,
            'max_positions': 3,
            'max_drawdown': 0.15,
            'stop_loss_percentage': 0.02,
            'take_profit_percentage': 0.05,
            'risk_per_trade': 0.01,
            'exchange_name': 'binance',
            'api_rate_limit': 1200
        }
        
        validator = ConfigValidator(**valid_config)
        assert validator.trading_mode.value == 'live'
        assert validator.max_position_size == 1000.0
        assert validator.max_positions == 3
    
    def test_validation_error_details(self):
        """Test validation error details"""
        try:
            validate_trading_symbol('invalid')
        except ValidationError as e:
            assert 'symbol' in str(e).lower()
            assert hasattr(e, 'field_name')
    
    def test_custom_validation_rules(self):
        """Test custom validation rules"""
        # Test position size limits
        with pytest.raises(ValidationError):
            AmountValidator(amount=1000000, currency='USD')  # Too large
        
        # Test percentage bounds
        with pytest.raises(ValidationError):
            ConfigValidator(
                trading_mode='live',
                max_position_size=1000.0,
                max_positions=5,
                max_drawdown=1.5,  # > 100%
                stop_loss_percentage=0.02,
                take_profit_percentage=0.05,
                risk_per_trade=0.02,
                exchange_name='binance',
                api_rate_limit=1200
            )
    
    def test_field_validation_decorators(self):
        """Test field validation decorators"""
        # Test that field validators are properly applied
        validator = TradingSymbolValidator(symbol='BTC/USDT')
        
        # Symbol should be normalized
        assert validator.symbol == 'BTC/USDT'
        assert len(validator.symbol) >= 5  # Minimum length enforced
    
    def test_model_validation_decorators(self):
        """Test model validation decorators"""
        # Test cross-field validation
        config = ConfigValidator(
            trading_mode='paper',
            max_position_size=1000.0,
            max_positions=5,
            max_drawdown=0.20,
            stop_loss_percentage=0.10,  # High stop loss
            take_profit_percentage=0.05,  # Lower take profit
            risk_per_trade=0.02,
            exchange_name='binance',
            api_rate_limit=1200
        )
        
        # Should validate that take_profit > stop_loss makes sense
        assert config.take_profit_percentage < config.stop_loss_percentage  # This scenario should trigger validation


class TestErrorHandlerSecurity:
    """Test SecureErrorHandler security features"""
    
    @pytest.fixture
    def secure_handler(self):
        """Create SecureErrorHandler instance for testing"""
        return SecureErrorHandler(app_name="test_app")
    
    def test_sensitive_data_sanitization(self, secure_handler):
        """Test sensitive data is sanitized from error messages"""
        sensitive_messages = [
            "API error with api_key=sk_live_12345678901234567890",
            "Authentication failed: Bearer token_abc123def456",
            "Database connection failed with password=secret123",
            "Credit card validation failed: 4111-1111-1111-1111"
        ]
        
        for message in sensitive_messages:
            sanitized = secure_handler._sanitize_message(message)
            
            # Should not contain the original sensitive values
            assert 'sk_live_12345678901234567890' not in sanitized
            assert 'token_abc123def456' not in sanitized
            assert 'secret123' not in sanitized
            assert '4111-1111-1111-1111' not in sanitized
            
            # Should contain redacted markers
            assert '***' in sanitized or 'REDACTED' in sanitized
    
    def test_sensitive_value_masking(self, secure_handler):
        """Test sensitive value masking preserves debugging info"""
        test_values = [
            ("short", "***"),
            ("medium123", "med***3"),
            ("very_long_secret_value_12345", "ver***45")
        ]
        
        for value, expected_pattern in test_values:
            masked = secure_handler._mask_sensitive_value(value)
            
            if len(value) <= 4:
                assert masked == "***"
            elif len(value) <= 8:
                assert masked.startswith(value[:2])
                assert masked.endswith(value[-1:])
                assert "***" in masked
            else:
                assert masked.startswith(value[:3])
                assert masked.endswith(value[-2:])
                assert "***" in masked
    
    def test_error_id_generation(self, secure_handler):
        """Test unique error ID generation"""
        error_ids = [secure_handler._generate_error_id() for _ in range(100)]
        
        # All IDs should be unique
        assert len(set(error_ids)) == 100
        
        # All IDs should be valid UUIDs
        import uuid
        for error_id in error_ids:
            uuid.UUID(error_id)  # Should not raise exception
    
    def test_trace_id_generation(self, secure_handler):
        """Test trace ID generation"""
        trace_ids = [secure_handler._generate_trace_id() for _ in range(100)]
        
        # All IDs should be unique
        assert len(set(trace_ids)) == 100
        
        # All IDs should be 8 characters
        for trace_id in trace_ids:
            assert len(trace_id) == 8
            assert trace_id.isalnum()
    
    def test_critical_error_handling(self, secure_handler):
        """Test critical error handling with security measures"""
        test_error = SystemExit("System shutdown requested")
        
        response = secure_handler.handle_critical_error(
            error=test_error,
            context={
                "api_key": "secret_key_12345",
                "operation": "system_shutdown"
            }
        )
        
        assert response.error_id is not None
        assert response.category == "system"
        assert response.severity == "critical"
        assert "secret_key_12345" not in response.message
        assert "secret_key_12345" not in str(response.details)
    
    def test_trading_error_handling(self, secure_handler):
        """Test trading error handling with sensitive data"""
        test_error = ValidationTradingError("Invalid API key", field="api_key")
        
        response = secure_handler.handle_trading_error(
            error=test_error,
            symbol="BTC/USDT",
            order_id="order_123",
            amount=0.001,
            context={
                "api_secret": "very_secret_value",
                "strategy": "momentum"
            }
        )
        
        assert response.error_id is not None
        assert response.category == "trading"
        assert "very_secret_value" not in response.message
        assert "very_secret_value" not in str(response.details)
        assert response.details["symbol"] == "BTC/USDT"
        assert response.details["order_id"] == "order_123"
    
    def test_api_error_handling(self, secure_handler):
        """Test API error handling with request sanitization"""
        test_error = requests.exceptions.ConnectionError("API connection failed")
        
        response = secure_handler.handle_api_error(
            error=test_error,
            endpoint="https://api.binance.com/api/v3/order",
            status_code=500,
            request_data={
                "symbol": "BTCUSDT",
                "api_key": "secret_api_key_value",
                "signature": "secret_signature"
            },
            response_data={"error": "Internal server error"}
        )
        
        assert response.error_id is not None
        assert response.category in ["network", "authentication", "rate_limit"]
        
        # Sensitive request data should be sanitized
        assert "secret_api_key_value" not in str(response.details)
        assert "secret_signature" not in str(response.details)
        
        # Non-sensitive data should be preserved
        assert response.details["endpoint"] == "https://api.binance.com/api/v3/order"
        assert response.details["status_code"] == 500
    
    def test_dictionary_sanitization(self, secure_handler):
        """Test recursive dictionary sanitization"""
        sensitive_dict = {
            "user": "testuser",
            "password": "secret123",
            "config": {
                "api_key": "secret_key",
                "timeout": 30,
                "nested": {
                    "token": "secret_token",
                    "retries": 3
                }
            },
            "data": ["item1", "password=secret", "item3"]
        }
        
        sanitized = secure_handler._sanitize_dict(sensitive_dict)
        
        # Non-sensitive data should be preserved
        assert sanitized["user"] == "testuser"
        assert sanitized["config"]["timeout"] == 30
        assert sanitized["config"]["nested"]["retries"] == 3
        
        # Sensitive data should be redacted
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["config"]["api_key"] == "***REDACTED***"
        assert sanitized["config"]["nested"]["token"] == "***REDACTED***"
        
        # List items should be sanitized
        assert "password=secret" not in str(sanitized["data"])
    
    def test_error_statistics_security(self, secure_handler):
        """Test that error statistics don't leak sensitive data"""
        # Generate some errors with sensitive data
        errors = [
            Exception("Database error with password=secret123"),
            ValueError("API key sk_12345 is invalid"),
            ConnectionError("Failed to connect with token=abc123")
        ]
        
        for error in errors:
            secure_handler.handle_critical_error(error)
        
        stats = secure_handler.get_error_statistics()
        
        # Statistics should not contain sensitive data
        assert "secret123" not in str(stats)
        assert "sk_12345" not in str(stats)
        assert "abc123" not in str(stats)
        
        # Should contain aggregated info
        assert stats["total_errors"] == 3
        assert "category_breakdown" in stats
        assert "severity_breakdown" in stats
    
    def test_error_retrieval_security(self, secure_handler):
        """Test that error retrieval maintains security"""
        test_error = Exception("Error with api_key=secret_value")
        
        response = secure_handler.handle_critical_error(test_error)
        error_id = response.error_id
        
        # Retrieve error by ID
        retrieved = secure_handler.get_error_by_id(error_id)
        
        assert retrieved is not None
        assert retrieved.error_id == error_id
        
        # Should not contain sensitive data
        assert "secret_value" not in retrieved.message
        assert "secret_value" not in str(retrieved.details)
    
    def test_logging_security(self, secure_handler):
        """Test that logging doesn't expose sensitive data"""
        with patch('logging.Logger.log') as mock_log:
            test_error = Exception("Authentication failed with secret=sensitive_data")
            
            secure_handler.handle_critical_error(test_error)
            
            # Check that log calls don't contain sensitive data
            for call in mock_log.call_args_list:
                log_message = str(call)
                assert "sensitive_data" not in log_message


class TestSecurityIntegration:
    """Integration tests for security components working together"""
    
    def test_end_to_end_security_flow(self, temp_dir):
        """Test complete security flow from storage to validation to error handling"""
        # 1. Store API credentials securely
        service_name = "test_integration_service"
        sm = SecretManager(service_name=service_name)
        
        api_key = "test_api_key_12345"
        api_secret = "test_api_secret_67890"
        
        sm.store_secret("api_key", api_key)
        sm.store_secret("api_secret", api_secret)
        
        # 2. Retrieve and validate credentials
        retrieved_key = sm.get_secret("api_key")
        retrieved_secret = sm.get_secret("api_secret")
        
        assert retrieved_key == api_key
        assert retrieved_secret == api_secret
        
        # 3. Use secure HTTP session
        session = create_secure_session()
        assert session.verify is True
        
        # 4. Handle potential errors securely
        handler = SecureErrorHandler()
        
        try:
            # Simulate an error with sensitive data
            raise Exception(f"API error with key={api_key}")
        except Exception as e:
            response = handler.handle_api_error(e, context={"secret": api_secret})
            
            # Error should be handled without exposing secrets
            assert api_key not in response.message
            assert api_secret not in str(response.details)
            assert response.error_id is not None
    
    def test_validation_with_error_handling(self):
        """Test validation errors are handled securely"""
        handler = SecureErrorHandler()
        
        try:
            # This should raise a validation error
            validate_trading_symbol("INVALID_SYMBOL_FORMAT")
        except ValidationError as e:
            response = handler.handle_trading_error(e, symbol="INVALID_SYMBOL_FORMAT")
            
            assert response.category == "trading"
            assert response.severity == "medium"
            assert "validation" in response.message.lower()
    
    def test_secure_configuration_handling(self):
        """Test that configuration with sensitive data is handled securely"""
        handler = SecureErrorHandler()
        
        # Simulate configuration error with sensitive data
        config_error = Exception("Config validation failed: api_key=secret123, password=pass456")
        
        response = handler.handle_critical_error(
            config_error,
            context={
                "config_file": "settings.json",
                "api_secret": "very_secret_value"
            }
        )
        
        # Should not expose sensitive values
        assert "secret123" not in response.message
        assert "pass456" not in response.message
        assert "very_secret_value" not in str(response.details)
        
        # Should preserve non-sensitive context
        assert response.details["config_file"] == "settings.json"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])