"""
Secure Secret Manager for Trading Bot
Provides encrypted storage and retrieval of sensitive data like API keys
"""

import logging
import os
from typing import Optional
from cryptography.fernet import Fernet
import keyring


class SecretManager:
    """
    Secure secret management class that encrypts and stores sensitive data
    using cryptography.fernet for encryption and keyring for secure storage
    """
    
    def __init__(self, service_name: str = "altcoin_trading_bot"):
        """
        Initialize the SecretManager
        
        Args:
            service_name: Name of the service for keyring storage
        """
        self.service_name = service_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption
        self.master_key = self._get_or_create_master_key()
        self.cipher_suite = Fernet(self.master_key)
        
        self.logger.info("SecretManager initialized successfully")
    
    def _get_or_create_master_key(self) -> bytes:
        """
        Retrieve existing master key or create a new one
        
        Returns:
            bytes: Master key for encryption
        """
        try:
            # Try to retrieve existing master key
            stored_key = keyring.get_password(self.service_name, "master_key")
            
            if stored_key:
                self.logger.debug("Retrieved existing master key")
                return stored_key.encode()
            
            # Generate new master key if none exists
            new_key = Fernet.generate_key()
            keyring.set_password(self.service_name, "master_key", new_key.decode())
            
            self.logger.info("Generated and stored new master key")
            return new_key
            
        except Exception as e:
            self.logger.error(f"Error managing master key: {e}")
            raise RuntimeError(f"Failed to initialize master key: {e}")
    
    def store_secret(self, key_name: str, secret_value: str) -> bool:
        """
        Encrypt and store a secret value
        
        Args:
            key_name: Name/identifier for the secret
            secret_value: The secret value to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not key_name or not secret_value:
                raise ValueError("Key name and secret value cannot be empty")
            
            # Encrypt the secret
            encrypted_secret = self.cipher_suite.encrypt(secret_value.encode())
            
            # Store encrypted secret in keyring
            keyring.set_password(
                self.service_name, 
                f"secret_{key_name}", 
                encrypted_secret.decode()
            )
            
            self.logger.info(f"Successfully stored secret: {key_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing secret {key_name}: {e}")
            return False
    
    def get_secret(self, key_name: str) -> Optional[str]:
        """
        Retrieve and decrypt a secret value
        
        Args:
            key_name: Name/identifier for the secret
            
        Returns:
            Optional[str]: Decrypted secret value or None if not found
        """
        try:
            if not key_name:
                raise ValueError("Key name cannot be empty")
            
            # Retrieve encrypted secret from keyring
            encrypted_secret = keyring.get_password(
                self.service_name, 
                f"secret_{key_name}"
            )
            
            if not encrypted_secret:
                self.logger.warning(f"Secret not found: {key_name}")
                return None
            
            # Decrypt the secret
            decrypted_secret = self.cipher_suite.decrypt(encrypted_secret.encode())
            
            self.logger.debug(f"Successfully retrieved secret: {key_name}")
            return decrypted_secret.decode()
            
        except Exception as e:
            self.logger.error(f"Error retrieving secret {key_name}: {e}")
            return None
    
    def delete_secret(self, key_name: str) -> bool:
        """
        Delete a stored secret
        
        Args:
            key_name: Name/identifier for the secret to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not key_name:
                raise ValueError("Key name cannot be empty")
            
            # Delete from keyring
            keyring.delete_password(self.service_name, f"secret_{key_name}")
            
            self.logger.info(f"Successfully deleted secret: {key_name}")
            return True
            
        except keyring.errors.PasswordDeleteError:
            self.logger.warning(f"Secret not found for deletion: {key_name}")
            return False
        except Exception as e:
            self.logger.error(f"Error deleting secret {key_name}: {e}")
            return False
    
    def list_secrets(self) -> list:
        """
        List all stored secret names (for debugging/management)
        Note: This is a simplified implementation as keyring doesn't provide
        a native way to list all keys
        
        Returns:
            list: List of secret names
        """
        # This is a limitation of keyring - it doesn't provide a list function
        # In a production environment, you might want to maintain a separate
        # index of stored secrets
        self.logger.warning("list_secrets() is not fully implemented due to keyring limitations")
        return []
    
    def update_secret(self, key_name: str, new_value: str) -> bool:
        """
        Update an existing secret with a new value
        
        Args:
            key_name: Name/identifier for the secret
            new_value: New secret value
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if secret exists
            if self.get_secret(key_name) is None:
                self.logger.warning(f"Secret does not exist for update: {key_name}")
                return False
            
            # Store new value (overwrites existing)
            return self.store_secret(key_name, new_value)
            
        except Exception as e:
            self.logger.error(f"Error updating secret {key_name}: {e}")
            return False
    
    def secret_exists(self, key_name: str) -> bool:
        """
        Check if a secret exists
        
        Args:
            key_name: Name/identifier for the secret
            
        Returns:
            bool: True if secret exists, False otherwise
        """
        return self.get_secret(key_name) is not None
    
    def reset_master_key(self) -> bool:
        """
        Reset the master key (WARNING: This will make all stored secrets unrecoverable)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete existing master key
            keyring.delete_password(self.service_name, "master_key")
            
            # Generate new master key
            self.master_key = self._get_or_create_master_key()
            self.cipher_suite = Fernet(self.master_key)
            
            self.logger.warning("Master key has been reset - all previous secrets are now unrecoverable")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting master key: {e}")
            return False


# Convenience functions for easy usage
def store_api_key(exchange_name: str, api_key: str, api_secret: str) -> bool:
    """
    Convenience function to store exchange API credentials
    
    Args:
        exchange_name: Name of the exchange (e.g., 'binance')
        api_key: API key
        api_secret: API secret
        
    Returns:
        bool: True if successful, False otherwise
    """
    sm = SecretManager()
    
    key_success = sm.store_secret(f"{exchange_name}_api_key", api_key)
    secret_success = sm.store_secret(f"{exchange_name}_api_secret", api_secret)
    
    return key_success and secret_success


def get_api_credentials(exchange_name: str) -> tuple:
    """
    Convenience function to retrieve exchange API credentials
    
    Args:
        exchange_name: Name of the exchange (e.g., 'binance')
        
    Returns:
        tuple: (api_key, api_secret) or (None, None) if not found
    """
    sm = SecretManager()
    
    api_key = sm.get_secret(f"{exchange_name}_api_key")
    api_secret = sm.get_secret(f"{exchange_name}_api_secret")
    
    return api_key, api_secret


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize SecretManager
    sm = SecretManager()
    
    # Store a test secret
    success = sm.store_secret("test_key", "test_secret_value")
    print(f"Store success: {success}")
    
    # Retrieve the secret
    retrieved = sm.get_secret("test_key")
    print(f"Retrieved: {retrieved}")
    
    # Test convenience functions
    store_result = store_api_key("binance", "test_api_key", "test_api_secret")
    print(f"Store API credentials: {store_result}")
    
    api_key, api_secret = get_api_credentials("binance")
    print(f"Retrieved API credentials: {api_key}, {api_secret}")