"""
JWT Authentication Middleware
=============================

Handles JWT token validation and user authentication.
"""

from flask import jsonify
from flask_jwt_extended import verify_jwt_in_request, get_jwt, create_access_token, create_refresh_token
from functools import wraps
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.secret_manager import SecretManager
from utils.error_handler import secure_error_handler

logger = logging.getLogger(__name__)


class AuthManager:
    """Manages authentication and authorization"""
    
    def __init__(self):
        self.secret_manager = SecretManager('trading_bot_api')
        self.allowed_users = self._load_allowed_users()
    
    def _load_allowed_users(self) -> Dict[str, Dict[str, Any]]:
        """Load allowed users from SecretManager"""
        users_data = self.secret_manager.get_secret('allowed_users')
        if not users_data:
            # Default admin user for initial setup
            default_admin = {
                'admin': {
                    'password_hash': self._hash_password('TradingBot2024'),  # Secure development password
                    'roles': ['admin', 'trader'],
                    'active': True
                }
            }
            self.secret_manager.store_secret('allowed_users', str(default_admin))
            return default_admin
        
        import ast
        return ast.literal_eval(users_data)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        import bcrypt
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, username: str, password: str) -> bool:
        """Verify user password"""
        import bcrypt
        
        user = self.allowed_users.get(username)
        if not user:
            return False
        
        # Check if user is active (default to True if not specified for backward compatibility)
        if not user.get('active', True):
            return False
        
        password_hash = user.get('password_hash', '')
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user data"""
        if not self.verify_password(username, password):
            return None
        
        user_data = self.allowed_users[username].copy()
        user_data['username'] = username
        user_data.pop('password_hash', None)  # Remove sensitive data
        
        # Handle both 'role' and 'roles' fields for backward compatibility
        if 'role' in user_data and 'roles' not in user_data:
            user_data['roles'] = [user_data['role']]
        elif 'roles' not in user_data:
            user_data['roles'] = []
        
        return user_data
    
    def create_tokens(self, user_data: Dict[str, Any]) -> Dict[str, str]:
        """Create access and refresh tokens"""
        identity = user_data['username']
        additional_claims = {
            'roles': user_data.get('roles', []),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        access_token = create_access_token(
            identity=identity,
            additional_claims=additional_claims
        )
        
        refresh_token = create_refresh_token(
            identity=identity,
            additional_claims=additional_claims
        )
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer'
        }


# Global auth manager instance
auth_manager = AuthManager()


def require_auth(f: Callable) -> Callable:
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            verify_jwt_in_request()
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return jsonify({'error': 'Authentication required'}), 401
    
    return decorated_function


def require_roles(*required_roles):
    """Decorator to require specific roles"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        @require_auth
        def decorated_function(*args, **kwargs):
            jwt_data = get_jwt()
            user_roles = jwt_data.get('roles', [])
            
            if not any(role in user_roles for role in required_roles):
                return jsonify({
                    'error': 'Insufficient permissions',
                    'required_roles': list(required_roles),
                    'user_roles': user_roles
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def require_admin(f: Callable) -> Callable:
    """Decorator to require admin role"""
    return require_roles('admin')(f)


def require_trader(f: Callable) -> Callable:
    """Decorator to require trader role"""
    return require_roles('trader', 'admin')(f)


def setup_jwt_callbacks(jwt):
    """Setup JWT callbacks for token validation"""
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({
            'error': 'Token has expired',
            'message': 'Please refresh your token or login again'
        }), 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({
            'error': 'Invalid token',
            'message': str(error)
        }), 401
    
    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({
            'error': 'Authorization required',
            'message': 'Request does not contain a valid access token'
        }), 401
    
    @jwt.revoked_token_loader
    def revoked_token_callback(jwt_header, jwt_payload):
        return jsonify({
            'error': 'Token has been revoked',
            'message': 'This token is no longer valid'
        }), 401
    
    @jwt.token_verification_failed_loader
    def failed_token_callback(jwt_header, jwt_payload):
        return jsonify({
            'error': 'Token verification failed',
            'message': 'Unable to verify token signature'
        }), 401


# Utility functions for use in routes
def get_current_user() -> Optional[str]:
    """Get current authenticated user"""
    try:
        verify_jwt_in_request()
        jwt_data = get_jwt()
        return jwt_data.get('sub')  # 'sub' contains the identity
    except:
        return None


def get_current_user_roles() -> list:
    """Get roles of current authenticated user"""
    try:
        verify_jwt_in_request()
        jwt_data = get_jwt()
        return jwt_data.get('roles', [])
    except:
        return []


def is_admin() -> bool:
    """Check if current user is admin"""
    return 'admin' in get_current_user_roles()


def is_trader() -> bool:
    """Check if current user is trader"""
    roles = get_current_user_roles()
    return 'trader' in roles or 'admin' in roles