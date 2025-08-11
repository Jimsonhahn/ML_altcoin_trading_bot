"""
Authentication API Routes
=========================

Handles user authentication and token management.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity, get_jwt
import logging
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from api.middleware.auth import auth_manager
from utils.error_handler import ValidationTradingError

logger = logging.getLogger(__name__)

bp = Blueprint('auth', __name__)


@bp.route('/login', methods=['POST'])
def login():
    """
    User login
    ---
    tags:
      - Authentication
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - username
              - password
            properties:
              username:
                type: string
              password:
                type: string
    responses:
      200:
        description: Login successful
        content:
          application/json:
            schema:
              type: object
              properties:
                access_token:
                  type: string
                refresh_token:
                  type: string
                token_type:
                  type: string
                user:
                  type: object
                  properties:
                    username:
                      type: string
                    roles:
                      type: array
                      items:
                        type: string
      401:
        description: Invalid credentials
    """
    data = request.json
    
    if not data or not data.get('username') or not data.get('password'):
        raise ValidationTradingError("Username and password are required")
    
    username = data['username']
    password = data['password']
    
    # Authenticate user
    user_data = auth_manager.authenticate_user(username, password)
    
    if not user_data:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Create tokens
    tokens = auth_manager.create_tokens(user_data)
    
    logger.info(f"User {username} logged in successfully")
    
    return jsonify({
        **tokens,
        'user': {
            'username': user_data['username'],
            'roles': user_data.get('roles', [])
        }
    }), 200


@bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh_token():
    """
    Refresh access token
    ---
    tags:
      - Authentication
    security:
      - BearerAuth: []
    responses:
      200:
        description: Token refreshed
        content:
          application/json:
            schema:
              type: object
              properties:
                access_token:
                  type: string
                token_type:
                  type: string
    """
    current_user = get_jwt_identity()
    jwt_data = get_jwt()
    
    # Create new access token
    access_token = create_access_token(
        identity=current_user,
        additional_claims={
            'roles': jwt_data.get('roles', []),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    )
    
    return jsonify({
        'access_token': access_token,
        'token_type': 'Bearer'
    }), 200


@bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """
    User logout
    ---
    tags:
      - Authentication
    security:
      - BearerAuth: []
    responses:
      200:
        description: Logout successful
    """
    current_user = get_jwt_identity()
    
    # In a production environment, you would add the JWT to a blacklist
    # For now, we'll just log the logout
    logger.info(f"User {current_user} logged out")
    
    return jsonify({'message': 'Logout successful'}), 200


@bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """
    Get current user profile
    ---
    tags:
      - Authentication
    security:
      - BearerAuth: []
    responses:
      200:
        description: User profile
        content:
          application/json:
            schema:
              type: object
              properties:
                username:
                  type: string
                roles:
                  type: array
                  items:
                    type: string
                login_time:
                  type: string
                  format: date-time
    """
    current_user = get_jwt_identity()
    jwt_data = get_jwt()
    
    return jsonify({
        'username': current_user,
        'roles': jwt_data.get('roles', []),
        'login_time': jwt_data.get('timestamp'),
        'token_expires': jwt_data.get('exp')
    }), 200


@bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """
    Change user password
    ---
    tags:
      - Authentication
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - current_password
              - new_password
            properties:
              current_password:
                type: string
              new_password:
                type: string
    responses:
      200:
        description: Password changed successfully
      400:
        description: Invalid current password
    """
    current_user = get_jwt_identity()
    data = request.json
    
    if not data or not data.get('current_password') or not data.get('new_password'):
        raise ValidationTradingError("Current password and new password are required")
    
    current_password = data['current_password']
    new_password = data['new_password']
    
    # Verify current password
    if not auth_manager.verify_password(current_user, current_password):
        return jsonify({'error': 'Invalid current password'}), 400
    
    # Update password
    new_password_hash = auth_manager._hash_password(new_password)
    auth_manager.allowed_users[current_user]['password_hash'] = new_password_hash
    
    # Update stored users
    auth_manager.secret_manager.store_secret('allowed_users', str(auth_manager.allowed_users))
    
    logger.info(f"User {current_user} changed password")
    
    return jsonify({'message': 'Password changed successfully'}), 200


@bp.route('/validate-token', methods=['GET'])
@jwt_required()
def validate_token():
    """
    Validate current token
    ---
    tags:
      - Authentication
    security:
      - BearerAuth: []
    responses:
      200:
        description: Token is valid
        content:
          application/json:
            schema:
              type: object
              properties:
                valid:
                  type: boolean
                username:
                  type: string
                roles:
                  type: array
                  items:
                    type: string
                expires_at:
                  type: integer
    """
    current_user = get_jwt_identity()
    jwt_data = get_jwt()
    
    return jsonify({
        'valid': True,
        'username': current_user,
        'roles': jwt_data.get('roles', []),
        'expires_at': jwt_data.get('exp')
    }), 200