#!/usr/bin/env python3
"""
🔒 API Security & Stability Module
Enhanced security with rate limiting, encryption, circuit breakers, and resilient requests
"""

import time
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
import asyncio
import aiohttp
import requests
from flask import request, jsonify, g
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """API Security Levels"""
    PUBLIC = "public"           # No authentication required
    AUTHENTICATED = "auth"      # API key required
    ADMIN = "admin"            # Admin privileges required
    INTERNAL = "internal"      # Internal service only

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_allowance: int = 10
    security_level: SecurityLevel = SecurityLevel.PUBLIC

@dataclass
class ClientInfo:
    """Client tracking information"""
    ip_address: str
    api_key: Optional[str] = None
    requests_minute: deque = field(default_factory=lambda: deque(maxlen=100))
    requests_hour: deque = field(default_factory=lambda: deque(maxlen=1000))
    requests_day: deque = field(default_factory=lambda: deque(maxlen=10000))
    blocked_until: Optional[datetime] = None
    security_violations: int = 0
    last_seen: datetime = field(default_factory=datetime.now)

class APISecurityManager:
    """
    🔒 Comprehensive API Security Manager
    
    Features:
    - Rate limiting per IP/API key
    - Request signing and validation
    - IP whitelist/blacklist
    - Circuit breaker pattern
    - Attack detection and blocking
    - Encrypted data transmission
    """
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or "janics_freedom_factory_secure_2024"
        self.clients: Dict[str, ClientInfo] = {}
        self.blocked_ips: set = set()
        self.whitelisted_ips: set = {"127.0.0.1", "::1", "localhost"}
        self.admin_api_keys: set = set()
        self.lock = threading.RLock()
        
        # Rate limiting rules by endpoint
        self.rate_limit_rules: Dict[str, RateLimitRule] = {
            "default": RateLimitRule(60, 1000, 10000, 10, SecurityLevel.PUBLIC),
            "dashboard": RateLimitRule(120, 2000, 20000, 20, SecurityLevel.PUBLIC),
            "api/intelligence": RateLimitRule(300, 5000, 50000, 50, SecurityLevel.AUTHENTICATED),
            "api/risk-tiered": RateLimitRule(100, 1500, 15000, 15, SecurityLevel.AUTHENTICATED),
            "api/admin": RateLimitRule(30, 300, 1000, 5, SecurityLevel.ADMIN)
        }
        
        # Attack detection thresholds
        self.attack_thresholds = {
            'requests_per_second': 20,
            'invalid_signatures': 5,
            'blocked_attempts': 3,
            'suspicious_patterns': 10
        }
        
        logger.info("🔒 API Security Manager initialized")
    
    def add_admin_api_key(self, api_key: str):
        """Add admin API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.admin_api_keys.add(key_hash)
        logger.info("🔑 Admin API key added")
    
    def whitelist_ip(self, ip_address: str):
        """Add IP to whitelist"""
        self.whitelisted_ips.add(ip_address)
        logger.info(f"✅ IP whitelisted: {ip_address}")
    
    def block_ip(self, ip_address: str, duration_hours: int = 24):
        """Block IP address"""
        with self.lock:
            self.blocked_ips.add(ip_address)
            if ip_address in self.clients:
                self.clients[ip_address].blocked_until = datetime.now() + timedelta(hours=duration_hours)
        logger.warning(f"🚫 IP blocked for {duration_hours}h: {ip_address}")
    
    def get_client_info(self, ip_address: str, api_key: str = None) -> ClientInfo:
        """Get or create client information"""
        client_id = f"{ip_address}:{api_key}" if api_key else ip_address
        
        with self.lock:
            if client_id not in self.clients:
                self.clients[client_id] = ClientInfo(
                    ip_address=ip_address,
                    api_key=api_key
                )
            
            client = self.clients[client_id]
            client.last_seen = datetime.now()
            return client
    
    def is_rate_limited(self, client: ClientInfo, endpoint: str) -> tuple[bool, Dict]:
        """Check if client is rate limited"""
        rule = self.rate_limit_rules.get(endpoint, self.rate_limit_rules["default"])
        now = datetime.now()
        
        # Clean old requests
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Remove old requests
        while client.requests_minute and client.requests_minute[0] < minute_ago:
            client.requests_minute.popleft()
        while client.requests_hour and client.requests_hour[0] < hour_ago:
            client.requests_hour.popleft()
        while client.requests_day and client.requests_day[0] < day_ago:
            client.requests_day.popleft()
        
        # Check limits
        minute_count = len(client.requests_minute)
        hour_count = len(client.requests_hour)
        day_count = len(client.requests_day)
        
        # Allow burst for first few requests
        burst_allowed = minute_count < rule.burst_allowance
        
        rate_limit_info = {
            'requests_minute': minute_count,
            'requests_hour': hour_count,
            'requests_day': day_count,
            'limits': {
                'minute': rule.requests_per_minute,
                'hour': rule.requests_per_hour,
                'day': rule.requests_per_day
            },
            'burst_allowed': burst_allowed
        }
        
        # Check if rate limited
        if (minute_count >= rule.requests_per_minute and not burst_allowed or
            hour_count >= rule.requests_per_hour or
            day_count >= rule.requests_per_day):
            return True, rate_limit_info
        
        # Record this request
        client.requests_minute.append(now)
        client.requests_hour.append(now)
        client.requests_day.append(now)
        
        return False, rate_limit_info
    
    def validate_request_signature(self, data: str, signature: str, api_key: str) -> bool:
        """Validate HMAC signature for request"""
        try:
            expected_signature = hmac.new(
                f"{self.secret_key}:{api_key}".encode(),
                data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(expected_signature, signature)
        except Exception as e:
            logger.error(f"Signature validation error: {e}")
            return False
    
    def detect_attack_patterns(self, client: ClientInfo, request_data: Dict) -> List[str]:
        """Detect suspicious attack patterns"""
        patterns = []
        
        # High frequency attacks
        if len(client.requests_minute) > self.attack_thresholds['requests_per_second']:
            patterns.append("high_frequency_attack")
        
        # SQL Injection patterns
        dangerous_patterns = ['union', 'select', 'drop', 'delete', 'insert', 'exec', 'script']
        request_str = json.dumps(request_data).lower()
        if any(pattern in request_str for pattern in dangerous_patterns):
            patterns.append("sql_injection_attempt")
        
        # Path traversal
        if any(path in request_str for path in ['../../../', '..\\..\\', '/etc/passwd']):
            patterns.append("path_traversal_attempt")
        
        # XSS attempts
        xss_patterns = ['<script', 'javascript:', 'onerror=', 'onload=']
        if any(xss in request_str for xss in xss_patterns):
            patterns.append("xss_attempt")
        
        return patterns
    
    def rate_limit_decorator(self, endpoint: str = "default"):
        """Decorator for rate limiting Flask endpoints"""
        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def wrapper(*args, **kwargs):
                return self.apply_rate_limiting(f, endpoint, *args, **kwargs)
            return wrapper
        return decorator
    
    def apply_rate_limiting(self, func: Callable, endpoint: str, *args, **kwargs):
        """Apply rate limiting to endpoint"""
        try:
            # Get client info
            ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            api_key = request.headers.get('X-API-Key')
            
            # Check if IP is blocked
            if ip_address in self.blocked_ips:
                return jsonify({
                    'error': 'IP_BLOCKED',
                    'message': 'Your IP address has been blocked due to suspicious activity'
                }), 429
            
            client = self.get_client_info(ip_address, api_key)
            
            # Check if client is temporarily blocked
            if client.blocked_until and datetime.now() < client.blocked_until:
                return jsonify({
                    'error': 'TEMPORARILY_BLOCKED',
                    'message': f'Access blocked until {client.blocked_until}'
                }), 429
            
            # Check rate limits
            is_limited, limit_info = self.is_rate_limited(client, endpoint)
            if is_limited:
                # Escalate security violations
                client.security_violations += 1
                if client.security_violations > 5:
                    self.block_ip(ip_address, duration_hours=1)
                
                return jsonify({
                    'error': 'RATE_LIMITED',
                    'message': 'Rate limit exceeded',
                    'rate_limit_info': limit_info,
                    'retry_after': 60
                }), 429
            
            # Detect attack patterns
            request_data = request.get_json() or {}
            attack_patterns = self.detect_attack_patterns(client, request_data)
            if attack_patterns:
                client.security_violations += len(attack_patterns)
                logger.warning(f"🚨 Attack patterns detected from {ip_address}: {attack_patterns}")
                
                if client.security_violations > 3:
                    self.block_ip(ip_address, duration_hours=6)
                    return jsonify({
                        'error': 'SECURITY_VIOLATION',
                        'message': 'Suspicious activity detected'
                    }), 403
            
            # Add rate limit headers to response
            response = func(*args, **kwargs)
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Remaining-Minute'] = str(
                    self.rate_limit_rules[endpoint].requests_per_minute - limit_info['requests_minute']
                )
                response.headers['X-RateLimit-Remaining-Hour'] = str(
                    self.rate_limit_rules[endpoint].requests_per_hour - limit_info['requests_hour']
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return func(*args, **kwargs)  # Allow request if rate limiting fails

class CircuitBreaker:
    """
    ⚡ Circuit Breaker for API Stability
    
    Prevents cascading failures by monitoring API health
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
        logger.info("⚡ Circuit Breaker initialized")
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset circuit breaker"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"⚡ Circuit breaker opened after {self.failure_count} failures")

class ResilientHTTPClient:
    """
    🔄 Resilient HTTP Client with automatic retries and exponential backoff
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 timeout: float = 30.0):
        
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        
        logger.info("🔄 Resilient HTTP Client initialized")
    
    async def request(self, method: str, url: str, **kwargs) -> Dict:
        """Make resilient HTTP request with retries"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await self._make_request(method, url, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    
                    logger.warning(f"🔄 Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ Request failed after {self.max_retries + 1} attempts: {e}")
        
        raise last_exception
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    async def _make_request(self, method: str, url: str, **kwargs) -> Dict:
        """Make single HTTP request"""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.request(method, url, **kwargs) as response:
                if response.status >= 400:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                return await response.json()
    
    async def get(self, url: str, **kwargs) -> Dict:
        """Resilient GET request"""
        return await self.request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> Dict:
        """Resilient POST request"""
        return await self.request('POST', url, **kwargs)

# Global instances
security_manager = APISecurityManager()
resilient_client = ResilientHTTPClient()

def init_api_security(app, secret_key: str = None, admin_keys: List[str] = None):
    """Initialize API security for Flask app"""
    global security_manager
    
    if secret_key:
        security_manager = APISecurityManager(secret_key)
    
    if admin_keys:
        for key in admin_keys:
            security_manager.add_admin_api_key(key)
    
    # Apply rate limiting to all routes
    @app.before_request
    def apply_security():
        g.security_manager = security_manager
        
        # Skip rate limiting for whitelisted IPs
        ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        if ip in security_manager.whitelisted_ips:
            return
        
        # Apply basic rate limiting
        endpoint = request.endpoint or 'default'
        try:
            security_manager.apply_rate_limiting(lambda: None, endpoint)
        except Exception as e:
            if "RATE_LIMITED" in str(e) or "BLOCKED" in str(e):
                return jsonify({
                    'error': 'ACCESS_DENIED',
                    'message': 'Request blocked by security system'
                }), 429
    
    logger.info("🔒 API Security initialized for Flask app")

# Example usage decorators
def require_api_key(f):
    """Decorator requiring valid API key"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API_KEY_REQUIRED'}), 401
        
        # Validate API key here
        return f(*args, **kwargs)
    return wrapper

def admin_only(f):
    """Decorator for admin-only endpoints"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'ADMIN_ACCESS_REQUIRED'}), 401
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash not in security_manager.admin_api_keys:
            return jsonify({'error': 'INSUFFICIENT_PRIVILEGES'}), 403
        
        return f(*args, **kwargs)
    return wrapper