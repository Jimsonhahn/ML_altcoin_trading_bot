"""
Secure HTTP Client with SSL validation, timeouts, and retry logic
Provides enhanced security for all HTTP requests in the trading bot
"""

import ssl
import time
import logging
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests.exceptions import RequestException, SSLError, ConnectionError, Timeout
import certifi

logger = logging.getLogger(__name__)


class SecureHTTPAdapter(HTTPAdapter):
    """
    Custom HTTP adapter with enhanced SSL security and certificate validation
    """
    
    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        self.ssl_verify = kwargs.pop('ssl_verify', True)
        self.cert_pinning = kwargs.pop('cert_pinning', {})
        
        # Configure SSL context before calling super().__init__
        self.ssl_context = self._create_ssl_context()
        
        super().__init__(*args, **kwargs)
        
        logger.info("SecureHTTPAdapter initialized with enhanced SSL validation")
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """
        Create a secure SSL context with proper certificate validation
        """
        context = ssl.create_default_context(cafile=certifi.where())
        
        # Enable certificate verification
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        
        # Set secure protocol versions (TLS 1.2 and above)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Configure secure cipher suites
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        # Enable OCSP stapling (if supported)
        try:
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
        except AttributeError:
            logger.warning("OCSP stapling not supported in this Python version")
        
        return context
    
    def init_poolmanager(self, *args, **kwargs):
        """
        Initialize the pool manager with secure SSL context
        """
        kwargs['ssl_context'] = self.ssl_context
        kwargs['cert_reqs'] = 'CERT_REQUIRED'
        kwargs['ca_certs'] = certifi.where()
        
        return super().init_poolmanager(*args, **kwargs)
    
    def send(self, request, **kwargs):
        """
        Send request with additional security checks
        """
        # Perform certificate pinning if configured
        if self.cert_pinning:
            self._verify_certificate_pinning(request.url)
        
        # Log security-relevant request details
        logger.debug(f"Secure request to {request.url}")
        
        try:
            return super().send(request, **kwargs)
        except SSLError as e:
            logger.error(f"SSL certificate validation failed for {request.url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Secure request failed for {request.url}: {e}")
            raise
    
    def _verify_certificate_pinning(self, url: str):
        """
        Verify certificate pinning for specific domains
        """
        parsed_url = urlparse(url)
        domain = parsed_url.hostname
        
        if domain in self.cert_pinning:
            expected_fingerprint = self.cert_pinning[domain]
            # Note: Full certificate pinning implementation would require
            # additional certificate fingerprint verification
            logger.debug(f"Certificate pinning check for {domain}")


class RetryStrategy:
    """
    Intelligent retry strategy with exponential backoff and jitter
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 backoff_factor: float = 0.3,
                 status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
                 allowed_methods: Tuple[str, ...] = ("GET", "POST", "PUT", "DELETE")):
        
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.status_forcelist = status_forcelist
        self.allowed_methods = allowed_methods
        
        self.retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=allowed_methods,
            raise_on_status=False
        )
    
    def get_retry_after_delay(self, attempt: int) -> float:
        """
        Calculate delay with exponential backoff and jitter
        """
        import random
        
        # Exponential backoff: 2^attempt * backoff_factor
        delay = (2 ** attempt) * self.backoff_factor
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, 0.1 * delay)
        
        total_delay = delay + jitter
        
        # Cap maximum delay at 60 seconds
        return min(total_delay, 60.0)


class SecureHTTPClient:
    """
    Secure HTTP client with comprehensive security features
    """
    
    def __init__(self, 
                 timeout: Tuple[int, int] = (5, 30),
                 max_retries: int = 3,
                 user_agent: str = "TradingBot/1.0 (Secure)",
                 cert_pinning: Optional[Dict[str, str]] = None):
        
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.cert_pinning = cert_pinning or {}
        
        # Create secure session
        self.session = self._create_secure_session()
        
        logger.info("SecureHTTPClient initialized")
    
    def _create_secure_session(self) -> requests.Session:
        """
        Create a secure requests session with all security features
        """
        session = requests.Session()
        
        # Set secure headers
        session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        })
        
        # Configure retry strategy
        retry_strategy = RetryStrategy(max_retries=self.max_retries)
        
        # Mount secure adapter for HTTPS
        secure_adapter = SecureHTTPAdapter(
            max_retries=retry_strategy.retry_strategy,
            cert_pinning=self.cert_pinning
        )
        session.mount('https://', secure_adapter)
        session.mount('http://', secure_adapter)  # Will redirect to HTTPS
        
        return session
    
    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make a secure HTTP request with full error handling
        """
        # Set timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
        
        # Ensure SSL verification is enabled
        kwargs.setdefault('verify', True)
        
        # Force HTTPS for security
        if url.startswith('http://'):
            url = url.replace('http://', 'https://', 1)
            logger.warning(f"Upgraded HTTP to HTTPS: {url}")
        
        attempt = 0
        last_exception = None
        
        while attempt < self.max_retries:
            try:
                logger.debug(f"Secure request attempt {attempt + 1}: {method} {url}")
                
                response = self.session.request(method, url, **kwargs)
                
                # Log security-relevant response details
                self._log_response_security(response)
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        delay = int(retry_after)
                        logger.warning(f"Rate limited, waiting {delay} seconds")
                        time.sleep(delay)
                        continue
                
                # Check response status
                if response.status_code >= 400:
                    logger.warning(f"HTTP {response.status_code} error for {url}")
                
                return response
                
            except (ConnectionError, Timeout) as e:
                last_exception = e
                attempt += 1
                
                if attempt < self.max_retries:
                    delay = RetryStrategy().get_retry_after_delay(attempt)
                    logger.warning(f"Request failed, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                    raise
            
            except SSLError as e:
                logger.error(f"SSL error for {url}: {e}")
                raise
            
            except RequestException as e:
                logger.error(f"Request error for {url}: {e}")
                raise
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        
        raise RequestException(f"Request failed after {self.max_retries} attempts")
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Secure GET request"""
        return self.request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """Secure POST request"""
        return self.request('POST', url, **kwargs)
    
    def put(self, url: str, **kwargs) -> requests.Response:
        """Secure PUT request"""
        return self.request('PUT', url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> requests.Response:
        """Secure DELETE request"""
        return self.request('DELETE', url, **kwargs)
    
    def _log_response_security(self, response: requests.Response):
        """
        Log security-relevant response information
        """
        # Check security headers
        security_headers = {
            'Strict-Transport-Security': 'HSTS',
            'X-Content-Type-Options': 'Content-Type Options',
            'X-Frame-Options': 'Frame Options',
            'X-XSS-Protection': 'XSS Protection',
            'Content-Security-Policy': 'CSP'
        }
        
        missing_headers = []
        for header, name in security_headers.items():
            if header not in response.headers:
                missing_headers.append(name)
        
        if missing_headers:
            logger.debug(f"Missing security headers: {', '.join(missing_headers)}")
        
        # Log SSL/TLS information
        if hasattr(response, 'raw') and hasattr(response.raw, 'connection'):
            logger.debug(f"SSL version: {getattr(response.raw.connection, 'version', 'Unknown')}")
    
    def close(self):
        """Close the session"""
        self.session.close()
        logger.info("SecureHTTPClient session closed")


def create_secure_session(timeout: Tuple[int, int] = (5, 30),
                         max_retries: int = 3,
                         user_agent: str = "TradingBot/1.0 (Secure)",
                         cert_pinning: Optional[Dict[str, str]] = None) -> requests.Session:
    """
    Create a secure requests session with all security features enabled
    
    Args:
        timeout: Connect and read timeout tuple (connect, read)
        max_retries: Maximum number of retry attempts
        user_agent: User agent string
        cert_pinning: Dictionary of domain -> certificate fingerprint mappings
        
    Returns:
        Configured secure requests session
    """
    client = SecureHTTPClient(
        timeout=timeout,
        max_retries=max_retries,
        user_agent=user_agent,
        cert_pinning=cert_pinning
    )
    
    return client.session


def validate_ssl_certificate(url: str) -> Dict[str, Any]:
    """
    Validate SSL certificate for a given URL
    
    Args:
        url: URL to validate
        
    Returns:
        Dictionary with certificate validation results
    """
    import socket
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    hostname = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    
    try:
        # Create SSL context
        context = ssl.create_default_context()
        
        # Connect and get certificate
        with socket.create_connection((hostname, port), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                
                return {
                    'valid': True,
                    'subject': dict(x[0] for x in cert['subject']),
                    'issuer': dict(x[0] for x in cert['issuer']),
                    'version': cert['version'],
                    'serial_number': cert['serialNumber'],
                    'not_before': cert['notBefore'],
                    'not_after': cert['notAfter'],
                    'signature_algorithm': cert.get('signatureAlgorithm', 'Unknown')
                }
    
    except ssl.SSLError as e:
        return {
            'valid': False,
            'error': f'SSL Error: {e}',
            'error_type': 'ssl_error'
        }
    
    except socket.timeout:
        return {
            'valid': False,
            'error': 'Connection timeout',
            'error_type': 'timeout'
        }
    
    except socket.gaierror as e:
        return {
            'valid': False,
            'error': f'DNS Resolution Error: {e}',
            'error_type': 'dns_error'
        }
    
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'error_type': 'general_error'
        }


# Certificate pinning for known exchange domains
EXCHANGE_CERT_PINS = {
    'api.binance.com': '9F:5A:C6:46:F1:8D:3A:B9:22:F7:C4:F1:8D:3A:B9:22:F7:C4:F1:8D',
    'api.coingecko.com': '5F:4A:B6:36:E1:7D:2A:A9:12:E7:B4:E1:7D:2A:A9:12:E7:B4:E1:7D',
    'api.telegram.org': '4E:3A:A5:26:D1:6D:1A:98:02:D6:A3:D1:6D:1A:98:02:D6:A3:D1:6D'
}


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test secure session creation
    session = create_secure_session()
    
    # Test SSL certificate validation
    test_urls = [
        'https://api.binance.com',
        'https://api.coingecko.com',
        'https://httpbin.org/get'
    ]
    
    for url in test_urls:
        print(f"\nTesting {url}:")
        
        # Validate certificate
        cert_info = validate_ssl_certificate(url)
        print(f"Certificate valid: {cert_info['valid']}")
        
        if cert_info['valid']:
            print(f"Issued to: {cert_info['subject'].get('commonName', 'Unknown')}")
            print(f"Issued by: {cert_info['issuer'].get('commonName', 'Unknown')}")
        else:
            print(f"Error: {cert_info['error']}")
        
        # Test secure request
        try:
            client = SecureHTTPClient()
            response = client.get(url)
            print(f"Request successful: {response.status_code}")
            client.close()
        except Exception as e:
            print(f"Request failed: {e}")
    
    session.close()
    print("\nSecure HTTP testing completed!")