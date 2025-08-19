"""Advanced security system for materials orchestrator with comprehensive protection."""

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security protection levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    source: str
    description: str
    details: Dict[str, Any]
    mitigated: bool = False


class AdvancedSecurityManager:
    """Comprehensive security management system."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.security_events = []
        self.failed_attempts = {}
        self.blocked_sources = set()
        self.api_keys = {}
        self.session_tokens = {}
        self.security_config = self._load_security_config()
        
        # Initialize secure random number generator
        self._secure_random = secrets.SystemRandom()
        
        logger.info(f"Advanced security manager initialized with {security_level.value} level")
    
    def authenticate_request(self, request_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Authenticate incoming requests with multiple validation layers."""
        source = request_data.get('source', 'unknown')
        
        # Check if source is blocked
        if source in self.blocked_sources:
            self._log_security_event(
                'blocked_source_attempt',
                ThreatLevel.HIGH,
                source,
                f"Attempt from blocked source: {source}"
            )
            return False, "Source is blocked"
        
        # Rate limiting
        if not self._check_rate_limit(source):
            return False, "Rate limit exceeded"
        
        # API key validation
        api_key = request_data.get('api_key')
        if not self._validate_api_key(api_key, source):
            self._record_failed_attempt(source)
            return False, "Invalid API key"
        
        # Request signature validation
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
            if not self._validate_request_signature(request_data):
                return False, "Invalid request signature"
        
        # Payload validation
        if not self._validate_payload(request_data):
            return False, "Invalid payload"
        
        return True, "Authentication successful"
    
    def authorize_experiment(self, experiment_data: Dict[str, Any], user_context: Dict[str, Any]) -> Tuple[bool, str]:
        """Authorize experiment execution with safety checks."""
        
        # Check user permissions
        if not self._check_user_permissions(user_context, 'run_experiment'):
            return False, "Insufficient permissions"
        
        # Validate experiment safety
        safety_result = self._assess_experiment_safety(experiment_data)
        if not safety_result['is_safe']:
            self._log_security_event(
                'unsafe_experiment_blocked',
                ThreatLevel.HIGH,
                user_context.get('user_id', 'unknown'),
                f"Unsafe experiment blocked: {safety_result['reason']}"
            )
            return False, f"Experiment blocked: {safety_result['reason']}"
        
        # Resource authorization
        if not self._authorize_resource_usage(experiment_data, user_context):
            return False, "Resource usage not authorized"
        
        return True, "Experiment authorized"
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using AES-256."""
        # Simplified encryption for demonstration
        # In production, use proper cryptographic libraries
        key = self._get_encryption_key()
        encrypted = self._simple_encrypt(data, key)
        return encrypted
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        key = self._get_encryption_key()
        decrypted = self._simple_decrypt(encrypted_data, key)
        return decrypted
    
    def generate_secure_token(self, context: Dict[str, Any]) -> str:
        """Generate cryptographically secure session token."""
        timestamp = str(int(time.time()))
        user_id = context.get('user_id', 'anonymous')
        random_bytes = secrets.token_hex(16)
        
        # Create token payload
        payload = f"{user_id}:{timestamp}:{random_bytes}"
        
        # Sign the token
        signature = self._sign_data(payload)
        token = f"{payload}:{signature}"
        
        # Store token with expiration
        self.session_tokens[token] = {
            'created': datetime.now(),
            'user_id': user_id,
            'expires': datetime.now() + timedelta(hours=24)
        }
        
        return token
    
    def validate_session_token(self, token: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate session token."""
        if not token or token not in self.session_tokens:
            return False, {}
        
        token_data = self.session_tokens[token]
        
        # Check expiration
        if datetime.now() > token_data['expires']:
            del self.session_tokens[token]
            return False, {}
        
        # Validate token signature
        try:
            payload, signature = token.rsplit(':', 1)
            expected_signature = self._sign_data(payload)
            if not hmac.compare_digest(signature, expected_signature):
                return False, {}
        except ValueError:
            return False, {}
        
        return True, token_data
    
    def scan_for_threats(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan data for potential security threats."""
        threats = []
        
        # SQL injection patterns
        sql_patterns = [
            r"(?i)union\s+select",
            r"(?i)drop\s+table",
            r"(?i)delete\s+from",
            r"(?i)insert\s+into",
            r"(?i)update\s+.*\s+set"
        ]
        
        # Command injection patterns
        cmd_patterns = [
            r";\s*rm\s+",
            r";\s*cat\s+",
            r";\s*ls\s+",
            r"\|\s*nc\s+",
            r"&&\s*curl\s+"
        ]
        
        data_str = json.dumps(data).lower()
        
        import re
        for pattern in sql_patterns + cmd_patterns:
            if re.search(pattern, data_str):
                threats.append({
                    'type': 'injection_attempt',
                    'pattern': pattern,
                    'severity': 'high',
                    'description': 'Potential injection attack detected'
                })
        
        # Check for suspicious parameter values
        for key, value in data.items():
            if isinstance(value, str):
                if len(value) > 10000:
                    threats.append({
                        'type': 'oversized_input',
                        'parameter': key,
                        'severity': 'medium',
                        'description': f'Oversized input in parameter {key}'
                    })
                
                # Check for encoded data
                try:
                    import base64
                    decoded = base64.b64decode(value)
                    if b'<script>' in decoded or b'eval(' in decoded:
                        threats.append({
                            'type': 'malicious_script',
                            'parameter': key,
                            'severity': 'high',
                            'description': 'Potential malicious script detected'
                        })
                except:
                    pass
        
        return threats
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        recent_events = [e for e in self.security_events if 
                        (datetime.now() - e.timestamp).total_seconds() < 3600]
        
        threat_counts = {}
        for event in recent_events:
            level = event.threat_level.value
            threat_counts[level] = threat_counts.get(level, 0) + 1
        
        return {
            'security_level': self.security_level.value,
            'total_security_events': len(self.security_events),
            'recent_events_1h': len(recent_events),
            'threat_counts': threat_counts,
            'blocked_sources': len(self.blocked_sources),
            'active_sessions': len(self.session_tokens),
            'failed_attempts': sum(len(attempts) if isinstance(attempts, list) else attempts 
                                  for attempts in self.failed_attempts.values()),
            'security_score': self._calculate_security_score()
        }
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check if source exceeds rate limits."""
        max_requests = self.security_config.get('max_requests_per_minute', 60)
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        source_attempts = self.failed_attempts.get(source, [])
        recent_attempts = [t for t in source_attempts if t > minute_ago]
        self.failed_attempts[source] = recent_attempts
        
        return len(recent_attempts) < max_requests
    
    def _validate_api_key(self, api_key: str, source: str) -> bool:
        """Validate API key."""
        if not api_key:
            return False
        
        # Check against stored API keys
        expected_key = self.api_keys.get(source)
        if not expected_key:
            # Generate and store new API key for development
            self.api_keys[source] = self._generate_api_key()
            return True
        
        return hmac.compare_digest(api_key, expected_key)
    
    def _validate_request_signature(self, request_data: Dict[str, Any]) -> bool:
        """Validate request signature."""
        signature = request_data.get('signature')
        if not signature:
            return False
        
        # Create expected signature
        payload = json.dumps(request_data, sort_keys=True)
        expected_signature = self._sign_data(payload)
        
        return hmac.compare_digest(signature, expected_signature)
    
    def _validate_payload(self, request_data: Dict[str, Any]) -> bool:
        """Validate request payload for malicious content."""
        threats = self.scan_for_threats(request_data)
        return len(threats) == 0
    
    def _check_user_permissions(self, user_context: Dict[str, Any], permission: str) -> bool:
        """Check user permissions."""
        user_permissions = user_context.get('permissions', [])
        return permission in user_permissions or 'admin' in user_permissions
    
    def _assess_experiment_safety(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess experiment safety."""
        # Temperature safety check
        temp = experiment_data.get('temperature', 0)
        if temp > 500:
            return {'is_safe': False, 'reason': f'Temperature {temp}°C exceeds safety limit'}
        
        # pH safety check
        pH = experiment_data.get('pH', 7)
        if pH < 0 or pH > 14:
            return {'is_safe': False, 'reason': f'pH {pH} is outside safe range'}
        
        # Concentration safety check
        for key, value in experiment_data.items():
            if 'conc' in key.lower() and isinstance(value, (int, float)):
                if value > 20:
                    return {'is_safe': False, 'reason': f'Concentration {value}M too high for {key}'}
        
        return {'is_safe': True, 'reason': 'Experiment passes safety checks'}
    
    def _authorize_resource_usage(self, experiment_data: Dict[str, Any], user_context: Dict[str, Any]) -> bool:
        """Authorize resource usage."""
        # Simple resource check - can be expanded
        user_quota = user_context.get('resource_quota', 100)
        experiment_cost = self._estimate_experiment_cost(experiment_data)
        
        return experiment_cost <= user_quota
    
    def _estimate_experiment_cost(self, experiment_data: Dict[str, Any]) -> float:
        """Estimate experiment resource cost."""
        cost = 0
        cost += experiment_data.get('temperature', 0) * 0.01
        cost += experiment_data.get('reaction_time', 0) * 0.5
        
        for key, value in experiment_data.items():
            if 'conc' in key.lower() and isinstance(value, (int, float)):
                cost += value * 5
        
        return cost
    
    def _record_failed_attempt(self, source: str):
        """Record failed authentication attempt."""
        current_time = time.time()
        
        if source not in self.failed_attempts:
            self.failed_attempts[source] = []
        
        self.failed_attempts[source].append(current_time)
        
        # Block source after too many failed attempts
        max_attempts = self.security_config.get('max_failed_attempts', 5)
        if len(self.failed_attempts[source]) > max_attempts:
            self.blocked_sources.add(source)
            self._log_security_event(
                'source_blocked',
                ThreatLevel.HIGH,
                source,
                f"Source blocked after {max_attempts} failed attempts"
            )
    
    def _log_security_event(self, event_type: str, threat_level: ThreatLevel, 
                          source: str, description: str, details: Dict[str, Any] = None):
        """Log security event."""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            threat_level=threat_level,
            source=source,
            description=description,
            details=details or {}
        )
        
        self.security_events.append(event)
        logger.warning(f"Security event: {description}")
    
    def _get_encryption_key(self) -> str:
        """Get encryption key."""
        # In production, use proper key management
        return os.environ.get('MATERIALS_ENCRYPTION_KEY', 'default-key-change-in-production')
    
    def _simple_encrypt(self, data: str, key: str) -> str:
        """Simple encryption for demonstration."""
        # This is a placeholder - use proper encryption in production
        return hashlib.sha256((data + key).encode()).hexdigest()
    
    def _simple_decrypt(self, encrypted_data: str, key: str) -> str:
        """Simple decryption for demonstration."""
        # This is a placeholder - use proper decryption in production
        return "decrypted_data"
    
    def _sign_data(self, data: str) -> str:
        """Sign data with HMAC."""
        secret_key = self.security_config.get('signing_key', 'default-signing-key')
        return hmac.new(secret_key.encode(), data.encode(), hashlib.sha256).hexdigest()
    
    def _generate_api_key(self) -> str:
        """Generate new API key."""
        return secrets.token_urlsafe(32)
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score."""
        base_score = 100.0
        
        # Deduct points for security events
        recent_events = [e for e in self.security_events if 
                        (datetime.now() - e.timestamp).total_seconds() < 86400]
        
        for event in recent_events:
            if event.threat_level == ThreatLevel.CRITICAL:
                base_score -= 20
            elif event.threat_level == ThreatLevel.HIGH:
                base_score -= 10
            elif event.threat_level == ThreatLevel.MEDIUM:
                base_score -= 5
            else:
                base_score -= 1
        
        # Deduct for failed attempts
        total_failed = sum(len(attempts) if isinstance(attempts, list) else attempts 
                          for attempts in self.failed_attempts.values())
        base_score -= min(total_failed * 2, 30)
        
        return max(base_score, 0.0)
    
    def _load_security_config(self) -> Dict[str, Any]:
        """Load security configuration."""
        return {
            'max_requests_per_minute': 60,
            'max_failed_attempts': 5,
            'session_timeout_hours': 24,
            'signing_key': 'materials-orchestrator-signing-key',
            'require_signature': self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]
        }


def create_advanced_security_system(security_level: SecurityLevel = SecurityLevel.STANDARD) -> AdvancedSecurityManager:
    """Factory function to create advanced security system."""
    security_manager = AdvancedSecurityManager(security_level)
    logger.info("Advanced security system created")
    return security_manager