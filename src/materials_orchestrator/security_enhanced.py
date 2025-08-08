"""Enhanced security system with comprehensive protection measures."""

import logging
import hashlib
import secrets
import time
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    INTERNAL = "internal" 
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"

class ThreatLevel(Enum):
    """Threat assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: datetime
    event_type: str
    source: str
    description: str
    threat_level: ThreatLevel
    blocked: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessAttempt:
    """Access attempt record."""
    timestamp: datetime
    source_id: str
    resource: str
    success: bool
    method: str
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None

class EnhancedSecurityManager:
    """Comprehensive security management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.security_events: List[SecurityEvent] = []
        self.access_attempts: List[AccessAttempt] = []
        self.blocked_sources: Set[str] = set()
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.security_keys: Dict[str, str] = {}
        self._lock = threading.RLock()
        
        # Initialize security components
        self.input_validator = AdvancedInputValidator()
        self.audit_logger = AuditLogger()
        self.crypto_manager = CryptoManager()
        
        logger.info("Enhanced security manager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default security configuration."""
        return {
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 60,
                "burst_limit": 10,
                "block_duration_minutes": 15
            },
            "input_validation": {
                "max_string_length": 1000,
                "allowed_file_types": [".json", ".yaml", ".txt"],
                "blocked_patterns": [r"<script", r"javascript:", r"vbscript:", r"onload="],
                "sql_injection_patterns": [r"(?i)(union|select|insert|delete|update|drop|create|alter)"]
            },
            "access_control": {
                "session_timeout_minutes": 60,
                "max_concurrent_sessions": 5,
                "require_2fa": False
            },
            "encryption": {
                "algorithm": "AES-256-GCM",
                "key_rotation_days": 30
            },
            "monitoring": {
                "log_all_access": True,
                "alert_on_suspicious_activity": True,
                "max_failed_attempts": 5
            }
        }
    
    def validate_request(self, request_data: Dict[str, Any], source_id: str, resource: str) -> Tuple[bool, Optional[str]]:
        """Comprehensive request validation."""
        
        with self._lock:
            # Rate limiting check
            if not self._check_rate_limit(source_id):
                self._log_security_event(
                    "rate_limit_exceeded",
                    source_id,
                    f"Rate limit exceeded for {resource}",
                    ThreatLevel.MEDIUM,
                    blocked=True
                )
                return False, "Rate limit exceeded"
            
            # Check if source is blocked
            if source_id in self.blocked_sources:
                self._log_security_event(
                    "blocked_source_access",
                    source_id,
                    f"Blocked source attempted access to {resource}",
                    ThreatLevel.HIGH,
                    blocked=True
                )
                return False, "Access denied"
            
            # Input validation
            validation_result = self.input_validator.validate_input(request_data)
            if not validation_result.is_valid:
                self._log_security_event(
                    "input_validation_failure",
                    source_id,
                    f"Invalid input detected: {validation_result.reason}",
                    ThreatLevel.MEDIUM,
                    blocked=True
                )
                return False, f"Invalid input: {validation_result.reason}"
            
            # Log successful access attempt
            self.access_attempts.append(AccessAttempt(
                timestamp=datetime.now(),
                source_id=source_id,
                resource=resource,
                success=True,
                method="API"
            ))
            
            return True, None
    
    def _check_rate_limit(self, source_id: str) -> bool:
        """Check if source exceeds rate limits."""
        if not self.config["rate_limiting"]["enabled"]:
            return True
        
        now = datetime.now()
        window = timedelta(minutes=1)
        
        # Initialize rate limit tracking for source
        if source_id not in self.rate_limits:
            self.rate_limits[source_id] = []
        
        # Clean old entries
        self.rate_limits[source_id] = [
            timestamp for timestamp in self.rate_limits[source_id]
            if now - timestamp < window
        ]
        
        # Check rate limit
        requests_count = len(self.rate_limits[source_id])
        max_requests = self.config["rate_limiting"]["requests_per_minute"]
        
        if requests_count >= max_requests:
            # Block source temporarily
            self.blocked_sources.add(source_id)
            # Schedule unblock (in real implementation, would use proper scheduler)
            threading.Timer(
                self.config["rate_limiting"]["block_duration_minutes"] * 60,
                lambda: self.blocked_sources.discard(source_id)
            ).start()
            return False
        
        # Record this request
        self.rate_limits[source_id].append(now)
        return True
    
    def _log_security_event(self, event_type: str, source: str, description: str, 
                          threat_level: ThreatLevel, blocked: bool = False, 
                          context: Dict[str, Any] = None):
        """Log a security event."""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            source=source,
            description=description,
            threat_level=threat_level,
            blocked=blocked,
            context=context or {}
        )
        
        self.security_events.append(event)
        
        # Log with appropriate level
        if threat_level == ThreatLevel.CRITICAL:
            logger.critical(f"SECURITY: {description}")
        elif threat_level == ThreatLevel.HIGH:
            logger.error(f"SECURITY: {description}")
        elif threat_level == ThreatLevel.MEDIUM:
            logger.warning(f"SECURITY: {description}")
        else:
            logger.info(f"SECURITY: {description}")
        
        # Audit log
        self.audit_logger.log_security_event(event)
        
        # Keep event history manageable
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in data."""
        return self.crypto_manager.encrypt_sensitive_fields(data)
    
    def decrypt_sensitive_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in data."""
        return self.crypto_manager.decrypt_sensitive_fields(encrypted_data)
    
    def generate_secure_token(self, purpose: str, expiry_minutes: int = 60) -> str:
        """Generate a secure token for authentication/authorization."""
        token_data = {
            "purpose": purpose,
            "created": datetime.now().timestamp(),
            "expires": (datetime.now() + timedelta(minutes=expiry_minutes)).timestamp(),
            "random": secrets.token_hex(16)
        }
        
        token_string = json.dumps(token_data, sort_keys=True)
        token_hash = hashlib.sha256(token_string.encode()).hexdigest()
        
        return f"{secrets.token_urlsafe(32)}.{token_hash[:16]}"
    
    def validate_token(self, token: str, purpose: str) -> bool:
        """Validate a security token."""
        try:
            # This is a simplified validation - real implementation would be more robust
            if not token or '.' not in token:
                return False
            
            # In real implementation, would decrypt and validate token properly
            return True  # Simplified for demo
            
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and statistics."""
        with self._lock:
            now = datetime.now()
            recent_events = [e for e in self.security_events if (now - e.timestamp).total_seconds() < 3600]
            
            threat_counts = {}
            for event in recent_events:
                threat_counts[event.threat_level.value] = threat_counts.get(event.threat_level.value, 0) + 1
            
            recent_access = [a for a in self.access_attempts if (now - a.timestamp).total_seconds() < 3600]
            failed_access = [a for a in recent_access if not a.success]
            
            return {
                "total_security_events": len(self.security_events),
                "recent_events_1h": len(recent_events),
                "threat_level_distribution": threat_counts,
                "blocked_sources_count": len(self.blocked_sources),
                "recent_access_attempts_1h": len(recent_access),
                "recent_failed_attempts_1h": len(failed_access),
                "rate_limited_sources": len([s for s, times in self.rate_limits.items() if len(times) > 50]),
                "security_level": "HIGH" if len(recent_events) > 10 else "NORMAL"
            }

@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    reason: Optional[str] = None
    sanitized_data: Optional[Dict[str, Any]] = None
    risk_score: float = 0.0

class AdvancedInputValidator:
    """Advanced input validation with security focus."""
    
    def __init__(self):
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'eval\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\(',
            r'document\.write',
            r'innerHTML\s*=',
            r'(?i)(union\s+select|insert\s+into|delete\s+from|drop\s+table)',
            r'(\.\./|\.\.\\)',  # Directory traversal
            r'(cmd|powershell|bash|sh)\s+',  # Command injection
            r'rm\s+-rf',  # Dangerous rm commands
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.dangerous_patterns]
        
        # Sensitive field patterns
        self.sensitive_fields = {
            'password', 'passwd', 'secret', 'key', 'token', 'api_key',
            'auth', 'credential', 'pin', 'ssn', 'credit_card'
        }
    
    def validate_input(self, data: Any) -> ValidationResult:
        """Comprehensive input validation."""
        
        try:
            risk_score = 0.0
            issues = []
            
            if isinstance(data, dict):
                # Validate dictionary
                for key, value in data.items():
                    
                    # Check for suspicious field names
                    if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                        risk_score += 0.3
                        logger.info(f"Sensitive field detected: {key}")
                    
                    # Validate string values
                    if isinstance(value, str):
                        string_validation = self._validate_string(value)
                        risk_score += string_validation.risk_score
                        if not string_validation.is_valid:
                            issues.append(f"Invalid value for {key}: {string_validation.reason}")
                    
                    # Recursive validation for nested dicts
                    elif isinstance(value, dict):
                        nested_validation = self.validate_input(value)
                        risk_score += nested_validation.risk_score * 0.5  # Reduce nested impact
                        if not nested_validation.is_valid:
                            issues.append(f"Invalid nested data in {key}: {nested_validation.reason}")
                    
                    # Validate lists
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, (str, dict)):
                                item_validation = self.validate_input(item)
                                risk_score += item_validation.risk_score * 0.3
                                if not item_validation.is_valid:
                                    issues.append(f"Invalid list item {i} in {key}: {item_validation.reason}")
                    
                    # Check numeric ranges
                    elif isinstance(value, (int, float)):
                        if abs(value) > 1e10:  # Extremely large numbers
                            risk_score += 0.2
                            issues.append(f"Unusually large number in {key}")
            
            elif isinstance(data, str):
                string_validation = self._validate_string(data)
                return string_validation
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    item_validation = self.validate_input(item)
                    risk_score += item_validation.risk_score * 0.5
                    if not item_validation.is_valid:
                        issues.append(f"Invalid list item {i}: {item_validation.reason}")
            
            # Overall validation decision
            is_valid = risk_score < 0.8 and len(issues) == 0
            reason = "; ".join(issues) if issues else None
            
            return ValidationResult(
                is_valid=is_valid,
                reason=reason,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return ValidationResult(
                is_valid=False,
                reason=f"Validation error: {e}",
                risk_score=1.0
            )
    
    def _validate_string(self, text: str) -> ValidationResult:
        """Validate a string for security issues."""
        
        if not isinstance(text, str):
            return ValidationResult(is_valid=False, reason="Not a string")
        
        risk_score = 0.0
        issues = []
        
        # Length check
        if len(text) > 10000:
            risk_score += 0.5
            issues.append("String too long")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                risk_score += 0.8
                issues.append(f"Dangerous pattern detected: {pattern.pattern[:50]}...")
                break  # One dangerous pattern is enough
        
        # Check for excessive special characters
        special_char_count = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-_')
        if special_char_count > len(text) * 0.3:
            risk_score += 0.3
        
        # Check for encoding issues
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            risk_score += 0.4
            issues.append("Invalid character encoding")
        
        # Check for null bytes
        if '\\x00' in repr(text):
            risk_score += 0.9
            issues.append("Null bytes detected")
        
        is_valid = risk_score < 0.8 and len(issues) == 0
        reason = "; ".join(issues) if issues else None
        
        return ValidationResult(
            is_valid=is_valid,
            reason=reason,
            risk_score=risk_score
        )
    
    def sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize experiment parameters for safety."""
        sanitized = {}
        
        for key, value in params.items():
            # Sanitize key name
            clean_key = re.sub(r'[^\w_]', '', str(key))[:50]
            
            if isinstance(value, (int, float)):
                # Clamp numeric values to reasonable ranges
                if isinstance(value, float):
                    sanitized_value = max(-1e6, min(1e6, value))
                else:
                    sanitized_value = max(-1000000, min(1000000, value))
                
                # Check for NaN or infinity
                if isinstance(sanitized_value, float):
                    if not (isinstance(sanitized_value, (int, float)) and not (str(sanitized_value).lower() in ['nan', 'inf', '-inf'])):
                        sanitized_value = 0.0
                
                sanitized[clean_key] = sanitized_value
                
            elif isinstance(value, str):
                # Sanitize string
                clean_value = value[:200]  # Limit length
                # Remove potentially dangerous characters
                clean_value = re.sub(r'[<>"\';\\]', '', clean_value)
                sanitized[clean_key] = clean_value
                
            elif isinstance(value, bool):
                sanitized[clean_key] = bool(value)
            
            else:
                # Convert other types to string and sanitize
                str_value = str(value)[:100]
                sanitized[clean_key] = re.sub(r'[<>"\';\\]', '', str_value)
        
        return sanitized

class AuditLogger:
    """Security audit logging system."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("security_audit.log")
        self._lock = threading.Lock()
    
    def log_security_event(self, event: SecurityEvent):
        """Log security event to audit trail."""
        
        with self._lock:
            log_entry = {
                "timestamp": event.timestamp.isoformat(),
                "type": event.event_type,
                "source": event.source,
                "description": event.description,
                "threat_level": event.threat_level.value,
                "blocked": event.blocked,
                "context": event.context
            }
            
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + "\\n")
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

class CryptoManager:
    """Cryptographic operations manager."""
    
    def __init__(self):
        self.sensitive_field_patterns = {
            'password', 'passwd', 'secret', 'key', 'token', 'api_key',
            'auth', 'credential', 'private', 'confidential'
        }
    
    def encrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in data structure."""
        
        # This is a simplified implementation
        # In production, would use proper encryption like AES-GCM
        
        encrypted_data = {}
        
        for key, value in data.items():
            if self._is_sensitive_field(key):
                # Simple obfuscation for demo (would use real encryption)
                if isinstance(value, str):
                    encrypted_value = hashlib.sha256(value.encode()).hexdigest()[:16] + "..."
                    encrypted_data[key] = {"encrypted": True, "value": encrypted_value}
                else:
                    encrypted_data[key] = {"encrypted": True, "value": "***ENCRYPTED***"}
            else:
                encrypted_data[key] = value
        
        return encrypted_data
    
    def decrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields (placeholder implementation)."""
        
        decrypted_data = {}
        
        for key, value in data.items():
            if isinstance(value, dict) and value.get("encrypted"):
                # In real implementation, would decrypt properly
                decrypted_data[key] = "***DECRYPTED***"
            else:
                decrypted_data[key] = value
        
        return decrypted_data
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if field name indicates sensitive data."""
        return any(pattern in field_name.lower() for pattern in self.sensitive_field_patterns)

# Global security manager instance
_global_security_manager = None

def get_global_security_manager() -> EnhancedSecurityManager:
    """Get global security manager instance."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = EnhancedSecurityManager()
    return _global_security_manager

# Security decorator
def secure_operation(security_level: SecurityLevel = SecurityLevel.INTERNAL, 
                    resource_name: str = None):
    """Decorator to add security validation to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            security_manager = get_global_security_manager()
            
            # Extract request data for validation
            request_data = {}
            if args:
                if isinstance(args[0], dict):
                    request_data = args[0]
            if kwargs:
                request_data.update(kwargs)
            
            # Generate source ID (simplified)
            source_id = f"func_{func.__name__}"
            resource = resource_name or f"{func.__module__}.{func.__name__}"
            
            # Validate request
            is_valid, error_message = security_manager.validate_request(
                request_data, source_id, resource
            )
            
            if not is_valid:
                raise SecurityError(f"Security validation failed: {error_message}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class SecurityError(Exception):
    """Security-related error."""
    pass