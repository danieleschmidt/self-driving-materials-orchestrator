"""Security features and access control for autonomous lab operations."""

import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access levels for lab operations."""

    READ_ONLY = "read_only"
    OPERATOR = "operator"
    ADMIN = "admin"
    SYSTEM = "system"


class OperationType(Enum):
    """Types of lab operations that can be controlled."""

    VIEW_EXPERIMENTS = "view_experiments"
    RUN_EXPERIMENT = "run_experiment"
    MANAGE_ROBOTS = "manage_robots"
    SYSTEM_CONFIG = "system_config"
    EMERGENCY_STOP = "emergency_stop"
    DATA_EXPORT = "data_export"


@dataclass
class User:
    """User account information."""

    username: str
    email: str
    access_level: AccessLevel
    permissions: Set[OperationType] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    max_experiment_duration: int = 86400  # 24 hours in seconds
    allowed_operations: Set[str] = field(
        default_factory=lambda: {"run_experiment", "view_data", "export_results"}
    )
    rate_limit_requests_per_minute: int = 60
    require_authentication: bool = True
    audit_logging_enabled: bool = True


class SecurityValidator:
    """Validates security constraints for lab operations."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.security_events: List[SecurityEvent] = []

    def validate_experiment_request(self, request: Dict[str, Any]) -> bool:
        """Validate experiment request for security compliance."""
        # Check basic security constraints
        if not self._validate_parameters(request.get("parameters", {})):
            return False
        if not self._validate_duration(request.get("duration", 0)):
            return False
        return True

    def _validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate experiment parameters for safety."""
        for key, value in parameters.items():
            if isinstance(value, str) and len(value) > 1000:
                return False
            if key.startswith("_") or key.startswith("__"):
                return False
        return True

    def _validate_duration(self, duration: float) -> bool:
        """Validate experiment duration."""
        return 0 < duration <= self.config.max_experiment_duration

    def log_security_event(self, event: "SecurityEvent") -> None:
        """Log a security event."""
        self.security_events.append(event)


@dataclass
class SecurityEvent:
    """Security-related event for audit logging."""

    timestamp: datetime
    event_type: str
    username: Optional[str]
    ip_address: Optional[str]
    operation: Optional[str]
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class SecurityManager:
    """Manages authentication, authorization, and security auditing."""

    def __init__(self, config_file: Optional[str] = None):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # key -> username
        self.security_events: List[SecurityEvent] = []
        self.config_file = config_file or "security_config.json"

        # Default permissions by access level
        self.default_permissions = {
            AccessLevel.READ_ONLY: {OperationType.VIEW_EXPERIMENTS},
            AccessLevel.OPERATOR: {
                OperationType.VIEW_EXPERIMENTS,
                OperationType.RUN_EXPERIMENT,
                OperationType.DATA_EXPORT,
            },
            AccessLevel.ADMIN: {
                OperationType.VIEW_EXPERIMENTS,
                OperationType.RUN_EXPERIMENT,
                OperationType.MANAGE_ROBOTS,
                OperationType.SYSTEM_CONFIG,
                OperationType.EMERGENCY_STOP,
                OperationType.DATA_EXPORT,
            },
            AccessLevel.SYSTEM: set(OperationType),  # All permissions
        }

        self._load_config()

    def _load_config(self) -> None:
        """Load security configuration from file."""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)

                # Load users
                for user_data in config.get("users", []):
                    user = User(
                        username=user_data["username"],
                        email=user_data["email"],
                        access_level=AccessLevel(user_data["access_level"]),
                        permissions=set(
                            OperationType(p) for p in user_data.get("permissions", [])
                        ),
                        created_at=datetime.fromisoformat(
                            user_data.get("created_at", datetime.now().isoformat())
                        ),
                        is_active=user_data.get("is_active", True),
                    )
                    self.users[user.username] = user

                # Load API keys
                self.api_keys = config.get("api_keys", {})

                logger.info(f"Loaded security config with {len(self.users)} users")

            except Exception as e:
                logger.error(f"Failed to load security config: {e}")
        else:
            # Create default admin user
            self._create_default_admin()

    def _create_default_admin(self) -> None:
        """Create default admin user for initial setup."""
        admin_user = User(
            username="admin",
            email="admin@localhost",
            access_level=AccessLevel.ADMIN,
            permissions=self.default_permissions[AccessLevel.ADMIN],
        )
        self.users["admin"] = admin_user

        # Generate API key for admin
        api_key = self.generate_api_key("admin")

        logger.warning(f"Created default admin user with API key: {api_key}")
        logger.warning("Please change the default configuration in production!")

    def save_config(self) -> None:
        """Save security configuration to file."""
        try:
            config = {
                "users": [
                    {
                        "username": user.username,
                        "email": user.email,
                        "access_level": user.access_level.value,
                        "permissions": [p.value for p in user.permissions],
                        "created_at": user.created_at.isoformat(),
                        "is_active": user.is_active,
                    }
                    for user in self.users.values()
                ],
                "api_keys": self.api_keys,
            }

            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)

            logger.info("Saved security configuration")

        except Exception as e:
            logger.error(f"Failed to save security config: {e}")

    def create_user(
        self,
        username: str,
        email: str,
        access_level: AccessLevel,
        custom_permissions: Optional[Set[OperationType]] = None,
    ) -> User:
        """Create a new user account."""
        if username in self.users:
            raise ValueError(f"User {username} already exists")

        permissions = custom_permissions or self.default_permissions.get(
            access_level, set()
        )

        user = User(
            username=username,
            email=email,
            access_level=access_level,
            permissions=permissions,
        )

        self.users[username] = user
        self._log_security_event(
            "user_created",
            username=username,
            success=True,
            details={"email": email, "access_level": access_level.value},
        )

        logger.info(f"Created user: {username} ({access_level.value})")
        return user

    def generate_api_key(self, username: str) -> str:
        """Generate API key for user."""
        if username not in self.users:
            raise ValueError(f"User {username} does not exist")

        # Generate secure random API key
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        self.api_keys[key_hash] = username

        self._log_security_event("api_key_generated", username=username, success=True)

        return api_key

    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate user by API key."""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            username = self.api_keys.get(key_hash)

            if username and username in self.users:
                user = self.users[username]
                if user.is_active and not self._is_user_locked(user):
                    user.last_login = datetime.now()
                    self._log_security_event(
                        "api_authentication", username=username, success=True
                    )
                    return user

            self._log_security_event(
                "api_authentication",
                username=username,
                success=False,
                details={"reason": "invalid_key_or_locked_user"},
            )
            return None

        except Exception as e:
            logger.error(f"API authentication error: {e}")
            return None

    def check_permission(self, user: User, operation: OperationType) -> bool:
        """Check if user has permission for operation."""
        if not user.is_active or self._is_user_locked(user):
            return False

        return operation in user.permissions

    def require_permission(self, user: User, operation: OperationType) -> None:
        """Require user to have permission for operation (raises exception if not)."""
        if not self.check_permission(user, operation):
            self._log_security_event(
                "unauthorized_access",
                username=user.username,
                success=False,
                details={"operation": operation.value},
            )
            raise PermissionError(
                f"User {user.username} does not have permission for {operation.value}"
            )

    def _is_user_locked(self, user: User) -> bool:
        """Check if user account is locked."""
        if user.locked_until is None:
            return False

        if datetime.now() > user.locked_until:
            # Unlock user
            user.locked_until = None
            user.failed_login_attempts = 0
            return False

        return True

    def record_failed_login(self, username: str) -> None:
        """Record failed login attempt and lock user if necessary."""
        if username in self.users:
            user = self.users[username]
            user.failed_login_attempts += 1

            # Lock user after 5 failed attempts for 15 minutes
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now() + timedelta(minutes=15)
                logger.warning(
                    f"User {username} locked due to repeated failed login attempts"
                )

        self._log_security_event("failed_login", username=username, success=False)

    def _log_security_event(
        self,
        event_type: str,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        operation: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log security event for audit trail."""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            username=username,
            ip_address=ip_address,
            operation=operation,
            success=success,
            details=details or {},
        )

        self.security_events.append(event)

        # Trim event history to last 10000 events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]

        # Log critical security events
        if not success or event_type in ["unauthorized_access", "failed_login"]:
            logger.warning(f"Security event: {event_type} - {username} - {success}")

    def get_security_summary(self) -> Dict[str, Any]:
        """Get security status summary."""
        recent_events = [
            e
            for e in self.security_events
            if e.timestamp > datetime.now() - timedelta(hours=24)
        ]

        failed_events = [e for e in recent_events if not e.success]

        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "locked_users": len(
                [u for u in self.users.values() if self._is_user_locked(u)]
            ),
            "api_keys_issued": len(self.api_keys),
            "events_last_24h": len(recent_events),
            "failed_events_last_24h": len(failed_events),
            "recent_failed_logins": len(
                [e for e in failed_events if e.event_type == "failed_login"]
            ),
            "unauthorized_attempts": len(
                [e for e in failed_events if e.event_type == "unauthorized_access"]
            ),
        }

    def export_audit_log(self, output_file: str, days: int = 30) -> None:
        """Export security audit log to file."""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            events = [e for e in self.security_events if e.timestamp > cutoff]

            audit_data = [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "username": event.username,
                    "ip_address": event.ip_address,
                    "operation": event.operation,
                    "success": event.success,
                    "details": event.details,
                }
                for event in events
            ]

            with open(output_file, "w") as f:
                json.dump(audit_data, f, indent=2)

            logger.info(f"Exported {len(audit_data)} audit events to {output_file}")

        except Exception as e:
            logger.error(f"Failed to export audit log: {e}")


class InputValidator:
    """Input validation and sanitization for security."""

    @staticmethod
    def validate_experiment_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize experiment parameters."""
        validated = {}

        # Define safe parameter ranges
        safe_ranges = {
            "precursor_A_conc": (0.01, 5.0),
            "precursor_B_conc": (0.01, 5.0),
            "temperature": (20, 500),  # Celsius
            "reaction_time": (0.1, 48),  # Hours
            "pH": (1, 14),
            "solvent_ratio": (0, 1),
        }

        for param, value in params.items():
            if param in safe_ranges:
                try:
                    # Convert to float and validate range
                    float_value = float(value)
                    min_val, max_val = safe_ranges[param]

                    if min_val <= float_value <= max_val:
                        validated[param] = float_value
                    else:
                        logger.warning(
                            f"Parameter {param} value {float_value} outside safe range {safe_ranges[param]}"
                        )
                        # Clamp to safe range
                        validated[param] = max(min_val, min(max_val, float_value))

                except (ValueError, TypeError):
                    logger.error(f"Invalid parameter value: {param} = {value}")
                    continue
            else:
                logger.warning(f"Unknown parameter: {param}")

        return validated

    @staticmethod
    def sanitize_string_input(input_str: str, max_length: int = 100) -> str:
        """Sanitize string input to prevent injection attacks."""
        if not isinstance(input_str, str):
            return ""

        # Remove potentially dangerous characters
        sanitized = "".join(c for c in input_str if c.isalnum() or c in " -._")

        # Limit length
        return sanitized[:max_length]

    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file path to prevent directory traversal."""
        try:
            path = Path(file_path).resolve()

            # Check for directory traversal attempts
            if ".." in str(path) or str(path).startswith("/"):
                return False

            # Only allow files in specific directories
            allowed_dirs = ["data", "results", "exports"]
            return any(
                str(path).startswith(allowed_dir) for allowed_dir in allowed_dirs
            )

        except Exception:
            return False


def create_security_manager(config_file: Optional[str] = None) -> SecurityManager:
    """Create and configure security manager."""
    return SecurityManager(config_file=config_file)
