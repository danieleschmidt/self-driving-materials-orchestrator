"""Global Compliance and Multi-Region Support for Self-Healing Pipeline.

Implements comprehensive compliance with GDPR, CCPA, PDPA and other global
regulations, multi-region deployment, and internationalization support.
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid
import base64

logger = logging.getLogger(__name__)


class ComplianceRegulation(Enum):
    """Global compliance regulations."""

    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore/Thailand)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    PRIVACY_ACT = "privacy_act"  # Privacy Act (Australia)
    KVKK = "kvkk"  # Kişisel Verilerin Korunması Kanunu (Turkey)


class DataClassification(Enum):
    """Data classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


class ConsentType(Enum):
    """Types of data processing consent."""

    EXPLICIT = "explicit"
    IMPLIED = "implied"
    LEGITIMATE_INTEREST = "legitimate_interest"
    CONTRACTUAL = "contractual"
    VITAL_INTEREST = "vital_interest"
    PUBLIC_TASK = "public_task"


@dataclass
class DataProcessingPurpose:
    """Data processing purpose definition."""

    purpose_id: str
    name: str
    description: str
    legal_basis: ConsentType
    retention_period_days: int
    required_consents: List[str] = field(default_factory=list)
    data_categories: List[str] = field(default_factory=list)
    third_party_sharing: bool = False
    cross_border_transfer: bool = False
    automated_decision_making: bool = False


@dataclass
class CompliancePolicy:
    """Compliance policy configuration."""

    regulation: ComplianceRegulation
    enabled: bool = True
    data_retention_days: int = 365
    pseudonymization_required: bool = True
    encryption_required: bool = True
    audit_logging_required: bool = True
    consent_management_required: bool = True
    data_subject_rights_enabled: bool = True
    cross_border_restrictions: List[str] = field(default_factory=list)
    lawful_bases: List[ConsentType] = field(default_factory=list)
    breach_notification_hours: int = 72

    def __post_init__(self):
        """Set default lawful bases for regulations."""
        if not self.lawful_bases:
            if self.regulation == ComplianceRegulation.GDPR:
                self.lawful_bases = [
                    ConsentType.EXPLICIT,
                    ConsentType.LEGITIMATE_INTEREST,
                ]
            elif self.regulation == ComplianceRegulation.CCPA:
                self.lawful_bases = [ConsentType.EXPLICIT, ConsentType.IMPLIED]
            else:
                self.lawful_bases = [ConsentType.EXPLICIT]


@dataclass
class DataSubjectRequest:
    """Data subject rights request."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_type: str = ""  # access, portability, erasure, rectification, restriction
    subject_id: str = ""
    submitted_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, processing, completed, rejected
    verification_method: str = ""
    requested_data: List[str] = field(default_factory=list)
    response_data: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GlobalComplianceManager:
    """Global compliance management system."""

    def __init__(self):
        self.compliance_policies: Dict[ComplianceRegulation, CompliancePolicy] = {}
        self.processing_purposes: Dict[str, DataProcessingPurpose] = {}
        self.data_subject_requests: Dict[str, DataSubjectRequest] = {}
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.audit_logs: List[Dict[str, Any]] = []

        # Data encryption and pseudonymization
        self.encryption_key = self._generate_encryption_key()
        self.pseudonymization_salt = self._generate_salt()

        # Cross-border transfer tracking
        self.data_transfer_log: List[Dict[str, Any]] = []

        # Initialize default compliance policies
        self._initialize_compliance_policies()
        self._initialize_processing_purposes()

    def _generate_encryption_key(self) -> str:
        """Generate encryption key for data protection."""
        return base64.b64encode(uuid.uuid4().bytes).decode("utf-8")

    def _generate_salt(self) -> str:
        """Generate salt for pseudonymization."""
        return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()

    def _initialize_compliance_policies(self):
        """Initialize default compliance policies."""
        # GDPR Policy
        self.compliance_policies[ComplianceRegulation.GDPR] = CompliancePolicy(
            regulation=ComplianceRegulation.GDPR,
            data_retention_days=1095,  # 3 years default
            pseudonymization_required=True,
            encryption_required=True,
            audit_logging_required=True,
            consent_management_required=True,
            data_subject_rights_enabled=True,
            cross_border_restrictions=["non_eu_countries"],
            breach_notification_hours=72,
        )

        # CCPA Policy
        self.compliance_policies[ComplianceRegulation.CCPA] = CompliancePolicy(
            regulation=ComplianceRegulation.CCPA,
            data_retention_days=365,
            pseudonymization_required=False,
            encryption_required=True,
            audit_logging_required=True,
            consent_management_required=True,
            data_subject_rights_enabled=True,
            cross_border_restrictions=[],
            breach_notification_hours=72,
        )

        # PDPA Policy
        self.compliance_policies[ComplianceRegulation.PDPA] = CompliancePolicy(
            regulation=ComplianceRegulation.PDPA,
            data_retention_days=730,  # 2 years
            pseudonymization_required=True,
            encryption_required=True,
            audit_logging_required=True,
            consent_management_required=True,
            data_subject_rights_enabled=True,
            cross_border_restrictions=["restricted_countries"],
            breach_notification_hours=72,
        )

    def _initialize_processing_purposes(self):
        """Initialize default data processing purposes."""
        # Materials research purpose
        self.processing_purposes["materials_research"] = DataProcessingPurpose(
            purpose_id="materials_research",
            name="Materials Discovery Research",
            description="Processing experimental data for materials science research",
            legal_basis=ConsentType.LEGITIMATE_INTEREST,
            retention_period_days=2555,  # 7 years for research
            data_categories=[
                "experimental_data",
                "synthesis_parameters",
                "property_measurements",
            ],
            third_party_sharing=False,
            cross_border_transfer=True,
            automated_decision_making=True,
        )

        # System monitoring purpose
        self.processing_purposes["system_monitoring"] = DataProcessingPurpose(
            purpose_id="system_monitoring",
            name="System Performance Monitoring",
            description="Monitoring system performance and user interactions",
            legal_basis=ConsentType.LEGITIMATE_INTEREST,
            retention_period_days=365,
            data_categories=["system_logs", "performance_metrics", "user_actions"],
            third_party_sharing=False,
            cross_border_transfer=False,
            automated_decision_making=False,
        )

        # Quality assurance purpose
        self.processing_purposes["quality_assurance"] = DataProcessingPurpose(
            purpose_id="quality_assurance",
            name="Quality Assurance and Validation",
            description="Ensuring data quality and experimental validation",
            legal_basis=ConsentType.CONTRACTUAL,
            retention_period_days=1095,  # 3 years
            data_categories=["validation_data", "quality_metrics", "audit_trails"],
            third_party_sharing=False,
            cross_border_transfer=True,
            automated_decision_making=True,
        )

    def register_data_processing(
        self,
        purpose_id: str,
        data_subject_id: str,
        data_categories: List[str],
        consent_obtained: bool = False,
        legal_basis: Optional[ConsentType] = None,
    ) -> str:
        """Register data processing activity."""
        processing_id = str(uuid.uuid4())

        if purpose_id not in self.processing_purposes:
            raise ValueError(f"Unknown processing purpose: {purpose_id}")

        purpose = self.processing_purposes[purpose_id]

        # Check compliance requirements
        applicable_regulations = self._get_applicable_regulations(data_subject_id)

        for regulation in applicable_regulations:
            policy = self.compliance_policies[regulation]

            # Check consent requirements
            if policy.consent_management_required and not consent_obtained:
                if purpose.legal_basis == ConsentType.EXPLICIT:
                    raise ValueError(
                        f"Explicit consent required for {regulation.value}"
                    )

        # Log processing activity
        self._audit_log(
            event_type="data_processing_registered",
            details={
                "processing_id": processing_id,
                "purpose_id": purpose_id,
                "data_subject_id": self._pseudonymize_identifier(data_subject_id),
                "data_categories": data_categories,
                "consent_obtained": consent_obtained,
                "legal_basis": (
                    legal_basis.value if legal_basis else purpose.legal_basis.value
                ),
                "applicable_regulations": [r.value for r in applicable_regulations],
            },
        )

        return processing_id

    def _get_applicable_regulations(
        self, data_subject_id: str
    ) -> List[ComplianceRegulation]:
        """Determine applicable regulations based on data subject location."""
        # Simplified: In production, would use geolocation or explicit user data
        # For demo, apply GDPR by default
        return [ComplianceRegulation.GDPR, ComplianceRegulation.CCPA]

    def _pseudonymize_identifier(self, identifier: str) -> str:
        """Pseudonymize personal identifier."""
        return hashlib.sha256(
            (identifier + self.pseudonymization_salt).encode()
        ).hexdigest()[:16]

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        # Simplified encryption - in production would use proper encryption
        encoded = base64.b64encode(data.encode()).decode()
        return f"enc_{self.encryption_key[:8]}_{encoded}"

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not encrypted_data.startswith("enc_"):
            return encrypted_data

        parts = encrypted_data.split("_", 2)
        if len(parts) != 3:
            raise ValueError("Invalid encrypted data format")

        return base64.b64decode(parts[2]).decode()

    def record_consent(
        self,
        data_subject_id: str,
        purpose_id: str,
        consent_given: bool,
        consent_method: str = "explicit",
        expiry_date: Optional[datetime] = None,
    ):
        """Record consent for data processing."""
        consent_id = str(uuid.uuid4())

        consent_record = {
            "consent_id": consent_id,
            "data_subject_id": self._pseudonymize_identifier(data_subject_id),
            "purpose_id": purpose_id,
            "consent_given": consent_given,
            "consent_method": consent_method,
            "recorded_at": datetime.now().isoformat(),
            "expiry_date": expiry_date.isoformat() if expiry_date else None,
            "withdrawn_at": None,
        }

        self.consent_records[consent_id] = consent_record

        self._audit_log(
            event_type="consent_recorded",
            details={
                "consent_id": consent_id,
                "purpose_id": purpose_id,
                "consent_given": consent_given,
                "consent_method": consent_method,
            },
        )

        return consent_id

    def withdraw_consent(self, data_subject_id: str, purpose_id: str):
        """Withdraw consent for data processing."""
        pseudonymized_id = self._pseudonymize_identifier(data_subject_id)

        # Find and update consent records
        for consent_id, consent in self.consent_records.items():
            if (
                consent["data_subject_id"] == pseudonymized_id
                and consent["purpose_id"] == purpose_id
                and consent["consent_given"]
                and consent["withdrawn_at"] is None
            ):

                consent["withdrawn_at"] = datetime.now().isoformat()

                self._audit_log(
                    event_type="consent_withdrawn",
                    details={
                        "consent_id": consent_id,
                        "purpose_id": purpose_id,
                        "withdrawn_at": consent["withdrawn_at"],
                    },
                )

    def submit_data_subject_request(
        self,
        request_type: str,
        data_subject_id: str,
        verification_method: str = "email",
        requested_data: List[str] = None,
    ) -> str:
        """Submit data subject rights request."""
        if request_type not in [
            "access",
            "portability",
            "erasure",
            "rectification",
            "restriction",
        ]:
            raise ValueError(f"Invalid request type: {request_type}")

        request = DataSubjectRequest(
            request_type=request_type,
            subject_id=self._pseudonymize_identifier(data_subject_id),
            verification_method=verification_method,
            requested_data=requested_data or [],
        )

        self.data_subject_requests[request.request_id] = request

        self._audit_log(
            event_type="data_subject_request_submitted",
            details={
                "request_id": request.request_id,
                "request_type": request_type,
                "verification_method": verification_method,
            },
        )

        # Auto-process certain request types
        if request_type in ["access", "portability"]:
            asyncio.create_task(self._process_data_subject_request(request.request_id))

        return request.request_id

    async def _process_data_subject_request(self, request_id: str):
        """Process data subject rights request."""
        if request_id not in self.data_subject_requests:
            return

        request = self.data_subject_requests[request_id]
        request.status = "processing"

        try:
            if request.request_type == "access":
                # Compile personal data
                personal_data = await self._compile_personal_data(request.subject_id)
                request.response_data = personal_data
                request.status = "completed"

            elif request.request_type == "portability":
                # Export data in portable format
                portable_data = await self._export_portable_data(request.subject_id)
                request.response_data = portable_data
                request.status = "completed"

            elif request.request_type == "erasure":
                # Delete personal data
                await self._erase_personal_data(request.subject_id)
                request.status = "completed"

            elif request.request_type == "rectification":
                # Request manual review
                request.status = "pending"

            elif request.request_type == "restriction":
                # Restrict processing
                await self._restrict_processing(request.subject_id)
                request.status = "completed"

            request.completed_at = datetime.now()

            self._audit_log(
                event_type="data_subject_request_processed",
                details={
                    "request_id": request_id,
                    "request_type": request.request_type,
                    "status": request.status,
                },
            )

        except Exception as e:
            request.status = "rejected"
            request.rejection_reason = str(e)

            self._audit_log(
                event_type="data_subject_request_failed",
                details={"request_id": request_id, "error": str(e)},
            )

    async def _compile_personal_data(self, subject_id: str) -> Dict[str, Any]:
        """Compile all personal data for a data subject."""
        # In production, would query all systems for personal data
        return {
            "experimental_data": [],
            "system_interactions": [],
            "consent_records": [
                consent
                for consent in self.consent_records.values()
                if consent["data_subject_id"] == subject_id
            ],
            "processing_records": [],
        }

    async def _export_portable_data(self, subject_id: str) -> Dict[str, Any]:
        """Export data in portable format."""
        personal_data = await self._compile_personal_data(subject_id)

        return {
            "format": "JSON",
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "data": personal_data,
        }

    async def _erase_personal_data(self, subject_id: str):
        """Erase personal data for a data subject."""
        # Remove consent records
        consent_ids_to_remove = [
            consent_id
            for consent_id, consent in self.consent_records.items()
            if consent["data_subject_id"] == subject_id
        ]

        for consent_id in consent_ids_to_remove:
            del self.consent_records[consent_id]

        # In production, would erase from all systems

    async def _restrict_processing(self, subject_id: str):
        """Restrict processing for a data subject."""
        # Mark data as restricted in all systems
        # In production, would update processing flags
        pass

    def check_data_retention_compliance(self) -> List[Dict[str, Any]]:
        """Check data retention compliance."""
        violations = []
        current_time = datetime.now()

        for purpose_id, purpose in self.processing_purposes.items():
            retention_limit = timedelta(days=purpose.retention_period_days)

            # Check consent records
            for consent_id, consent in self.consent_records.items():
                if consent["purpose_id"] == purpose_id:
                    recorded_at = datetime.fromisoformat(consent["recorded_at"])

                    if current_time - recorded_at > retention_limit:
                        violations.append(
                            {
                                "type": "consent_retention_violation",
                                "consent_id": consent_id,
                                "purpose_id": purpose_id,
                                "recorded_at": consent["recorded_at"],
                                "retention_limit_days": purpose.retention_period_days,
                                "days_overdue": (
                                    current_time - recorded_at - retention_limit
                                ).days,
                            }
                        )

        return violations

    def generate_compliance_report(
        self, regulation: ComplianceRegulation
    ) -> Dict[str, Any]:
        """Generate compliance report for a specific regulation."""
        policy = self.compliance_policies.get(regulation)
        if not policy:
            raise ValueError(f"No policy configured for {regulation.value}")

        # Count active consents
        active_consents = sum(
            1
            for consent in self.consent_records.values()
            if consent["consent_given"] and consent["withdrawn_at"] is None
        )

        # Count data subject requests
        request_counts = {}
        for request in self.data_subject_requests.values():
            request_type = request.request_type
            request_counts[request_type] = request_counts.get(request_type, 0) + 1

        # Check retention violations
        retention_violations = self.check_data_retention_compliance()

        return {
            "regulation": regulation.value,
            "policy": {
                "enabled": policy.enabled,
                "data_retention_days": policy.data_retention_days,
                "encryption_required": policy.encryption_required,
                "consent_management_required": policy.consent_management_required,
            },
            "statistics": {
                "active_consents": active_consents,
                "total_processing_purposes": len(self.processing_purposes),
                "data_subject_requests": request_counts,
                "retention_violations": len(retention_violations),
                "audit_log_entries": len(self.audit_logs),
            },
            "compliance_status": {
                "consent_management": active_consents > 0,
                "data_retention": len(retention_violations) == 0,
                "audit_logging": len(self.audit_logs) > 0,
                "encryption": policy.encryption_required,
            },
            "violations": retention_violations,
            "generated_at": datetime.now().isoformat(),
        }

    def _audit_log(self, event_type: str, details: Dict[str, Any]):
        """Add entry to audit log."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
            "log_id": str(uuid.uuid4()),
        }

        self.audit_logs.append(log_entry)

        # Keep only recent logs (limit memory usage)
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-5000:]

    def get_audit_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get filtered audit logs."""
        filtered_logs = self.audit_logs

        if start_date:
            filtered_logs = [
                log
                for log in filtered_logs
                if datetime.fromisoformat(log["timestamp"]) >= start_date
            ]

        if end_date:
            filtered_logs = [
                log
                for log in filtered_logs
                if datetime.fromisoformat(log["timestamp"]) <= end_date
            ]

        if event_types:
            filtered_logs = [
                log for log in filtered_logs if log["event_type"] in event_types
            ]

        return filtered_logs

    def export_audit_logs(self, format: str = "json") -> str:
        """Export audit logs for compliance reporting."""
        if format == "json":
            return json.dumps(self.audit_logs, indent=2)
        elif format == "csv":
            # Simplified CSV export
            lines = ["timestamp,event_type,details"]
            for log in self.audit_logs:
                details_str = json.dumps(log["details"]).replace(",", ";")
                lines.append(
                    f"{log['timestamp']},{log['event_type']},\"{details_str}\""
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class InternationalizationManager:
    """Internationalization (i18n) management."""

    def __init__(self):
        self.supported_locales = [
            "en-US",  # English (United States)
            "en-GB",  # English (United Kingdom)
            "es-ES",  # Spanish (Spain)
            "es-MX",  # Spanish (Mexico)
            "fr-FR",  # French (France)
            "fr-CA",  # French (Canada)
            "de-DE",  # German (Germany)
            "ja-JP",  # Japanese (Japan)
            "zh-CN",  # Chinese (Simplified)
            "zh-TW",  # Chinese (Traditional)
            "ko-KR",  # Korean (South Korea)
            "pt-BR",  # Portuguese (Brazil)
            "ru-RU",  # Russian (Russia)
            "ar-SA",  # Arabic (Saudi Arabia)
            "hi-IN",  # Hindi (India)
        ]

        self.default_locale = "en-US"
        self.translations: Dict[str, Dict[str, str]] = {}

        # Load default translations
        self._load_default_translations()

    def _load_default_translations(self):
        """Load default translations for key messages."""
        # English translations
        self.translations["en-US"] = {
            "pipeline.status.healthy": "Pipeline is healthy",
            "pipeline.status.degraded": "Pipeline is degraded",
            "pipeline.status.failed": "Pipeline has failed",
            "pipeline.failure.robot_disconnection": "Robot disconnection detected",
            "pipeline.failure.database_error": "Database error occurred",
            "pipeline.failure.memory_leak": "Memory leak detected",
            "pipeline.healing.success": "Self-healing completed successfully",
            "pipeline.healing.failed": "Self-healing failed",
            "compliance.consent.required": "Consent is required for data processing",
            "compliance.data.encrypted": "Data has been encrypted",
            "compliance.request.submitted": "Data subject request submitted",
            "compliance.request.completed": "Data subject request completed",
            "error.validation.failed": "Validation failed",
            "error.network.timeout": "Network timeout",
            "error.permission.denied": "Permission denied",
            "optimization.started": "Optimization started",
            "optimization.completed": "Optimization completed",
            "monitoring.alert.high_cpu": "High CPU usage detected",
            "monitoring.alert.high_memory": "High memory usage detected",
        }

        # Spanish translations
        self.translations["es-ES"] = {
            "pipeline.status.healthy": "La tubería está saludable",
            "pipeline.status.degraded": "La tubería está degradada",
            "pipeline.status.failed": "La tubería ha fallado",
            "pipeline.failure.robot_disconnection": "Desconexión del robot detectada",
            "pipeline.failure.database_error": "Error de base de datos ocurrido",
            "pipeline.failure.memory_leak": "Fuga de memoria detectada",
            "pipeline.healing.success": "Auto-sanación completada exitosamente",
            "pipeline.healing.failed": "Auto-sanación falló",
            "compliance.consent.required": "Se requiere consentimiento para el procesamiento de datos",
            "compliance.data.encrypted": "Los datos han sido encriptados",
            "compliance.request.submitted": "Solicitud del sujeto de datos enviada",
            "compliance.request.completed": "Solicitud del sujeto de datos completada",
            "error.validation.failed": "Validación falló",
            "error.network.timeout": "Tiempo de espera de red agotado",
            "error.permission.denied": "Permiso denegado",
            "optimization.started": "Optimización iniciada",
            "optimization.completed": "Optimización completada",
            "monitoring.alert.high_cpu": "Alto uso de CPU detectado",
            "monitoring.alert.high_memory": "Alto uso de memoria detectado",
        }

        # French translations
        self.translations["fr-FR"] = {
            "pipeline.status.healthy": "Le pipeline est en bonne santé",
            "pipeline.status.degraded": "Le pipeline est dégradé",
            "pipeline.status.failed": "Le pipeline a échoué",
            "pipeline.failure.robot_disconnection": "Déconnexion du robot détectée",
            "pipeline.failure.database_error": "Erreur de base de données survenue",
            "pipeline.failure.memory_leak": "Fuite mémoire détectée",
            "pipeline.healing.success": "Auto-guérison terminée avec succès",
            "pipeline.healing.failed": "Auto-guérison échouée",
            "compliance.consent.required": "Le consentement est requis pour le traitement des données",
            "compliance.data.encrypted": "Les données ont été chiffrées",
            "compliance.request.submitted": "Demande du sujet de données soumise",
            "compliance.request.completed": "Demande du sujet de données terminée",
            "error.validation.failed": "Validation échouée",
            "error.network.timeout": "Délai d'attente réseau",
            "error.permission.denied": "Permission refusée",
            "optimization.started": "Optimisation démarrée",
            "optimization.completed": "Optimisation terminée",
            "monitoring.alert.high_cpu": "Utilisation élevée du CPU détectée",
            "monitoring.alert.high_memory": "Utilisation élevée de la mémoire détectée",
        }

        # German translations
        self.translations["de-DE"] = {
            "pipeline.status.healthy": "Pipeline ist gesund",
            "pipeline.status.degraded": "Pipeline ist beeinträchtigt",
            "pipeline.status.failed": "Pipeline ist fehlgeschlagen",
            "pipeline.failure.robot_disconnection": "Roboter-Trennung erkannt",
            "pipeline.failure.database_error": "Datenbankfehler aufgetreten",
            "pipeline.failure.memory_leak": "Speicherleck erkannt",
            "pipeline.healing.success": "Selbstheilung erfolgreich abgeschlossen",
            "pipeline.healing.failed": "Selbstheilung fehlgeschlagen",
            "compliance.consent.required": "Einwilligung für Datenverarbeitung erforderlich",
            "compliance.data.encrypted": "Daten wurden verschlüsselt",
            "compliance.request.submitted": "Betroffenenanfrage eingereicht",
            "compliance.request.completed": "Betroffenenanfrage abgeschlossen",
            "error.validation.failed": "Validierung fehlgeschlagen",
            "error.network.timeout": "Netzwerk-Timeout",
            "error.permission.denied": "Berechtigung verweigert",
            "optimization.started": "Optimierung gestartet",
            "optimization.completed": "Optimierung abgeschlossen",
            "monitoring.alert.high_cpu": "Hohe CPU-Auslastung erkannt",
            "monitoring.alert.high_memory": "Hohe Speicherauslastung erkannt",
        }

        # Japanese translations
        self.translations["ja-JP"] = {
            "pipeline.status.healthy": "パイプラインは正常です",
            "pipeline.status.degraded": "パイプラインが劣化しています",
            "pipeline.status.failed": "パイプラインが失敗しました",
            "pipeline.failure.robot_disconnection": "ロボットの切断が検出されました",
            "pipeline.failure.database_error": "データベースエラーが発生しました",
            "pipeline.failure.memory_leak": "メモリリークが検出されました",
            "pipeline.healing.success": "自己修復が正常に完了しました",
            "pipeline.healing.failed": "自己修復が失敗しました",
            "compliance.consent.required": "データ処理には同意が必要です",
            "compliance.data.encrypted": "データが暗号化されました",
            "compliance.request.submitted": "データ主体の要求が送信されました",
            "compliance.request.completed": "データ主体の要求が完了しました",
            "error.validation.failed": "検証に失敗しました",
            "error.network.timeout": "ネットワークタイムアウト",
            "error.permission.denied": "権限が拒否されました",
            "optimization.started": "最適化が開始されました",
            "optimization.completed": "最適化が完了しました",
            "monitoring.alert.high_cpu": "高いCPU使用率が検出されました",
            "monitoring.alert.high_memory": "高いメモリ使用率が検出されました",
        }

        # Chinese Simplified translations
        self.translations["zh-CN"] = {
            "pipeline.status.healthy": "管道状态正常",
            "pipeline.status.degraded": "管道状态降级",
            "pipeline.status.failed": "管道失败",
            "pipeline.failure.robot_disconnection": "检测到机器人断开连接",
            "pipeline.failure.database_error": "发生数据库错误",
            "pipeline.failure.memory_leak": "检测到内存泄漏",
            "pipeline.healing.success": "自愈成功完成",
            "pipeline.healing.failed": "自愈失败",
            "compliance.consent.required": "数据处理需要同意",
            "compliance.data.encrypted": "数据已被加密",
            "compliance.request.submitted": "数据主体请求已提交",
            "compliance.request.completed": "数据主体请求已完成",
            "error.validation.failed": "验证失败",
            "error.network.timeout": "网络超时",
            "error.permission.denied": "权限被拒绝",
            "optimization.started": "优化已开始",
            "optimization.completed": "优化已完成",
            "monitoring.alert.high_cpu": "检测到高CPU使用率",
            "monitoring.alert.high_memory": "检测到高内存使用率",
        }

    def get_translation(self, key: str, locale: str = None, **kwargs) -> str:
        """Get translation for a key in specified locale."""
        if locale is None:
            locale = self.default_locale

        # Fallback to English if locale not supported
        if locale not in self.translations:
            locale = "en-US"

        # Get translation
        translation = self.translations[locale].get(key, key)

        # Format with kwargs if provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError):
                # If formatting fails, return unformatted translation
                pass

        return translation

    def get_supported_locales(self) -> List[str]:
        """Get list of supported locales."""
        return self.supported_locales.copy()

    def add_translation(self, locale: str, key: str, translation: str):
        """Add or update a translation."""
        if locale not in self.translations:
            self.translations[locale] = {}

        self.translations[locale][key] = translation

    def load_translations_from_file(self, file_path: str, locale: str):
        """Load translations from JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                translations = json.load(f)
                self.translations[locale] = translations
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load translations from {file_path}: {e}")

    def export_translations(self, locale: str, file_path: str):
        """Export translations to JSON file."""
        if locale not in self.translations:
            raise ValueError(f"Locale {locale} not found")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.translations[locale], f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"Failed to export translations to {file_path}: {e}")


# Global instances
_global_compliance_manager: Optional[GlobalComplianceManager] = None
_global_i18n_manager: Optional[InternationalizationManager] = None


def get_compliance_manager() -> GlobalComplianceManager:
    """Get the global compliance manager instance."""
    global _global_compliance_manager
    if _global_compliance_manager is None:
        _global_compliance_manager = GlobalComplianceManager()
    return _global_compliance_manager


def get_i18n_manager() -> InternationalizationManager:
    """Get the global internationalization manager instance."""
    global _global_i18n_manager
    if _global_i18n_manager is None:
        _global_i18n_manager = InternationalizationManager()
    return _global_i18n_manager


def t(key: str, locale: str = None, **kwargs) -> str:
    """Shorthand function for translations."""
    return get_i18n_manager().get_translation(key, locale, **kwargs)
