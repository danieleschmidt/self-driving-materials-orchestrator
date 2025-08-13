"""Global deployment and multi-region support for materials discovery."""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)

class Region(Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    ASIA_PACIFIC_1 = "ap-southeast-1"
    ASIA_PACIFIC_2 = "ap-northeast-1"

class ComplianceStandard(Enum):
    """Compliance standards."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    PDPA = "pdpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    HIPAA = "hipaa"

@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region: Region
    data_center: str
    compliance_standards: List[ComplianceStandard]
    data_residency_required: bool = False
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    audit_logging: bool = True
    backup_region: Optional[Region] = None
    max_latency_ms: int = 100
    preferred_languages: List[str] = field(default_factory=lambda: ["en"])

@dataclass
class GlobalConfig:
    """Global deployment configuration."""
    primary_region: Region
    regions: Dict[Region, RegionConfig]
    load_balancing_strategy: str = "latency_based"
    cross_region_replication: bool = True
    global_encryption_key: Optional[str] = None
    compliance_mode: str = "strict"
    data_classification: str = "confidential"

class GlobalDeploymentManager:
    """Manages global deployment and compliance."""
    
    def __init__(self, config: GlobalConfig):
        """Initialize global deployment manager."""
        self.config = config
        self.active_regions = {}
        self.compliance_validators = {}
        self.monitoring_enabled = True
        
        # Initialize region-specific configurations
        self._initialize_regions()
        
        # Setup compliance monitoring
        self._setup_compliance_monitoring()
        
        logger.info(f"Global deployment manager initialized for {len(self.config.regions)} regions")
    
    def _initialize_regions(self):
        """Initialize region-specific configurations."""
        for region, region_config in self.config.regions.items():
            try:
                # Initialize region
                self.active_regions[region] = {
                    'status': 'active',
                    'config': region_config,
                    'last_health_check': datetime.now(timezone.utc),
                    'latency': 0,
                    'experiment_count': 0,
                    'compliance_status': 'compliant'
                }
                
                # Setup region-specific compliance
                self._setup_region_compliance(region, region_config)
                
                logger.info(f"Region {region.value} initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize region {region.value}: {e}")
    
    def _setup_region_compliance(self, region: Region, config: RegionConfig):
        """Setup compliance for a specific region."""
        compliance_config = {
            'data_residency': config.data_residency_required,
            'encryption_at_rest': config.encryption_at_rest,
            'encryption_in_transit': config.encryption_in_transit,
            'audit_logging': config.audit_logging,
            'standards': [std.value for std in config.compliance_standards]
        }
        
        # GDPR-specific configurations
        if ComplianceStandard.GDPR in config.compliance_standards:
            compliance_config.update({
                'right_to_deletion': True,
                'data_portability': True,
                'consent_management': True,
                'privacy_by_design': True,
                'data_protection_officer': True
            })
        
        # CCPA-specific configurations
        if ComplianceStandard.CCPA in config.compliance_standards:
            compliance_config.update({
                'consumer_rights': True,
                'opt_out_mechanisms': True,
                'data_disclosure': True,
                'non_discrimination': True
            })
        
        # PDPA-specific configurations
        if ComplianceStandard.PDPA in config.compliance_standards:
            compliance_config.update({
                'notification_requirements': True,
                'consent_withdrawal': True,
                'data_breach_notification': True
            })
        
        self.compliance_validators[region] = compliance_config
        logger.info(f"Compliance configured for {region.value}: {compliance_config['standards']}")
    
    def _setup_compliance_monitoring(self):
        """Setup continuous compliance monitoring."""
        self._compliance_thread = threading.Thread(target=self._compliance_monitor_loop, daemon=True)
        self._compliance_thread.start()
        
    def _compliance_monitor_loop(self):
        """Continuous compliance monitoring loop."""
        while self.monitoring_enabled:
            try:
                for region in self.active_regions:
                    self._check_region_compliance(region)
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Compliance monitoring error: {e}")
                time.sleep(60)
    
    def _check_region_compliance(self, region: Region):
        """Check compliance for a specific region."""
        try:
            region_data = self.active_regions[region]
            compliance_config = self.compliance_validators[region]
            
            # Check encryption requirements
            if compliance_config['encryption_at_rest']:
                # Verify data encryption at rest
                pass  # Implementation would check actual encryption status
            
            if compliance_config['encryption_in_transit']:
                # Verify data encryption in transit
                pass  # Implementation would check TLS/SSL status
            
            # Check audit logging
            if compliance_config['audit_logging']:
                # Verify audit logs are being generated
                pass  # Implementation would check log generation
            
            # Update compliance status
            region_data['compliance_status'] = 'compliant'
            region_data['last_compliance_check'] = datetime.now(timezone.utc)
            
        except Exception as e:
            self.active_regions[region]['compliance_status'] = 'non_compliant'
            logger.error(f"Compliance check failed for {region.value}: {e}")
    
    def route_experiment(self, user_location: str, experiment_data: Dict[str, Any]) -> Region:
        """Route experiment to optimal region based on user location and compliance."""
        # Determine optimal region based on user location
        if user_location.startswith('us'):
            preferred_regions = [Region.US_EAST_1, Region.US_WEST_2]
        elif user_location.startswith('eu'):
            preferred_regions = [Region.EU_WEST_1, Region.EU_CENTRAL_1]
        elif user_location.startswith('ap'):
            preferred_regions = [Region.ASIA_PACIFIC_1, Region.ASIA_PACIFIC_2]
        else:
            preferred_regions = [self.config.primary_region]
        
        # Check compliance requirements
        data_sensitivity = experiment_data.get('data_classification', 'public')
        requires_residency = data_sensitivity in ['confidential', 'restricted']
        
        # Find best available region
        for region in preferred_regions:
            if region in self.active_regions:
                region_data = self.active_regions[region]
                
                # Check region health
                if region_data['status'] != 'active':
                    continue
                
                # Check compliance
                if region_data['compliance_status'] != 'compliant':
                    continue
                
                # Check data residency requirements
                region_config = region_data['config']
                if requires_residency and not region_config.data_residency_required:
                    # Find region with data residency capability
                    continue
                
                return region
        
        # Fallback to primary region
        return self.config.primary_region
    
    def validate_data_transfer(self, source_region: Region, target_region: Region, 
                             data_classification: str) -> Tuple[bool, str]:
        """Validate data transfer between regions."""
        source_config = self.config.regions.get(source_region)
        target_config = self.config.regions.get(target_region)
        
        if not source_config or not target_config:
            return False, "Invalid source or target region"
        
        # Check if cross-border transfer is allowed
        if data_classification in ['confidential', 'restricted']:
            # Check if both regions have appropriate compliance
            source_standards = set(std.value for std in source_config.compliance_standards)
            target_standards = set(std.value for std in target_config.compliance_standards)
            
            # Must have compatible compliance standards
            if not source_standards.intersection(target_standards):
                return False, "Incompatible compliance standards for data transfer"
        
        # Check encryption requirements
        if not (source_config.encryption_in_transit and target_config.encryption_in_transit):
            return False, "Encryption in transit required for cross-region transfer"
        
        return True, "Data transfer approved"
    
    def get_region_status(self) -> Dict[str, Any]:
        """Get status of all regions."""
        status = {
            'primary_region': self.config.primary_region.value,
            'total_regions': len(self.active_regions),
            'healthy_regions': 0,
            'compliant_regions': 0,
            'regions': {}
        }
        
        for region, region_data in self.active_regions.items():
            region_status = {
                'status': region_data['status'],
                'compliance_status': region_data['compliance_status'],
                'last_health_check': region_data['last_health_check'].isoformat(),
                'experiment_count': region_data['experiment_count'],
                'latency_ms': region_data['latency'],
                'data_center': region_data['config'].data_center,
                'compliance_standards': [std.value for std in region_data['config'].compliance_standards]
            }
            
            status['regions'][region.value] = region_status
            
            if region_data['status'] == 'active':
                status['healthy_regions'] += 1
            
            if region_data['compliance_status'] == 'compliant':
                status['compliant_regions'] += 1
        
        return status
    
    def handle_data_subject_request(self, request_type: str, user_id: str, 
                                  region: Region) -> Dict[str, Any]:
        """Handle data subject requests (GDPR, CCPA, etc.)."""
        region_config = self.config.regions.get(region)
        if not region_config:
            return {'success': False, 'error': 'Invalid region'}
        
        compliance_standards = [std.value for std in region_config.compliance_standards]
        
        if request_type == 'data_export':
            # Handle data portability request
            if 'gdpr' in compliance_standards or 'ccpa' in compliance_standards:
                return self._export_user_data(user_id, region)
            else:
                return {'success': False, 'error': 'Data export not supported in this region'}
        
        elif request_type == 'data_deletion':
            # Handle right to be forgotten
            if 'gdpr' in compliance_standards:
                return self._delete_user_data(user_id, region)
            else:
                return {'success': False, 'error': 'Data deletion not supported in this region'}
        
        elif request_type == 'consent_withdrawal':
            # Handle consent withdrawal
            return self._withdraw_consent(user_id, region)
        
        else:
            return {'success': False, 'error': f'Unknown request type: {request_type}'}
    
    def _export_user_data(self, user_id: str, region: Region) -> Dict[str, Any]:
        """Export all user data for portability."""
        try:
            # Implementation would collect all user data
            export_data = {
                'user_id': user_id,
                'region': region.value,
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'experiments': [],  # Would contain actual experiment data
                'preferences': {},  # Would contain user preferences
                'analytics': {}     # Would contain aggregated analytics
            }
            
            # Generate secure download link
            download_link = f"https://secure-export.materials-orchestrator.com/download/{user_id}"
            
            return {
                'success': True,
                'download_link': download_link,
                'expires_at': (datetime.now(timezone.utc)).isoformat(),
                'data_summary': {
                    'total_experiments': len(export_data['experiments']),
                    'data_size_mb': 1.5  # Estimated size
                }
            }
            
        except Exception as e:
            logger.error(f"Data export failed for user {user_id}: {e}")
            return {'success': False, 'error': 'Export failed'}
    
    def _delete_user_data(self, user_id: str, region: Region) -> Dict[str, Any]:
        """Delete all user data (right to be forgotten)."""
        try:
            # Implementation would delete all user data
            deleted_items = {
                'experiments': 0,     # Count of deleted experiments
                'preferences': 0,     # Count of deleted preferences
                'analytics': 0,       # Count of deleted analytics
                'audit_logs': 0       # Count of anonymized audit logs
            }
            
            # Log the deletion for audit purposes
            audit_entry = {
                'action': 'data_deletion',
                'user_id': user_id,
                'region': region.value,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'deleted_items': deleted_items
            }
            
            return {
                'success': True,
                'deletion_summary': deleted_items,
                'audit_reference': f"DEL-{user_id}-{int(time.time())}"
            }
            
        except Exception as e:
            logger.error(f"Data deletion failed for user {user_id}: {e}")
            return {'success': False, 'error': 'Deletion failed'}
    
    def _withdraw_consent(self, user_id: str, region: Region) -> Dict[str, Any]:
        """Withdraw user consent for data processing."""
        try:
            # Implementation would update consent status
            consent_record = {
                'user_id': user_id,
                'region': region.value,
                'consent_withdrawn_at': datetime.now(timezone.utc).isoformat(),
                'previous_consents': [],  # List of previously given consents
                'data_processing_stopped': True
            }
            
            return {
                'success': True,
                'consent_status': 'withdrawn',
                'effective_date': consent_record['consent_withdrawn_at']
            }
            
        except Exception as e:
            logger.error(f"Consent withdrawal failed for user {user_id}: {e}")
            return {'success': False, 'error': 'Consent withdrawal failed'}
    
    def generate_compliance_report(self, region: Optional[Region] = None) -> Dict[str, Any]:
        """Generate compliance report."""
        if region:
            regions_to_report = [region]
        else:
            regions_to_report = list(self.active_regions.keys())
        
        report = {
            'report_timestamp': datetime.now(timezone.utc).isoformat(),
            'report_type': 'compliance_status',
            'regions': {},
            'summary': {
                'total_regions': len(regions_to_report),
                'compliant_regions': 0,
                'non_compliant_regions': 0,
                'compliance_standards_coverage': {}
            }
        }
        
        all_standards = set()
        
        for region in regions_to_report:
            region_data = self.active_regions[region]
            region_config = region_data['config']
            
            # Collect compliance information
            standards = [std.value for std in region_config.compliance_standards]
            all_standards.update(standards)
            
            region_report = {
                'region': region.value,
                'compliance_status': region_data['compliance_status'],
                'compliance_standards': standards,
                'data_residency_required': region_config.data_residency_required,
                'encryption_at_rest': region_config.encryption_at_rest,
                'encryption_in_transit': region_config.encryption_in_transit,
                'audit_logging': region_config.audit_logging,
                'last_compliance_check': region_data.get('last_compliance_check', '').isoformat() if region_data.get('last_compliance_check') else '',
                'experiment_count': region_data['experiment_count']
            }
            
            report['regions'][region.value] = region_report
            
            if region_data['compliance_status'] == 'compliant':
                report['summary']['compliant_regions'] += 1
            else:
                report['summary']['non_compliant_regions'] += 1
        
        # Calculate standards coverage
        for standard in all_standards:
            regions_with_standard = [
                r for r in regions_to_report 
                if standard in [std.value for std in self.active_regions[r]['config'].compliance_standards]
            ]
            report['summary']['compliance_standards_coverage'][standard] = {
                'regions_count': len(regions_with_standard),
                'coverage_percentage': len(regions_with_standard) / len(regions_to_report) * 100
            }
        
        return report

class InternationalizationManager:
    """Manages internationalization and localization."""
    
    def __init__(self):
        """Initialize internationalization manager."""
        self.supported_languages = {
            'en': 'English',
            'es': 'Español',
            'fr': 'Français', 
            'de': 'Deutsch',
            'ja': '日本語',
            'zh': '中文',
            'ko': '한국어',
            'pt': 'Português',
            'ru': 'Русский',
            'ar': 'العربية'
        }
        
        self.translations = {}
        self.default_language = 'en'
        
        # Load translations
        self._load_translations()
        
    def _load_translations(self):
        """Load translation files."""
        # Base translations for key terms
        self.translations = {
            'en': {
                'experiment': 'Experiment',
                'campaign': 'Campaign',
                'optimization': 'Optimization',
                'materials': 'Materials',
                'discovery': 'Discovery',
                'autonomous': 'Autonomous',
                'laboratory': 'Laboratory',
                'analysis': 'Analysis',
                'results': 'Results',
                'parameters': 'Parameters',
                'success_rate': 'Success Rate',
                'acceleration_factor': 'Acceleration Factor',
                'band_gap': 'Band Gap',
                'efficiency': 'Efficiency',
                'stability': 'Stability',
                'target_reached': 'Target Reached',
                'convergence': 'Convergence',
                'performance': 'Performance'
            },
            'es': {
                'experiment': 'Experimento',
                'campaign': 'Campaña',
                'optimization': 'Optimización',
                'materials': 'Materiales',
                'discovery': 'Descubrimiento',
                'autonomous': 'Autónomo',
                'laboratory': 'Laboratorio',
                'analysis': 'Análisis',
                'results': 'Resultados',
                'parameters': 'Parámetros',
                'success_rate': 'Tasa de Éxito',
                'acceleration_factor': 'Factor de Aceleración',
                'band_gap': 'Banda Prohibida',
                'efficiency': 'Eficiencia',
                'stability': 'Estabilidad',
                'target_reached': 'Objetivo Alcanzado',
                'convergence': 'Convergencia',
                'performance': 'Rendimiento'
            },
            'de': {
                'experiment': 'Experiment',
                'campaign': 'Kampagne',
                'optimization': 'Optimierung',
                'materials': 'Materialien',
                'discovery': 'Entdeckung',
                'autonomous': 'Autonom',
                'laboratory': 'Labor',
                'analysis': 'Analyse',
                'results': 'Ergebnisse',
                'parameters': 'Parameter',
                'success_rate': 'Erfolgsrate',
                'acceleration_factor': 'Beschleunigungsfaktor',
                'band_gap': 'Bandlücke',
                'efficiency': 'Effizienz',
                'stability': 'Stabilität',
                'target_reached': 'Ziel Erreicht',
                'convergence': 'Konvergenz',
                'performance': 'Leistung'
            },
            'zh': {
                'experiment': '实验',
                'campaign': '活动',
                'optimization': '优化',
                'materials': '材料',
                'discovery': '发现',
                'autonomous': '自主',
                'laboratory': '实验室',
                'analysis': '分析',
                'results': '结果',
                'parameters': '参数',
                'success_rate': '成功率',
                'acceleration_factor': '加速因子',
                'band_gap': '带隙',
                'efficiency': '效率',
                'stability': '稳定性',
                'target_reached': '达到目标',
                'convergence': '收敛',
                'performance': '性能'
            }
        }
    
    def translate(self, key: str, language: str = None) -> str:
        """Translate a key to the specified language."""
        if language is None:
            language = self.default_language
        
        if language not in self.supported_languages:
            language = self.default_language
        
        if language in self.translations and key in self.translations[language]:
            return self.translations[language][key]
        
        # Fallback to English
        if key in self.translations.get('en', {}):
            return self.translations['en'][key]
        
        # Return key if no translation found
        return key
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def format_number(self, number: float, language: str = None, decimal_places: int = 2) -> str:
        """Format number according to locale."""
        if language is None:
            language = self.default_language
        
        # Basic number formatting based on language
        if language in ['en', 'zh', 'ja', 'ko']:
            return f"{number:.{decimal_places}f}"
        elif language in ['de', 'fr']:
            # Use comma as decimal separator
            formatted = f"{number:.{decimal_places}f}"
            return formatted.replace('.', ',')
        elif language in ['es', 'pt']:
            # Use comma as decimal separator, space as thousands separator
            formatted = f"{number:,.{decimal_places}f}"
            return formatted.replace(',', ' ').replace('.', ',')
        else:
            return f"{number:.{decimal_places}f}"
    
    def format_datetime(self, dt: datetime, language: str = None) -> str:
        """Format datetime according to locale."""
        if language is None:
            language = self.default_language
        
        # Basic datetime formatting based on language
        if language == 'en':
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        elif language == 'de':
            return dt.strftime("%d.%m.%Y %H:%M:%S")
        elif language == 'fr':
            return dt.strftime("%d/%m/%Y %H:%M:%S")
        elif language in ['zh', 'ja', 'ko']:
            return dt.strftime("%Y年%m月%d日 %H:%M:%S")
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S")

# Global instances
_global_deployment_manager = None
_global_i18n_manager = None

def get_global_deployment_manager() -> Optional[GlobalDeploymentManager]:
    """Get global deployment manager instance."""
    return _global_deployment_manager

def initialize_global_deployment(config: GlobalConfig) -> GlobalDeploymentManager:
    """Initialize global deployment manager."""
    global _global_deployment_manager
    _global_deployment_manager = GlobalDeploymentManager(config)
    return _global_deployment_manager

def get_global_i18n_manager() -> InternationalizationManager:
    """Get global internationalization manager instance."""
    global _global_i18n_manager
    if _global_i18n_manager is None:
        _global_i18n_manager = InternationalizationManager()
    return _global_i18n_manager