#!/usr/bin/env python3
"""
Robust Materials Discovery Example with Enhanced Security, Validation, and Monitoring

Demonstrates the complete Generation 2 robust implementation with:
- Comprehensive validation and error handling
- Advanced security measures
- Real-time monitoring and alerting
- Production-ready reliability features
"""

import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from materials_orchestrator import AutonomousLab, MaterialsObjective, BayesianPlanner
from materials_orchestrator.enhanced_validation import create_robust_validation_system, ValidationLevel
from materials_orchestrator.advanced_security import create_advanced_security_system, SecurityLevel
from materials_orchestrator.comprehensive_monitoring import create_comprehensive_monitoring, AlertSeverity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_robust_systems():
    """Set up all robust systems for production-ready operation."""
    print("🛡️  Setting up robust systems...")
    
    # Initialize validation system
    validator, error_handler = create_robust_validation_system()
    
    # Initialize security system  
    security_manager = create_advanced_security_system(SecurityLevel.HIGH)
    
    # Initialize monitoring system
    monitoring, alert_manager = create_comprehensive_monitoring()
    
    # Setup alert handlers
    def log_alert(alert):
        logger.warning(f"ALERT: {alert.name} - {alert.message}")
    
    alert_manager.add_notification_handler(AlertSeverity.ERROR, log_alert)
    alert_manager.add_notification_handler(AlertSeverity.CRITICAL, log_alert)
    
    print("✅ Robust systems initialized")
    return validator, error_handler, security_manager, monitoring, alert_manager


def demonstrate_validation_system(validator):
    """Demonstrate comprehensive validation capabilities."""
    print("\n🔍 Demonstrating Validation System")
    print("-" * 50)
    
    # Test various parameter sets
    test_cases = [
        {
            "name": "Normal parameters",
            "params": {
                "temperature": 150,
                "pH": 8.5,
                "precursor_A_conc": 1.0,
                "precursor_B_conc": 0.5,
                "reaction_time": 4
            }
        },
        {
            "name": "High temperature warning",
            "params": {
                "temperature": 350,
                "pH": 7.0,
                "precursor_A_conc": 2.0,
                "reaction_time": 2
            }
        },
        {
            "name": "Critical safety violation",
            "params": {
                "temperature": 450,
                "pH": 0.5,
                "precursor_A_conc": 15.0,
                "reaction_time": 50
            }
        },
        {
            "name": "Economic analysis",
            "params": {
                "temperature": 300,
                "reaction_time": 20,
                "precursor_A_conc": 3.0,
                "precursor_B_conc": 4.0
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 {test_case['name']}:")
        results = validator.validate_experiment_parameters(test_case['params'])
        
        if not results:
            print("  ✅ All validations passed")
        else:
            for result in results:
                icon = "🚨" if result.severity == "critical" else "⚠️" if result.severity == "warning" else "ℹ️"
                print(f"  {icon} {result.category.value}: {result.message}")
    
    # Show validation summary
    summary = validator.get_validation_summary()
    print(f"\n📊 Validation Summary:")
    print(f"   Total validations: {summary['total_validations']}")
    print(f"   Total checks: {summary['total_checks']}")
    print(f"   Critical issues: {summary['total_critical_issues']}")
    print(f"   Warnings: {summary['total_warnings']}")


def demonstrate_security_system(security_manager):
    """Demonstrate advanced security capabilities."""
    print("\n🔒 Demonstrating Security System") 
    print("-" * 50)
    
    # Test authentication
    test_requests = [
        {
            "name": "Valid request",
            "data": {
                "source": "lab_station_1",
                "api_key": "test_key",
                "action": "run_experiment"
            }
        },
        {
            "name": "Suspicious payload",
            "data": {
                "source": "unknown",
                "payload": "'; DROP TABLE experiments; --",
                "script": "<script>alert('xss')</script>"
            }
        },
        {
            "name": "High-risk experiment",
            "data": {
                "temperature": 600,
                "pH": -1,
                "precursor_A_conc": 50
            }
        }
    ]
    
    for test_request in test_requests:
        print(f"\n🔐 {test_request['name']}:")
        
        # Test authentication
        is_auth, auth_msg = security_manager.authenticate_request(test_request['data'])
        print(f"   Authentication: {'✅' if is_auth else '❌'} {auth_msg}")
        
        # Test threat scanning
        threats = security_manager.scan_for_threats(test_request['data'])
        if threats:
            print(f"   🚨 Threats detected: {len(threats)}")
            for threat in threats[:2]:  # Show first 2 threats
                print(f"     - {threat['type']}: {threat['description']}")
        else:
            print("   ✅ No threats detected")
        
        # Test experiment authorization if it's an experiment
        if 'temperature' in test_request['data']:
            user_context = {"user_id": "test_user", "permissions": ["run_experiment"]}
            is_authorized, auth_msg = security_manager.authorize_experiment(test_request['data'], user_context)
            print(f"   Experiment authorization: {'✅' if is_authorized else '❌'} {auth_msg}")
    
    # Generate secure token
    token = security_manager.generate_secure_token({"user_id": "demo_user"})
    print(f"\n🎟️  Generated secure token: {token[:20]}...")
    
    # Validate token
    is_valid, token_data = security_manager.validate_session_token(token)
    print(f"   Token validation: {'✅' if is_valid else '❌'}")
    
    # Show security status
    status = security_manager.get_security_status()
    print(f"\n📊 Security Status:")
    print(f"   Security level: {status['security_level']}")
    print(f"   Security events: {status['total_security_events']}")
    print(f"   Security score: {status['security_score']:.1f}/100")


def demonstrate_monitoring_system(monitoring):
    """Demonstrate comprehensive monitoring capabilities."""
    print("\n📊 Demonstrating Monitoring System")
    print("-" * 50)
    
    # Simulate some metrics
    print("📈 Recording sample metrics...")
    
    for i in range(10):
        # Simulate experiment metrics
        monitoring.increment_counter('experiments_total')
        monitoring.increment_counter('experiments_successful')
        monitoring.record_metric('experiment_duration', 120 + i * 10)
        monitoring.record_metric('queue_size', 5 - i * 0.5)
        
        time.sleep(0.1)  # Small delay to show progression
    
    # Record some system metrics
    monitoring.record_metric('system_cpu_load', 2.5)
    monitoring.record_metric('system_memory_mb', 512)
    monitoring.record_metric('experiment_failure_rate', 0.1)  # 10% failure rate
    
    # Get recent statistics
    stats = monitoring.get_metric_statistics('experiments_total', 1)
    print(f"   Experiments (last hour): {stats}")
    
    duration_stats = monitoring.get_metric_statistics('experiment_duration', 1)
    print(f"   Duration stats: avg={duration_stats.get('avg', 0):.1f}s")
    
    # Test alert triggering
    print("\n🚨 Testing alert system...")
    monitoring.record_metric('experiment_failure_rate', 0.6)  # Trigger high failure rate alert
    time.sleep(0.5)  # Allow alert processing
    
    # Get dashboard data
    dashboard = monitoring.get_monitoring_dashboard()
    print(f"\n📊 Dashboard Summary:")
    print(f"   System status: {dashboard['system_status']['status']}")
    print(f"   Active alerts: {dashboard['system_status']['active_alerts_count']}")
    print(f"   Performance: {dashboard['performance_summary']['success_rate_percent']:.1f}% success rate")


def run_robust_discovery_campaign(validator, security_manager, monitoring):
    """Run a materials discovery campaign with all robust systems active."""
    print("\n🚀 Running Robust Discovery Campaign")
    print("=" * 60)
    
    # Define objective with validation
    objective = MaterialsObjective(
        target_property="band_gap",
        target_range=(1.2, 1.6),
        optimization_direction="target",
        material_system="perovskites",
        success_threshold=1.4,
    )
    
    # Define parameter space
    param_space = {
        "precursor_A_conc": (0.1, 2.0),
        "precursor_B_conc": (0.1, 2.0),
        "temperature": (100, 300),
        "reaction_time": (1, 24),
        "pH": (3, 11),
        "solvent_ratio": (0, 1),
    }
    
    print(f"🎯 Objective: {objective.target_property} in range {objective.target_range}")
    
    # Initialize lab with robust systems
    lab = AutonomousLab(
        robots=["synthesis_robot", "characterization_robot"],
        instruments=["xrd", "uv_vis", "pl_spectrometer"],
        planner=BayesianPlanner(target_property="band_gap")
    )
    
    # Simulate security context
    security_context = {
        "user_id": "researcher_1",
        "permissions": ["run_experiment"],
        "api_key": "secure_key_12345"
    }
    
    print("🛡️  Security validation...")
    is_authorized, msg = security_manager.authorize_experiment(
        {"temperature": 200, "pH": 7}, security_context
    )
    if not is_authorized:
        print(f"❌ Campaign blocked: {msg}")
        return
    
    print("✅ Security cleared")
    
    # Record monitoring metrics for campaign start
    monitoring.increment_counter('campaigns_started')
    monitoring.record_metric('active_campaigns', 1)
    
    start_time = time.time()
    
    print("🏁 Starting campaign...")
    
    # Run campaign with robust error handling
    try:
        campaign = lab.run_campaign(
            objective=objective,
            param_space=param_space,
            initial_samples=10,
            max_experiments=50,
            stop_on_target=True,
            convergence_patience=15
        )
        
        campaign_duration = time.time() - start_time
        
        # Record campaign completion metrics
        monitoring.record_timer('campaign_duration', campaign_duration)
        monitoring.increment_counter('campaigns_completed')
        monitoring.record_metric('campaign_success_rate', campaign.success_rate)
        
        # Display results with robust analysis
        print("\n" + "=" * 60)
        print("🏆 ROBUST CAMPAIGN RESULTS")
        print("=" * 60)
        
        print(f"Campaign ID: {campaign.campaign_id}")
        print(f"Total experiments: {campaign.total_experiments}")
        print(f"Success rate: {campaign.success_rate:.1%}")
        print(f"Duration: {campaign_duration:.2f} seconds")
        
        if campaign.best_material:
            print(f"\n🥇 Best Material:")
            print(f"   Band gap: {campaign.best_properties.get('band_gap', 'N/A'):.3f} eV")
            print(f"   Efficiency: {campaign.best_properties.get('efficiency', 'N/A'):.1f}%")
            
            # Validate best parameters
            validation_results = validator.validate_experiment_parameters(
                campaign.best_material['parameters']
            )
            
            critical_issues = [r for r in validation_results if r.severity == "critical"]
            if critical_issues:
                print(f"⚠️  Best material has {len(critical_issues)} critical safety issues")
            else:
                print("✅ Best material passes all safety validations")
        
        # Performance analysis
        acceleration = 200 / max(campaign.total_experiments, 1)
        print(f"\n⚡ Performance Analysis:")
        print(f"   Acceleration: {acceleration:.1f}x vs traditional methods")
        print(f"   Efficiency: {campaign.successful_experiments / campaign.total_experiments:.1%} successful experiments")
        
        return campaign
        
    except Exception as e:
        logger.error(f"Campaign failed: {e}")
        monitoring.increment_counter('campaigns_failed')
        return None


def main():
    """Main execution function."""
    print("🧬 Robust Materials Discovery System - Generation 2")
    print("=" * 70)
    print("Demonstrating enhanced validation, security, and monitoring")
    
    try:
        # Setup robust systems
        validator, error_handler, security_manager, monitoring, alert_manager = setup_robust_systems()
        
        # Demonstrate each system
        demonstrate_validation_system(validator)
        demonstrate_security_system(security_manager)
        demonstrate_monitoring_system(monitoring)
        
        # Run full robust campaign
        campaign = run_robust_discovery_campaign(validator, security_manager, monitoring)
        
        if campaign:
            print("\n✅ Robust discovery campaign completed successfully!")
        else:
            print("\n❌ Campaign failed - robust error handling active")
        
        # Final system status
        print("\n📊 Final System Status:")
        
        validation_summary = validator.get_validation_summary()
        print(f"   Validations performed: {validation_summary['total_validations']}")
        
        security_status = security_manager.get_security_status()
        print(f"   Security score: {security_status['security_score']:.1f}/100")
        
        dashboard = monitoring.get_monitoring_dashboard()
        print(f"   System health: {dashboard['system_status']['status']}")
        
        # Stop monitoring
        monitoring.stop_monitoring()
        
    except KeyboardInterrupt:
        print("\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        return 1
    
    print("\n🎉 Robust systems demonstration completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())