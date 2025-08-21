#!/usr/bin/env python3
"""Validation script for next-generation AI enhancements.

This script validates that all next-generation modules can be imported
and basic functionality works correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def validate_imports():
    """Validate that all next-generation modules can be imported."""
    print("🔍 Validating Next-Generation AI Enhancement Imports...")
    print("=" * 60)
    
    try:
        # Test core module imports
        print("Testing core module imports...")
        
        from materials_orchestrator.autonomous_hypothesis_generator import (
            AutonomousHypothesisGenerator,
            ScientificHypothesis,
            HypothesisType,
            HypothesisConfidence
        )
        print("  ✅ Autonomous Hypothesis Generator")
        
        from materials_orchestrator.quantum_hybrid_optimizer import (
            QuantumHybridOptimizer,
            QuantumOptimizationProblem,
            OptimizationStrategy,
            QuantumBackend
        )
        print("  ✅ Quantum Hybrid Optimizer")
        
        from materials_orchestrator.federated_learning_coordinator import (
            FederatedLearningCoordinator,
            LabNode,
            FederatedModel,
            LabRole,
            PrivacyLevel
        )
        print("  ✅ Federated Learning Coordinator")
        
        from materials_orchestrator.realtime_adaptive_protocols import (
            AdaptiveProtocolEngine,
            ExperimentalCondition,
            RealTimeResult,
            AdaptationStrategy
        )
        print("  ✅ Real-Time Adaptive Protocols")
        
        print("\nTesting main module imports...")
        
        # Test main module imports
        from materials_orchestrator import (
            AutonomousHypothesisGenerator,
            QuantumHybridOptimizer,
            FederatedLearningCoordinator,
            AdaptiveProtocolEngine,
            generate_scientific_hypotheses,
            optimize_with_quantum_hybrid,
            create_federated_materials_network,
            process_realtime_experiment_data
        )
        print("  ✅ Main module exports")
        print("  ✅ Convenience functions")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def validate_basic_functionality():
    """Validate basic functionality of key components."""
    print("\n🧪 Validating Basic Functionality...")
    print("=" * 40)
    
    try:
        # Test Hypothesis Generator instantiation
        from materials_orchestrator.autonomous_hypothesis_generator import AutonomousHypothesisGenerator
        generator = AutonomousHypothesisGenerator()
        print("  ✅ Hypothesis Generator instantiation")
        
        # Test Quantum Optimizer instantiation
        from materials_orchestrator.quantum_hybrid_optimizer import QuantumHybridOptimizer
        optimizer = QuantumHybridOptimizer()
        print("  ✅ Quantum Optimizer instantiation")
        
        # Test Federated Coordinator instantiation
        from materials_orchestrator.federated_learning_coordinator import FederatedLearningCoordinator
        coordinator = FederatedLearningCoordinator("Test Lab")
        print("  ✅ Federated Coordinator instantiation")
        
        # Test Adaptive Engine instantiation
        from materials_orchestrator.realtime_adaptive_protocols import AdaptiveProtocolEngine
        engine = AdaptiveProtocolEngine()
        print("  ✅ Adaptive Engine instantiation")
        
        # Test data structure creation
        from materials_orchestrator.autonomous_hypothesis_generator import ScientificHypothesis, HypothesisType
        hypothesis = ScientificHypothesis(
            hypothesis_text="Test hypothesis",
            hypothesis_type=HypothesisType.PREDICTIVE
        )
        print("  ✅ Scientific Hypothesis creation")
        
        from materials_orchestrator.realtime_adaptive_protocols import ExperimentalCondition
        conditions = ExperimentalCondition(temperature=150.0, pressure=1.0)
        print("  ✅ Experimental Condition creation")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality error: {e}")
        return False

def validate_file_structure():
    """Validate that all required files exist."""
    print("\n📁 Validating File Structure...")
    print("=" * 30)
    
    required_files = [
        "src/materials_orchestrator/autonomous_hypothesis_generator.py",
        "src/materials_orchestrator/quantum_hybrid_optimizer.py", 
        "src/materials_orchestrator/federated_learning_coordinator.py",
        "src/materials_orchestrator/realtime_adaptive_protocols.py",
        "tests/test_next_generation_enhancements.py",
        "examples/next_generation_ai_discovery.py",
        "NEXT_GENERATION_AI_ENHANCEMENTS.md"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def main():
    """Main validation function."""
    print("🚀 NEXT-GENERATION AI ENHANCEMENTS VALIDATION")
    print("=" * 80)
    print()
    
    validation_results = []
    
    # Run validations
    validation_results.append(("File Structure", validate_file_structure()))
    validation_results.append(("Module Imports", validate_imports()))
    validation_results.append(("Basic Functionality", validate_basic_functionality()))
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 25)
    
    all_passed = True
    for test_name, result in validation_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("🚀 Next-Generation AI Enhancements are ready for use!")
        print()
        print("Key Capabilities Available:")
        print("  • 🧠 Autonomous Hypothesis Generation")
        print("  • ⚛️  Quantum-Hybrid Optimization")
        print("  • 🌐 Federated Learning Coordination")  
        print("  • 🔄 Real-Time Adaptive Protocols")
        print()
        print("Try running: python3 examples/next_generation_ai_discovery.py")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("Please check the error messages above and ensure all dependencies are installed.")
        return 1

if __name__ == "__main__":
    exit(main())