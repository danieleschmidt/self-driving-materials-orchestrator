"""Sample data fixtures for testing."""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np


class SampleDataGenerator:
    """Generate sample data for testing materials discovery workflows."""
    
    @staticmethod
    def perovskite_parameters() -> Dict[str, tuple]:
        """Standard perovskite parameter space."""
        return {
            "precursor_A_conc": (0.1, 2.0),
            "precursor_B_conc": (0.1, 2.0),
            "temperature": (100, 300),
            "reaction_time": (1, 24),
            "pH": (3, 11),
            "solvent_ratio": (0, 1)
        }
    
    @staticmethod
    def generate_experiment_data(
        n_experiments: int = 50,
        noise_level: float = 0.05,
        include_failures: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate realistic experiment data with noise and failures."""
        experiments = []
        param_space = SampleDataGenerator.perovskite_parameters()
        
        for i in range(n_experiments):
            # Generate random parameters
            params = {}
            for param, (min_val, max_val) in param_space.items():
                params[param] = np.random.uniform(min_val, max_val)
            
            # Simulate realistic band gap based on parameters
            # This is a simplified model for testing purposes
            temp_factor = (params["temperature"] - 200) / 100
            conc_factor = (params["precursor_A_conc"] + params["precursor_B_conc"]) / 2
            time_factor = np.log(params["reaction_time"]) / 3
            
            base_bandgap = 1.4 + 0.2 * temp_factor + 0.1 * conc_factor - 0.05 * time_factor
            
            # Add noise
            bandgap = base_bandgap + np.random.normal(0, noise_level)
            
            # Calculate efficiency (simplified correlation)
            if 1.2 <= bandgap <= 1.6:
                efficiency = 25 * (1 - abs(bandgap - 1.4) / 0.4) + np.random.normal(0, 2)
            else:
                efficiency = np.random.uniform(0, 10)
            
            # Simulate stability
            stability = max(0, min(1, 0.8 + np.random.normal(0, 0.1)))
            
            # Randomly include failures
            failed = include_failures and np.random.random() < 0.05
            
            experiment = {
                "experiment_id": f"exp_{i:04d}",
                "timestamp": datetime.now() - timedelta(hours=i),
                "parameters": params,
                "results": {} if failed else {
                    "band_gap": round(bandgap, 3),
                    "efficiency": round(max(0, efficiency), 1),
                    "stability": round(stability, 3)
                },
                "metadata": {
                    "status": "failed" if failed else "completed",
                    "duration_minutes": np.random.randint(30, 180),
                    "operator": "test_robot",
                    "campaign_id": "test_campaign_001"
                }
            }
            
            experiments.append(experiment)
        
        return experiments
    
    @staticmethod
    def generate_campaign_data() -> Dict[str, Any]:
        """Generate sample campaign data."""
        return {
            "campaign_id": "test_campaign_001",
            "objective": {
                "target_property": "band_gap",
                "target_range": [1.2, 1.6],
                "optimization_direction": "target",
                "material_system": "perovskites"
            },
            "parameter_space": SampleDataGenerator.perovskite_parameters(),
            "status": "running",
            "created_at": datetime.now() - timedelta(days=2),
            "experiments_completed": 45,
            "experiments_planned": 100,
            "best_result": {
                "band_gap": 1.413,
                "efficiency": 27.1,
                "stability": 0.820
            },
            "optimization_history": [
                {"iteration": 1, "best_value": 1.523, "acquisition_value": 0.245},
                {"iteration": 2, "best_value": 1.487, "acquisition_value": 0.198},
                {"iteration": 3, "best_value": 1.445, "acquisition_value": 0.167},
                {"iteration": 4, "best_value": 1.413, "acquisition_value": 0.134}
            ]
        }
    
    @staticmethod
    def robot_configurations() -> List[Dict[str, Any]]:
        """Sample robot configurations for testing."""
        return [
            {
                "robot_id": "opentrons_ot2_001",
                "name": "Opentrons OT-2 #001",
                "type": "liquid_handler",
                "status": "available",
                "capabilities": ["pipetting", "dispensing", "mixing"],
                "current_protocol": None,
                "last_calibration": datetime.now() - timedelta(days=1),
                "connection": {
                    "type": "http",
                    "host": "192.168.1.100",
                    "port": 31950,
                    "connected": True
                }
            },
            {
                "robot_id": "chemspeed_swing_001", 
                "name": "Chemspeed SWING #001",
                "type": "synthesizer",
                "status": "busy",
                "capabilities": ["synthesis", "heating", "stirring", "dispensing"],
                "current_protocol": "perovskite_synthesis_v2",
                "last_calibration": datetime.now() - timedelta(hours=6),
                "connection": {
                    "type": "serial",
                    "port": "/dev/ttyUSB0",
                    "baudrate": 9600,
                    "connected": True
                }
            }
        ]
    
    @staticmethod
    def instrument_configurations() -> List[Dict[str, Any]]:
        """Sample instrument configurations for testing."""
        return [
            {
                "instrument_id": "xrd_bruker_001",
                "name": "Bruker D8 XRD",
                "type": "xrd",
                "status": "available",
                "measurements": ["crystal_structure", "phase_identification"],
                "last_calibration": datetime.now() - timedelta(days=7),
                "queue_length": 0
            },
            {
                "instrument_id": "uvvis_perkin_001",
                "name": "PerkinElmer UV-Vis",
                "type": "spectroscopy",
                "status": "measuring",
                "measurements": ["absorbance", "band_gap"],
                "last_calibration": datetime.now() - timedelta(days=3),
                "queue_length": 2
            }
        ]


# Pre-generated test data
SAMPLE_EXPERIMENTS = SampleDataGenerator.generate_experiment_data(50)
SAMPLE_CAMPAIGN = SampleDataGenerator.generate_campaign_data()
SAMPLE_ROBOTS = SampleDataGenerator.robot_configurations()
SAMPLE_INSTRUMENTS = SampleDataGenerator.instrument_configurations()


def save_sample_data(filepath: str) -> None:
    """Save all sample data to JSON file."""
    data = {
        "experiments": SAMPLE_EXPERIMENTS,
        "campaign": SAMPLE_CAMPAIGN,
        "robots": SAMPLE_ROBOTS,
        "instruments": SAMPLE_INSTRUMENTS,
        "generated_at": datetime.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_sample_data(filepath: str) -> Dict[str, Any]:
    """Load sample data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)