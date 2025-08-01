"""Sample materials data for testing."""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Sample perovskite materials data
PEROVSKITE_SAMPLES = [
    {
        "composition": "MAPbI3",
        "precursors": {"MAI": 1.0, "PbI2": 1.0},
        "synthesis_conditions": {
            "temperature": 100,
            "time": 10,
            "solvent": "DMF",
            "concentration": 1.0
        },
        "properties": {
            "band_gap": 1.55,
            "efficiency": 15.3,
            "stability": 0.7
        }
    },
    {
        "composition": "FAPbI3",
        "precursors": {"FAI": 1.0, "PbI2": 1.0},
        "synthesis_conditions": {
            "temperature": 120,
            "time": 15,
            "solvent": "DMF:DMSO",
            "concentration": 1.2
        },
        "properties": {
            "band_gap": 1.48,
            "efficiency": 18.7,
            "stability": 0.6
        }
    },
    {
        "composition": "CsPbI3",
        "precursors": {"CsI": 1.0, "PbI2": 1.0},
        "synthesis_conditions": {
            "temperature": 150,
            "time": 20,
            "solvent": "DMSO",
            "concentration": 0.8
        },
        "properties": {
            "band_gap": 1.73,
            "efficiency": 12.1,
            "stability": 0.9
        }
    }
]

# Parameter space definitions
PARAMETER_SPACES = {
    "perovskite_synthesis": {
        "temperature": (80, 200),
        "time": (5, 30),
        "concentration": (0.5, 2.0),
        "pressure": (1.0, 3.0),
        "pH": (4, 10)
    },
    "organic_photovoltaic": {
        "annealing_temp": (100, 180),
        "solvent_ratio": (0.1, 0.9),
        "spin_speed": (1000, 4000),
        "thickness": (50, 300)
    },
    "catalyst_synthesis": {
        "calcination_temp": (300, 800),
        "impregnation_time": (1, 24),
        "metal_loading": (1, 20),
        "support_area": (50, 500)
    }
}

# Sample experiment sequences
EXPERIMENT_SEQUENCES = [
    {
        "id": "seq_001",
        "objective": "band_gap_optimization",
        "material_system": "perovskites",
        "experiments": [
            {
                "parameters": {"temperature": 100, "time": 10, "concentration": 1.0},
                "results": {"band_gap": 1.55, "efficiency": 15.3},
                "timestamp": datetime.now() - timedelta(days=5)
            },
            {
                "parameters": {"temperature": 120, "time": 15, "concentration": 1.2},
                "results": {"band_gap": 1.48, "efficiency": 18.7},
                "timestamp": datetime.now() - timedelta(days=4)
            },
            {
                "parameters": {"temperature": 110, "time": 12, "concentration": 1.1},
                "results": {"band_gap": 1.52, "efficiency": 16.8},
                "timestamp": datetime.now() - timedelta(days=3)
            }
        ]
    }
]

# Mock robot responses
ROBOT_RESPONSES = {
    "opentrons": {
        "status": "idle",
        "position": {"x": 0, "y": 0, "z": 50},
        "pipette": {"volume": 0, "tip_attached": False},
        "deck": {
            "A1": {"labware": "96-well-plate", "volume": 200},
            "B1": {"labware": "reagent-reservoir", "volume": 15000}
        }
    },
    "chemspeed": {
        "status": "ready",
        "temperature": 25.0,
        "stirrer_speed": 0,
        "reactor_volumes": [0, 0, 0, 0],
        "valve_positions": ["closed"] * 8
    }
}

# Simulation data for virtual experiments
def generate_virtual_experiment_result(parameters: Dict[str, float], 
                                     material_system: str = "perovskites",
                                     noise_level: float = 0.05) -> Dict[str, Any]:
    """Generate realistic virtual experiment results with noise."""
    
    if material_system == "perovskites":
        # Simple model for perovskite band gap
        temp = parameters.get("temperature", 100)
        conc = parameters.get("concentration", 1.0)
        time = parameters.get("time", 10)
        
        # Simplified relationship (not physically accurate, just for testing)
        base_gap = 1.6 - 0.001 * temp + 0.05 * conc - 0.002 * time
        base_efficiency = 20 * (1 - abs(base_gap - 1.5) / 0.3)
        
        # Add noise
        gap_noise = np.random.normal(0, noise_level)
        eff_noise = np.random.normal(0, noise_level * 20)
        
        return {
            "band_gap": max(0.5, min(3.0, base_gap + gap_noise)),
            "efficiency": max(0, min(25, base_efficiency + eff_noise)),
            "stability": np.random.uniform(0.5, 1.0),
            "synthesis_success": np.random.choice([True, False], p=[0.85, 0.15])
        }
    
    else:
        # Generic results for other material systems
        return {
            "property_1": np.random.uniform(0, 100),
            "property_2": np.random.uniform(-10, 10),
            "synthesis_success": np.random.choice([True, False], p=[0.8, 0.2])
        }

# Database test data
DATABASE_TEST_COLLECTIONS = {
    "experiments": [
        {
            "_id": "exp_001",
            "campaign_id": "camp_001",
            "parameters": {"temperature": 100, "concentration": 1.0},
            "results": {"band_gap": 1.55, "efficiency": 15.3},
            "timestamp": datetime.now(),
            "status": "completed"
        },
        {
            "_id": "exp_002", 
            "campaign_id": "camp_001",
            "parameters": {"temperature": 120, "concentration": 1.2},
            "results": {"band_gap": 1.48, "efficiency": 18.7},
            "timestamp": datetime.now(),
            "status": "completed"
        }
    ],
    "campaigns": [
        {
            "_id": "camp_001",
            "name": "Perovskite Band Gap Optimization",
            "objective": "minimize band_gap variance",
            "material_system": "perovskites",
            "status": "active",
            "created_at": datetime.now(),
            "experiments_count": 2
        }
    ],
    "materials": [
        {
            "_id": "mat_001",
            "composition": "MAPbI3",
            "properties": {"band_gap": 1.55, "efficiency": 15.3},
            "synthesis_route": "solution_processing",
            "stability": 0.7
        }
    ]
}

# Performance test data
PERFORMANCE_TEST_SIZES = [10, 50, 100, 500, 1000]

def generate_large_dataset(size: int) -> List[Dict[str, Any]]:
    """Generate large dataset for performance testing."""
    data = []
    for i in range(size):
        data.append({
            "id": f"exp_{i:06d}",
            "parameters": {
                "temperature": np.random.uniform(80, 200),
                "concentration": np.random.uniform(0.5, 2.0),
                "time": np.random.uniform(5, 30)
            },
            "results": {
                "band_gap": np.random.uniform(1.0, 2.0),
                "efficiency": np.random.uniform(5, 25)
            },
            "timestamp": datetime.now() - timedelta(days=np.random.randint(0, 365))
        })
    return data