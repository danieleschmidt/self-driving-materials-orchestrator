"""Experimental protocol definitions and templates."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import timedelta


@dataclass
class ProtocolStep:
    """Individual step in an experimental protocol."""
    
    action: str
    robot: Optional[str] = None
    instrument: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: Optional[timedelta] = None
    parallel_with: Optional[List[int]] = None  # Step indices to run in parallel
    wait_for: Optional[List[int]] = None       # Step indices to wait for
    description: Optional[str] = None


@dataclass 
class ExperimentalProtocol:
    """Complete experimental protocol definition."""
    
    name: str
    description: str
    steps: List[ProtocolStep] = field(default_factory=list)
    materials: Dict[str, Any] = field(default_factory=dict)
    expected_duration: Optional[timedelta] = None
    safety_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert protocol to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "action": step.action,
                    "robot": step.robot,
                    "instrument": step.instrument,
                    "parameters": step.parameters,
                    "description": step.description,
                }
                for step in self.steps
            ],
            "materials": self.materials,
            "safety_notes": self.safety_notes,
        }


class SynthesisProtocol:
    """Templates for common synthesis protocols."""
    
    @staticmethod
    def perovskite_synthesis(parameters: Dict[str, Any]) -> ExperimentalProtocol:
        """Generate perovskite synthesis protocol.
        
        Args:
            parameters: Synthesis parameters
            
        Returns:
            Complete synthesis protocol
        """
        precursor_a_conc = parameters.get("precursor_A_conc", 1.0)
        precursor_b_conc = parameters.get("precursor_B_conc", 1.0)
        temperature = parameters.get("temperature", 100)
        reaction_time = parameters.get("reaction_time", 4)
        solvent_ratio = parameters.get("solvent_ratio", 0.5)
        
        steps = [
            ProtocolStep(
                action="dispense",
                robot="liquid_handler",
                parameters={
                    "volume": precursor_a_conc * 100,  # μL
                    "source": "precursor_A_stock",
                    "dest": "reaction_vial_1",
                },
                description=f"Dispense {precursor_a_conc}M precursor A"
            ),
            ProtocolStep(
                action="dispense", 
                robot="liquid_handler",
                parameters={
                    "volume": precursor_b_conc * 100,
                    "source": "precursor_B_stock", 
                    "dest": "reaction_vial_1",
                },
                description=f"Dispense {precursor_b_conc}M precursor B"
            ),
            ProtocolStep(
                action="dispense",
                robot="liquid_handler", 
                parameters={
                    "volume": 200,  # Base solvent volume
                    "source": "DMF",
                    "dest": "reaction_vial_1",
                },
                description="Add DMF solvent"
            ),
            ProtocolStep(
                action="dispense",
                robot="liquid_handler",
                parameters={
                    "volume": int(200 * (1 - solvent_ratio)),
                    "source": "DMSO", 
                    "dest": "reaction_vial_1",
                },
                description="Add DMSO solvent"
            ),
            ProtocolStep(
                action="mix",
                robot="liquid_handler",
                parameters={
                    "location": "reaction_vial_1",
                    "volume": 50,
                    "repetitions": 10,
                },
                description="Mix precursors and solvents"
            ),
            ProtocolStep(
                action="move",
                robot="synthesizer",
                parameters={
                    "source": "reaction_vial_1",
                    "dest": "heating_block_1",
                },
                description="Transfer to heating block"
            ),
            ProtocolStep(
                action="heat",
                robot="synthesizer",
                parameters={
                    "temperature": temperature,
                    "duration": reaction_time * 3600,  # Convert hours to seconds
                    "location": "heating_block_1",
                },
                description=f"Heat to {temperature}°C for {reaction_time} hours"
            ),
            ProtocolStep(
                action="cool",
                robot="synthesizer", 
                parameters={
                    "target_temperature": 25,
                    "location": "heating_block_1",
                },
                description="Cool to room temperature"
            ),
            ProtocolStep(
                action="move",
                robot="synthesizer",
                parameters={
                    "source": "heating_block_1",
                    "dest": "sample_holder_1",
                },
                description="Transfer to sample holder"
            ),
        ]
        
        return ExperimentalProtocol(
            name="Perovskite Synthesis",
            description="Solution-based perovskite synthesis protocol",
            steps=steps,
            materials={
                "precursor_A": "PbI2",
                "precursor_B": "MAI", 
                "solvent_1": "DMF",
                "solvent_2": "DMSO",
            },
            expected_duration=timedelta(hours=reaction_time + 1),
            safety_notes=[
                "Use fume hood for all operations",
                "Wear appropriate PPE",
                "Handle lead compounds with care",
            ]
        )
    
    @staticmethod
    def catalyst_synthesis(parameters: Dict[str, Any]) -> ExperimentalProtocol:
        """Generate catalyst synthesis protocol."""
        metal_conc = parameters.get("metal_concentration", 0.1)
        support_mass = parameters.get("support_mass", 1.0)  # grams
        calcination_temp = parameters.get("calcination_temperature", 400)
        
        steps = [
            ProtocolStep(
                action="weigh",
                robot="balance_robot",
                parameters={
                    "target_mass": support_mass,
                    "material": "support_powder",
                    "container": "synthesis_crucible_1",
                },
                description=f"Weigh {support_mass}g support material"
            ),
            ProtocolStep(
                action="dispense",
                robot="liquid_handler",
                parameters={
                    "volume": metal_conc * 1000,  # Convert to μL
                    "source": "metal_precursor_solution",
                    "dest": "synthesis_crucible_1",
                },
                description="Add metal precursor solution"
            ),
            ProtocolStep(
                action="stir",
                robot="synthesizer",
                parameters={
                    "location": "synthesis_crucible_1",
                    "speed": 300,  # RPM
                    "duration": 3600,  # 1 hour
                },
                description="Stir mixture for impregnation"
            ),
            ProtocolStep(
                action="dry",
                robot="synthesizer",
                parameters={
                    "temperature": 120,
                    "duration": 7200,  # 2 hours
                    "location": "synthesis_crucible_1",
                },
                description="Dry at 120°C"
            ),
            ProtocolStep(
                action="calcine",
                robot="furnace_robot",
                parameters={
                    "temperature": calcination_temp,
                    "duration": 14400,  # 4 hours
                    "atmosphere": "air",
                    "ramp_rate": 5,  # °C/min
                },
                description=f"Calcine at {calcination_temp}°C"
            ),
        ]
        
        return ExperimentalProtocol(
            name="Supported Catalyst Synthesis",
            description="Incipient wetness impregnation catalyst synthesis",
            steps=steps,
            materials={
                "support": "Al2O3",
                "metal_precursor": "Metal nitrate solution",
            },
            expected_duration=timedelta(hours=8),
            safety_notes=[
                "Use proper ventilation during calcination",
                "Allow adequate cooling time",
                "Handle hot materials with appropriate tools",
            ]
        )


class CharacterizationProtocol:
    """Templates for materials characterization protocols."""
    
    @staticmethod
    def comprehensive_characterization(sample_id: str) -> ExperimentalProtocol:
        """Generate comprehensive characterization protocol.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Complete characterization protocol
        """
        steps = [
            ProtocolStep(
                action="measure",
                instrument="xrd",
                parameters={
                    "sample_id": sample_id,
                    "scan_range": (10, 80),  # 2-theta degrees
                    "step_size": 0.02,
                    "time_per_step": 1.0,
                },
                description="X-ray diffraction analysis"
            ),
            ProtocolStep(
                action="measure",
                instrument="uv_vis",
                parameters={
                    "sample_id": sample_id,
                    "wavelength_range": (300, 800),  # nm
                    "measurement_mode": "transmission",
                },
                description="UV-Vis spectroscopy", 
                parallel_with=[0]  # Can run in parallel with XRD
            ),
            ProtocolStep(
                action="measure",
                instrument="pl_spectrometer",
                parameters={
                    "sample_id": sample_id,
                    "excitation_wavelength": 405,  # nm
                    "emission_range": (450, 800),
                },
                description="Photoluminescence spectroscopy",
                wait_for=[1]  # Wait for UV-Vis to complete
            ),
            ProtocolStep(
                action="measure",
                instrument="sem",
                parameters={
                    "sample_id": sample_id,
                    "magnification": 10000,
                    "voltage": 15,  # kV
                    "spot_size": 3,
                },
                description="SEM morphology analysis"
            ),
        ]
        
        return ExperimentalProtocol(
            name="Comprehensive Characterization",
            description="Multi-technique materials characterization",
            steps=steps,
            expected_duration=timedelta(hours=2),
            safety_notes=[
                "Handle samples carefully to avoid contamination",
                "Follow instrument-specific safety procedures",
            ]
        )
    
    @staticmethod
    def optical_characterization(sample_id: str) -> ExperimentalProtocol:
        """Generate optical characterization protocol."""
        steps = [
            ProtocolStep(
                action="measure",
                instrument="uv_vis",
                parameters={
                    "sample_id": sample_id,
                    "wavelength_range": (200, 1000),
                    "measurement_mode": "reflection",
                },
                description="UV-Vis-NIR absorption"
            ),
            ProtocolStep(
                action="measure", 
                instrument="pl_spectrometer",
                parameters={
                    "sample_id": sample_id,
                    "excitation_wavelength": 365,
                    "emission_range": (400, 900),
                },
                description="Photoluminescence mapping"
            ),
            ProtocolStep(
                action="measure",
                instrument="ellipsometer",
                parameters={
                    "sample_id": sample_id,
                    "wavelength_range": (400, 1000),
                    "angle_of_incidence": 70,
                },
                description="Ellipsometry for optical constants"
            ),
        ]
        
        return ExperimentalProtocol(
            name="Optical Characterization",
            description="Comprehensive optical property analysis",
            steps=steps,
            expected_duration=timedelta(minutes=45),
            safety_notes=[
                "Avoid exposure to UV light",
                "Handle samples in clean environment",
            ]
        )