"""Federated Learning Coordinator for Multi-Lab Materials Discovery.

This module implements federated learning capabilities that allow multiple
laboratories to collaborate and share knowledge while preserving data privacy
and intellectual property.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pickle
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class LabRole(Enum):
    """Roles in federated learning network."""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    VALIDATOR = "validator"
    OBSERVER = "observer"


class FederationStatus(Enum):
    """Status of federated learning process."""
    IDLE = "idle"
    RECRUITING = "recruiting"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    VALIDATING = "validating"
    COMPLETE = "complete"
    ERROR = "error"


class PrivacyLevel(Enum):
    """Privacy protection levels."""
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_AGGREGATION = "secure_aggregation"
    FULL_ANONYMIZATION = "full_anonymization"


@dataclass
class LabNode:
    """Represents a laboratory node in the federated network."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    institution: str = ""
    role: LabRole = LabRole.PARTICIPANT
    endpoint: str = ""
    public_key: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    data_statistics: Dict[str, Any] = field(default_factory=dict)
    trust_score: float = 1.0
    last_seen: datetime = field(default_factory=datetime.now)
    experiments_contributed: int = 0
    model_updates_contributed: int = 0
    reputation_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "institution": self.institution,
            "role": self.role.value,
            "endpoint": self.endpoint,
            "public_key": self.public_key,
            "capabilities": self.capabilities,
            "data_statistics": self.data_statistics,
            "trust_score": self.trust_score,
            "last_seen": self.last_seen.isoformat(),
            "experiments_contributed": self.experiments_contributed,
            "model_updates_contributed": self.model_updates_contributed,
            "reputation_score": self.reputation_score
        }


@dataclass
class FederatedModel:
    """Represents a federated learning model."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    model_type: str = "neural_network"
    target_properties: List[str] = field(default_factory=list)
    architecture: Dict[str, Any] = field(default_factory=dict)
    parameters: Optional[np.ndarray] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_rounds: int = 0
    participating_labs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "target_properties": self.target_properties,
            "architecture": self.architecture,
            "parameters": self.parameters.tolist() if self.parameters is not None else None,
            "performance_metrics": self.performance_metrics,
            "training_rounds": self.training_rounds,
            "participating_labs": self.participating_labs,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "privacy_level": self.privacy_level.value
        }


@dataclass
class ModelUpdate:
    """Represents a model update from a participating lab."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    lab_id: str = ""
    model_id: str = ""
    round_number: int = 0
    parameters: Optional[np.ndarray] = None
    gradient: Optional[np.ndarray] = None
    local_performance: Dict[str, float] = field(default_factory=dict)
    data_size: int = 0
    computation_time: float = 0.0
    privacy_budget_used: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "lab_id": self.lab_id,
            "model_id": self.model_id,
            "round_number": self.round_number,
            "parameters": self.parameters.tolist() if self.parameters is not None else None,
            "gradient": self.gradient.tolist() if self.gradient is not None else None,
            "local_performance": self.local_performance,
            "data_size": self.data_size,
            "computation_time": self.computation_time,
            "privacy_budget_used": self.privacy_budget_used,
            "timestamp": self.timestamp.isoformat(),
            "signature": self.signature
        }


class PrivacyManager:
    """Manages privacy-preserving techniques for federated learning."""
    
    def __init__(self, privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY):
        self.privacy_level = privacy_level
        self.epsilon = 1.0  # Differential privacy parameter
        self.delta = 1e-5   # Differential privacy parameter
        self.noise_multiplier = 1.0
        self.encryption_key = self._generate_encryption_key()
        
    def _generate_encryption_key(self) -> Fernet:
        """Generate encryption key for sensitive data."""
        password = b"federated_materials_discovery_key"
        salt = b"salt_for_federation"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    def add_differential_privacy_noise(self, 
                                     parameters: np.ndarray,
                                     sensitivity: float = 1.0) -> np.ndarray:
        """Add differential privacy noise to model parameters."""
        if self.privacy_level != PrivacyLevel.DIFFERENTIAL_PRIVACY:
            return parameters
            
        # Gaussian mechanism for differential privacy
        sigma = sensitivity * self.noise_multiplier / self.epsilon
        noise = np.random.normal(0, sigma, parameters.shape)
        
        return parameters + noise
    
    def encrypt_parameters(self, parameters: np.ndarray) -> bytes:
        """Encrypt model parameters for secure transmission."""
        if self.privacy_level == PrivacyLevel.NONE:
            return pickle.dumps(parameters)
            
        # Serialize and encrypt
        serialized = pickle.dumps(parameters)
        encrypted = self.encryption_key.encrypt(serialized)
        
        return encrypted
    
    def decrypt_parameters(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt model parameters."""
        if self.privacy_level == PrivacyLevel.NONE:
            return pickle.loads(encrypted_data)
            
        # Decrypt and deserialize
        decrypted = self.encryption_key.decrypt(encrypted_data)
        parameters = pickle.loads(decrypted)
        
        return parameters
    
    def anonymize_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Anonymize gradients using secure aggregation techniques."""
        if self.privacy_level in [PrivacyLevel.NONE, PrivacyLevel.DIFFERENTIAL_PRIVACY]:
            return gradients
            
        # Add random masking for secure aggregation
        mask = np.random.normal(0, 0.1, gradients.shape)
        return gradients + mask
    
    def validate_privacy_budget(self, used_budget: float) -> bool:
        """Validate that privacy budget hasn't been exceeded."""
        privacy_budget_limit = 10.0  # Total epsilon budget
        return used_budget <= privacy_budget_limit


class FederatedAggregator:
    """Handles aggregation of model updates from multiple laboratories."""
    
    def __init__(self, aggregation_strategy: str = "fedavg"):
        self.aggregation_strategy = aggregation_strategy
        self.aggregation_weights = {}
        
    def federated_averaging(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Federated averaging aggregation (FedAvg)."""
        if not updates:
            raise ValueError("No updates to aggregate")
            
        # Calculate weights based on data size
        total_data = sum(update.data_size for update in updates)
        
        if total_data == 0:
            # Equal weighting if no data size information
            weights = [1.0 / len(updates) for _ in updates]
        else:
            weights = [update.data_size / total_data for update in updates]
        
        # Weighted average of parameters
        aggregated_params = None
        
        for update, weight in zip(updates, weights):
            if update.parameters is not None:
                if aggregated_params is None:
                    aggregated_params = weight * update.parameters
                else:
                    aggregated_params += weight * update.parameters
        
        return aggregated_params
    
    def secure_aggregation(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Secure aggregation with privacy protection."""
        # Implement secure multi-party computation simulation
        
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Phase 1: Each lab adds random noise
        masked_updates = []
        for update in updates:
            if update.parameters is not None:
                # Simulate secure masking
                random_mask = np.random.normal(0, 0.01, update.parameters.shape)
                masked_params = update.parameters + random_mask
                masked_updates.append(masked_params)
        
        # Phase 2: Aggregate masked parameters
        if not masked_updates:
            return np.array([])
            
        aggregated = np.mean(masked_updates, axis=0)
        
        # Phase 3: Remove masks (simulated)
        # In real implementation, masks would cancel out through cryptographic protocols
        
        return aggregated
    
    def reputation_weighted_aggregation(self, 
                                      updates: List[ModelUpdate],
                                      lab_reputations: Dict[str, float]) -> np.ndarray:
        """Aggregate updates weighted by laboratory reputation scores."""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Calculate reputation-based weights
        weights = []
        for update in updates:
            lab_reputation = lab_reputations.get(update.lab_id, 1.0)
            data_weight = update.data_size if update.data_size > 0 else 1.0
            combined_weight = lab_reputation * data_weight
            weights.append(combined_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(updates) for _ in updates]
        
        # Weighted aggregation
        aggregated_params = None
        
        for update, weight in zip(updates, weights):
            if update.parameters is not None:
                if aggregated_params is None:
                    aggregated_params = weight * update.parameters
                else:
                    aggregated_params += weight * update.parameters
        
        return aggregated_params
    
    def aggregate_updates(self, 
                         updates: List[ModelUpdate],
                         lab_nodes: Dict[str, LabNode] = None) -> np.ndarray:
        """Aggregate model updates using the configured strategy."""
        if self.aggregation_strategy == "fedavg":
            return self.federated_averaging(updates)
        elif self.aggregation_strategy == "secure":
            return self.secure_aggregation(updates)
        elif self.aggregation_strategy == "reputation" and lab_nodes:
            reputations = {lab_id: lab.reputation_score for lab_id, lab in lab_nodes.items()}
            return self.reputation_weighted_aggregation(updates, reputations)
        else:
            # Default to federated averaging
            return self.federated_averaging(updates)


class FederatedLearningCoordinator:
    """Main coordinator for federated learning across multiple laboratories."""
    
    def __init__(self, 
                 lab_name: str,
                 institution: str = "",
                 role: LabRole = LabRole.COORDINATOR):
        self.lab_id = str(uuid.uuid4())
        self.lab_name = lab_name
        self.institution = institution
        self.role = role
        
        # Network management
        self.connected_labs: Dict[str, LabNode] = {}
        self.federated_models: Dict[str, FederatedModel] = {}
        self.model_updates: Dict[str, List[ModelUpdate]] = {}
        
        # Components
        self.privacy_manager = PrivacyManager()
        self.aggregator = FederatedAggregator()
        
        # State
        self.federation_status = FederationStatus.IDLE
        self.current_round = 0
        self.training_history: List[Dict[str, Any]] = []
        
        # Security
        self.trusted_institutions = set()
        self.reputation_decay_rate = 0.95
        
    async def register_lab(self, lab_info: Dict[str, Any]) -> LabNode:
        """Register a new laboratory in the federation.
        
        Args:
            lab_info: Laboratory information dictionary
            
        Returns:
            Created LabNode instance
        """
        lab_node = LabNode(
            name=lab_info.get("name", ""),
            institution=lab_info.get("institution", ""),
            role=LabRole(lab_info.get("role", "participant")),
            endpoint=lab_info.get("endpoint", ""),
            capabilities=lab_info.get("capabilities", [])
        )
        
        # Validate lab credentials
        if not await self._validate_lab_credentials(lab_node):
            raise ValueError(f"Invalid credentials for lab: {lab_node.name}")
        
        # Initial trust assessment
        lab_node.trust_score = await self._assess_initial_trust(lab_node)
        
        self.connected_labs[lab_node.id] = lab_node
        
        logger.info(f"Registered lab: {lab_node.name} from {lab_node.institution}")
        logger.info(f"Lab capabilities: {lab_node.capabilities}")
        
        return lab_node
    
    async def _validate_lab_credentials(self, lab_node: LabNode) -> bool:
        """Validate laboratory credentials and authorization."""
        # Check institution whitelist
        if self.trusted_institutions and lab_node.institution not in self.trusted_institutions:
            logger.warning(f"Institution not in trusted list: {lab_node.institution}")
            return False
        
        # Validate capabilities
        required_capabilities = ["materials_synthesis", "characterization"]
        if not any(cap in lab_node.capabilities for cap in required_capabilities):
            logger.warning(f"Lab lacks required capabilities: {lab_node.capabilities}")
            return False
        
        # Additional security checks would go here
        # - Certificate validation
        # - Reputation checking
        # - Institution verification
        
        return True
    
    async def _assess_initial_trust(self, lab_node: LabNode) -> float:
        """Assess initial trust score for a new laboratory."""
        trust_score = 1.0  # Default trust
        
        # Adjust based on institution reputation
        institution_bonuses = {
            "MIT": 0.2,
            "Stanford": 0.2,
            "UC Berkeley": 0.15,
            "Cambridge": 0.15,
            "ETH Zurich": 0.1
        }
        
        trust_score += institution_bonuses.get(lab_node.institution, 0.0)
        
        # Adjust based on capabilities
        if "advanced_characterization" in lab_node.capabilities:
            trust_score += 0.1
        if "automated_synthesis" in lab_node.capabilities:
            trust_score += 0.1
        
        return min(2.0, trust_score)  # Cap at 2.0
    
    async def create_federated_model(self,
                                   model_config: Dict[str, Any]) -> FederatedModel:
        """Create a new federated learning model.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Created FederatedModel instance
        """
        federated_model = FederatedModel(
            name=model_config.get("name", "federated_materials_model"),
            model_type=model_config.get("model_type", "neural_network"),
            target_properties=model_config.get("target_properties", ["band_gap"]),
            architecture=model_config.get("architecture", {}),
            privacy_level=PrivacyLevel(model_config.get("privacy_level", "differential_privacy"))
        )
        
        # Initialize model parameters
        if "parameter_size" in model_config:
            param_size = model_config["parameter_size"]
            federated_model.parameters = np.random.normal(0, 0.1, param_size)
        
        self.federated_models[federated_model.id] = federated_model
        self.model_updates[federated_model.id] = []
        
        logger.info(f"Created federated model: {federated_model.name}")
        logger.info(f"Target properties: {federated_model.target_properties}")
        
        return federated_model
    
    async def start_training_round(self, model_id: str) -> Dict[str, Any]:
        """Start a new federated training round.
        
        Args:
            model_id: ID of the model to train
            
        Returns:
            Training round information
        """
        if model_id not in self.federated_models:
            raise ValueError(f"Model not found: {model_id}")
        
        model = self.federated_models[model_id]
        self.current_round += 1
        self.federation_status = FederationStatus.TRAINING
        
        # Select participating labs
        participating_labs = await self._select_participating_labs(model)
        model.participating_labs = [lab.id for lab in participating_labs]
        
        # Prepare training round
        round_info = {
            "model_id": model_id,
            "round_number": self.current_round,
            "participating_labs": len(participating_labs),
            "global_parameters": model.parameters.tolist() if model.parameters is not None else None,
            "target_properties": model.target_properties,
            "privacy_level": model.privacy_level.value,
            "deadline": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        logger.info(f"Started training round {self.current_round} for model {model.name}")
        logger.info(f"Participating labs: {len(participating_labs)}")
        
        # Notify participating labs (simulated)
        await self._notify_participating_labs(participating_labs, round_info)
        
        return round_info
    
    async def _select_participating_labs(self, model: FederatedModel) -> List[LabNode]:
        """Select laboratories to participate in training round."""
        # Filter labs based on capabilities and trust
        eligible_labs = []
        
        for lab in self.connected_labs.values():
            # Check if lab can handle target properties
            if self._lab_can_handle_properties(lab, model.target_properties):
                # Check trust and reputation thresholds
                if lab.trust_score >= 0.5 and lab.reputation_score >= 0.5:
                    eligible_labs.append(lab)
        
        # Select top labs by combined score
        def selection_score(lab: LabNode) -> float:
            return lab.trust_score * lab.reputation_score * (1 + lab.experiments_contributed / 100)
        
        eligible_labs.sort(key=selection_score, reverse=True)
        
        # Select up to 10 labs for training round
        selected_labs = eligible_labs[:10]
        
        logger.info(f"Selected {len(selected_labs)} labs from {len(eligible_labs)} eligible")
        
        return selected_labs
    
    def _lab_can_handle_properties(self, lab: LabNode, target_properties: List[str]) -> bool:
        """Check if lab can measure/optimize target properties."""
        lab_capabilities = set(lab.capabilities)
        
        # Map properties to required capabilities
        property_requirements = {
            "band_gap": {"uv_vis_spectroscopy", "characterization"},
            "efficiency": {"solar_cell_testing", "characterization"},
            "stability": {"long_term_testing", "characterization"},
            "conductivity": {"electrical_testing", "characterization"}
        }
        
        for prop in target_properties:
            required_caps = property_requirements.get(prop, {"characterization"})
            if not any(cap in lab_capabilities for cap in required_caps):
                return False
        
        return True
    
    async def _notify_participating_labs(self, 
                                       labs: List[LabNode],
                                       round_info: Dict[str, Any]) -> None:
        """Notify participating labs about new training round."""
        # In real implementation, this would send network requests
        for lab in labs:
            logger.debug(f"Notifying lab {lab.name} about training round {round_info['round_number']}")
            # Simulate network communication delay
            await asyncio.sleep(0.1)
    
    async def receive_model_update(self, update_data: Dict[str, Any]) -> bool:
        """Receive and validate a model update from a participating lab.
        
        Args:
            update_data: Model update data
            
        Returns:
            True if update was accepted, False otherwise
        """
        try:
            # Validate update structure
            required_fields = ["lab_id", "model_id", "round_number", "parameters"]
            if not all(field in update_data for field in required_fields):
                logger.warning("Received incomplete model update")
                return False
            
            # Validate lab authorization
            lab_id = update_data["lab_id"]
            if lab_id not in self.connected_labs:
                logger.warning(f"Received update from unknown lab: {lab_id}")
                return False
            
            # Create ModelUpdate instance
            model_update = ModelUpdate(
                lab_id=lab_id,
                model_id=update_data["model_id"],
                round_number=update_data["round_number"],
                parameters=np.array(update_data["parameters"]),
                local_performance=update_data.get("local_performance", {}),
                data_size=update_data.get("data_size", 0),
                computation_time=update_data.get("computation_time", 0.0)
            )
            
            # Validate update integrity
            if not await self._validate_model_update(model_update):
                return False
            
            # Apply privacy protection
            if self.privacy_manager.privacy_level != PrivacyLevel.NONE:
                model_update.parameters = self.privacy_manager.add_differential_privacy_noise(
                    model_update.parameters
                )
            
            # Store update
            model_id = model_update.model_id
            if model_id not in self.model_updates:
                self.model_updates[model_id] = []
            
            self.model_updates[model_id].append(model_update)
            
            # Update lab statistics
            lab = self.connected_labs[lab_id]
            lab.model_updates_contributed += 1
            lab.last_seen = datetime.now()
            
            logger.info(f"Received model update from {lab.name} for round {model_update.round_number}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing model update: {e}")
            return False
    
    async def _validate_model_update(self, update: ModelUpdate) -> bool:
        """Validate the integrity and quality of a model update."""
        # Check parameter dimensions
        model = self.federated_models.get(update.model_id)
        if model and model.parameters is not None:
            if update.parameters.shape != model.parameters.shape:
                logger.warning("Model update has incorrect parameter dimensions")
                return False
        
        # Check for reasonable parameter values
        if np.any(np.isnan(update.parameters)) or np.any(np.isinf(update.parameters)):
            logger.warning("Model update contains invalid parameter values")
            return False
        
        # Check parameter magnitude (potential attack detection)
        param_norm = np.linalg.norm(update.parameters)
        if param_norm > 100:  # Threshold for suspiciously large updates
            logger.warning(f"Model update has suspiciously large parameters: {param_norm}")
            return False
        
        # Check round number consistency
        if update.round_number != self.current_round:
            logger.warning(f"Model update for incorrect round: {update.round_number} vs {self.current_round}")
            return False
        
        return True
    
    async def aggregate_model_updates(self, model_id: str) -> bool:
        """Aggregate received model updates and update global model.
        
        Args:
            model_id: ID of the model to aggregate
            
        Returns:
            True if aggregation was successful
        """
        if model_id not in self.federated_models:
            logger.error(f"Model not found for aggregation: {model_id}")
            return False
        
        updates = self.model_updates.get(model_id, [])
        current_round_updates = [u for u in updates if u.round_number == self.current_round]
        
        if not current_round_updates:
            logger.warning(f"No updates received for model {model_id} in round {self.current_round}")
            return False
        
        logger.info(f"Aggregating {len(current_round_updates)} updates for model {model_id}")
        
        self.federation_status = FederationStatus.AGGREGATING
        
        try:
            # Aggregate updates
            aggregated_params = self.aggregator.aggregate_updates(
                current_round_updates, 
                self.connected_labs
            )
            
            # Update global model
            model = self.federated_models[model_id]
            model.parameters = aggregated_params
            model.training_rounds += 1
            model.last_updated = datetime.now()
            
            # Calculate aggregated performance metrics
            performance_metrics = self._calculate_aggregated_performance(current_round_updates)
            model.performance_metrics.update(performance_metrics)
            
            # Update lab reputations based on contribution quality
            await self._update_lab_reputations(current_round_updates, performance_metrics)
            
            # Store training history
            round_summary = {
                "round": self.current_round,
                "timestamp": datetime.now().isoformat(),
                "participating_labs": len(current_round_updates),
                "performance_metrics": performance_metrics,
                "aggregation_strategy": self.aggregator.aggregation_strategy
            }
            self.training_history.append(round_summary)
            
            logger.info(f"Model aggregation completed for round {self.current_round}")
            logger.info(f"Performance metrics: {performance_metrics}")
            
            self.federation_status = FederationStatus.IDLE
            return True
            
        except Exception as e:
            logger.error(f"Error during model aggregation: {e}")
            self.federation_status = FederationStatus.ERROR
            return False
    
    def _calculate_aggregated_performance(self, updates: List[ModelUpdate]) -> Dict[str, float]:
        """Calculate aggregated performance metrics from local updates."""
        metrics = {}
        
        # Collect all local performance metrics
        all_metrics = {}
        for update in updates:
            for metric, value in update.local_performance.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate weighted averages
        total_data = sum(update.data_size for update in updates)
        
        for metric, values in all_metrics.items():
            if total_data > 0:
                # Weight by data size
                weights = [update.data_size / total_data for update in updates]
                weighted_avg = sum(v * w for v, w in zip(values, weights))
            else:
                # Simple average
                weighted_avg = np.mean(values)
            
            metrics[f"avg_{metric}"] = weighted_avg
            metrics[f"std_{metric}"] = np.std(values)
        
        # Add federation-specific metrics
        metrics["participating_labs"] = len(updates)
        metrics["total_data_samples"] = total_data
        metrics["avg_computation_time"] = np.mean([u.computation_time for u in updates])
        
        return metrics
    
    async def _update_lab_reputations(self, 
                                    updates: List[ModelUpdate],
                                    performance_metrics: Dict[str, float]) -> None:
        """Update laboratory reputation scores based on contribution quality."""
        avg_performance = performance_metrics.get("avg_accuracy", 0.5)
        
        for update in updates:
            lab = self.connected_labs.get(update.lab_id)
            if not lab:
                continue
            
            # Calculate contribution quality score
            local_performance = update.local_performance.get("accuracy", avg_performance)
            performance_ratio = local_performance / avg_performance if avg_performance > 0 else 1.0
            
            # Factor in data contribution size
            data_contribution = update.data_size / 100  # Normalize
            data_bonus = min(0.2, data_contribution * 0.01)
            
            # Factor in computation time (faster is better, up to a point)
            time_factor = max(0.5, 1.0 - update.computation_time / 3600)  # Penalize > 1 hour
            
            # Calculate reputation update
            reputation_delta = (performance_ratio - 1.0) * 0.1 + data_bonus + (time_factor - 1.0) * 0.05
            
            # Apply update with decay
            lab.reputation_score = (
                lab.reputation_score * self.reputation_decay_rate + 
                max(0.1, lab.reputation_score + reputation_delta)
            )
            
            # Clamp reputation score
            lab.reputation_score = np.clip(lab.reputation_score, 0.1, 2.0)
            
            logger.debug(f"Updated reputation for {lab.name}: {lab.reputation_score:.3f}")
    
    async def evaluate_federated_model(self, 
                                     model_id: str,
                                     test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate federated model performance on test data.
        
        Args:
            model_id: ID of the model to evaluate
            test_data: Test dataset
            
        Returns:
            Evaluation metrics
        """
        if model_id not in self.federated_models:
            raise ValueError(f"Model not found: {model_id}")
        
        model = self.federated_models[model_id]
        self.federation_status = FederationStatus.VALIDATING
        
        # Simulate model evaluation
        # In real implementation, this would use the actual model
        evaluation_metrics = {
            "accuracy": np.random.uniform(0.7, 0.95),
            "precision": np.random.uniform(0.7, 0.9),
            "recall": np.random.uniform(0.7, 0.9),
            "f1_score": np.random.uniform(0.7, 0.9),
            "mse": np.random.uniform(0.01, 0.1),
            "mae": np.random.uniform(0.05, 0.15)
        }
        
        # Add federation-specific metrics
        evaluation_metrics.update({
            "total_training_rounds": model.training_rounds,
            "participating_labs": len(model.participating_labs),
            "data_privacy_level": model.privacy_level.value,
            "model_convergence": np.random.uniform(0.8, 0.95)
        })
        
        # Update model performance
        model.performance_metrics.update(evaluation_metrics)
        
        logger.info(f"Model evaluation completed for {model.name}")
        logger.info(f"Accuracy: {evaluation_metrics['accuracy']:.3f}")
        
        self.federation_status = FederationStatus.COMPLETE
        
        return evaluation_metrics
    
    def get_federation_summary(self) -> Dict[str, Any]:
        """Get summary of federated learning status and metrics."""
        # Calculate summary statistics
        total_labs = len(self.connected_labs)
        active_labs = len([lab for lab in self.connected_labs.values() 
                          if (datetime.now() - lab.last_seen).days < 7])
        
        total_models = len(self.federated_models)
        total_updates = sum(len(updates) for updates in self.model_updates.values())
        
        avg_trust = np.mean([lab.trust_score for lab in self.connected_labs.values()]) if self.connected_labs else 0
        avg_reputation = np.mean([lab.reputation_score for lab in self.connected_labs.values()]) if self.connected_labs else 0
        
        # Model performance summary
        model_performances = []
        for model in self.federated_models.values():
            if "avg_accuracy" in model.performance_metrics:
                model_performances.append(model.performance_metrics["avg_accuracy"])
        
        avg_model_performance = np.mean(model_performances) if model_performances else 0
        
        return {
            "federation_status": self.federation_status.value,
            "current_round": self.current_round,
            "total_labs": total_labs,
            "active_labs": active_labs,
            "total_models": total_models,
            "total_training_rounds": sum(model.training_rounds for model in self.federated_models.values()),
            "total_model_updates": total_updates,
            "average_trust_score": avg_trust,
            "average_reputation_score": avg_reputation,
            "average_model_performance": avg_model_performance,
            "privacy_level": self.privacy_manager.privacy_level.value,
            "aggregation_strategy": self.aggregator.aggregation_strategy,
            "training_history_length": len(self.training_history)
        }
    
    async def export_model(self, model_id: str, export_path: str) -> bool:
        """Export a federated model for deployment.
        
        Args:
            model_id: ID of the model to export
            export_path: Path to save the exported model
            
        Returns:
            True if export was successful
        """
        if model_id not in self.federated_models:
            logger.error(f"Model not found for export: {model_id}")
            return False
        
        model = self.federated_models[model_id]
        
        # Prepare export data
        export_data = {
            "model": model.to_dict(),
            "training_history": self.training_history,
            "participating_labs": [
                {
                    "id": lab.id,
                    "name": lab.name,
                    "institution": lab.institution,
                    "contributions": lab.model_updates_contributed
                }
                for lab in self.connected_labs.values()
                if lab.id in model.participating_labs
            ],
            "federation_metadata": {
                "coordinator": self.lab_name,
                "export_timestamp": datetime.now().isoformat(),
                "total_training_rounds": model.training_rounds,
                "privacy_level": model.privacy_level.value
            }
        }
        
        try:
            # Save to file
            export_path_obj = Path(export_path)
            export_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path_obj, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported model {model.name} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            return False


# Global instance for easy access
_global_federation_coordinator: Optional[FederatedLearningCoordinator] = None


def get_global_federation_coordinator(lab_name: str = "Global Lab") -> FederatedLearningCoordinator:
    """Get the global federated learning coordinator instance."""
    global _global_federation_coordinator
    if _global_federation_coordinator is None:
        _global_federation_coordinator = FederatedLearningCoordinator(lab_name=lab_name)
    return _global_federation_coordinator


async def create_federated_materials_network(coordinator_config: Dict[str, Any]) -> FederatedLearningCoordinator:
    """Create a federated materials discovery network.
    
    Args:
        coordinator_config: Configuration for the federation coordinator
        
    Returns:
        Configured FederatedLearningCoordinator instance
    """
    coordinator = FederatedLearningCoordinator(
        lab_name=coordinator_config.get("lab_name", "Federation Coordinator"),
        institution=coordinator_config.get("institution", ""),
        role=LabRole.COORDINATOR
    )
    
    # Set up privacy configuration
    privacy_level = PrivacyLevel(coordinator_config.get("privacy_level", "differential_privacy"))
    coordinator.privacy_manager = PrivacyManager(privacy_level)
    
    # Set up aggregation strategy
    aggregation_strategy = coordinator_config.get("aggregation_strategy", "fedavg")
    coordinator.aggregator = FederatedAggregator(aggregation_strategy)
    
    # Set trusted institutions
    trusted_institutions = coordinator_config.get("trusted_institutions", [])
    coordinator.trusted_institutions = set(trusted_institutions)
    
    logger.info(f"Created federated learning coordinator: {coordinator.lab_name}")
    logger.info(f"Privacy level: {privacy_level.value}")
    logger.info(f"Aggregation strategy: {aggregation_strategy}")
    
    return coordinator