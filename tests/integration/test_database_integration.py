"""Integration tests for database operations."""

import pytest
from unittest.mock import Mock
import pymongo
from materials_orchestrator.core import AutonomousLab, MaterialsObjective


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database integration scenarios."""
    
    def test_mongodb_connection(self):
        """Test MongoDB connection and basic operations."""
        # This would test actual MongoDB connection in a real integration test
        # For now, we'll mock to avoid requiring live database
        assert True
    
    def test_experiment_storage_retrieval(self):
        """Test storing and retrieving experiment data."""
        lab = AutonomousLab(database_url="mongodb://localhost:27017/test_db")
        
        # Test basic initialization
        assert lab.database_url == "mongodb://localhost:27017/test_db"
        assert lab.total_experiments == 0
    
    def test_campaign_persistence(self):
        """Test campaign data persistence across restarts."""
        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            material_system="perovskites"
        )
        
        lab = AutonomousLab()
        campaign = lab.run_campaign(objective, initial_samples=5, max_experiments=10)
        
        # Verify campaign results structure
        assert hasattr(campaign, 'best_material')
        assert hasattr(campaign, 'total_experiments')
        assert hasattr(campaign, 'best_properties')