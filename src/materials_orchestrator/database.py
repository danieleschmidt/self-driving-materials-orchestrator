"""Database integration for experiment data management."""

from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import json
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pymongo
    from pymongo import MongoClient, ASCENDING, DESCENDING
    PYMONGO_AVAILABLE = True
except ImportError:
    logger.warning("pymongo not available, using file-based storage")
    PYMONGO_AVAILABLE = False


@dataclass
class ExperimentRecord:
    """Structured experiment record for database storage."""
    id: str
    campaign_id: str
    timestamp: datetime
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    status: str
    duration: Optional[float] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DatabaseInterface(ABC):
    """Abstract interface for experiment databases."""
    
    @abstractmethod
    def store_experiment(self, experiment: ExperimentRecord) -> bool:
        """Store an experiment record."""
        pass
    
    @abstractmethod
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Retrieve a specific experiment."""
        pass
    
    @abstractmethod
    def query_experiments(
        self,
        campaign_id: Optional[str] = None,
        status: Optional[str] = None,
        date_range: Optional[tuple] = None,
        limit: Optional[int] = None
    ) -> List[ExperimentRecord]:
        """Query experiments with filters."""
        pass
    
    @abstractmethod
    def get_campaign_summary(self, campaign_id: str) -> Dict[str, Any]:
        """Get summary statistics for a campaign."""
        pass


class FileDatabase(DatabaseInterface):
    """File-based database implementation for development and testing."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.experiments_file = self.data_dir / "experiments.jsonl"
        
    def store_experiment(self, experiment: ExperimentRecord) -> bool:
        """Store experiment to JSONL file."""
        try:
            # Convert to dict and handle datetime serialization
            exp_dict = asdict(experiment)
            exp_dict['timestamp'] = experiment.timestamp.isoformat()
            
            with open(self.experiments_file, 'a') as f:
                f.write(json.dumps(exp_dict) + '\n')
            
            logger.debug(f"Stored experiment {experiment.id} to file")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store experiment {experiment.id}: {e}")
            return False
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Retrieve experiment by ID from file."""
        try:
            for exp_dict in self._read_experiments():
                if exp_dict['id'] == experiment_id:
                    return self._dict_to_experiment(exp_dict)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve experiment {experiment_id}: {e}")
            return None
    
    def query_experiments(
        self,
        campaign_id: Optional[str] = None,
        status: Optional[str] = None,
        date_range: Optional[tuple] = None,
        limit: Optional[int] = None
    ) -> List[ExperimentRecord]:
        """Query experiments with filters."""
        results = []
        count = 0
        
        try:
            for exp_dict in self._read_experiments():
                # Apply filters
                if campaign_id and exp_dict.get('campaign_id') != campaign_id:
                    continue
                if status and exp_dict.get('status') != status:
                    continue
                if date_range:
                    exp_time = datetime.fromisoformat(exp_dict['timestamp'])
                    if not (date_range[0] <= exp_time <= date_range[1]):
                        continue
                
                results.append(self._dict_to_experiment(exp_dict))
                count += 1
                
                if limit and count >= limit:
                    break
                    
        except Exception as e:
            logger.error(f"Failed to query experiments: {e}")
        
        return results
    
    def get_campaign_summary(self, campaign_id: str) -> Dict[str, Any]:
        """Get campaign summary statistics."""
        experiments = self.query_experiments(campaign_id=campaign_id)
        
        if not experiments:
            return {"campaign_id": campaign_id, "total_experiments": 0}
        
        successful = [e for e in experiments if e.status == "completed"]
        failed = [e for e in experiments if e.status == "failed"]
        
        # Calculate statistics
        total_duration = sum(e.duration or 0 for e in experiments)
        avg_duration = total_duration / len(experiments) if experiments else 0
        
        return {
            "campaign_id": campaign_id,
            "total_experiments": len(experiments),
            "successful_experiments": len(successful),
            "failed_experiments": len(failed),
            "success_rate": len(successful) / len(experiments) if experiments else 0,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "start_time": min(e.timestamp for e in experiments).isoformat(),
            "end_time": max(e.timestamp for e in experiments).isoformat(),
        }
    
    def _read_experiments(self) -> Iterator[Dict[str, Any]]:
        """Read experiments from JSONL file."""
        if not self.experiments_file.exists():
            return
            
        with open(self.experiments_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    
    def _dict_to_experiment(self, exp_dict: Dict[str, Any]) -> ExperimentRecord:
        """Convert dictionary to ExperimentRecord."""
        exp_dict['timestamp'] = datetime.fromisoformat(exp_dict['timestamp'])
        return ExperimentRecord(**exp_dict)


class MongoDatabase(DatabaseInterface):
    """MongoDB-based database implementation for production."""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", database_name: str = "materials_discovery"):
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required for MongoDB support")
            
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        self.experiments_collection = None
        self._connect()
    
    def _connect(self):
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.database_name]
            self.experiments_collection = self.db.experiments
            
            # Create indexes for better query performance
            self.experiments_collection.create_index("id", unique=True)
            self.experiments_collection.create_index("campaign_id")
            self.experiments_collection.create_index("timestamp")
            self.experiments_collection.create_index("status")
            
            logger.info(f"Connected to MongoDB: {self.database_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def store_experiment(self, experiment: ExperimentRecord) -> bool:
        """Store experiment in MongoDB."""
        try:
            exp_dict = asdict(experiment)
            self.experiments_collection.insert_one(exp_dict)
            logger.debug(f"Stored experiment {experiment.id} in MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store experiment {experiment.id}: {e}")
            return False
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Retrieve experiment by ID from MongoDB."""
        try:
            doc = self.experiments_collection.find_one({"id": experiment_id})
            if doc:
                doc.pop('_id', None)  # Remove MongoDB ObjectId
                return ExperimentRecord(**doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve experiment {experiment_id}: {e}")
            return None
    
    def query_experiments(
        self,
        campaign_id: Optional[str] = None,
        status: Optional[str] = None,
        date_range: Optional[tuple] = None,
        limit: Optional[int] = None
    ) -> List[ExperimentRecord]:
        """Query experiments with filters from MongoDB."""
        try:
            query = {}
            
            if campaign_id:
                query["campaign_id"] = campaign_id
            if status:
                query["status"] = status
            if date_range:
                query["timestamp"] = {"$gte": date_range[0], "$lte": date_range[1]}
            
            cursor = self.experiments_collection.find(query)
            
            if limit:
                cursor = cursor.limit(limit)
            
            results = []
            for doc in cursor:
                doc.pop('_id', None)  # Remove MongoDB ObjectId
                results.append(ExperimentRecord(**doc))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to query experiments: {e}")
            return []
    
    def get_campaign_summary(self, campaign_id: str) -> Dict[str, Any]:
        """Get campaign summary using MongoDB aggregation."""
        try:
            pipeline = [
                {"$match": {"campaign_id": campaign_id}},
                {
                    "$group": {
                        "_id": "$campaign_id",
                        "total_experiments": {"$sum": 1},
                        "successful_experiments": {
                            "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                        },
                        "failed_experiments": {
                            "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                        },
                        "total_duration": {"$sum": "$duration"},
                        "start_time": {"$min": "$timestamp"},
                        "end_time": {"$max": "$timestamp"},
                    }
                }
            ]
            
            result = list(self.experiments_collection.aggregate(pipeline))
            
            if result:
                stats = result[0]
                stats["campaign_id"] = campaign_id
                stats["success_rate"] = (
                    stats["successful_experiments"] / stats["total_experiments"]
                    if stats["total_experiments"] > 0 else 0
                )
                stats["average_duration"] = (
                    stats["total_duration"] / stats["total_experiments"]
                    if stats["total_experiments"] > 0 else 0
                )
                stats.pop('_id', None)
                return stats
            else:
                return {"campaign_id": campaign_id, "total_experiments": 0}
                
        except Exception as e:
            logger.error(f"Failed to get campaign summary: {e}")
            return {"campaign_id": campaign_id, "error": str(e)}
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()


def create_database(database_type: str = "file", **kwargs) -> DatabaseInterface:
    """Factory function to create database instance."""
    if database_type == "mongodb" and PYMONGO_AVAILABLE:
        return MongoDatabase(**kwargs)
    else:
        if database_type == "mongodb" and not PYMONGO_AVAILABLE:
            logger.warning("MongoDB requested but pymongo not available, using file database")
        return FileDatabase(**kwargs)


class ExperimentTracker:
    """High-level interface for tracking experiments."""
    
    def __init__(self, database: Optional[DatabaseInterface] = None):
        self.database = database or create_database()
    
    def track_experiment(self, experiment, campaign_id: str) -> bool:
        """Track an experiment from core.Experiment object."""
        record = ExperimentRecord(
            id=experiment.id,
            campaign_id=campaign_id,
            timestamp=experiment.timestamp,
            parameters=experiment.parameters,
            results=experiment.results,
            status=experiment.status,
            duration=experiment.duration,
            metadata=experiment.metadata
        )
        
        return self.database.store_experiment(record)
    
    def get_campaign_experiments(self, campaign_id: str) -> List[ExperimentRecord]:
        """Get all experiments for a campaign."""
        return self.database.query_experiments(campaign_id=campaign_id)
    
    def get_successful_experiments(self, campaign_id: str) -> List[ExperimentRecord]:
        """Get only successful experiments for a campaign."""
        return self.database.query_experiments(campaign_id=campaign_id, status="completed")
    
    def get_recent_experiments(self, hours: int = 24) -> List[ExperimentRecord]:
        """Get experiments from the last N hours."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        return self.database.query_experiments(date_range=(start_time, end_time))
    
    def analyze_parameter_performance(
        self, 
        campaign_id: str, 
        target_property: str
    ) -> Dict[str, Any]:
        """Analyze which parameters lead to better performance."""
        experiments = self.get_successful_experiments(campaign_id)
        
        if not experiments:
            return {"error": "No successful experiments found"}
        
        # Extract parameter values and target property values
        param_data = {}
        target_values = []
        
        for exp in experiments:
            if target_property in exp.results:
                target_values.append(exp.results[target_property])
                
                for param, value in exp.parameters.items():
                    if param not in param_data:
                        param_data[param] = []
                    param_data[param].append(value)
        
        if not target_values:
            return {"error": f"No experiments with {target_property} found"}
        
        # Calculate correlations (simplified)
        analysis = {
            "target_property": target_property,
            "num_experiments": len(target_values),
            "target_mean": sum(target_values) / len(target_values),
            "target_std": (
                sum((x - sum(target_values)/len(target_values))**2 for x in target_values) / len(target_values)
            )**0.5,
            "parameter_ranges": {}
        }
        
        for param, values in param_data.items():
            analysis["parameter_ranges"][param] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
            }
        
        return analysis
