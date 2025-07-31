// MongoDB initialization script for materials discovery database

db = db.getSiblingDB('materials_discovery');

// Create collections with validation
db.createCollection('experiments', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['timestamp', 'parameters', 'results', 'metadata'],
      properties: {
        timestamp: {
          bsonType: 'date',
          description: 'Experiment timestamp is required'
        },
        parameters: {
          bsonType: 'object',
          description: 'Experiment parameters object is required'
        },
        results: {
          bsonType: 'object',
          description: 'Experiment results object is required'
        },
        metadata: {
          bsonType: 'object',
          required: ['operator', 'campaign_id'],
          properties: {
            operator: {
              bsonType: 'string',
              description: 'Operator identification is required'
            },
            campaign_id: {
              bsonType: 'string',
              description: 'Campaign ID is required'
            }
          }
        }
      }
    }
  }
});

db.createCollection('campaigns', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['campaign_id', 'objective', 'status', 'created_at'],
      properties: {
        campaign_id: {
          bsonType: 'string',
          description: 'Campaign ID is required'
        },
        objective: {
          bsonType: 'object',
          description: 'Campaign objective is required'
        },
        status: {
          enum: ['active', 'paused', 'completed', 'failed'],
          description: 'Status must be one of: active, paused, completed, failed'
        },
        created_at: {
          bsonType: 'date',
          description: 'Creation timestamp is required'
        }
      }
    }
  }
});

// Create indexes for performance
db.experiments.createIndex({ 'timestamp': -1 });
db.experiments.createIndex({ 'metadata.campaign_id': 1 });
db.experiments.createIndex({ 'metadata.operator': 1 });
db.experiments.createIndex({ 'results.band_gap': 1 });
db.experiments.createIndex({ 'results.efficiency': -1 });

db.campaigns.createIndex({ 'campaign_id': 1 }, { unique: true });
db.campaigns.createIndex({ 'status': 1 });
db.campaigns.createIndex({ 'created_at': -1 });

// Create user with appropriate permissions
db.createUser({
  user: 'materials_user',
  pwd: 'materials_password',
  roles: [
    {
      role: 'readWrite',
      db: 'materials_discovery'
    }
  ]
});

print('Materials discovery database initialized successfully!');