# ADR-0001: Database Technology Selection

**Status:** Accepted  
**Date:** 2025-08-01  
**Deciders:** Development Team, Materials Science SMEs

## Context

The self-driving materials orchestrator needs to store and manage several types of data:

- Experimental parameters and results
- Material property measurements
- Campaign metadata and provenance
- Robot execution logs
- ML model training data

The database must support:
- Complex nested documents (experimental protocols)
- Time-series data (measurements over time)
- Rich querying capabilities for materials discovery
- Horizontal scaling for large campaigns
- Integration with Python ML ecosystem

## Decision

We will use **MongoDB** as our primary database technology.

## Consequences

### Positive
- **Flexible schema**: Easy to store complex experimental protocols and nested data structures
- **Rich query language**: Supports complex aggregation pipelines for materials analysis
- **Python integration**: Excellent pymongo driver and integration with pandas
- **Horizontal scaling**: Built-in sharding support for large datasets
- **JSON/BSON format**: Natural fit for REST API data exchange
- **Indexing**: Powerful indexing capabilities for material property searches

### Negative
- **Consistency model**: Eventually consistent in distributed setups
- **Memory usage**: Higher memory footprint compared to relational databases
- **SQL familiarity**: Team needs to learn MongoDB query syntax
- **Complex transactions**: Limited cross-document transaction support

### Risks
- **Vendor lock-in**: MongoDB-specific query language and features
- **Schema evolution**: Need careful planning for data migration
- **Backup complexity**: Requires MongoDB-specific backup strategies

## Alternatives Considered

### PostgreSQL with JSONB
- **Pros**: ACID compliance, familiar SQL, good JSON support
- **Cons**: Less flexible for nested documents, more complex ORM mapping
- **Decision**: Rejected due to complex experimental protocol storage requirements

### InfluxDB
- **Pros**: Excellent for time-series measurement data
- **Cons**: Poor support for complex document structures, limited querying
- **Decision**: Rejected as specialized for time-series only

### Elasticsearch
- **Pros**: Excellent search capabilities, good for analytics
- **Cons**: Not designed as primary database, complex cluster management
- **Decision**: Rejected as search engine, not transactional database

### SQLite
- **Pros**: Simple setup, good for development
- **Cons**: No horizontal scaling, poor concurrent write performance
- **Decision**: Rejected due to scalability limitations

## Implementation Notes

- Use MongoDB 5.0+ for improved transaction support
- Implement proper indexing strategy for material property queries
- Use replica sets for high availability
- Consider read replicas for analytics workloads
- Implement proper backup and disaster recovery procedures