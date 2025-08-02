# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the self-driving-materials-orchestrator project.

## What are ADRs?

Architecture Decision Records (ADRs) are short text documents that capture important architectural decisions made during the project, along with their context and consequences.

## ADR Format

We use the template suggested by Michael Nygard in his article "Documenting Architecture Decisions". Each ADR should include:

1. **Title** - A short noun phrase describing the architectural decision
2. **Status** - Proposed, Accepted, Deprecated, or Superseded
3. **Context** - What is the issue that we're trying to solve?
4. **Decision** - What is the change that we're proposing and/or doing?
5. **Consequences** - What becomes easier or more difficult to do because of this change?

## Current ADRs

- [ADR-0000: Use Architecture Decision Records](0000-use-architecture-decision-records.md)
- [ADR-0001: Choose Python as Primary Language](0001-choose-python-as-primary-language.md)
- [ADR-0002: Use MongoDB for Data Storage](0002-use-mongodb-for-data-storage.md)
- [ADR-0003: Adopt Bayesian Optimization for Experiment Planning](0003-adopt-bayesian-optimization.md)
- [ADR-0004: Use Docker for Containerization](0004-use-docker-for-containerization.md)

## Creating New ADRs

1. Copy the template from `0000-adr-template.md`
2. Rename to the next available number
3. Fill in the sections
4. Submit for review through pull request
5. Update this README with the new ADR

## ADR Lifecycle

- **Proposed**: Under discussion
- **Accepted**: Decision is final and being implemented
- **Deprecated**: No longer recommended but not actively harmful
- **Superseded**: Replaced by a newer decision (link to replacement)

## Contributing

When making significant architectural decisions:
1. Create an ADR documenting the decision
2. Seek input from relevant stakeholders
3. Update implementation to reflect the decision
4. Keep ADRs up to date as architecture evolves