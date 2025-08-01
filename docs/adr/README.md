# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Self-Driving Materials Orchestrator project.

## ADR Format

We use a simplified ADR format based on Michael Nygard's template:

```markdown
# ADR-XXXX: Title

**Status:** [Proposed | Accepted | Superseded | Deprecated]
**Date:** YYYY-MM-DD
**Deciders:** [List of people involved in the decision]

## Context

What is the issue that we're seeing that is motivating this decision or change?

## Decision

What is the change that we're proposing or have agreed to implement?

## Consequences

What becomes easier or more difficult to do and any risks introduced by this change?

## Alternatives Considered

What other options were considered and why were they rejected?
```

## Current ADRs

1. [ADR-0001: Database Technology Selection](adr-0001-database-selection.md)
2. [ADR-0002: Robot Communication Architecture](adr-0002-robot-communication.md)
3. [ADR-0003: Optimization Framework Choice](adr-0003-optimization-framework.md)
4. [ADR-0004: API Design Pattern](adr-0004-api-design.md)
5. [ADR-0005: Containerization Strategy](adr-0005-containerization.md)

## Creating New ADRs

When creating a new ADR:

1. Use the next sequential number (ADR-XXXX)
2. Use a descriptive title that summarizes the decision
3. Follow the format template above
4. Add the new ADR to the list above
5. Commit the ADR with your implementation changes

## ADR Guidelines

- **Be specific**: Focus on architectural decisions, not implementation details
- **Provide context**: Explain why the decision was needed
- **Document alternatives**: Show what other options were considered
- **Update status**: Mark as Superseded when decisions change
- **Keep it concise**: ADRs should be readable in 5-10 minutes