# Self-Healing Pipeline Benchmark Report

## Executive Summary

**Study Date**: 2025-08-18 12:11:14
**Total Algorithms Tested**: 6
**Trials per Algorithm**: 30
**Test Duration**: 120 seconds

### Key Findings

#### 🏆 Algorithm Performance Rankings

1. **quantum_annealing** (Score: 4.346)
   - Success Rate: 30.0%
   - Avg Healing Time: 0.082s
   - Availability: 30.0%

2. **hybrid_quantum_classical** (Score: 3.830)
   - Success Rate: 20.0%
   - Avg Healing Time: 0.093s
   - Availability: 20.0%

3. **qaoa_optimization** (Score: 3.555)
   - Success Rate: 26.7%
   - Avg Healing Time: 0.101s
   - Availability: 26.7%

4. **vqe_hybrid** (Score: 3.103)
   - Success Rate: 36.7%
   - Avg Healing Time: 0.119s
   - Availability: 36.7%

5. **classical_healing** (Score: 2.523)
   - Success Rate: 20.0%
   - Avg Healing Time: 0.144s
   - Availability: 20.0%

6. **distributed_raft** (Score: 2.268)
   - Success Rate: 23.3%
   - Avg Healing Time: 0.162s
   - Availability: 23.3%

## Detailed Performance Metrics

| Algorithm | Success Rate | Avg Healing Time | Throughput | MTTR | MTBF |
|-----------|--------------|------------------|------------|------|------|
| classical_healing | 20.0% | 0.144s | 4.7 ops/s | 0.144s | 0.3s |
| quantum_annealing | 30.0% | 0.082s | 6.6 ops/s | 0.082s | 0.2s |
| qaoa_optimization | 26.7% | 0.101s | 5.9 ops/s | 0.101s | 0.2s |
| vqe_hybrid | 36.7% | 0.119s | 5.2 ops/s | 0.119s | 0.3s |
| distributed_raft | 23.3% | 0.162s | 4.5 ops/s | 0.162s | 0.3s |
| hybrid_quantum_classical | 20.0% | 0.093s | 6.5 ops/s | 0.093s | 0.2s |

## Test Configuration

- **Number of Trials**: 30
- **Max Concurrent Failures**: 3
- **Failure Injection Rate**: 0.08 failures/second
- **System Load Multiplier**: 1.1
- **Quantum Noise Level**: 0.015

## Statistical Significance

- **classical_healing_vs_quantum_annealing**: 0.100
- **classical_healing_vs_qaoa_optimization**: 0.067
- **classical_healing_vs_vqe_hybrid**: 0.167
- **classical_healing_vs_distributed_raft**: 0.033
- **classical_healing_vs_hybrid_quantum_classical**: 0.000
- **quantum_annealing_vs_qaoa_optimization**: 0.033
- **quantum_annealing_vs_vqe_hybrid**: 0.067
- **quantum_annealing_vs_distributed_raft**: 0.067
- **quantum_annealing_vs_hybrid_quantum_classical**: 0.100
- **qaoa_optimization_vs_vqe_hybrid**: 0.100
- **qaoa_optimization_vs_distributed_raft**: 0.033
- **qaoa_optimization_vs_hybrid_quantum_classical**: 0.067
- **vqe_hybrid_vs_distributed_raft**: 0.133
- **vqe_hybrid_vs_hybrid_quantum_classical**: 0.167
- **distributed_raft_vs_hybrid_quantum_classical**: 0.033

## Recommendations

### Production Deployment
**Recommended Algorithm**: vqe_hybrid

This algorithm achieved:
- 36.7% success rate
- 0.119s average healing time
- 36.7% availability
