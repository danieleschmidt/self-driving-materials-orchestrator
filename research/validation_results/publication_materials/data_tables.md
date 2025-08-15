# Supplementary Data Tables

## Table S1: Detailed Performance Metrics

| Algorithm | Success Rate | Avg Time | Min Time | Max Time | Std Time | MTTR | MTBF |
|-----------|--------------|----------|----------|----------|----------|------|------|
| classical_healing | 0.800 | 1.169s | 1.005s | 1.385s | 0.109s | 1.169s | 600.0s |
| quantum_annealing | 0.933 | 0.573s | 0.409s | 0.798s | 0.100s | 0.573s | 1800.0s |
| qaoa_optimization | 1.000 | 0.813s | 0.643s | 0.996s | 0.108s | 0.813s | ∞ |
| vqe_hybrid | 0.867 | 0.668s | 0.510s | 0.882s | 0.112s | 0.668s | 900.0s |
| distributed_raft | 0.967 | 1.006s | 0.811s | 1.196s | 0.104s | 1.006s | 3600.0s |
| hybrid_quantum_classical | 1.000 | 0.423s | 0.207s | 0.587s | 0.124s | 0.423s | ∞ |

## Table S2: Statistical Test Results

| Comparison | t-statistic | p-value | Effect Size | Significant |
|------------|-------------|---------|-------------|-------------|
| classical_healing_vs_quantum_annealing | 21.758 | 0.010 | 5.618 | Yes |
| classical_healing_vs_qaoa_optimization | 13.279 | 0.010 | 3.429 | Yes |
| classical_healing_vs_vqe_hybrid | 16.297 | 0.010 | 4.208 | Yes |
| classical_healing_vs_distributed_raft | 5.619 | 0.010 | 1.451 | Yes |
| classical_healing_vs_hybrid_quantum_classical | 32.336 | 0.010 | 8.349 | Yes |
| quantum_annealing_vs_qaoa_optimization | -8.468 | 0.010 | -2.186 | Yes |
| quantum_annealing_vs_vqe_hybrid | -5.242 | 0.010 | -1.353 | Yes |
| quantum_annealing_vs_distributed_raft | -13.823 | 0.010 | -3.569 | Yes |
| quantum_annealing_vs_hybrid_quantum_classical | 6.614 | 0.010 | 1.708 | Yes |
| qaoa_optimization_vs_vqe_hybrid | 4.684 | 0.010 | 1.209 | Yes |
| qaoa_optimization_vs_distributed_raft | -6.937 | 0.010 | -1.791 | Yes |
| qaoa_optimization_vs_hybrid_quantum_classical | 11.428 | 0.010 | 2.951 | Yes |
| vqe_hybrid_vs_distributed_raft | -12.705 | 0.010 | -3.280 | Yes |
| vqe_hybrid_vs_hybrid_quantum_classical | 7.568 | 0.010 | 1.954 | Yes |
| distributed_raft_vs_hybrid_quantum_classical | 22.108 | 0.010 | 5.708 | Yes |
