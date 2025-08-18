# Supplementary Data Tables

## Table S1: Detailed Performance Metrics

| Algorithm | Success Rate | Avg Time | Min Time | Max Time | Std Time | MTTR | MTBF |
|-----------|--------------|----------|----------|----------|----------|------|------|
| classical_healing | 0.833 | 1.220s | 1.001s | 1.397s | 0.109s | 1.220s | 720.0s |
| quantum_annealing | 0.900 | 0.575s | 0.400s | 0.783s | 0.113s | 0.575s | 1200.0s |
| qaoa_optimization | 0.900 | 0.831s | 0.601s | 0.995s | 0.112s | 0.831s | 1200.0s |
| vqe_hybrid | 0.800 | 0.623s | 0.500s | 0.874s | 0.100s | 0.623s | 600.0s |
| distributed_raft | 0.900 | 0.976s | 0.822s | 1.171s | 0.110s | 0.976s | 1200.0s |
| hybrid_quantum_classical | 1.000 | 0.377s | 0.201s | 0.593s | 0.117s | 0.377s | ∞ |

## Table S2: Statistical Test Results

| Comparison | t-statistic | p-value | Effect Size | Significant |
|------------|-------------|---------|-------------|-------------|
| classical_healing_vs_quantum_annealing | 20.913 | 0.010 | 5.400 | Yes |
| classical_healing_vs_qaoa_optimization | 14.071 | 0.010 | 3.633 | Yes |
| classical_healing_vs_vqe_hybrid | 22.488 | 0.010 | 5.806 | Yes |
| classical_healing_vs_distributed_raft | 9.501 | 0.010 | 2.453 | Yes |
| classical_healing_vs_hybrid_quantum_classical | 34.128 | 0.010 | 8.812 | Yes |
| quantum_annealing_vs_qaoa_optimization | -9.377 | 0.010 | -2.421 | Yes |
| quantum_annealing_vs_vqe_hybrid | -2.342 | 0.050 | -0.605 | No |
| quantum_annealing_vs_distributed_raft | -13.183 | 0.010 | -3.404 | Yes |
| quantum_annealing_vs_hybrid_quantum_classical | 8.861 | 0.010 | 2.288 | Yes |
| qaoa_optimization_vs_vqe_hybrid | 7.151 | 0.010 | 1.846 | Yes |
| qaoa_optimization_vs_distributed_raft | -5.255 | 0.010 | -1.357 | Yes |
| qaoa_optimization_vs_hybrid_quantum_classical | 15.191 | 0.010 | 3.922 | Yes |
| vqe_hybrid_vs_distributed_raft | -13.212 | 0.010 | -3.411 | Yes |
| vqe_hybrid_vs_hybrid_quantum_classical | 8.435 | 0.010 | 2.178 | Yes |
| distributed_raft_vs_hybrid_quantum_classical | 18.887 | 0.010 | 4.877 | Yes |
