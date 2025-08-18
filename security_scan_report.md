# Security Scan Report

**Scan Date:** Mon Aug 18 12:15:05 UTC 2025
**Total Files Scanned:** 15

⚠️  **Total Issues Found:** 16

## Issue Summary
- **Hardcoded Secrets:** 2
- **Insecure Random:** 28
- **Sql Injection:** 8

## Detailed Findings

### /root/repo/src/materials_orchestrator/core.py

**Insecure Random:**
- Line 287: `if random.random() < 0.05:  # 5% failure rate...`

### /root/repo/src/materials_orchestrator/dashboard.py

**Hardcoded Secrets:**
- Line 261: `if st.button("▶️ Process Queue", key="process_queue"):...`

**Insecure Random:**
- Line 472: `50 + 30 * abs(h - 12) / 12 + np.random.randint(-10, 10) for h in hours...`
- Line 475: `60 + 20 * abs(h - 14) / 14 + np.random.randint(-5, 15) for h in hours...`
- Line 478: `30 + 40 * abs(h - 10) / 10 + np.random.randint(-15, 5) for h in hours...`

### /root/repo/src/materials_orchestrator/dashboard/app.py

**Sql Injection:**
- Line 365: `time_str = dt.strftime("%H:%M:%S")...`
- Line 537: `"Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),...`

### /root/repo/src/materials_orchestrator/distributed_computing.py

**Insecure Random:**
- Line 753: `base_exp = np.random.choice(successful_experiments)...`

### /root/repo/src/materials_orchestrator/distributed_self_healing.py

**Insecure Random:**
- Line 258: `if peer != self.node_id and random.random() > 0.3:  # 70% chance of vote...`

### /root/repo/src/materials_orchestrator/global_compliance.py

**Sql Injection:**
- Line 835: `translation = translation.format(**kwargs)...`

### /root/repo/src/materials_orchestrator/global_deployment.py

**Sql Injection:**
- Line 666: `return dt.strftime("%Y-%m-%d %H:%M:%S")...`
- Line 668: `return dt.strftime("%d.%m.%Y %H:%M:%S")...`
- Line 670: `return dt.strftime("%d/%m/%Y %H:%M:%S")...`
- Line 672: `return dt.strftime("%Y年%m月%d日 %H:%M:%S")...`
- Line 674: `return dt.strftime("%Y-%m-%d %H:%M:%S")...`

### /root/repo/src/materials_orchestrator/ml_enhanced.py

**Insecure Random:**
- Line 379: `+ (high - param_avg) * random.random() * 0.7...`
- Line 384: `- (param_avg - low) * random.random() * 0.7...`
- Line 391: `value = low + (high - low) * random.random()...`
- Line 402: `value = low + (high - low) * random.random()...`
- Line 404: `value = low + (high - low) * random.random()...`

### /root/repo/src/materials_orchestrator/quality_gates.py

**Hardcoded Secrets:**
- Line 839: `test_key = "test_key"...`

### /root/repo/src/materials_orchestrator/quantum_enhanced_optimization.py

**Insecure Random:**
- Line 340: `lambda: np.random.random(), default_value=0.5...`
- Line 559: `lambda: np.random.random(), default_value=0.5...`
- Line 562: `lambda: np.random.random(), default_value=0.5...`
- Line 680: `return safe_numerical_operation(lambda: np.random.random(), default_value=0.5)...`
- Line 688: `lambda: np.random.choice(len(population), tournament_size, replace=False),...`
- Line 711: `safe_numerical_operation(lambda: np.random.random(), default_value=0.5)...`
- Line 727: `safe_numerical_operation(lambda: np.random.random(), default_value=0.5)...`
- Line 826: `lambda: np.random.choice(list(parameter_space.keys())),...`

### /root/repo/src/materials_orchestrator/quantum_enhanced_pipeline_guard.py

**Insecure Random:**
- Line 240: `state_index = np.random.choice(len(probabilities), p=probabilities)...`
- Line 309: `if random.random() < self.quantum_classical_ratio:...`

### /root/repo/src/materials_orchestrator/robots/drivers.py

**Insecure Random:**
- Line 47: `if random.random() < 0.05:  # 5% failure rate...`
- Line 262: `return random.random() > 0.1  # 90% success rate...`

### /root/repo/src/materials_orchestrator/robust_error_handling.py

**Insecure Random:**
- Line 245: `delay *= 0.5 + random.random() * 0.5...`

### /root/repo/src/materials_orchestrator/scalability.py

**Insecure Random:**
- Line 563: `if random.random() < 0.05:  # 5% failure rate...`

### /root/repo/src/materials_orchestrator/utils.py

**Insecure Random:**
- Line 74: `return random.random()...`
- Line 76: `return [random.random() for _ in range(args[0])]...`
- Line 79: `[random.random() for _ in range(args[1])]...`


## Additional Security Checks
⚠️  scripts/deploy.sh has overly permissive permissions
⚠️  start_production.sh has overly permissive permissions
⚠️  Potential secret in environment: KEY
⚠️  Potential secret in environment: TOKEN
