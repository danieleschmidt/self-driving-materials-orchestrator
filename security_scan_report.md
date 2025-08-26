# Security Scan Report

**Scan Date:** Tue Aug 26 18:53:14 UTC 2025
**Total Files Scanned:** 24

⚠️  **Total Issues Found:** 26

## Issue Summary
- **Dangerous Functions:** 3
- **Hardcoded Secrets:** 1
- **Insecure Random:** 60
- **Sql Injection:** 12

## Detailed Findings

### /root/repo/src/materials_orchestrator/advanced_security.py

**Dangerous Functions:**
- Line 243: `if b"<script>" in decoded or b"eval(" in decoded:...`

### /root/repo/src/materials_orchestrator/autonomous_benchmarking.py

**Sql Injection:**
- Line 546: `f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",...`
- Line 675: `timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")...`

### /root/repo/src/materials_orchestrator/autonomous_research_coordinator.py

**Insecure Random:**
- Line 221: `local_random = random.Random(seed)...`
- Line 224: `target_property = local_random.choice(target_properties)...`
- Line 852: `"success": random.random() > 0.1,  # 90% success rate...`

**Sql Injection:**
- Line 1107: `f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",...`

### /root/repo/src/materials_orchestrator/breakthrough_scientific_ai.py

**Sql Injection:**
- Line 380: `return template.format(**pattern.get("description_params", {}))...`

### /root/repo/src/materials_orchestrator/core.py

**Insecure Random:**
- Line 320: `if random.random() < 0.05:  # 5% failure rate...`

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

### /root/repo/src/materials_orchestrator/federated_learning_coordinator.py

**Dangerous Functions:**
- Line 220: `return pickle.loads(encrypted_data)...`
- Line 224: `parameters = pickle.loads(decrypted)...`

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

### /root/repo/src/materials_orchestrator/quantum_accelerated_discovery.py

**Insecure Random:**
- Line 672: `r1, r2 = np.random.random(2)...`
- Line 877: `random_val = min_val + np.random.random() * (max_val - min_val)...`
- Line 1065: `rand_val = np.random.random()...`
- Line 1134: `if np.random.random() < mutation_rate:...`
- Line 1142: `if np.random.random() < tunneling_prob:...`
- Line 1188: `tournament_indices = np.random.choice(...`
- Line 1199: `winner_idx = np.random.choice(tournament_indices)...`
- Line 1215: `rand_val = np.random.random()...`
- Line 1288: `if np.random.random() < 0.02:  # 2% tunneling probability...`
- Line 1291: `direction = 1 if np.random.random() < 0.5 else -1...`
- Line 1355: `architecture[param] = np.random.choice(options)...`

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
- Line 47: `return random.random()...`
- Line 302: `state_index = np.random.choice(len(probabilities), p=probabilities)...`
- Line 371: `if random.random() < self.quantum_classical_ratio:...`

### /root/repo/src/materials_orchestrator/quantum_hybrid_optimizer.py

**Insecure Random:**
- Line 215: `if np.random.random() < accept_probability:...`

### /root/repo/src/materials_orchestrator/realtime_adaptive_protocols.py

**Insecure Random:**
- Line 673: `if np.random.random() < 0.5:...`
- Line 675: `np.random.choice([-1, 1])...`
- Line 680: `if np.random.random() < 0.5:...`
- Line 682: `np.random.choice([-1, 1])...`
- Line 687: `if np.random.random() < 0.3:...`
- Line 689: `np.random.choice([-1, 1])...`
- Line 815: `if np.random.random() < 0.3:...`
- Line 818: `np.random.choice([-1, 1]) * 0.3 * old_conditions.temperature...`
- Line 820: `new_conditions.ph += np.random.choice([-1, 1]) * 1.0...`

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

### /root/repo/src/materials_orchestrator/secure_random.py

**Insecure Random:**
- Line 72: `return _stdlib_random.random()...`
- Line 87: `return _stdlib_random.randint(a, b)...`
- Line 101: `return _stdlib_random.choice(sequence)...`

### /root/repo/src/materials_orchestrator/utils.py

**Insecure Random:**
- Line 139: `return random.random()...`
- Line 141: `return [random.random() for _ in range(args[0])]...`
- Line 144: `[random.random() for _ in range(args[1])]...`

### /root/repo/src/materials_orchestrator/virtual_laboratory.py

**Insecure Random:**
- Line 71: `defect_factor = random.uniform(-0.1, 0.1) if random.random() < 0.3 else 0...`
- Line 122: `structure_confidence = 0.95 if random.random() < self.precision else 0.7...`
- Line 173: `if random.random() < self.error_rate:...`
- Line 361: `if random.random() < 0.9:  # 90% success rate for efficiency measurement...`


## Additional Security Checks
⚠️  scripts/deploy.sh has overly permissive permissions
⚠️  start_production.sh has overly permissive permissions
⚠️  Potential secret in environment: KEY
⚠️  Potential secret in environment: TOKEN
