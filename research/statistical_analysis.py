#!/usr/bin/env python3
"""
Statistical Analysis Module for Self-Healing Pipeline Research
==========================================================

Advanced statistical analysis and hypothesis testing for comparing
self-healing algorithms and quantum optimization strategies.

Author: Terragon Labs
Date: August 2025
License: MIT
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class StatisticalTest(Enum):
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    KRUSKAL_WALLIS = "kruskal_wallis"
    ANOVA = "anova"
    CHI_SQUARE = "chi_square"
    WILCOXON = "wilcoxon"
    FRIEDMAN = "friedman"

@dataclass
class StatisticalResult:
    """Results from statistical hypothesis testing"""
    test_type: StatisticalTest
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    significant: bool
    
    @property
    def significance_level(self) -> str:
        """Return significance level interpretation"""
        if self.p_value < 0.001:
            return "extremely significant (p < 0.001)"
        elif self.p_value < 0.01:
            return "highly significant (p < 0.01)"
        elif self.p_value < 0.05:
            return "significant (p < 0.05)"
        elif self.p_value < 0.1:
            return "marginally significant (p < 0.1)"
        else:
            return "not significant (p ≥ 0.1)"

@dataclass
class EffectSizeAnalysis:
    """Effect size analysis for practical significance"""
    cohens_d: float
    r_squared: float
    practical_significance: str
    
    @property
    def effect_interpretation(self) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(self.cohens_d)
        if abs_d < 0.2:
            return "negligible effect"
        elif abs_d < 0.5:
            return "small effect"
        elif abs_d < 0.8:
            return "medium effect"
        else:
            return "large effect"

class HypothesisTestFramework:
    """Framework for statistical hypothesis testing"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results_cache = {}
    
    def compare_algorithms(self, 
                          algorithm_data: Dict[str, List[float]], 
                          metric_name: str) -> Dict[str, StatisticalResult]:
        """Compare multiple algorithms using appropriate statistical tests"""
        results = {}
        
        # Pairwise comparisons
        algorithms = list(algorithm_data.keys())
        
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                data1 = algorithm_data[algo1]
                data2 = algorithm_data[algo2]
                
                # Choose appropriate test
                if self._check_normality(data1) and self._check_normality(data2):
                    if self._check_equal_variance(data1, data2):
                        result = self._independent_t_test(data1, data2)
                    else:
                        result = self._welch_t_test(data1, data2)
                else:
                    result = self._mann_whitney_test(data1, data2)
                
                comparison_key = f"{algo1}_vs_{algo2}_{metric_name}"
                results[comparison_key] = result
        
        # Overall comparison (ANOVA or Kruskal-Wallis)
        if len(algorithms) > 2:
            all_data = list(algorithm_data.values())
            if all(self._check_normality(data) for data in all_data):
                overall_result = self._anova_test(all_data, list(algorithm_data.keys()))
            else:
                overall_result = self._kruskal_wallis_test(all_data, list(algorithm_data.keys()))
            
            results[f"overall_{metric_name}"] = overall_result
        
        return results
    
    def _check_normality(self, data: List[float], alpha: float = 0.05) -> bool:
        """Check if data follows normal distribution using Shapiro-Wilk test"""
        if len(data) < 3:
            return False
        
        try:
            statistic, p_value = stats.shapiro(data)
            return p_value > alpha
        except:
            return False
    
    def _check_equal_variance(self, data1: List[float], data2: List[float], alpha: float = 0.05) -> bool:
        """Check for equal variances using Levene's test"""
        try:
            statistic, p_value = stats.levene(data1, data2)
            return p_value > alpha
        except:
            return False
    
    def _independent_t_test(self, data1: List[float], data2: List[float]) -> StatisticalResult:
        """Perform independent samples t-test"""
        statistic, p_value = stats.ttest_ind(data1, data2)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                             (len(data2) - 1) * np.var(data2, ddof=1)) / 
                            (len(data1) + len(data2) - 2))
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        # Confidence interval for difference in means
        se_diff = pooled_std * np.sqrt(1/len(data1) + 1/len(data2))
        df = len(data1) + len(data2) - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        mean_diff = np.mean(data1) - np.mean(data2)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        interpretation = f"Independent t-test: mean difference = {mean_diff:.4f}, {self._interpret_result(p_value)}"
        
        return StatisticalResult(
            test_type=StatisticalTest.T_TEST,
            statistic=statistic,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            significant=p_value < self.alpha
        )
    
    def _welch_t_test(self, data1: List[float], data2: List[float]) -> StatisticalResult:
        """Perform Welch's t-test (unequal variances)"""
        statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        
        # Calculate effect size
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        # Approximate confidence interval
        mean_diff = np.mean(data1) - np.mean(data2)
        se_diff = np.sqrt(np.var(data1, ddof=1)/len(data1) + np.var(data2, ddof=1)/len(data2))
        
        # Welch-Satterthwaite equation for degrees of freedom
        s1_sq = np.var(data1, ddof=1)
        s2_sq = np.var(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        df = ((s1_sq/n1 + s2_sq/n2)**2) / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
        
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        interpretation = f"Welch's t-test: mean difference = {mean_diff:.4f}, {self._interpret_result(p_value)}"
        
        return StatisticalResult(
            test_type=StatisticalTest.T_TEST,
            statistic=statistic,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            significant=p_value < self.alpha
        )
    
    def _mann_whitney_test(self, data1: List[float], data2: List[float]) -> StatisticalResult:
        """Perform Mann-Whitney U test (non-parametric)"""
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(data1), len(data2)
        u1 = statistic
        u2 = n1 * n2 - u1
        effect_size = 1 - (2 * min(u1, u2)) / (n1 * n2)
        
        # Bootstrap confidence interval (simplified)
        ci_lower, ci_upper = self._bootstrap_ci(data1, data2, np.median)
        
        interpretation = f"Mann-Whitney U test: U = {statistic:.2f}, {self._interpret_result(p_value)}"
        
        return StatisticalResult(
            test_type=StatisticalTest.MANN_WHITNEY,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            significant=p_value < self.alpha
        )
    
    def _anova_test(self, data_groups: List[List[float]], group_names: List[str]) -> StatisticalResult:
        """Perform one-way ANOVA"""
        statistic, p_value = stats.f_oneway(*data_groups)
        
        # Calculate eta-squared (effect size)
        all_data = np.concatenate(data_groups)
        group_means = [np.mean(group) for group in data_groups]
        overall_mean = np.mean(all_data)
        
        ss_between = sum(len(group) * (mean - overall_mean)**2 for group, mean in zip(data_groups, group_means))
        ss_total = sum((x - overall_mean)**2 for x in all_data)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        interpretation = f"One-way ANOVA: F = {statistic:.4f}, {self._interpret_result(p_value)}"
        
        return StatisticalResult(
            test_type=StatisticalTest.ANOVA,
            statistic=statistic,
            p_value=p_value,
            effect_size=eta_squared,
            confidence_interval=(0, 0),  # Not applicable for ANOVA
            interpretation=interpretation,
            significant=p_value < self.alpha
        )
    
    def _kruskal_wallis_test(self, data_groups: List[List[float]], group_names: List[str]) -> StatisticalResult:
        """Perform Kruskal-Wallis test (non-parametric ANOVA)"""
        statistic, p_value = stats.kruskal(*data_groups)
        
        # Calculate epsilon-squared (effect size)
        n_total = sum(len(group) for group in data_groups)
        epsilon_squared = (statistic - len(data_groups) + 1) / (n_total - len(data_groups))
        
        interpretation = f"Kruskal-Wallis test: H = {statistic:.4f}, {self._interpret_result(p_value)}"
        
        return StatisticalResult(
            test_type=StatisticalTest.KRUSKAL_WALLIS,
            statistic=statistic,
            p_value=p_value,
            effect_size=epsilon_squared,
            confidence_interval=(0, 0),  # Not applicable
            interpretation=interpretation,
            significant=p_value < self.alpha
        )
    
    def _bootstrap_ci(self, data1: List[float], data2: List[float], 
                     statistic_func, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(data1, len(data1), replace=True)
            sample2 = np.random.choice(data2, len(data2), replace=True)
            stat_diff = statistic_func(sample1) - statistic_func(sample2)
            bootstrap_stats.append(stat_diff)
        
        ci_lower = np.percentile(bootstrap_stats, (self.alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - self.alpha/2) * 100)
        
        return ci_lower, ci_upper
    
    def _interpret_result(self, p_value: float) -> str:
        """Interpret p-value"""
        if p_value < 0.001:
            return "extremely significant (p < 0.001)"
        elif p_value < 0.01:
            return "highly significant (p < 0.01)"
        elif p_value < 0.05:
            return "significant (p < 0.05)"
        elif p_value < 0.1:
            return "marginally significant (p < 0.1)"
        else:
            return "not significant (p ≥ 0.1)"

class PowerAnalysis:
    """Statistical power analysis for experimental design"""
    
    @staticmethod
    def calculate_sample_size(effect_size: float, power: float = 0.8, alpha: float = 0.05) -> int:
        """Calculate required sample size for given effect size and power"""
        try:
            from scipy.stats import norm
            
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(power)
            
            # For two-sample t-test
            n = 2 * ((z_alpha + z_beta) / effect_size)**2
            
            return int(np.ceil(n))
        except:
            # Fallback calculation
            return max(30, int(16 / (effect_size**2)))
    
    @staticmethod
    def calculate_achieved_power(sample_size: int, effect_size: float, alpha: float = 0.05) -> float:
        """Calculate achieved statistical power"""
        try:
            from scipy.stats import norm
            
            z_alpha = norm.ppf(1 - alpha/2)
            z_effect = effect_size * np.sqrt(sample_size / 2)
            
            power = norm.cdf(z_effect - z_alpha) + norm.cdf(-z_effect - z_alpha)
            
            return min(1.0, max(0.0, power))
        except:
            # Fallback calculation
            return min(1.0, effect_size * np.sqrt(sample_size) / 4)

class BayesianAnalysis:
    """Bayesian statistical analysis framework"""
    
    def __init__(self):
        self.prior_params = {}
    
    def bayesian_t_test(self, data1: List[float], data2: List[float], 
                       prior_mean_diff: float = 0, prior_var_diff: float = 1) -> Dict[str, float]:
        """Perform Bayesian t-test"""
        # Simplified Bayesian analysis
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        
        # Posterior for mean difference
        observed_diff = mean1 - mean2
        observed_var = var1/n1 + var2/n2
        
        # Bayesian updating
        posterior_precision = 1/prior_var_diff + 1/observed_var
        posterior_var = 1/posterior_precision
        posterior_mean = posterior_var * (prior_mean_diff/prior_var_diff + observed_diff/observed_var)
        
        # Credible interval
        ci_lower = posterior_mean - 1.96 * np.sqrt(posterior_var)
        ci_upper = posterior_mean + 1.96 * np.sqrt(posterior_var)
        
        # Probability that difference > 0
        prob_positive = 1 - stats.norm.cdf(0, posterior_mean, np.sqrt(posterior_var))
        
        return {
            "posterior_mean": posterior_mean,
            "posterior_var": posterior_var,
            "credible_interval": (ci_lower, ci_upper),
            "prob_difference_positive": prob_positive,
            "bayes_factor": self._calculate_bayes_factor(observed_diff, observed_var, prior_var_diff)
        }
    
    def _calculate_bayes_factor(self, observed_diff: float, observed_var: float, prior_var: float) -> float:
        """Calculate approximate Bayes factor"""
        # Simplified BF calculation
        bf = np.sqrt(prior_var / (prior_var + observed_var)) * \
             np.exp(-0.5 * observed_diff**2 * (1/observed_var - 1/(prior_var + observed_var)))
        return bf

class MetaAnalysis:
    """Meta-analysis framework for combining multiple studies"""
    
    def __init__(self):
        self.studies = []
    
    def add_study(self, effect_size: float, variance: float, sample_size: int, study_name: str = ""):
        """Add study results to meta-analysis"""
        self.studies.append({
            "effect_size": effect_size,
            "variance": variance,
            "weight": 1 / variance,
            "sample_size": sample_size,
            "name": study_name
        })
    
    def fixed_effects_analysis(self) -> Dict[str, float]:
        """Perform fixed-effects meta-analysis"""
        if not self.studies:
            return {}
        
        weights = [study["weight"] for study in self.studies]
        effect_sizes = [study["effect_size"] for study in self.studies]
        
        total_weight = sum(weights)
        pooled_effect = sum(w * es for w, es in zip(weights, effect_sizes)) / total_weight
        pooled_variance = 1 / total_weight
        pooled_se = np.sqrt(pooled_variance)
        
        # Confidence interval
        ci_lower = pooled_effect - 1.96 * pooled_se
        ci_upper = pooled_effect + 1.96 * pooled_se
        
        # Heterogeneity test (Q statistic)
        q_statistic = sum(w * (es - pooled_effect)**2 for w, es in zip(weights, effect_sizes))
        df = len(self.studies) - 1
        p_heterogeneity = 1 - stats.chi2.cdf(q_statistic, df) if df > 0 else 1.0
        
        # I-squared
        i_squared = max(0, (q_statistic - df) / q_statistic) if q_statistic > 0 else 0
        
        return {
            "pooled_effect_size": pooled_effect,
            "pooled_variance": pooled_variance,
            "confidence_interval": (ci_lower, ci_upper),
            "q_statistic": q_statistic,
            "p_heterogeneity": p_heterogeneity,
            "i_squared": i_squared,
            "num_studies": len(self.studies)
        }

class StatisticalReporter:
    """Generate statistical analysis reports"""
    
    def __init__(self, output_dir: str = "/root/repo/research/statistical_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_statistical_report(self, analysis_results: Dict[str, Any], 
                                  study_name: str = "statistical_analysis") -> str:
        """Generate comprehensive statistical analysis report"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"{study_name}_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(self._generate_statistical_markdown(analysis_results, study_name))
        
        # Save raw results as JSON
        json_file = self.output_dir / f"{study_name}_data_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        return str(report_file)
    
    def _generate_statistical_markdown(self, results: Dict[str, Any], study_name: str) -> str:
        """Generate markdown content for statistical report"""
        report = f"""# Statistical Analysis Report: {study_name}

## Executive Summary

This report presents the statistical analysis results for the self-healing pipeline benchmarking study.

### Key Statistical Findings

"""
        
        # Add hypothesis test results
        if "hypothesis_tests" in results:
            report += "## Hypothesis Testing Results\n\n"
            for test_name, test_result in results["hypothesis_tests"].items():
                if hasattr(test_result, 'test_type'):
                    report += f"### {test_name}\n\n"
                    report += f"- **Test Type**: {test_result.test_type.value}\n"
                    report += f"- **Statistic**: {test_result.statistic:.4f}\n"
                    report += f"- **P-value**: {test_result.p_value:.6f}\n"
                    report += f"- **Effect Size**: {test_result.effect_size:.4f}\n"
                    report += f"- **Significance**: {test_result.significance_level}\n"
                    report += f"- **Interpretation**: {test_result.interpretation}\n\n"
        
        # Add power analysis
        if "power_analysis" in results:
            report += "## Power Analysis\n\n"
            power_results = results["power_analysis"]
            report += f"- **Recommended Sample Size**: {power_results.get('recommended_sample_size', 'N/A')}\n"
            report += f"- **Achieved Power**: {power_results.get('achieved_power', 'N/A'):.3f}\n"
            report += f"- **Minimum Detectable Effect**: {power_results.get('min_detectable_effect', 'N/A'):.3f}\n\n"
        
        # Add Bayesian analysis
        if "bayesian_analysis" in results:
            report += "## Bayesian Analysis\n\n"
            bayes_results = results["bayesian_analysis"]
            report += f"- **Posterior Mean Difference**: {bayes_results.get('posterior_mean', 'N/A'):.4f}\n"
            report += f"- **95% Credible Interval**: {bayes_results.get('credible_interval', 'N/A')}\n"
            report += f"- **Probability of Positive Effect**: {bayes_results.get('prob_difference_positive', 'N/A'):.3f}\n"
            report += f"- **Bayes Factor**: {bayes_results.get('bayes_factor', 'N/A'):.3f}\n\n"
        
        # Add meta-analysis
        if "meta_analysis" in results:
            report += "## Meta-Analysis\n\n"
            meta_results = results["meta_analysis"]
            report += f"- **Pooled Effect Size**: {meta_results.get('pooled_effect_size', 'N/A'):.4f}\n"
            report += f"- **95% Confidence Interval**: {meta_results.get('confidence_interval', 'N/A')}\n"
            report += f"- **Heterogeneity (I²)**: {meta_results.get('i_squared', 'N/A'):.1%}\n"
            report += f"- **Number of Studies**: {meta_results.get('num_studies', 'N/A')}\n\n"
        
        report += "## Conclusions and Recommendations\n\n"
        report += "Based on the statistical analysis:\n\n"
        report += "1. **Statistical Significance**: Detailed hypothesis testing results provide evidence for algorithm performance differences\n"
        report += "2. **Effect Sizes**: Practical significance assessed through effect size calculations\n"
        report += "3. **Power Analysis**: Sample size recommendations for future studies\n"
        report += "4. **Bayesian Inference**: Probabilistic interpretation of results\n"
        
        return report

# Example usage and demonstration
async def demonstrate_statistical_analysis():
    """Demonstrate statistical analysis capabilities"""
    print("🧮 Statistical Analysis Framework Demonstration")
    print("=" * 50)
    
    # Simulate algorithm performance data
    classical_data = np.random.normal(0.85, 0.1, 50).clip(0, 1).tolist()
    quantum_data = np.random.normal(0.92, 0.08, 50).clip(0, 1).tolist()
    hybrid_data = np.random.normal(0.95, 0.06, 50).clip(0, 1).tolist()
    
    algorithm_data = {
        "classical_healing": classical_data,
        "quantum_annealing": quantum_data,
        "hybrid_quantum": hybrid_data
    }
    
    # Perform hypothesis testing
    hypothesis_framework = HypothesisTestFramework()
    test_results = hypothesis_framework.compare_algorithms(algorithm_data, "success_rate")
    
    # Power analysis
    power_analysis = PowerAnalysis()
    sample_size = power_analysis.calculate_sample_size(effect_size=0.5, power=0.8)
    achieved_power = power_analysis.calculate_achieved_power(50, 0.5)
    
    # Bayesian analysis
    bayesian_framework = BayesianAnalysis()
    bayesian_results = bayesian_framework.bayesian_t_test(classical_data, quantum_data)
    
    # Compile results
    analysis_results = {
        "hypothesis_tests": test_results,
        "power_analysis": {
            "recommended_sample_size": sample_size,
            "achieved_power": achieved_power,
            "min_detectable_effect": 0.5
        },
        "bayesian_analysis": bayesian_results
    }
    
    # Generate report
    reporter = StatisticalReporter()
    report_file = reporter.generate_statistical_report(analysis_results, "demo_analysis")
    
    print(f"📊 Statistical analysis completed!")
    print(f"📋 Report saved: {report_file}")
    
    # Display key results
    print("\n🔍 KEY FINDINGS:")
    for test_name, result in test_results.items():
        if hasattr(result, 'significant'):
            status = "✅ Significant" if result.significant else "❌ Not Significant"
            print(f"  {test_name}: {status} (p = {result.p_value:.4f})")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_statistical_analysis())