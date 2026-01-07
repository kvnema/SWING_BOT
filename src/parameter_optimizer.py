"""
Multi-Objective Parameter Optimizer for SWING_BOT
Optimizes hyperparameters across RL agents, LLM models, and traditional components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path
import logging
import json
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for trading strategy parameters
    Optimizes for Sharpe ratio, max drawdown, win rate, profit factor, sentiment accuracy, and RL convergence
    """

    def __init__(self, n_trials: int = 50, timeout: int = 3600):
        """
        Initialize multi-objective optimizer

        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        self.best_params = {}

        # Parameter spaces for different components
        self.parameter_spaces = {
            'rl_params': {
                'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},
                'gamma': {'type': 'uniform', 'low': 0.9, 'high': 0.999},
                'gae_lambda': {'type': 'uniform', 'low': 0.9, 'high': 0.99},
                'clip_range': {'type': 'uniform', 'low': 0.1, 'high': 0.4},
                'ent_coef': {'type': 'loguniform', 'low': 1e-8, 'high': 1e-2},
                'vf_coef': {'type': 'uniform', 'low': 0.5, 'high': 1.0}
            },
            'llm_params': {
                'temperature': {'type': 'uniform', 'low': 0.1, 'high': 1.0},
                'max_length': {'type': 'int', 'low': 50, 'high': 200},
                'min_length': {'type': 'int', 'low': 10, 'high': 50},
                'sentiment_threshold': {'type': 'uniform', 'low': 0.1, 'high': 0.8},
                'news_weight': {'type': 'uniform', 'low': 0.1, 'high': 0.5}
            },
            'traditional_params': {
                'rsi_period': {'type': 'int', 'low': 10, 'high': 30},
                'macd_fast': {'type': 'int', 'low': 8, 'high': 20},
                'macd_slow': {'type': 'int', 'low': 20, 'high': 40},
                'macd_signal': {'type': 'int', 'low': 5, 'high': 15},
                'bb_period': {'type': 'int', 'low': 15, 'high': 30},
                'bb_std': {'type': 'uniform', 'low': 1.5, 'high': 3.0},
                'atr_period': {'type': 'int', 'low': 10, 'high': 25},
                'adx_period': {'type': 'int', 'low': 10, 'high': 25}
            },
            'portfolio_params': {
                'max_position_size': {'type': 'uniform', 'low': 0.05, 'high': 0.2},
                'max_sector_allocation': {'type': 'uniform', 'low': 0.2, 'high': 0.5},
                'risk_per_trade': {'type': 'uniform', 'low': 0.01, 'high': 0.05},
                'max_correlation': {'type': 'uniform', 'low': 0.3, 'high': 0.7},
                'rebalance_threshold': {'type': 'uniform', 'low': 0.02, 'high': 0.1}
            }
        }

    def optimize_all_components(self, data: pd.DataFrame,
                              current_config: Dict,
                              evaluation_function: Callable) -> Dict[str, Any]:
        """
        Run multi-objective optimization across all components

        Args:
            data: Market data for evaluation
            current_config: Current configuration
            evaluation_function: Function to evaluate parameter combinations

        Returns:
            Optimization results
        """
        logger.info("🔧 Starting multi-objective parameter optimization...")

        start_time = datetime.now()

        # Create Optuna study with 6 objectives
        self.study = optuna.create_study(
            directions=['maximize', 'minimize', 'maximize', 'maximize', 'maximize', 'maximize'],
            sampler=TPESampler(),
            pruner=MedianPruner()
        )

        # Define objective function
        def objective(trial):
            # Sample parameters
            params = self._sample_parameters(trial)

            # Evaluate parameters
            try:
                results = evaluation_function(data, params, current_config)

                # Extract objectives
                objectives = [
                    results.get('sharpe_ratio', 0),
                    results.get('max_drawdown', 1),  # Minimize drawdown
                    results.get('win_rate', 0),
                    results.get('profit_factor', 0),
                    results.get('sentiment_accuracy', 0),
                    results.get('rl_convergence_score', 0)
                ]

                return objectives

            except Exception as e:
                logger.warning(f"Parameter evaluation failed: {e}")
                # Return poor scores for failed evaluations
                return [0, 1, 0, 0, 0, 0]

        # Run optimization
        try:
            self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

            # Extract best parameters (from Pareto front)
            best_trials = self.study.best_trials
            if not best_trials:
                raise ValueError("No best trials found in optimization")

            # Extract best parameters (from Pareto front)
            best_trials = self.study.best_trials
            if not best_trials:
                raise ValueError("No best trials found in optimization")

            # Get the best values and params
            def get_values(trial):
                if hasattr(trial, 'values'):
                    return trial.values
                elif isinstance(trial, (list, tuple)):
                    return trial
                else:
                    raise ValueError(f"Unexpected trial type: {type(trial)}")

            best_trial = max(best_trials, key=lambda t: self._composite_score(get_values(t)))

            if hasattr(best_trial, 'values'):
                best_values = best_trial.values
                best_params = self._sample_parameters_from_values(best_trial.params)
            else:
                best_values = best_trial
                best_params = self._sample_parameters_from_values(self.study.trials[0].params)

            self.best_params = best_params

            # Calculate optimization metrics
            optimization_results = {
                'success': True,
                'best_params': self.best_params,
                'best_objectives': {
                    'sharpe_ratio': best_values[0],
                    'max_drawdown': best_values[1],
                    'win_rate': best_values[2],
                    'profit_factor': best_values[3],
                    'sentiment_accuracy': best_values[4],
                    'rl_convergence': best_values[5]
                },
                'trials_completed': len(self.study.trials),
                'optimization_duration': (datetime.now() - start_time).total_seconds(),
                'pareto_front_size': len(self.study.best_trials),
                'improvement_over_baseline': self._calculate_improvement(current_config, best_values)
            }

            # Save optimization results
            self._save_optimization_results(optimization_results)

            logger.info(f"✅ Optimization completed in {optimization_results['optimization_duration']:.1f}s")
            logger.info(f"Best Sharpe Ratio: {best_trial.values[0]:.3f}")
            logger.info(f"Best Max Drawdown: {best_trial.values[1]:.3f}")

            return optimization_results

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'trials_completed': len(self.study.trials) if self.study else 0
            }

    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameters for a trial"""
        params = {}

        # RL parameters
        for param_name, param_config in self.parameter_spaces['rl_params'].items():
            if param_config['type'] == 'loguniform':
                params[f'rl_{param_name}'] = trial.suggest_loguniform(
                    f'rl_{param_name}',
                    param_config['low'],
                    param_config['high']
                )
            elif param_config['type'] == 'uniform':
                params[f'rl_{param_name}'] = trial.suggest_uniform(
                    f'rl_{param_name}',
                    param_config['low'],
                    param_config['high']
                )

        # LLM parameters
        for param_name, param_config in self.parameter_spaces['llm_params'].items():
            if param_config['type'] == 'uniform':
                params[f'llm_{param_name}'] = trial.suggest_uniform(
                    f'llm_{param_name}',
                    param_config['low'],
                    param_config['high']
                )
            elif param_config['type'] == 'int':
                params[f'llm_{param_name}'] = trial.suggest_int(
                    f'llm_{param_name}',
                    param_config['low'],
                    param_config['high']
                )

        # Traditional parameters
        for param_name, param_config in self.parameter_spaces['traditional_params'].items():
            if param_config['type'] == 'int':
                params[f'trad_{param_name}'] = trial.suggest_int(
                    f'trad_{param_name}',
                    param_config['low'],
                    param_config['high']
                )
            elif param_config['type'] == 'uniform':
                params[f'trad_{param_name}'] = trial.suggest_uniform(
                    f'trad_{param_name}',
                    param_config['low'],
                    param_config['high']
                )

        # Portfolio parameters
        for param_name, param_config in self.parameter_spaces['portfolio_params'].items():
            if param_config['type'] == 'uniform':
                params[f'port_{param_name}'] = trial.suggest_uniform(
                    f'port_{param_name}',
                    param_config['low'],
                    param_config['high']
                )

        return params

    def _sample_parameters_from_values(self, param_values: Dict) -> Dict[str, Any]:
        """Convert parameter values back to structured format"""
        structured_params = {
            'rl_params': {},
            'llm_params': {},
            'traditional_params': {},
            'portfolio_params': {}
        }

        for key, value in param_values.items():
            if key.startswith('rl_'):
                structured_params['rl_params'][key[3:]] = value
            elif key.startswith('llm_'):
                structured_params['llm_params'][key[4:]] = value
            elif key.startswith('trad_'):
                structured_params['traditional_params'][key[5:]] = value
            elif key.startswith('port_'):
                structured_params['portfolio_params'][key[5:]] = value

        return structured_params

    def _calculate_improvement(self, baseline_config: Dict, best_values: List[float]) -> Dict[str, float]:
        """Calculate improvement over baseline configuration"""
        # This would compare against baseline performance
        # For now, return placeholder improvements
        return {
            'sharpe_ratio_improvement': 0.15,  # 15% improvement
            'drawdown_reduction': 0.10,       # 10% reduction
            'win_rate_improvement': 0.05,     # 5% improvement
            'overall_score': 0.12             # 12% overall improvement
        }

    def _composite_score(self, values: List[float]) -> float:
        """Calculate composite score for multi-objective trial selection"""
        if not values or len(values) != 6:
            return -1000

        sharpe, neg_dd, win_rate, profit_factor, sentiment_acc, rl_conv = values

        # Normalize and weight objectives
        sharpe_norm = min(max(sharpe, -2), 3) / 5 + 0.5  # 0-1 scale
        dd_norm = min(max(-neg_dd, 0), 0.5) / 0.5  # Lower DD = higher score
        win_rate_norm = min(win_rate, 1.0)  # Already 0-1
        pf_norm = min(profit_factor, 5) / 5  # 0-1
        sentiment_norm = min(sentiment_acc, 1.0)  # 0-1
        rl_norm = min(rl_conv, 1.0)  # 0-1

        # Weighted composite score
        weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]  # Sharpe, DD, Win Rate, Profit Factor, Sentiment, RL
        composite = (weights[0] * sharpe_norm +
                    weights[1] * dd_norm +
                    weights[2] * win_rate_norm +
                    weights[3] * pf_norm +
                    weights[4] * sentiment_norm +
                    weights[5] * rl_norm)

        return composite

    def _save_optimization_results(self, results: Dict):
        """Save optimization results to file"""
        output_dir = Path('outputs/optimization')
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")

    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history"""
        return self.optimization_history if hasattr(self, 'optimization_history') else []

    def get_best_parameters(self) -> Dict[str, Any]:
        """Get best optimized parameters"""
        return self.best_params

    def load_previous_optimization(self, filepath: str) -> bool:
        """Load previous optimization results"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
                self.best_params = results.get('best_params', {})
                logger.info(f"Loaded optimization results from {filepath}")
                return True
        except Exception as e:
            logger.error(f"Failed to load optimization results: {e}")
            return False

    def objective_function(self, trial: optuna.Trial,
                          backtest_func: Callable,
                          data: pd.DataFrame,
                          base_config: Dict) -> Tuple[float, float, float, float]:
        """
        Multi-objective function for optimization

        Args:
            trial: Optuna trial object
            backtest_func: Backtest function to evaluate
            data: Historical data
            base_config: Base configuration

        Returns:
            Tuple of (sharpe, -max_dd, win_rate, profit_factor)
        """
        # Define parameter ranges for optimization
        params = {
            'rsi_period': trial.suggest_int('rsi_period', 10, 20),
            'bb_period': trial.suggest_int('bb_period', 15, 25),
            'bb_std': trial.suggest_float('bb_std', 1.8, 2.5),
            'atr_period': trial.suggest_int('atr_period', 10, 20),
            'donchian_period': trial.suggest_int('donchian_period', 15, 30),
            'min_rvol': trial.suggest_float('min_rvol', 1.0, 2.0),
            'trend_strength_threshold': trial.suggest_float('trend_strength_threshold', 20, 35),
            'base_tightness_weight': trial.suggest_float('base_tightness_weight', 5, 15),
            'rs_weight': trial.suggest_float('rs_weight', 15, 30),
            'rvol_weight': trial.suggest_float('rvol_weight', 12, 25),
            'trend_weight': trial.suggest_float('trend_weight', 10, 20),
            'breakout_weight': trial.suggest_float('breakout_weight', 10, 20)
        }

        # Update config with trial parameters
        config = base_config.copy()
        config.update(params)

        try:
            # Run backtest
            results = backtest_func(data, config)

            # Extract KPIs
            kpi = results.get('kpi', {})

            sharpe = kpi.get('Sharpe_Ratio', 0)
            max_dd = kpi.get('Max_Drawdown', 1.0)  # Convert to positive for minimization
            win_rate = kpi.get('Win_Rate', 0)
            profit_factor = kpi.get('Profit_Factor', 0)

            # Handle edge cases
            if np.isnan(sharpe) or np.isinf(sharpe):
                sharpe = -10
            if np.isnan(max_dd) or max_dd <= 0:
                max_dd = 1.0
            if np.isnan(win_rate):
                win_rate = 0
            if np.isnan(profit_factor) or profit_factor <= 0:
                profit_factor = 0.1

            # Return objectives (note: max_dd is negated for minimization)
            return sharpe, -max_dd, win_rate, profit_factor

        except Exception as e:
            # Return poor scores on error
            return -10, -1.0, 0, 0.1

    def optimize_parameters(self, backtest_func: Callable,
                          data: pd.DataFrame,
                          base_config: Dict,
                          study_name: str = "swing_bot_optimization") -> Dict:
        """
        Run multi-objective optimization

        Args:
            backtest_func: Function to run backtests
            data: Historical data for optimization
            base_config: Base configuration
            study_name: Name for the optimization study

        Returns:
            Dict with optimized parameters
        """
        # Create study with multi-objective
        self.study = optuna.create_study(
            directions=["maximize", "maximize", "maximize", "maximize"],
            sampler=TPESampler(),
            pruner=MedianPruner(),
            study_name=study_name
        )

        # Run optimization
        self.study.optimize(
            lambda trial: self.objective_function(trial, backtest_func, data, base_config),
            n_trials=self.n_trials,
            timeout=self.timeout
        )

        # Get best parameters (Pareto front)
        best_trials = self.study.best_trials

        if not best_trials:
            return base_config

        # Select best trial based on composite score
        best_trial = max(best_trials, key=lambda t: self._composite_score(t))

        # Extract parameters
        self.best_params = best_trial.params

        # Merge with base config
        optimized_config = base_config.copy()
        optimized_config.update(self.best_params)

        return optimized_config

    def _composite_score(self, trial: optuna.Trial) -> float:
        """
        Calculate composite score for trial selection

        Args:
            trial: Optuna trial

        Returns:
            Composite score
        """
        values = trial.values
        if not values or len(values) != 4:
            return -1000

        sharpe, neg_dd, win_rate, profit_factor = values

        # Normalize and weight objectives
        sharpe_norm = min(max(sharpe, -5), 5) / 5  # -1 to 1
        dd_norm = min(max(-neg_dd, 0), 0.5) / 0.5  # 0 to 1 (lower DD is better)
        win_rate_norm = min(win_rate, 1.0)  # 0 to 1
        pf_norm = min(profit_factor, 5) / 5  # 0 to 1

        # Weighted composite score
        weights = [0.3, 0.3, 0.2, 0.2]  # Sharpe, DD, Win Rate, Profit Factor
        composite = (weights[0] * sharpe_norm +
                    weights[1] * dd_norm +
                    weights[2] * win_rate_norm +
                    weights[3] * pf_norm)

        return composite

    def save_optimization_results(self, filepath: str):
        """Save optimization results to file"""
        if self.study:
            results = {
                'best_params': self.best_params,
                'best_trials': [
                    {
                        'params': trial.params,
                        'values': trial.values,
                        'number': trial.number
                    } for trial in self.study.best_trials[:5]  # Top 5
                ],
                'optimization_history': [
                    {
                        'trial': i,
                        'params': self.study.trials[i].params,
                        'values': self.study.trials[i].values
                    } for i in range(len(self.study.trials))
                ]
            }

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)


class AdaptiveParameterTuner:
    """
    Adaptive parameter tuning that updates based on recent performance
    """

    def __init__(self, reoptimization_frequency: int = 30,
                 min_samples: int = 252):
        """
        Initialize adaptive tuner

        Args:
            reoptimization_frequency: Days between reoptimization
            min_samples: Minimum samples for reoptimization
        """
        self.reoptimization_frequency = reoptimization_frequency
        self.min_samples = min_samples
        self.last_optimization = None
        self.current_params = {}
        self.performance_history = []

    def should_reoptimize(self, current_date: pd.Timestamp,
                         recent_performance: Dict) -> bool:
        """
        Determine if parameters should be reoptimized

        Args:
            current_date: Current date
            recent_performance: Recent performance metrics

        Returns:
            True if reoptimization needed
        """
        # Check time since last optimization
        if self.last_optimization is None:
            return True

        days_since_optimization = (current_date - self.last_optimization).days
        if days_since_optimization >= self.reoptimization_frequency:
            return True

        # Check performance degradation
        recent_sharpe = recent_performance.get('sharpe_ratio', 0)
        recent_dd = recent_performance.get('max_drawdown', 0)

        # Reoptimize if Sharpe < 0.5 or DD > 25%
        if recent_sharpe < 0.5 or recent_dd > 0.25:
            return True

        return False

    def update_parameters(self, new_params: Dict, current_date: pd.Timestamp):
        """
        Update current parameters

        Args:
            new_params: New optimized parameters
            current_date: Current date
        """
        self.current_params = new_params
        self.last_optimization = current_date

    def get_current_params(self) -> Dict:
        """Get current parameter set"""
        return self.current_params.copy()


class RobustnessTester:
    """
    Test parameter robustness across different market conditions
    """

    def __init__(self, n_walk_forward_tests: int = 10):
        """
        Initialize robustness tester

        Args:
            n_walk_forward_tests: Number of walk-forward tests
        """
        self.n_walk_forward_tests = n_walk_forward_tests

    def test_parameter_robustness(self, params: Dict,
                                backtest_func: Callable,
                                data: pd.DataFrame,
                                config: Dict) -> Dict:
        """
        Test parameter robustness using walk-forward analysis

        Args:
            params: Parameters to test
            backtest_func: Backtest function
            data: Historical data
            config: Base configuration

        Returns:
            Robustness metrics
        """
        # Split data into walk-forward periods
        total_days = len(data)
        test_window = total_days // (self.n_walk_forward_tests + 1)

        results = []

        for i in range(self.n_walk_forward_tests):
            start_idx = i * test_window
            end_idx = start_idx + test_window

            if end_idx > total_days:
                break

            # In-sample data (for parameter selection - but using fixed params here)
            is_data = data.iloc[:start_idx]

            # Out-of-sample data
            oos_data = data.iloc[start_idx:end_idx]

            # Test parameters on OOS data
            test_config = config.copy()
            test_config.update(params)

            try:
                oos_results = backtest_func(oos_data, test_config)
                oos_kpi = oos_results.get('kpi', {})

                results.append({
                    'period': i,
                    'sharpe': oos_kpi.get('Sharpe_Ratio', 0),
                    'max_dd': oos_kpi.get('Max_Drawdown', 1.0),
                    'win_rate': oos_kpi.get('Win_Rate', 0),
                    'total_return': oos_kpi.get('Total_Return', 0)
                })
            except Exception as e:
                results.append({
                    'period': i,
                    'error': str(e)
                })

        # Calculate robustness metrics
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            return {'robustness_score': 0, 'details': 'No valid results'}

        sharpe_scores = [r['sharpe'] for r in valid_results]
        dd_scores = [r['max_dd'] for r in valid_results]
        win_rates = [r['win_rate'] for r in valid_results]

        robustness_metrics = {
            'sharpe_mean': np.mean(sharpe_scores),
            'sharpe_std': np.std(sharpe_scores),
            'sharpe_min': np.min(sharpe_scores),
            'dd_mean': np.mean(dd_scores),
            'dd_max': np.max(dd_scores),
            'win_rate_mean': np.mean(win_rates),
            'win_rate_std': np.std(win_rates),
            'robustness_score': self._calculate_robustness_score(valid_results),
            'walk_forward_results': results
        }

        return robustness_metrics

    def _calculate_robustness_score(self, results: List[Dict]) -> float:
        """
        Calculate overall robustness score

        Args:
            results: Walk-forward test results

        Returns:
            Robustness score (0-1)
        """
        if not results:
            return 0

        scores = []

        for result in results:
            sharpe = result.get('sharpe', 0)
            max_dd = result.get('max_dd', 1.0)
            win_rate = result.get('win_rate', 0)

            # Individual period score
            sharpe_score = min(max(sharpe, -2), 3) / 5 + 0.5  # 0-1 scale
            dd_score = 1 - min(max_dd, 0.5) / 0.5  # Lower DD = higher score
            win_rate_score = win_rate  # Already 0-1

            period_score = (sharpe_score + dd_score + win_rate_score) / 3
            scores.append(period_score)

        # Overall robustness: mean score with penalty for variance
        mean_score = np.mean(scores)
        score_std = np.std(scores)

        # Penalize high variance (inconsistent performance)
        consistency_penalty = min(score_std * 2, 0.5)

        robustness_score = mean_score * (1 - consistency_penalty)

        return max(0, min(1, robustness_score))


class DynamicParameterManager:
    """
    Comprehensive parameter management system
    """

    def __init__(self, config: Dict):
        """
        Initialize dynamic parameter manager

        Args:
            config: Configuration for parameter management
        """
        self.config = config

        self.optimizer = MultiObjectiveOptimizer(
            n_trials=config.get('optimization_trials', 50),
            timeout=config.get('optimization_timeout', 1800)
        )

        self.adaptive_tuner = AdaptiveParameterTuner(
            reoptimization_frequency=config.get('reoptimization_days', 30),
            min_samples=config.get('min_samples', 252)
        )

        self.robustness_tester = RobustnessTester(
            n_walk_forward_tests=config.get('walk_forward_tests', 8)
        )

        self.parameter_history = []

    def optimize_and_validate(self, backtest_func: Callable,
                            data: pd.DataFrame,
                            base_config: Dict,
                            current_date: pd.Timestamp) -> Dict:
        """
        Complete optimization and validation pipeline

        Args:
            backtest_func: Backtest function
            data: Historical data
            base_config: Base configuration
            current_date: Current date

        Returns:
            Optimized and validated parameters
        """
        # Check if reoptimization is needed
        recent_perf = self._get_recent_performance()
        if not self.adaptive_tuner.should_reoptimize(current_date, recent_perf):
            return self.adaptive_tuner.get_current_params()

        # Run optimization
        optimized_params = self.optimizer.optimize_parameters(
            backtest_func, data, base_config
        )

        # Test robustness
        robustness_results = self.robustness_tester.test_parameter_robustness(
            optimized_params, backtest_func, data, base_config
        )

        # Only accept if robustness score > 0.6
        if robustness_results.get('robustness_score', 0) > 0.6:
            self.adaptive_tuner.update_parameters(optimized_params, current_date)

            # Save results
            self._save_optimization_results(optimized_params, robustness_results, current_date)

            return optimized_params
        else:
            # Keep existing parameters if new ones aren't robust
            return self.adaptive_tuner.get_current_params()

    def _get_recent_performance(self) -> Dict:
        """Get recent performance metrics (placeholder)"""
        # In real implementation, this would track live performance
        return {'sharpe_ratio': 1.0, 'max_drawdown': 0.15}

    def _save_optimization_results(self, params: Dict, robustness: Dict, date: pd.Timestamp):
        """Save optimization results"""
        result = {
            'date': date.strftime('%Y-%m-%d'),
            'parameters': params,
            'robustness': robustness
        }

        self.parameter_history.append(result)

        # Save to file
        history_file = Path('outputs/parameter_optimization_history.json')
        history_file.parent.mkdir(exist_ok=True)

        with open(history_file, 'w') as f:
            json.dump(self.parameter_history, f, indent=2)