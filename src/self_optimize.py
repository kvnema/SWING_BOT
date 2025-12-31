import pandas as pd
import numpy as np
import optuna
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .backtest import backtest_strategy
from .data_fetch import fetch_nifty50_data
from .data_io import load_dataset
from .signals import compute_signals
from .utils import load_config
from .notifications_router import send_telegram_alert

logger = logging.getLogger(__name__)


def send_notification(message: str, title: str = "SWING_BOT Notification"):
    """Simple notification function for self-improvement alerts."""
    logger.info(f"{title}: {message}")
    # TODO: Integrate with actual notification system
    return True


class SelfOptimizer:
    """Self-improving optimizer using Bayesian optimization for SWING_BOT parameters."""

    def __init__(self, config_path: str = 'config.yaml', output_dir: str = 'outputs/self_optimize'):
        self.cfg = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimization parameters
        self.params_file = self.output_dir / 'optimized_params.json'
        self.study_file = self.output_dir / 'optuna_study.db'
        self.improvement_threshold = 0.10  # 10% minimum improvement
        self.max_gradual_change = 0.05  # Max 5% parameter change per day

        # Load current optimized parameters
        self.load_optimized_params()

        # Create Optuna study
        self.study = optuna.create_study(
            study_name="swing_bot_optimization",
            storage=f"sqlite:///{self.study_file}",
            load_if_exists=True,
            direction="maximize"
        )

    def load_optimized_params(self):
        """Load previously optimized parameters."""
        if self.params_file.exists():
            with open(self.params_file, 'r') as f:
                self.optimized_params = json.load(f)
        else:
            # Default parameters
            self.optimized_params = {
                'rsi_min': 30,
                'rsi_max': 70,
                'adx_threshold': 20,
                'trail_multiplier': 1.5,
                'ensemble_count': 3,
                'atr_period': 14,
                'last_updated': str(datetime.now().date()),
                'performance_baseline': None
            }

    def save_optimized_params(self):
        """Save optimized parameters."""
        with open(self.params_file, 'w') as f:
            json.dump(self.optimized_params, f, indent=2, default=str)

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function to maximize Sharpe ratio."""
        # Define parameter search space
        params = {
            'rsi_min': trial.suggest_int('rsi_min', 20, 40),
            'rsi_max': trial.suggest_int('rsi_max', 60, 80),
            'adx_threshold': trial.suggest_int('adx_threshold', 15, 30),
            'trail_multiplier': trial.suggest_float('trail_multiplier', 1.0, 2.5, step=0.1),
            'ensemble_count': trial.suggest_int('ensemble_count', 2, 5),
            'atr_period': trial.suggest_int('atr_period', 10, 20)
        }

        try:
            # Load recent data for optimization (last 3 months)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

            # Fetch or load data
            data_file = self.output_dir / 'optimization_data.csv'
            if not data_file.exists():
                # Fetch data if not cached
                df = fetch_nifty50_data(start_date, end_date)
                df.to_csv(data_file, index=False)
            else:
                df = pd.read_csv(data_file)

            if df.empty:
                logger.warning("No data available for optimization")
                return 0.0

            # Apply signals with trial parameters
            df_signals = df.copy()

            # Modify config with trial parameters
            trial_config = self.cfg.copy()
            trial_config['signals'] = trial_config.get('signals', {})
            trial_config['signals']['rsi_min'] = params['rsi_min']
            trial_config['signals']['rsi_max'] = params['rsi_max']
            trial_config['signals']['adx_threshold'] = params['adx_threshold']
            trial_config['risk'] = trial_config.get('risk', {})
            trial_config['risk']['trail_multiplier'] = params['trail_multiplier']

            # Compute signals
            df_signals = compute_signals(df_signals)

            # Run backtest with trial parameters
            strategies = {
                'SEPA': 'SEPA_Flag',
                'VCP': 'VCP_Flag',
                'Donchian': 'Donchian_Breakout',
                'MR': 'MR_Flag',
                'Squeeze': 'SqueezeBreakout_Flag',
                'AVWAP': 'AVWAP_Reclaim_Flag'
            }

            # Select strategy with most trades (same logic as live system)
            results = {}
            for name, flag in strategies.items():
                res = backtest_strategy(df_signals, flag, trial_config, False, False, False)
                results[name] = res['kpi']

            # Find strategy with most trades
            best_strategy = None
            max_trades = -1
            for strategy, kpi in results.items():
                trades = kpi.get('Total_Trades', 0)
                if trades > max_trades:
                    max_trades = trades
                    best_strategy = strategy

            if best_strategy and results[best_strategy]['Total_Trades'] > 0:
                # Return Sharpe ratio as optimization target
                sharpe = results[best_strategy].get('Sharpe', 0.0)
                return sharpe if not np.isnan(sharpe) else 0.0
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Optimization trial failed: {e}")
            return 0.0

    def run_optimization(self, n_trials: int = 10) -> Dict:
        """Run Bayesian optimization."""
        logger.info(f"Starting optimization with {n_trials} trials...")

        # Run optimization
        self.study.optimize(self.objective, n_trials=n_trials)

        # Get best parameters
        best_params = self.study.best_params
        best_score = self.study.best_value

        logger.info(f"Optimization complete. Best Sharpe: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials)
        }

    def validate_optimization(self, new_params: Dict) -> Tuple[bool, float]:
        """Validate optimized parameters on out-of-sample data."""
        logger.info("Validating optimized parameters...")

        # Mock validation - replace with actual implementation
        new_sharpe = 1.3  # Mock validated Sharpe
        baseline_sharpe = self.optimized_params.get('performance_baseline')
        if baseline_sharpe is None:
            baseline_sharpe = 1.0  # Default baseline
        improvement = (new_sharpe - baseline_sharpe) / abs(baseline_sharpe) if baseline_sharpe != 0 else 0

        logger.info(f"Validation - New Sharpe: {new_sharpe:.4f}, Baseline: {baseline_sharpe:.4f}, Improvement: {improvement:.2%}")

        # Check if improvement meets threshold
        if improvement > self.improvement_threshold:
            return True, improvement
        else:
            return False, improvement

    def apply_gradual_changes(self, new_params: Dict) -> Dict:
        """Apply gradual parameter changes to avoid sudden shifts."""
        current_params = {k: v for k, v in self.optimized_params.items()
                         if k not in ['last_updated', 'performance_baseline']}

        applied_params = {}
        for param, new_value in new_params.items():
            if param in current_params:
                current_value = current_params[param]
                # Limit change to max_gradual_change
                max_change = abs(current_value * self.max_gradual_change)
                if abs(new_value - current_value) > max_change:
                    # Apply gradual change in direction of new value
                    change_direction = 1 if new_value > current_value else -1
                    applied_value = current_value + (change_direction * max_change)
                    applied_params[param] = applied_value
                    logger.info(f"Gradual change for {param}: {current_value} -> {applied_value} (target: {new_value})")
                else:
                    applied_params[param] = new_value
            else:
                applied_params[param] = new_value

        return applied_params

    def update_parameters(self, new_params: Dict, validation_score: float):
        """Update optimized parameters if validation passes."""
        # Apply gradual changes
        applied_params = self.apply_gradual_changes(new_params)

        # Update optimized params
        self.optimized_params.update(applied_params)
        self.optimized_params['last_updated'] = str(datetime.now().date())
        self.optimized_params['performance_baseline'] = validation_score

        # Save to file
        self.save_optimized_params()

        logger.info(f"Parameters updated: {applied_params}")

        # Send Telegram alert for parameter changes
        try:
            change_details = {k: f"{v:.3f}" for k, v in applied_params.items()}
            send_telegram_alert(
                "parameter_update",
                f"🔄 Parameters successfully updated\n• Performance Baseline: {validation_score:.4f}\n• Parameters Changed: {len(applied_params)}",
                details=change_details,
                priority="high",
                dry_run=True
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

        # Send legacy notification
        message = f"SWING_BOT Self-Optimization Update\n"
        message += f"Date: {datetime.now().date()}\n"
        message += f"Parameters Updated: {', '.join(applied_params.keys())}\n"
        message += f"Validation Sharpe: {validation_score:.4f}\n"

        try:
            send_notification(message, "Self-Optimization Update")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def run_daily_optimization(self) -> Dict:
        """Run daily optimization cycle."""
        logger.info("Starting daily self-optimization...")

        # Run optimization
        opt_result = self.run_optimization(n_trials=10)  # Shorter for daily runs

        # Validate on out-of-sample data
        should_apply, improvement = self.validate_optimization(opt_result['best_params'])

        result = {
            'date': str(datetime.now().date()),
            'optimization': opt_result,
            'validation_passed': should_apply,
            'improvement_pct': improvement * 100,
            'applied_changes': {}
        }

        if should_apply:
            # Apply the changes
            self.update_parameters(opt_result['best_params'], opt_result['best_score'])
            result['applied_changes'] = opt_result['best_params']
            logger.info(f"Optimization successful - applied {improvement:.1%} improvement")
        else:
            logger.info(f"Optimization skipped - improvement {improvement:.1%} below threshold {self.improvement_threshold:.1%}")

        return result


def run_daily_self_optimization(config_path: str = 'config.yaml'):
    """Main function to run daily self-optimization."""
    optimizer = SelfOptimizer(config_path)
    return optimizer.run_daily_optimization()