#!/usr/bin/env python3
"""
SWING_BOT Calibration Snapshot Script

Creates weekly snapshot of confidence calibration for monitoring.
Usage: python scripts/calibration_snapshot.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import get_ist_now

def main():
    """Create calibration snapshot."""

    # Setup logging
    log_dir = Path('outputs/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    import logging
    logging.basicConfig(
        filename=log_dir / f'calibration_snapshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting calibration snapshot")

        # Load confidence report
        confidence_file = Path('outputs/confidence_report.csv')
        if not confidence_file.exists():
            logger.error(f"Confidence report not found: {confidence_file}")
            print(f"❌ Confidence report not found: {confidence_file}")
            sys.exit(1)

        df = pd.read_csv(confidence_file)
        logger.info(f"Loaded confidence data: {len(df)} records")

        # Create snapshot directory
        snapshot_dir = Path('outputs/calibration_snapshots')
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        today = get_ist_now().date()
        snapshot_file = snapshot_dir / f'calibration_{today}.png'

        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'SWING_BOT Confidence Calibration - {today}', fontsize=16)

        # 1. Confidence distribution
        if 'DecisionConfidence' in df.columns:
            sns.histplot(data=df, x='DecisionConfidence', bins=20, ax=ax1)
            ax1.set_title('Decision Confidence Distribution')
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Frequency')

        # 2. Confidence by strategy
        if 'Strategy' in df.columns and 'DecisionConfidence' in df.columns:
            strategy_conf = df.groupby('Strategy')['DecisionConfidence'].mean().sort_values(ascending=False)
            strategy_conf.plot(kind='bar', ax=ax2)
            ax2.set_title('Average Confidence by Strategy')
            ax2.set_xlabel('Strategy')
            ax2.set_ylabel('Average Confidence')
            ax2.tick_params(axis='x', rotation=45)

        # 3. Top 10 symbols by confidence
        if 'Symbol' in df.columns and 'DecisionConfidence' in df.columns:
            top_symbols = df.nlargest(10, 'DecisionConfidence')[['Symbol', 'DecisionConfidence']]
            top_symbols.plot(kind='barh', x='Symbol', y='DecisionConfidence', ax=ax3)
            ax3.set_title('Top 10 Symbols by Confidence')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Symbol')

        # 4. Confidence vs other metrics
        if 'DecisionConfidence' in df.columns and 'RSI' in df.columns:
            sns.scatterplot(data=df, x='RSI', y='DecisionConfidence', ax=ax4, alpha=0.6)
            ax4.set_title('Confidence vs RSI')
            ax4.set_xlabel('RSI')
            ax4.set_ylabel('Confidence Score')

        plt.tight_layout()
        plt.savefig(snapshot_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Calibration snapshot saved: {snapshot_file}")

        # Save summary stats
        summary_file = snapshot_dir / f'summary_{today}.txt'
        with open(summary_file, 'w') as f:
            f.write(f"SWING_BOT Calibration Summary - {today}\n")
            f.write("=" * 50 + "\n\n")

            if 'DecisionConfidence' in df.columns:
                f.write("Confidence Statistics:\n")
                f.write(f"  Mean: {df['DecisionConfidence'].mean():.3f}\n")
                f.write(f"  Median: {df['DecisionConfidence'].median():.3f}\n")
                f.write(f"  Std: {df['DecisionConfidence'].std():.3f}\n")
                f.write(f"  Min: {df['DecisionConfidence'].min():.3f}\n")
                f.write(f"  Max: {df['DecisionConfidence'].max():.3f}\n\n")

            if 'Strategy' in df.columns:
                f.write("Strategy Distribution:\n")
                strategy_counts = df['Strategy'].value_counts()
                for strategy, count in strategy_counts.items():
                    f.write(f"  {strategy}: {count}\n")
                f.write("\n")

            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Snapshot saved: {snapshot_file}\n")

        logger.info(f"Summary saved: {summary_file}")
        print(f"✅ Calibration snapshot created: {snapshot_file}")

        # Clean up old snapshots (keep last 4 weeks)
        cleanup_old_snapshots(snapshot_dir, logger)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

def cleanup_old_snapshots(snapshot_dir, logger):
    """Clean up snapshots older than 4 weeks."""
    try:
        today = get_ist_now().date()
        cutoff_date = today - timedelta(weeks=4)

        for file in snapshot_dir.glob('*.png'):
            try:
                # Extract date from filename (calibration_YYYY-MM-DD.png)
                date_str = file.stem.split('_')[1]
                file_date = datetime.strptime(date_str, '%Y-%m-%d').date()

                if file_date < cutoff_date:
                    file.unlink()
                    logger.info(f"Cleaned up old snapshot: {file}")
            except (ValueError, IndexError):
                continue  # Skip files with unexpected naming

    except Exception as e:
        logger.warning(f"Error during cleanup: {str(e)}")

if __name__ == '__main__':
    main()