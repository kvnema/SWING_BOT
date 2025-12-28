"""
SWING_BOT Sector Analysis
Sector-based analysis and filtering for better diversification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# NSE Sector Classification (simplified)
SECTOR_MAPPING = {
    # NIFTY 50 Sectors
    'RELIANCE': 'ENERGY',
    'TCS': 'IT',
    'HDFCBANK': 'BANKING',
    'ICICIBANK': 'BANKING',
    'INFY': 'IT',
    'HINDUNILVR': 'CONSUMER',
    'ITC': 'CONSUMER',
    'KOTAKBANK': 'BANKING',
    'LT': 'INFRASTRUCTURE',
    'AXISBANK': 'BANKING',
    'MARUTI': 'AUTO',
    'BAJFINANCE': 'FINANCIAL_SERVICES',
    'BHARTIARTL': 'TELECOM',
    'HCLTECH': 'IT',
    'ASIANPAINT': 'CONSUMER',
    'TITAN': 'CONSUMER',
    'BAJAJFINSV': 'FINANCIAL_SERVICES',
    'ADANIPORTS': 'INFRASTRUCTURE',
    'TATASTEEL': 'METALS',
    'NESTLEIND': 'CONSUMER',
    'ULTRACEMCO': 'CEMENT',
    'WIPRO': 'IT',
    'TECHM': 'IT',
    'POWERGRID': 'UTILITIES',
    'NTPC': 'UTILITIES',
    'JSWSTEEL': 'METALS',
    'GRASIM': 'CEMENT',
    'INDUSINDBK': 'BANKING',
    'HINDALCO': 'METALS',
    'DRREDDY': 'PHARMA',
    'CIPLA': 'PHARMA',
    'SHREECEM': 'CEMENT',
    'BRITANNIA': 'CONSUMER',
    'EICHERMOT': 'AUTO',
    'APOLLOHOSP': 'HEALTHCARE',
    'DIVISLAB': 'PHARMA',
    'UPL': 'CHEMICALS',
    'HEROMOTOCO': 'AUTO',
    'ADANIENT': 'CONGLOMERATE',
    'COALINDIA': 'ENERGY',
    'BPCL': 'ENERGY',
    'SUNPHARMA': 'PHARMA',
    'ONGC': 'ENERGY',
    'SBILIFE': 'INSURANCE',
    'HDFCLIFE': 'INSURANCE',
    'BAJAJ-AUTO': 'AUTO',
    'TATAMOTORS': 'AUTO',
    'M&M': 'AUTO',

    # Additional PSU stocks
    'NTPC': 'PSU_UTILITIES',
    'POWERGRID': 'PSU_UTILITIES',
    'COALINDIA': 'PSU_ENERGY',
    'ONGC': 'PSU_ENERGY',
    'GAIL': 'PSU_ENERGY',
    'NMDC': 'PSU_METALS',
    'SAIL': 'PSU_METALS',
    'HAL': 'PSU_DEFENSE',
    'BEL': 'PSU_DEFENSE',
    'BEML': 'PSU_DEFENSE',
    'CONCOR': 'PSU_LOGISTICS',
    'IRFC': 'PSU_FINANCIAL',
    'PFC': 'PSU_FINANCIAL',
    'REC': 'PSU_FINANCIAL',
}

class SectorAnalyzer:
    """Analyze sectors for relative strength and diversification."""

    def __init__(self):
        self.sector_data = {}
        self.relative_strength = {}

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        clean_symbol = symbol.replace('.NS', '').replace('NS:', '')
        return SECTOR_MAPPING.get(clean_symbol, 'UNKNOWN')

    def analyze_sector_strength(self, symbols: List[str], lookback_days: int = 60) -> Dict[str, float]:
        """Analyze relative strength of different sectors."""
        from .data_fetch import fetch_market_index_data

        sector_performance = {}
        sector_counts = {}

        for symbol in symbols:
            try:
                # Get recent data
                df = fetch_market_index_data(symbol, lookback_days)
                if df.empty or len(df) < 30:
                    continue

                # Calculate returns
                start_price = df.iloc[0]['Close']
                end_price = df.iloc[-1]['Close']
                total_return = (end_price - start_price) / start_price * 100

                # Get sector
                sector = self.get_sector(symbol)

                # Accumulate sector performance
                if sector not in sector_performance:
                    sector_performance[sector] = 0
                    sector_counts[sector] = 0

                sector_performance[sector] += total_return
                sector_counts[sector] += 1

            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                continue

        # Calculate average performance per sector
        sector_avg_performance = {}
        for sector in sector_performance:
            if sector_counts[sector] > 0:
                sector_avg_performance[sector] = sector_performance[sector] / sector_counts[sector]

        # Sort by performance
        sorted_sectors = sorted(sector_avg_performance.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"Sector analysis complete: {len(sorted_sectors)} sectors analyzed")
        return dict(sorted_sectors)

    def get_sector_diversification_score(self, positions: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate sector diversification score for current positions."""
        sector_exposure = {}
        total_value = 0

        for symbol, position in positions.items():
            sector = self.get_sector(symbol)
            value = position.get('quantity', 0) * position.get('entry_price', 0)

            sector_exposure[sector] = sector_exposure.get(sector, 0) + value
            total_value += value

        # Calculate diversification metrics
        diversification = {}
        if total_value > 0:
            # Herfindahl-Hirschman Index (lower = more diversified)
            hhi = sum((exposure / total_value) ** 2 for exposure in sector_exposure.values())

            # Maximum sector concentration
            max_concentration = max(sector_exposure.values()) / total_value if sector_exposure else 0

            diversification = {
                'hhi_index': hhi,
                'max_sector_pct': max_concentration * 100,
                'sector_count': len(sector_exposure),
                'diversification_score': 1 - hhi  # Higher = more diversified
            }

        return diversification

    def filter_by_sector_limits(self, signals: List[Dict], current_positions: Dict[str, Dict],
                               max_sector_pct: float = 0.25) -> List[Dict]:
        """Filter signals based on sector exposure limits."""
        filtered_signals = []

        for signal in signals:
            symbol = signal['symbol']
            sector = self.get_sector(symbol)

            # Calculate current sector exposure
            sector_value = 0
            total_value = 0

            for pos_symbol, position in current_positions.items():
                pos_sector = self.get_sector(pos_symbol)
                pos_value = position.get('quantity', 0) * position.get('entry_price', 0)

                if pos_sector == sector:
                    sector_value += pos_value
                total_value += pos_value

            # Add potential new position
            new_position_value = signal.get('quantity', 0) * signal.get('entry_price', 0)
            total_value += new_position_value
            sector_value += new_position_value

            # Check sector limit
            if total_value > 0:
                sector_pct = sector_value / total_value
                if sector_pct <= max_sector_pct:
                    filtered_signals.append(signal)
                else:
                    logger.info(f"Rejected {symbol} ({sector}): would exceed {max_sector_pct*100:.0f}% sector limit")
            else:
                filtered_signals.append(signal)

        return filtered_signals

    def get_sector_recommendations(self, sector_strength: Dict[str, float],
                                 top_n: int = 3) -> List[str]:
        """Get recommended sectors based on relative strength."""
        # Get top performing sectors
        top_sectors = list(sector_strength.keys())[:top_n]

        recommendations = []
        for sector in top_sectors:
            strength = sector_strength[sector]
            if strength > 5:  # At least 5% outperformance
                recommendations.append(sector)

        return recommendations

    def get_sector_rotation_signals(self, symbols: List[str], lookback_days: int = 90) -> Dict[str, str]:
        """Identify sector rotation opportunities."""
        sector_strength = self.analyze_sector_strength(symbols, lookback_days)

        # Identify leading and lagging sectors
        if not sector_strength:
            return {}

        sorted_sectors = sorted(sector_strength.items(), key=lambda x: x[1], reverse=True)

        leading_sectors = [sector for sector, strength in sorted_sectors[:3] if strength > 0]
        lagging_sectors = [sector for sector, strength in sorted_sectors[-3:] if strength < -5]

        return {
            'leading_sectors': leading_sectors,
            'lagging_sectors': lagging_sectors,
            'rotation_opportunity': len(leading_sectors) > 0 and len(lagging_sectors) > 0
        }

def get_sector_analysis(symbols: List[str] = None) -> Dict:
    """Get comprehensive sector analysis."""
    if symbols is None:
        # Default NIFTY 50 symbols
        symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS'
        ]

    analyzer = SectorAnalyzer()

    # Analyze sector strength
    sector_strength = analyzer.analyze_sector_strength(symbols)

    # Get sector rotation signals
    rotation_signals = analyzer.get_sector_rotation_signals(symbols)

    # Get recommendations
    recommendations = analyzer.get_sector_recommendations(sector_strength)

    return {
        'sector_strength': sector_strength,
        'rotation_signals': rotation_signals,
        'recommendations': recommendations,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }

def print_sector_analysis(analysis: Dict):
    """Print formatted sector analysis."""
    print("\n" + "="*60)
    print("SECTOR ANALYSIS REPORT")
    print("="*60)

    print("\n📊 Sector Relative Strength (Last 60 Days):")
    for sector, strength in analysis['sector_strength'].items():
        emoji = "🟢" if strength > 5 else "🔴" if strength < -5 else "🟡"
        print(f"{emoji} {sector}: {strength:+.1f}%")

    rotation = analysis['rotation_signals']
    print("\n🔄 Sector Rotation Signals:")
    print(f"Leading Sectors: {', '.join(rotation['leading_sectors'])}")
    print(f"Lagging Sectors: {', '.join(rotation['lagging_sectors'])}")
    print(f"Rotation Opportunity: {'Yes' if rotation['rotation_opportunity'] else 'No'}")

    print("\n🎯 Recommended Sectors:")
    if analysis['recommendations']:
        for sector in analysis['recommendations']:
            print(f"  • {sector}")
    else:
        print("  No strong sector recommendations at this time")

    print(f"\n⏰ Analysis Time: {analysis['analysis_timestamp'][:19]}")

if __name__ == "__main__":
    # Example usage
    analysis = get_sector_analysis()
    print_sector_analysis(analysis)