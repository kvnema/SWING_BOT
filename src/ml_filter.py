"""
Machine Learning Signal Filtering for SWING_BOT
Adds predictive filtering to improve signal quality and reduce false positives
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MLSignalFilter:
    """
    ML-based signal filtering using historical trade outcomes to predict success probability
    """

    def __init__(self, min_confidence: float = 0.30, model_type: str = 'rf',
                 lookback_window: int = 252, min_samples: int = 100):
        """
        Initialize ML signal filter

        Args:
            min_confidence: Minimum confidence threshold for signal acceptance
            model_type: 'rf' (Random Forest) or 'gb' (Gradient Boosting)
            lookback_window: Historical window for training (trading days)
            min_samples: Minimum samples required for training
        """
        self.min_confidence = min_confidence
        self.model_type = model_type
        self.lookback_window = lookback_window
        self.min_samples = min_samples
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ML features from technical indicators and market data
        """
        features = pd.DataFrame(index=df.index)

        # Technical features
        features['rsi_14'] = df['RSI14'].fillna(50)
        features['macd_signal'] = df['MACD_Signal'].fillna(0)
        features['macd_hist'] = df['MACD_Hist'].fillna(0)
        features['bb_width'] = df['BB_BandWidth'].fillna(0.1)

        # Calculate BB position if not available
        if 'BB_Position' in df.columns:
            features['bb_position'] = df['BB_Position'].fillna(0.5)
        elif all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
            bb_upper = df['BB_Upper']
            bb_lower = df['BB_Lower']
            bb_range = bb_upper - bb_lower
            features['bb_position'] = ((df['Close'] - bb_lower) / bb_range.replace(0, 1)).fillna(0.5)
        else:
            features['bb_position'] = 0.5  # Neutral position

        features['atr_14'] = df['ATR14'].fillna(1)
        features['volume_ratio'] = df['RVOL20'].fillna(1)
        features['trend_strength'] = df['ADX14'].fillna(25)

        # Price action features
        features['close_to_ema20'] = (df['Close'] / df['EMA20'] - 1).fillna(0)
        features['close_to_ema50'] = (df['Close'] / df['EMA50'] - 1).fillna(0)
        features['close_to_ema200'] = (df['Close'] / df['EMA200'] - 1).fillna(0)

        # Momentum features
        features['rs_roc'] = df['RS_ROC20'].fillna(0)
        features['momentum_12m'] = df['TS_Momentum'].fillna(0)

        # Volatility features
        features['bb_width_pctile'] = df['BB_BandWidth'].rolling(60).rank(pct=True).fillna(0.5)
        features['volume_pctile'] = df['RVOL20'].rolling(60).rank(pct=True).fillna(0.5)

        # Market regime features
        features['trend_ok'] = df['Trend_OK'].fillna(0)
        features['minervini_trend'] = df['Minervini_Trend'].fillna(0)
        features['index_up_regime'] = df['IndexUpRegime'].fillna(1)

        # Lagged features (previous day values)
        lag_features = ['rsi_14', 'macd_hist', 'bb_position', 'volume_ratio']
        for feat in lag_features:
            features[f'{feat}_lag1'] = features[feat].shift(1).fillna(features[feat].median())

        self.feature_columns = features.columns.tolist()
        return features

    def _create_target(self, df: pd.DataFrame, flag_col: str, forward_days: int = 5) -> pd.Series:
        """
        Create target variable based on forward returns
        """
        # Calculate forward returns
        forward_returns = []
        for days in range(1, forward_days + 1):
            ret = df['Close'].shift(-days) / df['Close'] - 1
            forward_returns.append(ret)

        # Target: 1 if any forward return > 2% (profitable trade), else 0
        target = pd.concat(forward_returns, axis=1).max(axis=1) > 0.02
        target = target.astype(int)

        # Only consider targets where signal was present
        target = target.where(df[flag_col] == 1, np.nan)
        return target

    def train(self, df: pd.DataFrame, flag_col: str) -> Dict:
        """
        Train ML model on historical data

        Args:
            df: Historical data with signals and indicators
            flag_col: Signal column name

        Returns:
            Training metrics
        """
        # Create features and target
        features = self._create_features(df)
        target = self._create_target(df, flag_col)

        # Remove NaN values
        valid_data = pd.concat([features, target], axis=1).dropna()
        if len(valid_data) < self.min_samples:
            return {'trained': False, 'reason': f'Insufficient samples: {len(valid_data)} < {self.min_samples}'}

        X = valid_data[self.feature_columns]
        y = valid_data[target.name]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize model
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='precision')

        # Train final model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        else:
            feature_importance = {}

        return {
            'trained': True,
            'cv_precision_mean': cv_scores.mean(),
            'cv_precision_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'samples_used': len(valid_data)
        }

    def predict_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict success confidence for current signals

        Args:
            df: Current data with indicators

        Returns:
            Series with confidence scores (0-1)
        """
        if not self.is_trained:
            return pd.Series(0.5, index=df.index)

        features = self._create_features(df)
        features_scaled = self.scaler.transform(features)

        # Get prediction probabilities
        proba = self.model.predict_proba(features_scaled)[:, 1]
        return pd.Series(proba, index=df.index)

    def filter_signals(self, df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
        """
        Filter signals based on ML confidence

        Args:
            df: DataFrame with signals
            flag_col: Signal column to filter

        Returns:
            DataFrame with filtered signals
        """
        if not self.is_trained:
            return df

        # Get confidence scores
        confidence = self.predict_confidence(df)

        # Filter signals below confidence threshold
        filtered_df = df.copy()
        low_confidence_mask = (confidence < self.min_confidence) & (df[flag_col] == 1)
        filtered_df.loc[low_confidence_mask, flag_col] = 0

        # Add confidence column for analysis
        filtered_df[f'{flag_col}_confidence'] = confidence

        return filtered_df

    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'min_confidence': self.min_confidence,
                'model_type': self.model_type
            }
            joblib.dump(model_data, filepath)

    def load_model(self, filepath: str):
        """Load trained model from disk"""
        if Path(filepath).exists():
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.min_confidence = model_data['min_confidence']
            self.model_type = model_data['model_type']
            self.is_trained = True
            return True
        return False


class SentimentFilter:
    """
    Alternative data filter using sentiment analysis
    """

    def __init__(self, min_sentiment: float = 0.0, api_key: Optional[str] = None):
        """
        Initialize sentiment filter

        Args:
            min_sentiment: Minimum sentiment score (-1 to 1) for signal acceptance
            api_key: API key for sentiment service (placeholder for future implementation)
        """
        self.min_sentiment = min_sentiment
        self.api_key = api_key

    def get_sentiment_score(self, symbol: str, date: pd.Timestamp) -> float:
        """
        Get sentiment score for symbol on date
        Placeholder implementation - would integrate with actual sentiment API
        """
        # Placeholder: return neutral sentiment
        # In real implementation, this would call sentiment APIs
        return 0.0

    def filter_by_sentiment(self, df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
        """
        Filter signals based on sentiment scores

        Args:
            df: DataFrame with signals
            flag_col: Signal column to filter

        Returns:
            DataFrame with sentiment-filtered signals
        """
        filtered_df = df.copy()

        # For now, skip sentiment filtering if no API key
        if not self.api_key:
            return filtered_df

        # Placeholder sentiment filtering logic
        # In real implementation:
        # 1. Get sentiment for each symbol/date
        # 2. Filter signals where sentiment < min_sentiment

        return filtered_df