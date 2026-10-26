import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Tuple
from queue import Queue

# --- Backtester Interface Imports ---
# These assume the file is placed in src/strategy/ and run from the root directory.
try:
    from src.engine.base import Strategy, DataHandler
    from src.engine.events import SignalEvent, SignalDirection
except ImportError:
    # Mocking classes for pure standalone execution if paths are broken
    class Strategy:
        pass


    class DataHandler:
        pass


    class SignalDirection:
        LONG = "LONG"
        SHORT = "SHORT"
        EXIT = "EXIT"


    class SignalEvent:
        def __init__(self, symbol, datetime, signal_type, strength=1.0):
            self.symbol = symbol
            self.datetime = datetime
            self.signal_type = signal_type
            self.strength = strength


# ==========================================
# 1. MACHINE LEARNING & FEATURE ENGINEERING
# ==========================================

def custom_directional_loss(preds: np.ndarray, train_data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom Asymmetric Objective Function.
    Penalizes the model 5x more if the predicted direction (sign)
    does not match the actual direction.
    """
    labels = train_data.get_label()
    residual = preds - labels

    # If the signs differ, apply a 5x penalty to the gradient and hessian
    penalty_multiplier = np.where(np.sign(preds) != np.sign(labels), 5.0, 1.0)

    grad = penalty_multiplier * residual
    hess = penalty_multiplier * np.ones_like(residual)

    return grad, hess


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes stationary features and shifts the target to prevent Look-Ahead Bias.
    Requires columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    df = df.copy()
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Calculate stationary features (Returns, Volatility, Moving Averages)
    df['Ret_1d'] = df['Close'].pct_change()
    df['Ret_5d'] = df['Close'].pct_change(5)
    df['Vol_20d'] = df['Ret_1d'].rolling(20).std()

    # Relative volume (Current Volume vs 20-day Average)
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # --- LOOK-AHEAD BIAS PREVENTION ---
    # The target is STRICTLY tomorrow's return.
    df['Target_Return'] = df['Ret_1d'].shift(-1)

    # Drop rows with NaN values created by rolling windows and shifting
    df.dropna(inplace=True)

    return df


# ==========================================
# 2. BACKTESTER STRATEGY CLASS
# ==========================================

class MlPredictiveStrategy(Strategy):
    """
    Event-driven strategy that uses a pre-trained LightGBM model to predict
    forward returns and scales position sizing based on recent volatility.
    """

    def __init__(self, symbol: str, data_handler: DataHandler, model: lgb.Booster, lookback: int = 20):
        self.symbol = symbol
        self.data_handler = data_handler
        self.model = model
        self.lookback = lookback

        # Engine will automatically assign this queue at runtime
        self.events: Optional[Queue] = None
        self.feature_cols = ['Ret_1d', 'Ret_5d', 'Vol_20d', 'Vol_Ratio']

    def calculate_signals(self, event: Any) -> None:
        # 1. Fetch minimum required bars for feature calculation
        bars = self.data_handler.get_latest_bars(self.symbol, n=self.lookback)
        if len(bars) < self.lookback:
            return

        # 2. Convert bars to DataFrame to calculate today's features
        df_bars = pd.DataFrame(bars)

        # Calculate today's features matching the training logic
        current_ret_1d = (df_bars['Close'].iloc[-1] - df_bars['Close'].iloc[-2]) / df_bars['Close'].iloc[-2]
        current_ret_5d = (df_bars['Close'].iloc[-1] - df_bars['Close'].iloc[-6]) / df_bars['Close'].iloc[-6]
        current_vol_20d = df_bars['Close'].pct_change().std()
        current_vol_ratio = df_bars['Volume'].iloc[-1] / df_bars['Volume'].mean()

        # Prepare feature vector for prediction
        X_current = pd.DataFrame([[current_ret_1d, current_ret_5d, current_vol_20d, current_vol_ratio]],
                                 columns=self.feature_cols)

        # 3. Predict tomorrow's return
        pred_return = self.model.predict(X_current)[0]

        # 4. SHARPE OPTIMIZATION: Volatility-adjusted signal strength
        # High volatility reduces position size; low volatility increases it.
        raw_strength = abs(pred_return) / (current_vol_20d + 1e-6)
        strength = min(raw_strength, 1.0)  # Cap leverage at 1.0

        # 5. Emit SignalEvent based on directional prediction
        dt = df_bars['Date'].iloc[-1]

        if pred_return > 0.001:  # Threshold to avoid trading on noise
            signal_type = SignalDirection.LONG
        elif pred_return < -0.001:
            signal_type = SignalDirection.SHORT
        else:
            signal_type = SignalDirection.EXIT

        signal = SignalEvent(
            symbol=self.symbol,
            datetime=dt,
            signal_type=signal_type,
            strength=strength
        )

        if self.events is not None:
            self.events.put(signal)


# ==========================================
# 3. STANDALONE EXECUTION & VISUALIZATION
# ==========================================

def run_standalone(data_path: str):
    print(f"Loading data from {data_path}...")
    try:
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        print(f"Could not load data: {e}. Generating dummy data for demonstration...")
        # Fallback dummy data generation if file is missing
        dates = pd.date_range('2020-01-01', periods=1000, freq='B')
        prices = np.cumprod(1 + np.random.normal(0.0005, 0.015, 1000)) * 100
        df = pd.DataFrame({'Date': dates, 'Close': prices, 'Volume': np.random.randint(1000, 10000, 1000)})
        df['Open'] = df['Close'] * 0.99
        df['High'] = df['Close'] * 1.01
        df['Low'] = df['Close'] * 0.98

    # 1. Prepare Data
    df_clean = prepare_data(df)
    features = ['Ret_1d', 'Ret_5d', 'Vol_20d', 'Vol_Ratio']

    # 2. Chronological Train/Test Split (Preventing Look-Ahead)
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:].copy()

    X_train = train_df[features]
    y_train = train_df['Target_Return']
    X_test = test_df[features]
    y_test = test_df['Target_Return']

    print("Training LightGBM with custom directional loss...")
    train_data = lgb.Dataset(X_train, label=y_train)

    params = {
        'learning_rate': 0.05,
        'max_depth': 4,
        'num_leaves': 15,
        'feature_fraction': 0.8,
        'verbose': -1,
        'objective': custom_directional_loss
    }

    # Train using the custom asymmetric objective
    model = lgb.train(params, train_data, num_boost_round=100)

    print("Running predictions on unseen test data...")
    test_df['Predicted_Return'] = model.predict(X_test)

    # Calculate Implied Predicted Price for charting
    # Implied Price T+1 = Actual Price T * (1 + Predicted Return T)
    test_df['Implied_Predicted_Price'] = test_df['Close'] * (1 + test_df['Predicted_Return'])
    # Shift actual close by 1 backward so it aligns chronologically with the prediction made for it
    test_df['Actual_Next_Close'] = test_df['Close'].shift(-1)

    # Calculate Strategy Returns (Sharpe Optimization)
    # Position = sign of prediction, scaled by (Predicted_Return / Volatility)
    test_df['Target_Position'] = np.where(test_df['Predicted_Return'] > 0, 1, -1)
    test_df['Strength'] = (abs(test_df['Predicted_Return']) / (test_df['Vol_20d'] + 1e-6)).clip(0, 1)
    test_df['Strategy_Return'] = test_df['Target_Position'] * test_df['Strength'] * test_df['Target_Return']

    # Metrics
    annualized_sharpe = np.sqrt(252) * (test_df['Strategy_Return'].mean() / test_df['Strategy_Return'].std())
    test_df['Cum_Strategy'] = (1 + test_df['Strategy_Return']).cumprod()
    test_df['Cum_Buy_Hold'] = (1 + test_df['Target_Return']).cumprod()

    print(f"\n--- Results ---")
    print(f"Test Set Sharpe Ratio: {annualized_sharpe:.2f}")

    # 3. Plotting
    plt.figure(figsize=(14, 10))

    # Subplot 1: Implied Predicted Price vs Actual Movement
    plt.subplot(2, 1, 1)
    # Plotting a 50-day slice to clearly see the prediction dynamics vs actuals
    slice_df = test_df.iloc[50:100]
    plt.plot(slice_df['Date'], slice_df['Actual_Next_Close'], label='Actual Next Day Price', color='blue', marker='o',
             markersize=4)
    plt.plot(slice_df['Date'], slice_df['Implied_Predicted_Price'], label='Implied Predicted Price', color='orange',
             linestyle='--')
    plt.title('Prediction Engine: Actual vs Predicted (50-Day Window)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Cumulative Returns
    plt.subplot(2, 1, 2)
    plt.plot(test_df['Date'], test_df['Cum_Strategy'], label=f'Strategy (Sharpe: {annualized_sharpe:.2f})',
             color='green')
    plt.plot(test_df['Date'], test_df['Cum_Buy_Hold'], label='Buy & Hold', color='gray', alpha=0.6)
    plt.title('Out-of-Sample Cumulative Returns')
    plt.ylabel('Growth (1.0 = Starting Equity)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML Prediction Strategy Backtest.")
    parser.add_argument("--data", type=str, default="data.parquet", help="Path to your CSV or Parquet dataset.")
    args = parser.parse_args()

    run_standalone(args.data)