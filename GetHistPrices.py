# Copyright 2019 Gehtsoft USA LLC

# Licensed under the license derived from the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

# http://fxcodebase.com/licenses/open-source/license.html

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from datetime import datetime, timedelta

import pandas as pd
from config_loader import ConfigLoader
from forexconnect import ForexConnect, fxcorepy
import common_samples

timeframe_minutes = {
    "m1": "00001",
    "m5": "00005",
    "m15": "00015",
    "m30": "00030",
    "H1": "00060",
    "H4": "00240",
    "D1": "01440",
    "W1": "10080",
    "M1": "43200",
}
def parse_args():
    parser = argparse.ArgumentParser(description='Process command parameters.')
    common_samples.add_main_arguments(parser)
    common_samples.add_instrument_timeframe_arguments(parser)
    common_samples.add_date_arguments(parser)
    common_samples.add_max_bars_arguments(parser)
    parser.add_argument('-o', '--output',
                        metavar="OUTPUT_FOLDER",
                        required=True,
                        help='Output folder path for CSV file.')
    parser.add_argument('--need_more_features',
                        type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=True,
                        help='Need more features (default: True). Use --need_more_features false to skip.')
    parser.add_argument('--sma',
                        metavar="SMA_PERIOD",
                        type=int,
                        default=14,
                        help='Simple Moving Average period (default: 14).')
    parser.add_argument('--atr',
                        metavar="ATR_PERIOD",
                        type=int,
                        default=14,
                        help='Average True Range period (default: 14).')
    parser.add_argument('--rsi',
                        metavar="RSI_PERIOD",
                        type=int,
                        default=14,
                        help='Relative Strength Index period (default: 14).')
    parser.add_argument('--ema',
                        metavar="EMA_PERIOD",
                        type=int,
                        default=14,
                        help='Exponential Moving Average period (default: 14).')
    parser.add_argument('--macd_fast',
                        metavar="MACD_PERIOD",
                        type=int,
                        default=12,
                        help='MACD Fast period (default: 12).')
    parser.add_argument('--macd_slow',
                        metavar="MACD_SLOW_PERIOD",
                        type=int,
                        default=26,
                        help='MACD Slow period (default: 26).')
    parser.add_argument('--macd_signal',
                        metavar="MACD_SIGNAL_PERIOD",
                        type=int,
                        default=9,
                        help='MACD Signal period (default: 9).')
    parser.add_argument('--psar_af_start',
                        metavar="PSAR_AF_START",
                        type=float,
                        default=0.02,
                        help='Parabolic SAR acceleration factor start (default: 0.02).')
    parser.add_argument('--psar_af_step',
                        metavar="PSAR_AF_STEP",
                        type=float,
                        default=0.02,
                        help='Parabolic SAR acceleration factor step (default: 0.02).')
    parser.add_argument('--psar_af_max',
                        metavar="PSAR_AF_MAX",
                        type=float,
                        default=0.2,
                        help='Parabolic SAR acceleration factor max (default: 0.2).')
    parser.add_argument('--bb_period',
                        metavar="BB_PERIOD",
                        type=int,
                        default=20,
                        help='Bollinger Bands period (default: 20).')
    parser.add_argument('--bb_sigma',
                        metavar="BB_SIGMA",
                        type=float,
                        default=2,
                        help='Bollinger Bands standard deviation multiplier (default: 2).')
    args = parser.parse_args()
    return args


def calculate_atr(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, 
                  period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    ATR measures market volatility by decomposing the entire range of an asset price.
    
    Parameters:
    high_prices: pandas Series of high prices
    low_prices: pandas Series of low prices
    close_prices: pandas Series of close prices
    period: ATR period (default 14)
    
    Returns:
    pandas Series with ATR values
    """
    # Calculate True Range
    # TR = max(high - low, abs(high - previous_close), abs(low - previous_close))
    
    high_low = high_prices - low_prices
    high_close = (high_prices - close_prices.shift(1)).abs()
    low_close = (low_prices - close_prices.shift(1)).abs()
    
    # True Range is the maximum of the three
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR as exponential moving average of True Range
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    # Fill initial NaN values with the simple moving average
    atr.iloc[:period] = true_range.iloc[:period].rolling(window=period, min_periods=1).mean()
    
    return atr

def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) for a series of close prices
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    
    Parameters:
    close_prices: pandas Series of close prices
    period: RSI period (default 14)
    
    Returns:
    pandas Series with RSI values (0-100)
    """
    # Calculate price changes (difference between consecutive prices)
    delta = close_prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)  # Keep positive changes, set negatives to 0
    losses = -delta.where(delta < 0, 0)  # Keep negative changes (made positive), set positives to 0
    
    # Calculate the initial smoothed averages using Simple Moving Average for the first period
    avg_gain = gains.rolling(window=period, min_periods=1).mean()
    avg_loss = losses.rolling(window=period, min_periods=1).mean()
    
    # Apply Wilder's smoothing (exponential smoothing) for subsequent periods
    # This is the traditional RSI calculation method
    for i in range(period, len(gains)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gains.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + losses.iloc[i]) / period
    
    # Calculate Relative Strength (RS) and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Handle edge cases:
    # When avg_loss is 0, RSI should be 100 (all gains, no losses)
    # When avg_gain is 0, RSI should be 0 (all losses, no gains)
    # When both are 0, RSI should be 50 (neutral)
    rsi = rsi.fillna(50)  # Fill NaN values with neutral RSI
    
    return rsi

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA)
    
    Parameters:
    prices: pandas Series of prices
    period: EMA period
    
    Returns:
    pandas Series with EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()

def calculate_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Parameters:
    prices: pandas Series of prices
    fast_period: Fast EMA period (default 12)
    slow_period: Slow EMA period (default 26)
    signal_period: Signal line EMA period (default 9)
    
    Returns:
    tuple: (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, histogram, signal_line

def calculate_psar(high, low, close, af_start=0.02, af_step=0.02, af_max=0.2):
    length = len(close)
    psar = pd.Series(index=close.index, dtype=float)
    af = af_start
    trend = 1  # 1 = uptrend, -1 = downtrend
    ep = low.iloc[0]  # extreme point
    psar.iloc[0] = high.iloc[0]
    
    for i in range(1, length):
        if trend == 1:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
            if low.iloc[i] < psar.iloc[i]:
                trend = -1
                psar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
            elif high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_step, af_max)
        else:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
            if high.iloc[i] > psar.iloc[i]:
                trend = 1
                psar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start
            elif low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_step, af_max)
    
    return psar

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, sigma: float = 2) -> tuple:
    """
    Calculate Bollinger Bands
    
    Parameters:
    prices: pandas Series of prices
    period: Moving average period (default 20)
    sigma: Number of standard deviations (default 2)
    
    Returns:
    tuple: (middle_band, upper_band, lower_band)
    """
    middle_band = prices.rolling(window=period, min_periods=1).mean()
    std = prices.rolling(window=period, min_periods=1).std()
    upper_band = middle_band + (std * sigma)
    lower_band = middle_band - (std * sigma)
    
    return middle_band, upper_band, lower_band

def calculate_obv(close_prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV)
    
    OBV measures buying and selling pressure as a cumulative indicator,
    adding volume on up days and subtracting on down days.
    
    Parameters:
    close_prices: pandas Series of close prices
    volume: pandas Series of volume
    
    Returns:
    pandas Series with OBV values
    """
    # Calculate price changes
    price_change = close_prices.diff()
    
    # Initialize OBV series
    obv = pd.Series(index=close_prices.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]  # First OBV equals first volume
    
    # Calculate OBV
    for i in range(1, len(close_prices)):
        if price_change.iloc[i] > 0:  # Price went up
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif price_change.iloc[i] < 0:  # Price went down
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:  # Price unchanged
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def main():
    args = parse_args()
    creds = ConfigLoader.load()
    print("Credentials loaded from config.ini")
    str_user_id = creds['login']
    str_password = creds['password']
    str_url = creds['url']
    str_connection = creds['connection']
    str_session_id = args.session
    str_pin = args.pin
    str_instrument = args.i
    str_timeframe = args.timeframe
    quotes_count = args.quotescount
    date_from = args.datefrom
    date_to = args.dateto
    output_folder = args.output
    need_more_features = args.need_more_features
    sma_period = args.sma
    atr_period = args.atr
    rsi_period = args.rsi   
    ema_period = args.ema
    macd_fast_period = args.macd_fast
    macd_slow_period = args.macd_slow
    macd_signal_period = args.macd_signal
    psar_af_start = args.psar_af_start
    psar_af_step = args.psar_af_step
    psar_af_max = args.psar_af_max
    bb_period = args.bb_period
    bb_sigma = args.bb_sigma
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate filename: instrument_timeframe_minutes.csv
    instrument_clean = str_instrument.replace('/', '')
    minutes = timeframe_minutes.get(str_timeframe, str_timeframe)
    csv_filename = f"{instrument_clean}_{minutes}.csv"
    csv_path = os.path.join(output_folder, csv_filename)

    with ForexConnect() as fx:
        try:
            fx.login(str_user_id, str_password, str_url,
                     str_connection, str_session_id, str_pin,
                     common_samples.session_status_changed)

            print("")
            print("Requesting a price history...")
            # Add 1 month buffer before date_from for indicator calculation
            if date_from:
                date_from_buffered = date_from - timedelta(days=31)
            else:
                date_from_buffered = date_from
            history = fx.get_history(str_instrument, str_timeframe, date_from_buffered, date_to, quotes_count)
            current_unit, _ = ForexConnect.parse_timeframe(str_timeframe)
           
            date_format = '%Y.%m.%d %H:%M:%S'
            
            # Convert history to pandas DataFrame and write to CSV
            if current_unit == fxcorepy.O2GTimeFrameUnit.TICK:
                data = []
                for row in history:
                    data.append({
                        'DateTime': pd.to_datetime(str(row['Date'])).strftime(date_format),
                        'Bid': round(row['Bid'], 5),
                        'Ask': round(row['Ask'], 5)
                    })
                df = pd.DataFrame(data, columns=['DateTime', 'Bid', 'Ask'])
                df = df.sort_values('DateTime', ascending=False)
                df.to_csv(csv_path, index=False)
                print(f"Tick data written to: {csv_path}")
                print(f"Records: {len(df)}")
            else:
                data = []
                for row in history:
                    data.append({
                        'DateTime': pd.to_datetime(str(row['Date'])).strftime(date_format),
                        'Open': round(row['BidOpen'], 5),
                        'High': round(row['BidHigh'], 5),
                        'Low': round(row['BidLow'], 5),
                        'Close': round(row['BidClose'], 5),
                        'Volume': row['Volume']
                    })
                df = pd.DataFrame(data, columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                # Sort by DateTime ascending (oldest first) - required for indicator calculations
                df = df.sort_values('DateTime', ascending=True).reset_index(drop=True)
                if need_more_features:
                    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], atr_period)
                    df['RSI'] = calculate_rsi(df['Close'], rsi_period)
                    df['MACD'], df['MACD_Hist'], df['MACD_Signal'] = calculate_macd(df['Close'], macd_fast_period, macd_slow_period, macd_signal_period)
                    df['PSAR'] = calculate_psar(df['High'], df['Low'], df['Close'], af_start=psar_af_start, af_step=psar_af_step, af_max=psar_af_max)
                    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'], period=bb_period, sigma=bb_sigma)
                    df['OBV'] = calculate_obv(df['Close'], df['Volume'])
                    df['SMA'] = df['Close'].rolling(window=sma_period).mean().round(5)
                    df['EMA'] = calculate_ema(df['Close'], ema_period)
                else:
                    print("No more features requested")
                    
                # Filter to original date_from (remove buffer period)
                if date_from:
                    # Convert to tz-naive for comparison
                    date_from_naive = date_from.replace(tzinfo=None) if hasattr(date_from, 'replace') else date_from
                    df = df[pd.to_datetime(df['DateTime']) >= date_from_naive].reset_index(drop=True)
                df.to_csv(csv_path, index=False)
                print(f"Price history written to: {csv_path}")
                print(f"Records: {len(df)}")
        except Exception as e:
            common_samples.print_exception(e)
        try:
            fx.logout()
        except Exception as e:
            common_samples.print_exception(e)


if __name__ == "__main__":
    main()
