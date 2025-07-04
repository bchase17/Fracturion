import numpy as np
import pandas as pd
import datetime
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def generate_all_features(df):

    df = df.sort_index(ascending=True)
    
    # =======================
    # Basic SMAs and Ratios
    # =======================
    for window in [10, 25, 50, 100, 200]:
        df[f'QQQ_SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'QQQ_EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        if window in (50, 100, 200):
            df[f'num_days_{window}'] = 0

    for window in [50, 100, 200]:
        for i in range(1, len(df)):
            prev = df.loc[i - 1, f'num_days_{window}']
            price = df.loc[i, 'Close']
            sma = df.loc[i, f'QQQ_SMA_{window}']
            if price > sma:
                df.loc[i, f'num_days_{window}'] = prev + 1 if prev >= 0 else 0
            elif price < sma:
                df.loc[i, f'num_days_{window}'] = prev - 1 if prev <= 0 else 0
            else:
                df.loc[i, f'num_days_{window}'] = 0

        df[f'num_days_{window}'] = df[f'num_days_{window}'].apply(lambda x: int(5 * round(x / 5)))

    # ============================
    # Relative Position Features
    # ============================
    def rows_since_max(x): return len(x) - x.argmax() - 1
    def rows_since_min(x): return len(x) - x.argmin() - 1

    for window in [10, 30, 60, 120]:
        df[f'Rel_Max_{window}'] = (df['High'] / df['High'].rolling(window=window).max()).round(3)
        df[f'Rel_Min_{window}'] = (df['Low'] / df['Low'].rolling(window=window).min()).round(3)
        df[f'Max_{window}_Rows_Since'] = df['High'].rolling(window=window).apply(rows_since_max, raw=True)
        df[f'Min_{window}_Rows_Since'] = df['Low'].rolling(window=window).apply(rows_since_min, raw=True)

    for a, b in [(50, 100), (50, 200), (100, 200), (10, 25), (10, 50), (10, 100), (10, 200), (25, 50), (25, 100), (25, 200)]:
        df[f'{a}_SMA_{b}'] = (df[f'QQQ_SMA_{a}'] / df[f'QQQ_SMA_{b}']).round(3)
        df[f'{a}_EMA_{b}'] = (df[f'QQQ_EMA_{a}'] / df[f'QQQ_EMA_{b}']).round(3)
        df[f'{a}_ESMA_{b}'] = (df[f'QQQ_EMA_{a}'] / df[f'QQQ_SMA_{b}']).round(3)

    def round_to_nearest_point05(x):
        return round(x * 20) / 20  # 1 / 0.05 = 20 steps per unit

    
    for window in [10, 25, 50, 100, 200]:
        df[f'QQQ_SMA_{window}'] = (df['Close'] / df[f'QQQ_SMA_{window}']).round(3)
        df[f'QQQ_EMA_{window}'] = (df['Close'] / df[f'QQQ_EMA_{window}']).round(3)
        
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        df[f'QQQ_EMA_{window}_r5'] = df[f'QQQ_EMA_{window}'].apply(round_to_nearest_point05)
        df[f'QQQ_SMA_{window}_r5'] = df[f'QQQ_SMA_{window}'].apply(round_to_nearest_point05)

    # ================
    # RSI Variants
    # ================
    def RSI(data, period):
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        RS = gain / loss
        return (100 - (100 / (1 + RS))).round(0)

    df['RSI_7'] = RSI(df, 7)
    df['RSI_14'] = RSI(df, 14)
    df['RSI_21'] = RSI(df, 21)

    # ================
    # MACD
    # ================
    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (ema_fast - ema_slow).round(3)
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean().round(3)

    # ================
    # Bollinger Bands
    # ================
    bb_mid = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = (bb_mid + 2 * bb_std).round(3) / df['Close']
    df['BB_Lower'] = (bb_mid - 2 * bb_std).round(3) / df['Close']
    df['BB_Mid'] = ((bb_mid * bb_std)/ df['Close']).round(3)

    # ================
    # VOLUME
    # ================

    # OBV Core
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Momentum / Deviation
    windows = [5, 10]
    for w in windows:
        df[f'OBV_ROC{w}'] = df['OBV'].pct_change(periods=w).round(3)
        df[f'OBV_Z{w}'] = ((df['OBV'] - df['OBV'].rolling(w).mean()) / df['OBV'].rolling(w).std()).round(3)
    
    df['UpMask'] = df['Close'] > df['Close'].shift(1)
    df['DownMask'] = df['Close'] < df['Close'].shift(1)
    df['UpVolume'] = df['Volume'] * df['UpMask']
    df['DownVolume'] = df['Volume'] * df['DownMask']
    windows = [10, 25, 50, 100]
    #df = calculate_obv_volume_ratio(df, windows)
    for w in windows:
        up = df['UpVolume'].rolling(w).sum()
        down = df['DownVolume'].rolling(w).sum()

        ratio = up / down.replace(0, np.nan)
        ratio.fillna(1_000_000, inplace=True)  # Up-only
        ratio[df['UpVolume'].rolling(w).sum() == 0] = 0  # Down-only

        df[f'Vol_Ratio_{w}'] = ratio.round(3)

    # Chaikin Money Flow (CMF)
    def CMF(data, period=20):
        mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
        mfv = mfm * data['Volume']
        cmf = mfv.rolling(window=period).sum() / data['Volume'].rolling(window=period).sum()
        return cmf.round(3)
    
    df['CMF_20'] = CMF(df, 20)
    df['CMF_10'] = CMF(df, 10)
     
    # Volume Rate of Change (VROC)
    windows = [3, 5, 10]
    for w in windows:
        df[f'VROC_{w}'] = df['Volume'].pct_change(periods=w).round(3)

    # Normalized Volume Spike
    windows = [10, 20, 40]
    for w in windows:
        df[f'Vol_Spike_{w}'] = (df['Volume'] / df['Volume'].rolling(w).median()).round(3)

    # Accumulation/Distribution Line (ADL)
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    df['ADL'] = (mfm * df['Volume']).cumsum().round(3)

    # ================
    # ATR
    # ================
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)
    windows = [7, 14, 21]
    for w in windows:
        df[f'ATR_{w}'] = tr.rolling(w).mean().round(1)

    # ================
    # ADX & DI
    # ================
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
    dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(14).mean()

    df['plus_DI'] = plus_di.round(0)
    df['minus_DI'] = minus_di.round(0)
    df['ADX'] = adx.round(0)

    # ================
    # Volatility
    # ================
    vol_5 = df['Close'].rolling(window=5).std().round(3)
    vol_10 = df['Close'].rolling(window=10).std().round(3)
    vol_25 = df['Close'].rolling(window=25).std().round(3)

    new_cols = {
        'vol_5': vol_5,
        'vol_10': vol_10,
        'vol_25': vol_25,
        'Price_Vol_Ratio_5': (df['Close'] / vol_5).round(3),
        'Price_Vol_Ratio_10': (df['Close'] / vol_10).round(3),
        'Price_Vol_Ratio_25': (df['Close'] / vol_25).round(3)
    }

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # ================
    # VIX External Data
    # ================
    vix_data = yf.Ticker("^VXN").history(period='1d', start='2005-03-10')
    # Resetting the index will turn the Date index into a column
    vix_data = vix_data.reset_index()[['Date', 'Close', 'High', 'Low', 'Volume']]
    # Convert the Date column to 'YYYY-MM-DD' format (if not already)
    vix_data['Date'] = pd.to_datetime(vix_data['Date']).dt.strftime('%Y-%m-%d')
    vix_data['VIX'] = vix_data['Close']

    ['VIX_5_change', 'VIX_crossover', 'VIX_1_change']
    # Convert the Date column to 'YYYY-MM-DD' format (if not already)
    vix_data = vix_data.sort_index(ascending=True)
    vix_data['VIX_rolling_std'] = vix_data['VIX'].rolling(window=5).std().round(1)
    vix_data['VIX_short_ma'] = vix_data['VIX'].rolling(window=3).mean()
    vix_data['VIX_long_ma'] = vix_data['VIX'].rolling(window=20).mean()
    vix_data['VIX_crossover'] = np.where(vix_data['VIX_short_ma'] > vix_data['VIX_long_ma'], 1, -1)
    vix_data['VIX_5_change'] = vix_data['VIX'].pct_change(periods=5).round(3)
    vix_data['VIX_1_change'] = vix_data['VIX'].pct_change(periods=1).round(3)
    vix_data['VIX_10_change'] = vix_data['VIX'].pct_change(periods=10).round(3)

    df = pd.merge(df, vix_data[['Date', 'VIX', 'VIX_rolling_std', 'VIX_crossover', 'VIX_5_change', 'VIX_1_change', 'VIX_10_change']],
                    on='Date', how='inner')

    # ================
    # Experimental Features
    # ================
    # CCI (Commodity Channel Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=14).mean()
    md = tp.rolling(window=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI_14'] = ((tp - ma) / (0.015 * md)).round(3)

    # Williams %R
    highest_high = df['High'].rolling(window=14).max()
    lowest_low = df['Low'].rolling(window=14).min()
    df['Williams_%R_14'] = ((highest_high - df['Close']) / (highest_high - lowest_low) * -100).round(3)

    # Z-scores
    zs = [5, 10, 25, 50]
    for z in zs:
        df[f'Zscore_{z}'] = ((df['Close'] - df['Close'].rolling(z).mean()) / df['Close'].rolling(z).std()).round(3)
        df[f'Zscore_{z}'] = df[f'Zscore_{z}'].replace([np.inf, -np.inf], np.nan)

    # ================
    # Convert All Non-Binary Variables into moving averages
    # ================
    windows   = [5, 10, 25, 50]
    
    temporal_targets = ['RSI_7', 'RSI_14', 'RSI_21', 'MACD', 'Signal_Line', 'CCI_14', 'Williams_%R_14', 'BB_Upper', 'BB_Lower', 'BB_Mid',
    'OBV', 'OBV_ROC5', 'OBV_ROC10', 'OBV_Z5', 'OBV_Z10', 'CMF_20', 'ADL', 'VROC_5', 'Vol_Spike_10', 'Vol_Spike_20',
    'Vol_Spike_40', 'Vol_Ratio_10', 'Vol_Ratio_25', 'Vol_Ratio_50', 'Vol_Ratio_100', 'ATR_7', 'ATR_14', 'ATR_21',
    'vol_5', 'vol_10', 'vol_25', 'Price_Vol_Ratio_5', 'Price_Vol_Ratio_10', 'Price_Vol_Ratio_25', 'plus_DI', 'minus_DI',
    'ADX', 'VIX', 'VIX_rolling_std', 'VIX_1_change', 'VIX_5_change', 'Zscore_5', 'Zscore_10', 'Zscore_25', 'Zscore_50']

    # dict‑comp to build every new Series, then concat once
    # Simple Moving Average
    new_cols = {
        f'{var}_MA{w}': df[var].rolling(w).mean().div(df[var])
        for w in windows
        for var in temporal_targets
    }

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)  # one write

    # Exponential Moving Average
    new_cols = {
        f'{var}_EA{w}': df[var].ewm(span=w, adjust=False).mean().div(df[var])
        for w in windows
        for var in temporal_targets
    }
    
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)  # one write

    def generate_slope_features(df):
        slope_windows = [5, 10, 25, 50]

        slope_targets = [
            'Close', 'RSI_7', 'RSI_14', 'RSI_21', 'MACD', 'Signal_Line', 'CCI_14', 'Williams_%R_14',
            'OBV', 'OBV_ROC5', 'OBV_ROC10', 'OBV_Z5', 'OBV_Z10', 'CMF_20', 'ADL', 'VROC_5',
            'Vol_Spike_10', 'Vol_Spike_20', 'Vol_Spike_40', 'Vol_Ratio_10', 'Vol_Ratio_25',
            'Vol_Ratio_50', 'Vol_Ratio_100', 'ATR_7', 'ATR_14', 'ATR_21', 'plus_DI', 'minus_DI',
            'ADX', 'VIX', 'VIX_rolling_std', 'VIX_1_change', 'VIX_5_change',
            'Zscore_5', 'Zscore_10', 'Zscore_25', 'Zscore_50',
            'Price_Vol_Ratio_5', 'Price_Vol_Ratio_10', 'Price_Vol_Ratio_25'
        ]

        def fast_slope(series, w):
            x = np.arange(w)
            x_mean = x.mean()
            denominator = ((x - x_mean) ** 2).sum()
            return (
                series.rolling(w).apply(
                    lambda y: ((x - x_mean) * (y - y.mean())).sum() / denominator,
                    raw=True
                )
            )

        slope_features = {}
        for w in slope_windows:
            for var in slope_targets:
                if var in df.columns:
                    slope_features[f'{var}_slope{w}'] = fast_slope(df[var], w).round(5)

        return pd.concat([df, pd.DataFrame(slope_features, index=df.index)], axis=1)
    
    df = generate_slope_features(df)

    return df

def regime(df_sma_returns): 
    
    df_sma = df_sma_returns.copy()

    # Daily Returns
    df_sma["Daily Return"] = df_sma["Close"].pct_change()
    # Intraday Volatility: High-Low Percentage Range
    df_sma["Intraday Change"] = (df_sma["High"] - df_sma["Low"]) / df_sma["Low"]
    df_sma["Intraday Change"] *= np.sign(df_sma["Daily Return"])
    df_sma["Intraday Volatility"] = df_sma["Intraday Change"].rolling(window=10).std()

    # === 3. Liquidity Score Calculation ===
    # Calculate VWAP
    df_sma["Typical Price"] = (df_sma["High"] + df_sma["Low"] + df_sma["Close"]) / 3
    # Rolling 252-day median values for normalization
    df_sma["Median Volume"] = df_sma["Volume"].rolling(window=50).median()
    df_sma["Median Close"] = df_sma["Close"].rolling(window=50).median()

    # Liquidity Score (using your formula)
    df_sma["Liquidity Score"] = (df_sma["Volume"] * df_sma["Typical Price"]) / (df_sma["Median Volume"] * df_sma["Median Close"])
    df_sma['Regime'] = (df_sma["Liquidity Score"] * (df_sma["Intraday Volatility"] * 100)).round(2)

    return df_sma

def final_df(ticker, returns, lb):

    # Define the ticker symbol
    tickerSymbol = ticker

    # Get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)
    start_date = (datetime.today() - relativedelta(years=lb)).strftime('%Y-%m-%d')

    tickerDf = tickerData.history(period='1d', start=start_date)
    
    # Resetting the index will turn the Date index into a column
    df_sma = tickerDf.reset_index()[['Date', 'Close', 'High', 'Low', 'Volume']]

    # Convert the Date column to 'YYYY-MM-DD' format (if not already)
    df_sma['Date'] = pd.to_datetime(df_sma['Date']).dt.strftime('%Y-%m-%d')
    #df_sma = pd.concat([df_sma, pd.DataFrame([main_row])], ignore_index=True)

    df_sma = generate_all_features(df_sma)
    df_sma = regime(df_sma)
    p40, p80 = np.percentile(df_sma['Regime'].dropna(), [40, 80])
    df_sma['Regime_Category'] = np.where(
        df_sma['Regime'] < p40, 0,
        np.where(df_sma['Regime'] > p80, 2, 1)
    )

    df_sma = df_sma.dropna()
    df_sma_clean = df_sma.copy()
    df_sma_clean = pd.DataFrame(df_sma_clean)
    df_sma_clean = df_sma_clean.sort_index(ascending=True)

    def add_column_based_on_future_value(df, days):
        future_return = (df['Close'].shift(-days) - df['Close']) / df['Close']

        if days >= 15:
            df[f'Return_{days}'] = np.where(
                future_return > 0.00, 1,
                np.where(future_return < -0.00, 0, np.nan)
            )
        else:
            df[f'Return_{days}'] = (future_return > 0).astype(int)

        return df

    # Apply return logic for each target horizon
    for r in returns:
        df_sma_returns = add_column_based_on_future_value(df_sma_clean, r)

    df_sma_returns = df_sma_returns.sort_index(ascending=False)

    return df_sma_returns
