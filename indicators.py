"""
技术指标模块
"""
import pandas as pd
import numpy as np


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range (EMA平滑)"""
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_adx(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    返回 DataFrame，包含列：adx, plus_di, minus_di
    """
    h, l, c = df['high'], df['low'], df['close']

    # Directional Movement
    up   = h.diff()
    down = -l.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    # True Range
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr      = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(span=period, adjust=False).mean()

    return pd.DataFrame({'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di})


def add_indicators(df: pd.DataFrame,
                   atr_period: int,
                   ema_period: int,
                   adx_period: int,
                   lookback: int) -> pd.DataFrame:
    """一次性把所有指标列附加到 df 并返回副本"""
    df = df.copy()

    df['atr']  = calc_atr(df, atr_period)
    df['ema']  = calc_ema(df['close'], ema_period)

    adx_df     = calc_adx(df, adx_period)
    df['adx']  = adx_df['adx']
    df['plus_di']  = adx_df['plus_di']
    df['minus_di'] = adx_df['minus_di']

    # shift(1) 避免当根K线用自己的数据判断突破（lookahead bias）
    df['recent_high'] = df['high'].rolling(lookback).max().shift(1)
    df['recent_low']  = df['low'].rolling(lookback).min().shift(1)

    return df
