"""
策略参数配置
"""
from dataclasses import dataclass, field


@dataclass
class StrategyParams:
    # ── 突破参数 ──────────────────────────────────────
    lookback: int = 20          # 突破周期：过去N根K线的最高/最低
    atr_period: int = 14        # ATR计算周期
    atr_mult: float = 0.5       # 假突破过滤：突破幅度需 > ATR * atr_mult

    # ── 趋势确认参数 ─────────────────────────────────
    ema_period: int = 20        # EMA周期（趋势确认 & 回踩判断）
    adx_period: int = 14        # ADX周期
    adx_threshold: float = 25.0 # ADX < 此值时不开仓（震荡过滤）
    use_adx_filter: bool = True # 是否启用ADX过滤

    # ── 回踩判断参数 ─────────────────────────────────
    pullback_bars: int = 10     # 突破后最多等待N根K线完成回踩确认
    # 回踩有效区域：close 在 [breakout_level, breakout_level + atr*pullback_zone_mult]
    pullback_zone_mult: float = 2.0

    # ── 出场参数 ─────────────────────────────────────
    # exit_mode: 'fixed_r' | 'atr_trail' | 'ema_exit'
    exit_mode: str = 'fixed_r'
    risk_reward: float = 3.0    # fixed_r模式下的盈亏比（TP = entry ± risk * rr）
    atr_trail_mult: float = 2.0 # atr_trail模式下的trailing stop倍数
    sl_atr_mult: float = 0.5    # 止损设置：breakout_level ± ATR * sl_atr_mult

    # ── 风控参数 ─────────────────────────────────────
    risk_pct: float = 0.01      # 每笔交易风险占账户资金的比例
    max_position_pct: float = 0.20  # 单笔最大占用本金比例（默认20%）


@dataclass
class BacktestConfig:
    symbol: str = '000001'          # 股票代码（A股6位）
    start_date: str = '2020-01-01'
    end_date: str = '2024-12-31'
    initial_capital: float = 100_000.0
    adjust: str = 'qfq'             # 复权方式：qfq前复权 hfq后复权
    allow_short: bool = False       # A股默认不允许做空
