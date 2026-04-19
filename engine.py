"""
回测引擎 —— 状态机实现 Breakout Pullback Strategy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List

from .config import StrategyParams
from .indicators import add_indicators


# ─────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    direction: int           # 1=多  -1=空
    entry_bar: int
    entry_date: object
    entry_price: float
    stop_loss: float
    take_profit: float
    breakout_level: float
    atr_at_entry: float
    size: float              # 仓位（股数/手数），由1%风险计算

    exit_bar:    Optional[int]   = None
    exit_date:   object          = None
    exit_price:  Optional[float] = None
    pnl:         float           = 0.0
    pnl_r:       float           = 0.0   # 以初始风险R为单位的盈亏
    exit_reason: str             = ''


# ─────────────────────────────────────────────────────────────
# 状态枚举
# ─────────────────────────────────────────────────────────────

IDLE             = 'idle'
WAIT_PB_LONG     = 'wait_pullback_long'
WAIT_PB_SHORT    = 'wait_pullback_short'


# ─────────────────────────────────────────────────────────────
# 回测引擎
# ─────────────────────────────────────────────────────────────

class Backtester:

    def __init__(self, params: StrategyParams, allow_short: bool = False):
        self.p           = params
        self.allow_short = allow_short
        self.trades:      List[Trade] = []
        self.equity_curve: List[float] = []

    # ── 主回测循环 ────────────────────────────────────────────

    def run(self, df: pd.DataFrame, initial_capital: float = 100_000.0) -> 'Backtester':
        p = self.p
        self.initial_capital = initial_capital

        df = add_indicators(
            df,
            atr_period=p.atr_period,
            ema_period=p.ema_period,
            adx_period=p.adx_period,
            lookback=p.lookback,
        )
        df = df.dropna().reset_index()   # 转成行索引，方便逐行遍历

        capital  = initial_capital
        position: Optional[Trade] = None

        state            = IDLE
        breakout_level   = None
        breakout_atr     = None
        breakout_bar     = None
        pullback_touched  = False   # 是否经历了真正的回踩

        equity = []

        for i in range(len(df)):
            row  = df.iloc[i]
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            atr  = row['atr']
            ema  = row['ema']
            adx  = row['adx']
            date = row['date']
            rh   = row['recent_high']   # 前N根最高
            rl   = row['recent_low']    # 前N根最低

            # ── 1. 持仓出场检查 ───────────────────────────────
            if position is not None:
                exit_price  = None
                exit_reason = ''

                if position.direction == 1:     # 多头出场
                    if l <= position.stop_loss:
                        exit_price  = position.stop_loss
                        exit_reason = 'stop_loss'
                    elif h >= position.take_profit and p.exit_mode == 'fixed_r':
                        exit_price  = position.take_profit
                        exit_reason = 'take_profit'
                    elif p.exit_mode == 'ema_exit' and c < ema:
                        exit_price  = c
                        exit_reason = 'ema_exit'
                    elif p.exit_mode == 'atr_trail':
                        trail = c - p.atr_trail_mult * atr
                        position.stop_loss = max(position.stop_loss, trail)
                        if l <= position.stop_loss:
                            exit_price  = position.stop_loss
                            exit_reason = 'trail_stop'

                else:                           # 空头出场
                    if h >= position.stop_loss:
                        exit_price  = position.stop_loss
                        exit_reason = 'stop_loss'
                    elif l <= position.take_profit and p.exit_mode == 'fixed_r':
                        exit_price  = position.take_profit
                        exit_reason = 'take_profit'
                    elif p.exit_mode == 'ema_exit' and c > ema:
                        exit_price  = c
                        exit_reason = 'ema_exit'
                    elif p.exit_mode == 'atr_trail':
                        trail = c + p.atr_trail_mult * atr
                        position.stop_loss = min(position.stop_loss, trail)
                        if h >= position.stop_loss:
                            exit_price  = position.stop_loss
                            exit_reason = 'trail_stop'

                if exit_price is not None:
                    risk   = abs(position.entry_price - position.stop_loss) * position.size
                    pnl    = (exit_price - position.entry_price) * position.direction * position.size
                    pnl_r  = pnl / risk if risk > 0 else 0

                    position.exit_bar    = i
                    position.exit_date   = date
                    position.exit_price  = exit_price
                    position.pnl         = pnl
                    position.pnl_r       = pnl_r
                    position.exit_reason = exit_reason

                    capital += pnl
                    self.trades.append(position)
                    position  = None
                    state     = IDLE
                    breakout_level  = None
                    breakout_bar    = None
                    pullback_touched = False

            equity.append(capital)

            # 已有持仓，跳过入场逻辑
            if position is not None:
                continue

            # ── 2. ADX 过滤 ───────────────────────────────────
            if p.use_adx_filter and adx < p.adx_threshold:
                # 震荡市：重置等待状态（可选，保守做法）
                state = IDLE
                breakout_level = None
                breakout_bar   = None
                continue

            # ── 3. 等待超时重置 ───────────────────────────────
            if breakout_bar is not None and (i - breakout_bar) > p.pullback_bars:
                state            = IDLE
                breakout_level   = None
                breakout_atr     = None
                breakout_bar     = None
                pullback_touched  = False

            # ── 4. 状态机 ─────────────────────────────────────

            if state == IDLE:
                # 做多突破检测
                if pd.notna(rh) and c > rh and (c - rh) > p.atr_mult * atr:
                    state           = WAIT_PB_LONG
                    breakout_level  = rh
                    breakout_atr    = atr
                    breakout_bar    = i
                    pullback_touched = False

                # 做空突破检测（A股默认关闭）
                elif self.allow_short and pd.notna(rl) and c < rl and (rl - c) > p.atr_mult * atr:
                    state           = WAIT_PB_SHORT
                    breakout_level  = rl
                    breakout_atr    = atr
                    breakout_bar    = i
                    pullback_touched = False

            elif state == WAIT_PB_LONG:
                # 价格收盘跌破突破点 → 突破失败，重置
                if c < breakout_level:
                    state = IDLE
                    breakout_level  = None
                    breakout_bar    = None
                    pullback_touched = False
                    continue

                # 回踩触及区域：close 回落到 (breakout_level, breakout_level + zone)
                pullback_zone_top = breakout_level + p.pullback_zone_mult * breakout_atr
                if breakout_level < c <= pullback_zone_top:
                    pullback_touched = True

                # 入场条件（回踩确认后）：
                #   ① 已经历回踩（pullback_touched）
                #   ② 当前阳线（收 > 开）
                #   ③ 收盘站上 EMA（趋势确认）
                #   ④ 收盘高于突破点（不能在关键位下方入场）
                bullish_candle = c > o
                if pullback_touched and bullish_candle and c > ema and c > breakout_level:
                    sl             = breakout_level - p.sl_atr_mult * breakout_atr
                    risk_per_unit  = c - sl
                    if risk_per_unit <= 0:
                        continue
                    tp             = c + risk_per_unit * p.risk_reward
                    risk_amount    = capital * p.risk_pct
                    size           = risk_amount / risk_per_unit
                    # 仓位上限：单笔占用不超过 max_position_pct
                    max_size       = (capital * p.max_position_pct) / c
                    size           = min(size, max_size)

                    position = Trade(
                        direction=1,
                        entry_bar=i,
                        entry_date=date,
                        entry_price=c,
                        stop_loss=sl,
                        take_profit=tp,
                        breakout_level=breakout_level,
                        atr_at_entry=breakout_atr,
                        size=size,
                    )
                    state           = IDLE
                    breakout_level  = None
                    breakout_bar    = None
                    pullback_touched = False

            elif state == WAIT_PB_SHORT:
                if c > breakout_level:
                    state = IDLE
                    breakout_level  = None
                    breakout_bar    = None
                    pullback_touched = False
                    continue

                pullback_zone_bot = breakout_level - p.pullback_zone_mult * breakout_atr
                if pullback_zone_bot <= c < breakout_level:
                    pullback_touched = True

                bearish_candle = c < o
                if pullback_touched and bearish_candle and c < ema and c < breakout_level:
                    sl             = breakout_level + p.sl_atr_mult * breakout_atr
                    risk_per_unit  = sl - c
                    if risk_per_unit <= 0:
                        continue
                    tp             = c - risk_per_unit * p.risk_reward
                    risk_amount    = capital * p.risk_pct
                    size           = risk_amount / risk_per_unit
                    max_size       = (capital * p.max_position_pct) / c
                    size           = min(size, max_size)

                    position = Trade(
                        direction=-1,
                        entry_bar=i,
                        entry_date=date,
                        entry_price=c,
                        stop_loss=sl,
                        take_profit=tp,
                        breakout_level=breakout_level,
                        atr_at_entry=breakout_atr,
                        size=size,
                    )
                    state           = IDLE
                    breakout_level  = None
                    breakout_bar    = None
                    pullback_touched = False

        # 回测结束，强制平掉未关仓的持仓
        if position is not None:
            last = df.iloc[-1]
            c    = last['close']
            risk = abs(position.entry_price - position.stop_loss) * position.size
            pnl  = (c - position.entry_price) * position.direction * position.size
            position.exit_bar    = len(df) - 1
            position.exit_date   = last['date']
            position.exit_price  = c
            position.pnl         = pnl
            position.pnl_r       = pnl / risk if risk > 0 else 0
            position.exit_reason = 'end_of_data'
            capital += pnl
            self.trades.append(position)
            equity[-1] = capital

        self.equity_curve = equity
        self.df_indexed   = df   # 保存带指标的 df 供报告使用
        return self
