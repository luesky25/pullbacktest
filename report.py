"""
绩效统计与可视化
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Backtester

# Windows 中文字体兼容
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────────────────────
# 统计计算
# ─────────────────────────────────────────────────────────────

def compute_stats(bt: 'Backtester') -> dict:
    trades  = bt.trades
    equity  = np.array(bt.equity_curve, dtype=float)
    initial = bt.initial_capital

    if not trades:
        return {}

    pnls    = np.array([t.pnl for t in trades])
    winners = pnls[pnls > 0]
    losers  = pnls[pnls <= 0]

    win_rate      = len(winners) / len(trades)
    avg_win       = winners.mean() if len(winners) else 0
    avg_loss      = losers.mean()  if len(losers)  else 0
    profit_factor = (winners.sum() / abs(losers.sum())) if losers.sum() != 0 else float('inf')
    total_pnl     = pnls.sum()
    total_return  = total_pnl / initial

    # 最大回撤
    rolling_max   = np.maximum.accumulate(equity)
    drawdown      = (equity - rolling_max) / rolling_max
    max_dd        = drawdown.min()
    max_dd_end    = drawdown.argmin()
    max_dd_start  = equity[:max_dd_end].argmax()

    # 夏普（年化，假设每根K线=1交易日）
    daily_ret     = pd.Series(equity).pct_change().dropna()
    sharpe        = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                     if daily_ret.std() > 0 else 0)

    # 平均持仓
    hold_bars     = [t.exit_bar - t.entry_bar for t in trades if t.exit_bar is not None]
    avg_hold      = np.mean(hold_bars) if hold_bars else 0

    # 出场原因
    exit_reasons  = pd.Series([t.exit_reason for t in trades]).value_counts().to_dict()

    # 多空分布
    longs  = [t for t in trades if t.direction ==  1]
    shorts = [t for t in trades if t.direction == -1]

    return dict(
        total_trades=len(trades),
        long_trades=len(longs),
        short_trades=len(shorts),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        total_pnl=total_pnl,
        total_return=total_return,
        final_capital=initial + total_pnl,
        max_dd=max_dd,
        max_dd_start=max_dd_start,
        max_dd_end=max_dd_end,
        sharpe=sharpe,
        avg_hold=avg_hold,
        exit_reasons=exit_reasons,
    )


def print_report(bt: 'Backtester', symbol: str = '') -> dict:
    stats = compute_stats(bt)
    if not stats:
        print("⚠  回测期间没有产生任何交易，请调整参数或数据范围。")
        return {}

    sep = "=" * 56
    print(sep)
    print(f"   Breakout Pullback Strategy 回测报告  {symbol}")
    print(sep)
    print(f"  交易次数    : {stats['total_trades']}  "
          f"（多 {stats['long_trades']} / 空 {stats['short_trades']}）")
    print(f"  胜率        : {stats['win_rate']:.1%}")
    print(f"  平均盈利    : {stats['avg_win']:>12,.2f}")
    print(f"  平均亏损    : {stats['avg_loss']:>12,.2f}")
    print(f"  盈亏比      : {stats['profit_factor']:.2f}")
    print(f"  净收益      : {stats['total_pnl']:>12,.2f}")
    print(f"  总收益率    : {stats['total_return']:.2%}")
    print(f"  最终资金    : {stats['final_capital']:>12,.2f}")
    print(f"  最大回撤    : {stats['max_dd']:.2%}")
    print(f"  夏普比率    : {stats['sharpe']:.2f}")
    print(f"  平均持仓    : {stats['avg_hold']:.1f} 根K线")
    print(sep)
    print("  出场原因分布:")
    for reason, cnt in stats['exit_reasons'].items():
        pct = cnt / stats['total_trades']
        bar = '█' * int(pct * 20)
        print(f"    {reason:<18}: {cnt:>3}次  {pct:.1%}  {bar}")
    print(sep)
    return stats


# ─────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────

def plot_result(bt: 'Backtester', symbol: str = '', save_path: str = 'backtest_result.png'):
    df     = bt.df_indexed
    trades = bt.trades
    equity = bt.equity_curve
    p      = bt.p

    fig, axes = plt.subplots(
        4, 1, figsize=(18, 14),
        gridspec_kw={'height_ratios': [4, 1.5, 1, 1]},
        sharex=False,
    )

    x     = np.arange(len(df))
    dates = df['date'].values

    # ── 子图1：价格 + EMA + 交易标记 ─────────────────────────
    ax1 = axes[0]
    ax1.plot(x, df['close'], color='#555555', linewidth=0.8, label='收盘价')
    ax1.plot(x, df['ema'],   color='#F5A623', linewidth=1.4,
             label=f'EMA{p.ema_period}', alpha=0.9)

    for t in trades:
        color_in  = '#2ECC71' if t.direction == 1 else '#E74C3C'
        color_out = '#E74C3C' if t.direction == 1 else '#2ECC71'
        mk_in     = '^' if t.direction == 1 else 'v'

        ax1.scatter(t.entry_bar, t.entry_price,
                    marker=mk_in, color=color_in, s=90, zorder=6)
        ax1.axhline(t.stop_loss,   color='#E74C3C', linewidth=0.4,
                    linestyle=':', alpha=0.4)
        ax1.axhline(t.take_profit, color='#2ECC71', linewidth=0.4,
                    linestyle=':', alpha=0.4)

        if t.exit_bar is not None:
            mk_out = 'v' if t.direction == 1 else '^'
            pnl_color = '#2ECC71' if t.pnl > 0 else '#E74C3C'
            ax1.scatter(t.exit_bar, t.exit_price,
                        marker=mk_out, color=pnl_color, s=70, zorder=6)
            ax1.annotate(
                f"{t.pnl_r:+.1f}R",
                xy=(t.exit_bar, t.exit_price),
                xytext=(4, 6), textcoords='offset points',
                fontsize=6.5, color=pnl_color,
            )

    long_patch  = mpatches.Patch(color='#2ECC71', label='多头入场 ▲ / 出场 ▼')
    short_patch = mpatches.Patch(color='#E74C3C', label='空头入场 ▼ / 出场 ▲')
    ax1.legend(handles=[long_patch, short_patch], loc='upper left', fontsize=8)
    ax1.set_title(f'Breakout Pullback Strategy  |  {symbol}', fontsize=13)
    ax1.set_ylabel('价格')
    ax1.grid(axis='y', alpha=0.25)

    # ── 子图2：权益曲线 ───────────────────────────────────────
    ax2 = axes[1]
    eq_arr = np.array(equity)
    ax2.plot(eq_arr, color='#3498DB', linewidth=1.2, label='权益曲线')
    ax2.axhline(bt.initial_capital, color='gray', linestyle='--', linewidth=0.8)

    # 填色区分盈亏
    ax2.fill_between(range(len(eq_arr)), bt.initial_capital, eq_arr,
                     where=eq_arr >= bt.initial_capital,
                     alpha=0.25, color='#2ECC71', label='盈利区域')
    ax2.fill_between(range(len(eq_arr)), bt.initial_capital, eq_arr,
                     where=eq_arr < bt.initial_capital,
                     alpha=0.25, color='#E74C3C', label='亏损区域')
    ax2.set_ylabel('资金')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(axis='y', alpha=0.25)

    # ── 子图3：ADX ────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(x, df['adx'], color='#9B59B6', linewidth=1.0, label='ADX')
    ax3.axhline(p.adx_threshold, color='red', linestyle='--',
                linewidth=0.8, label=f'阈值={p.adx_threshold}')
    ax3.fill_between(x, 0, df['adx'],
                     where=df['adx'] >= p.adx_threshold,
                     alpha=0.15, color='#9B59B6')
    ax3.set_ylabel('ADX')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(axis='y', alpha=0.25)

    # ── 子图4：逐笔PnL（条形图）─────────────────────────────
    ax4 = axes[3]
    if trades:
        trade_bars  = [t.entry_bar for t in trades]
        trade_pnl_r = [t.pnl_r    for t in trades]
        colors = ['#2ECC71' if r > 0 else '#E74C3C' for r in trade_pnl_r]
        ax4.bar(range(len(trades)), trade_pnl_r, color=colors, width=0.6)
        ax4.axhline(0, color='gray', linewidth=0.8)
        ax4.set_ylabel('盈亏(R)')
        ax4.set_xlabel('交易序号')
        ax4.grid(axis='y', alpha=0.25)

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图表已保存：{save_path}")
