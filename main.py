"""
Breakout Pullback Strategy 主入口
────────────────────────────────
用法：
    python -m breakout_pullback.main
    python -m breakout_pullback.main --symbol 600519 --start 2019-01-01 --end 2024-12-31
    python -m breakout_pullback.main --exit_mode atr_trail --adx_threshold 20
"""
import argparse
import sys
import os

# 允许直接运行 main.py（非包模式）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakout_pullback.config  import StrategyParams, BacktestConfig
from breakout_pullback.data    import load_stock
from breakout_pullback.engine  import Backtester
from breakout_pullback.report  import print_report, plot_result


def parse_args():
    parser = argparse.ArgumentParser(description='Breakout Pullback Strategy Backtest')

    # 数据参数
    parser.add_argument('--symbol',    default='000001', help='A股代码，默认000001')
    parser.add_argument('--start',     default='2020-01-01')
    parser.add_argument('--end',       default='2024-12-31')
    parser.add_argument('--capital',   type=float, default=100_000.0)
    parser.add_argument('--adjust',    default='qfq', choices=['qfq','hfq',''])

    # 策略参数
    parser.add_argument('--lookback',        type=int,   default=20)
    parser.add_argument('--atr_period',      type=int,   default=14)
    parser.add_argument('--atr_mult',        type=float, default=0.5,
                        help='假突破过滤：突破幅度 > ATR * atr_mult')
    parser.add_argument('--ema_period',      type=int,   default=20)
    parser.add_argument('--adx_period',      type=int,   default=14)
    parser.add_argument('--adx_threshold',   type=float, default=25.0)
    parser.add_argument('--no_adx_filter',   action='store_true',
                        help='禁用ADX震荡过滤')
    parser.add_argument('--pullback_bars',   type=int,   default=10,
                        help='突破后最多等待N根K线确认回踩')
    parser.add_argument('--pullback_zone',   type=float, default=2.0,
                        help='回踩有效区域宽度（ATR倍数）')

    # 出场参数
    parser.add_argument('--exit_mode',    default='fixed_r',
                        choices=['fixed_r', 'atr_trail', 'ema_exit'])
    parser.add_argument('--rr',           type=float, default=2.0,
                        help='fixed_r模式盈亏比（TP = entry + risk * rr）')
    parser.add_argument('--trail_mult',   type=float, default=2.0,
                        help='atr_trail模式trailing stop ATR倍数')
    parser.add_argument('--sl_atr_mult',  type=float, default=0.5,
                        help='止损位 = breakout_level ± ATR * sl_atr_mult')

    # 风控
    parser.add_argument('--risk_pct',     type=float, default=0.01,
                        help='每笔风险占账户比例，默认1%%')

    # 其他
    parser.add_argument('--allow_short',  action='store_true',
                        help='允许做空（A股默认关闭）')
    parser.add_argument('--no_plot',      action='store_true',
                        help='跳过绘图')
    parser.add_argument('--save',         default='backtest_result.png',
                        help='图表保存路径')
    parser.add_argument('--refresh',      action='store_true',
                        help='强制重新下载数据（忽略本地缓存）')

    return parser.parse_args()


def main():
    args = parse_args()

    params = StrategyParams(
        lookback          = args.lookback,
        atr_period        = args.atr_period,
        atr_mult          = args.atr_mult,
        ema_period        = args.ema_period,
        adx_period        = args.adx_period,
        adx_threshold     = args.adx_threshold,
        use_adx_filter    = not args.no_adx_filter,
        pullback_bars     = args.pullback_bars,
        pullback_zone_mult= args.pullback_zone,
        exit_mode         = args.exit_mode,
        risk_reward       = args.rr,
        atr_trail_mult    = args.trail_mult,
        sl_atr_mult       = args.sl_atr_mult,
        risk_pct          = args.risk_pct,
    )

    print(f"\n{'─'*56}")
    print(f"  股票代码   : {args.symbol}")
    print(f"  日期区间   : {args.start} ~ {args.end}")
    print(f"  初始资金   : {args.capital:,.0f}")
    print(f"  出场模式   : {args.exit_mode}  盈亏比={args.rr}")
    print(f"  ADX过滤    : {'启用 ≥' + str(args.adx_threshold) if not args.no_adx_filter else '关闭'}")
    print(f"  允许做空   : {args.allow_short}")
    print(f"{'─'*56}")

    print(f"\n[1/3] 正在加载 {args.symbol} 历史数据...")
    df = load_stock(args.symbol, args.start, args.end,
                    adjust=args.adjust, refresh=args.refresh)
    print(f"      数据加载完成，共 {len(df)} 根K线  "
          f"({df.index[0].date()} ~ {df.index[-1].date()})")

    print("[2/3] 开始回测...")
    bt = Backtester(params, allow_short=args.allow_short)
    bt.run(df, initial_capital=args.capital)
    print(f"      回测完成，共产生 {len(bt.trades)} 笔交易")

    print("[3/3] 生成报告...\n")
    print_report(bt, symbol=args.symbol)

    if not args.no_plot:
        plot_result(bt, symbol=args.symbol, save_path=args.save)


if __name__ == '__main__':
    main()
