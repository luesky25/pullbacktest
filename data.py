"""
数据获取模块 —— baostock（免费，自有TCP协议，不受HTTP代理影响）

baostock 优势：
  - 自有 TCP 长连接，不走 HTTP 代理，网络稳定
  - 免费，无需付费，login() 即可用
  - 覆盖 A股日K、周K、月K、5/15/30/60分钟线
  - 数据来源：上交所 / 深交所 官方

股票代码格式：
  - 沪市（6开头）→ sh.600519
  - 深市（0/3开头）→ sz.000001 / sz.300750

复权参数 adjustflag：
  '1' = 后复权   '2' = 前复权   '3' = 不复权

缓存策略：
  首次下载保存到 data_cache/<symbol>_<adjust>.parquet
  后续直接读本地，毫秒级加载
"""
from __future__ import annotations

import time
from pathlib import Path

import baostock as bs
import pandas as pd

_CACHE_DIR = Path(__file__).parent.parent / 'data_cache'
_CACHE_DIR.mkdir(exist_ok=True)

# baostock adjustflag 映射
_ADJUST_MAP = {'qfq': '2', 'hfq': '1', '': '3', 'none': '3'}

# baostock 股票代码前缀映射（6开头=沪市，其余=深市）
def _bs_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    if symbol.startswith(('6',)):
        return f'sh.{symbol}'
    return f'sz.{symbol}'


# ─────────────────────────────────────────────────────────────
# 缓存读写
# ─────────────────────────────────────────────────────────────

def _cache_path(symbol: str, adjust: str) -> Path:
    return _CACHE_DIR / f'{symbol}_{adjust}.parquet'

def _cache_path_csv(symbol: str, adjust: str) -> Path:
    return _CACHE_DIR / f'{symbol}_{adjust}.csv'

def _load_cache(symbol: str, adjust: str) -> pd.DataFrame | None:
    p = _cache_path(symbol, adjust)
    c = _cache_path_csv(symbol, adjust)
    if p.exists():
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        return df
    if c.exists():
        df = pd.read_csv(c, index_col='date', parse_dates=True)
        return df
    return None

def _save_cache(df: pd.DataFrame, symbol: str, adjust: str) -> None:
    p = _cache_path(symbol, adjust)
    try:
        df.to_parquet(p)
        print(f"  [缓存] 已保存 → {p.name}  ({len(df)} 根K线)")
    except Exception:
        c = _cache_path_csv(symbol, adjust)
        df.to_csv(c)
        print(f"  [缓存] 已保存 → {c.name}  ({len(df)} 根K线)"
              f"\n  提示：pip install pyarrow 可改用更快的 Parquet 格式")


# ─────────────────────────────────────────────────────────────
# baostock 拉取
# ─────────────────────────────────────────────────────────────

def _fetch_bs(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """通过 baostock 拉取 A股日K线"""
    bs_sym   = _bs_symbol(symbol)
    adj_flag = _ADJUST_MAP.get(adjust, '2')
    fields   = 'date,open,high,low,close,volume,amount,turn,pctChg'

    print(f"  [baostock] 下载 {symbol} ({bs_sym})  {start} ~ {end} ...")
    t0 = time.time()

    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            bs_sym, fields,
            start_date=start, end_date=end,
            frequency='d', adjustflag=adj_flag,
        )
        if rs.error_code != '0':
            raise RuntimeError(f"baostock 查询失败: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()

    if not rows:
        raise RuntimeError(f"baostock 返回空数据，请检查股票代码 {symbol} 和日期区间")

    df = pd.DataFrame(rows, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna().sort_index()

    print(f"  完成：{len(df)} 根K线  耗时 {time.time()-t0:.1f}s")
    return df


# ─────────────────────────────────────────────────────────────
# 公共接口
# ─────────────────────────────────────────────────────────────

def load_stock(
    symbol:  str,
    start:   str,
    end:     str,
    adjust:  str  = 'qfq',
    refresh: bool = False,
) -> pd.DataFrame:
    """
    加载A股日K线，本地缓存优先。

    Args:
        symbol:  6位代码，如 '000001'（平安银行）'600519'（茅台）
        start:   '2018-01-01'
        end:     '2024-12-31'
        adjust:  'qfq' 前复权 | 'hfq' 后复权 | '' 不复权
        refresh: True → 强制重新下载

    Returns:
        DataFrame（DatetimeIndex，open/high/low/close/volume）
    """
    req_start = pd.Timestamp(start)
    req_end   = pd.Timestamp(end)

    if not refresh:
        cached = _load_cache(symbol, adjust)
        if cached is not None:
            cs, ce = cached.index[0], cached.index[-1]

            # 完全命中（允许7天误差，应对节假日/非交易日）
            slack = pd.Timedelta(days=7)
            if cs <= req_start + slack and ce >= req_end - slack:
                print(f"  [缓存命中] {symbol}  {cs.date()} ~ {ce.date()}")
                return cached.loc[start:end]

            # 部分命中：只补充缺失段
            print(f"  [缓存部分] 需要 {start}~{end}，缓存 {cs.date()}~{ce.date()}，补充下载...")
            parts = []
            if req_start < cs:
                parts.append(_fetch_bs(symbol, start, str(cs.date()), adjust))
            if req_end > ce:
                parts.append(_fetch_bs(symbol, str(ce.date()), end, adjust))
            merged = pd.concat([cached] + parts).sort_index()
            merged = merged[~merged.index.duplicated(keep='last')]
            _save_cache(merged, symbol, adjust)
            return merged.loc[start:end]

    # 无缓存 / 强制刷新
    df = _fetch_bs(symbol, start, end, adjust)
    _save_cache(df, symbol, adjust)
    return df


# ─────────────────────────────────────────────────────────────
# 批量预下载
# ─────────────────────────────────────────────────────────────

def download_batch(
    symbols: list[str],
    start:   str = '2010-01-01',
    end:     str = '2026-04-17',
    adjust:  str = 'qfq',
) -> None:
    """
    批量预下载并缓存，后续回测秒级加载。

    示例：
        from breakout_pullback.data import download_batch
        download_batch(['000001', '600519', '300750', '002415'])
    """
    total = len(symbols)
    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{total}] {sym}")
        try:
            load_stock(sym, start, end, adjust, refresh=False)
        except Exception as e:
            print(f"  ❌ 失败: {e}")
        time.sleep(0.1)   # baostock 有 TCP 长连接，间隔短即可
    print(f"\n✅ 批量完成，缓存目录：{_CACHE_DIR}")
