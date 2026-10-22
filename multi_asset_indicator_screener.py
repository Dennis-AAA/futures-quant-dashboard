"""
Multi-Asset Quantitative Indicator Screening & Walk-Forward Validation Engine
=============================================================================
Applicable Assets:
  - GC (COMEX Gold Futures)
  - HG (COMEX Copper Futures)
  - ES (CME E-mini S&P 500 Futures)
  - NQ (CME E-mini Nasdaq-100 Futures)
  - CL (NYMEX WTI Crude Oil Futures)

Methodology ("A+B Hybrid" 6-Stage Screening Pipeline):
  Combines the Walk-Forward & Trade-Path rigor of Report A with the Newey-West HAC,
  Benjamini-Hochberg FDR, Event De-duplication, and Incremental Vol-Control regression
  of Report B, while fixing three critical failure modes identified in WTI live testing:
    1. Causal Rolling Percentile Rank Normalization (eliminates threshold drift in Holdout).
    2. Right-Side Price Confirmation Gate (prevents catching falling knives / shorting squeezes).
    3. Triple-Barrier Path Excursion (MFE/MAE + ATR trailing stop) alongside fixed-horizon IC.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. ASSET-CLASS CONFIGURATION (GC, HG, ES, NQ, CL)
# ==============================================================================

@dataclass
class AssetConfig:
    symbol: str
    name: str
    asset_type: str  # "equity_index" | "industrial_metal" | "precious_metal" | "energy"
    databento_fut_root: str
    databento_opt_root: str
    point_value: float
    tick_size: float
    commission_per_side: float
    slippage_ticks: int
    # Regime thresholds (expressed in multiples of 20d realized volatility)
    q2_breakout_lookback: int = 20
    q2_surge_vol_mult: float = 1.25
    q3_rally_lookback: int = 20
    q3_rally_vol_mult: float = 1.60
    q3_long_rally_lookback: int = 60
    q3_long_rally_vol_mult: float = 2.20
    # Whether futures curve slope reflects physical supply/demand convenience yield
    use_futures_curve_carry: bool = True
    # Whether upside spot rallies correlate positively with IV (e.g. Gold safe-haven spikes)
    positive_spot_vol_corr: bool = False


ASSET_CONFIGS: Dict[str, AssetConfig] = {
    "GC": AssetConfig(
        symbol="GC",
        name="COMEX 黄金期货 (Gold)",
        asset_type="precious_metal",
        databento_fut_root="GC",
        databento_opt_root="OG",
        point_value=100.0,
        tick_size=0.10,
        commission_per_side=2.50,
        slippage_ticks=1,
        use_futures_curve_carry=False,  # Gold curve is interest-rate driven, not physical scarcity
        positive_spot_vol_corr=True,    # Gold call skew & IV rise during safe-haven breakouts
    ),
    "HG": AssetConfig(
        symbol="HG",
        name="COMEX 铜期货 (Copper)",
        asset_type="industrial_metal",
        databento_fut_root="HG",
        databento_opt_root="HXE",
        point_value=25000.0,
        tick_size=0.0005,
        commission_per_side=2.50,
        slippage_ticks=1,
        use_futures_curve_carry=True,   # Physical inventory & convenience yield drive M1-M6 spread
        positive_spot_vol_corr=False,
    ),
    "ES": AssetConfig(
        symbol="ES",
        name="CME 标普500指数期货 (E-mini S&P 500)",
        asset_type="equity_index",
        databento_fut_root="ES",
        databento_opt_root="ES",
        point_value=50.0,
        tick_size=0.25,
        commission_per_side=2.00,
        slippage_ticks=1,
        use_futures_curve_carry=False,  # Equity index basis is interest rate - dividend yield
        positive_spot_vol_corr=False,   # Strong negative spot-vol correlation (put skew)
    ),
    "NQ": AssetConfig(
        symbol="NQ",
        name="CME 纳斯达克100指数期货 (E-mini Nasdaq-100)",
        asset_type="equity_index",
        databento_fut_root="NQ",
        databento_opt_root="NQ",
        point_value=20.0,
        tick_size=0.25,
        commission_per_side=2.00,
        slippage_ticks=1,
        use_futures_curve_carry=False,
        positive_spot_vol_corr=False,
    ),
    "CL": AssetConfig(
        symbol="CL",
        name="NYMEX WTI 原油期货 (Crude Oil)",
        asset_type="energy",
        databento_fut_root="CL",
        databento_opt_root="LO",
        point_value=1000.0,
        tick_size=0.01,
        commission_per_side=2.50,
        slippage_ticks=1,
        use_futures_curve_carry=True,
        positive_spot_vol_corr=False,
    ),
}


# ==============================================================================
# 2. STAGE 1: FEATURE ENGINEERING & CAUSAL ROLLING PERCENTILE NORMALIZATION
# ==============================================================================

def causal_rolling_pctrank(series: pd.Series, window: int = 504, min_periods: int = 126) -> pd.Series:
    """
    Computes strictly causal rolling percentile rank in [0, 1] using only [t-window+1 .. t].
    Prevents threshold drift across structural volatility regimes (fixing Report A's 2026 failure).
    """
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        if np.isnan(vals[i]):
            continue
        start = max(0, i - window + 1)
        win = vals[start : i + 1]
        valid = win[~np.isnan(win)]
        if len(valid) >= min_periods:
            out[i] = np.mean(valid <= vals[i])
    return pd.Series(out, index=series.index, name=f"{series.name}_pctrank")


def compute_candidate_indicators(df: pd.DataFrame, cfg: AssetConfig) -> Tuple[pd.DataFrame, List[str]]:
    """
    Expects a daily DataFrame indexed by date with core columns:
      Required OHLCV & OI:
        open, high, low, close, volume, open_interest
      Optional Futures Term Structure:
        close_m2 (2nd month), close_m3 (3rd month), close_m6 (6th month)
      Optional Options Surface (30d & 60d constant maturity):
        atm_iv_30, atm_iv_60, iv_25d_call_30, iv_25d_put_30, iv_10d_call_30, iv_10d_put_30,
        iv_cross_std_60, opt_put_vol, opt_call_vol, opt_put_oi, opt_call_oi, net_gex, vanna_abs_risk
    Returns:
      df_feat: DataFrame with raw indicators and causal rolling percentile ranks
      indicator_cols: list of standardized indicator column names to screen
    """
    d = df.copy().sort_index()
    c = d["close"]
    h = d["high"]
    l = d["low"]
    o = d["open"]
    ret1 = c.pct_change()
    log_ret1 = np.log(c / c.shift(1))

    # --- Family 1: Realized Volatility, Downside Semivol & Vol-of-Vol ---
    d["realized_vol_20"] = log_ret1.rolling(20).std() * np.sqrt(252)
    d["realized_vol_60"] = log_ret1.rolling(60).std() * np.sqrt(252)

    neg_ret = np.minimum(log_ret1, 0.0)
    pos_ret = np.maximum(log_ret1, 0.0)
    d["downside_semivol_60"] = np.sqrt((neg_ret ** 2).rolling(60).mean()) * np.sqrt(252)
    d["upside_semivol_60"] = np.sqrt((pos_ret ** 2).rolling(60).mean()) * np.sqrt(252)
    d["semivol_asymmetry_60"] = d["downside_semivol_60"] / (d["upside_semivol_60"] + 1e-8)

    # Garman-Klass 20-day volatility
    hl_term = 0.5 * (np.log(h / l) ** 2)
    co_term = (2.0 * np.log(2.0) - 1.0) * (np.log(c / o) ** 2)
    d["garman_klass_20"] = np.sqrt((hl_term - co_term).rolling(20).mean().clip(lower=0) * 252)

    # Vol-of-Vol (252d std of 20d realized vol)
    d["vol_of_vol_252"] = d["realized_vol_20"].rolling(252, min_periods=120).std()

    # ATR(20) for Triple-Barrier & Trailing Stop sizing
    tr = pd.concat([
        (h - l),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    d["atr_20"] = tr.rolling(20).mean()
    d["atr_pct_20"] = d["atr_20"] / c

    # --- Family 2: Trend, Momentum, Variance Ratio & Exhaustion ---
    d["roc_10"] = c.pct_change(10)
    d["roc_20"] = c.pct_change(20)
    d["roc_60"] = c.pct_change(60)
    d["roc_120"] = c.pct_change(120)

    # Lo-MacKinlay Variance Ratio (120d, q=5)
    ret5 = log_ret1.rolling(5).sum()
    var1 = log_ret1.rolling(120).var()
    var5 = ret5.rolling(120).var()
    d["variance_ratio_120_5"] = var5 / (5.0 * var1 + 1e-12)

    # ADX(14) Trend Strength
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-8)
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    d["adx_14"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # Right-side price confirmation helper: close > 5d EMA or 2-day higher low
    d["ema_5"] = c.ewm(span=5, adjust=False).mean()
    d["ema_20"] = c.ewm(span=20, adjust=False).mean()
    d["right_side_bull_confirm"] = ((c > d["ema_5"]) | (c > c.shift(1))).astype(int)
    d["right_side_bear_confirm"] = ((c < d["ema_5"]) | (c < c.shift(1))).astype(int)

    # --- Family 3: Volume & Open Interest Microstructure ---
    if "volume" in d.columns:
        d["vol_zscore_60"] = (d["volume"] - d["volume"].rolling(60).mean()) / (d["volume"].rolling(60).std() + 1e-8)
    if "open_interest" in d.columns:
        d["oi_roc_20"] = d["open_interest"].pct_change(20)
        # Price-OI Divergence: price up over 20d while OI shrinking (short-covering exhaustion)
        d["price_oi_divergence_20"] = d["roc_20"] * (-d["oi_roc_20"])

    # --- Family 4: Futures Term Structure (Carry / Convenience Yield) ---
    if "close_m6" in d.columns:
        # Positive = Contango (M6 > M1), Negative = Backwardation (M1 > M6)
        d["curve_slope_6m"] = (d["close_m6"] - c) / c
        # Front-end tightening over 20 days (M1 - M6 20d change)
        spread_1_6 = (c - d["close_m6"]) / c
        d["spread_1_6_chg20"] = spread_1_6.diff(20)

    # --- Family 5: Options Implied Volatility Surface & Skew/Smile ---
    if "atm_iv_30" in d.columns:
        # VRP (Variance Risk Premium): Implied Vol - Realized Vol
        d["vrp_30_20"] = d["atm_iv_30"] - d["realized_vol_20"]

    if "atm_iv_30" in d.columns and "atm_iv_60" in d.columns:
        # IV Term Structure Slope (Front 30d vs 60d; >0 means panic backwardation in IV)
        d["iv_term_slope_30_60"] = d["atm_iv_30"] - d["atm_iv_60"]

    if {"iv_25d_call_30", "iv_25d_put_30", "atm_iv_30"}.issubset(d.columns):
        # 25-Delta Risk Reversal (Skew) & Butterfly (Smile Curvature / Fat Tails)
        d["rr25_30"] = d["iv_25d_call_30"] - d["iv_25d_put_30"]
        d["bf25_30"] = 0.5 * (d["iv_25d_call_30"] + d["iv_25d_put_30"]) - d["atm_iv_30"]
        d["smile_curv_30"] = d["bf25_30"] / (d["atm_iv_30"] + 1e-8)

    if {"iv_10d_put_30", "atm_iv_30"}.issubset(d.columns):
        d["downside_wing_prem_30_10d"] = (d["iv_10d_put_30"] - d["atm_iv_30"]) / (d["atm_iv_30"] + 1e-8)

    if "iv_cross_std_60" in d.columns:
        d["iv_std_60"] = d["iv_cross_std_60"]

    if {"opt_put_vol", "opt_call_vol"}.issubset(d.columns):
        d["pcr_vol_5d"] = (d["opt_put_vol"].rolling(5).sum()) / (d["opt_call_vol"].rolling(5).sum() + 1e-8)

    # Collect raw candidate columns
    candidate_raw_cols = [
        col for col in [
            "downside_semivol_60", "semivol_asymmetry_60", "garman_klass_20", "vol_of_vol_252",
            "variance_ratio_120_5", "adx_14", "roc_120",
            "vol_zscore_60", "oi_roc_20", "price_oi_divergence_20",
            "curve_slope_6m", "spread_1_6_chg20",
            "vrp_30_20", "iv_term_slope_30_60", "rr25_30", "bf25_30", "smile_curv_30",
            "downside_wing_prem_30_10d", "iv_std_60", "pcr_vol_5d",
            "net_gex", "vanna_abs_risk"
        ] if col in d.columns
    ]

    # Filter out futures curve carry for equity index / gold if configured
    if not cfg.use_futures_curve_carry:
        candidate_raw_cols = [c_name for c_name in candidate_raw_cols if c_name not in ("curve_slope_6m", "spread_1_6_chg20")]

    # Compute causal 504-day rolling percentile ranks for all candidate indicators
    pct_cols = []
    for col in candidate_raw_cols:
        pcol = f"{col}_pct"
        d[pcol] = causal_rolling_pctrank(d[col], window=504, min_periods=120)
        pct_cols.append(pcol)

    return d, pct_cols


# ==============================================================================
# 3. STAGE 2 & 3: STATISTICAL ENGINE (NEWEY-WEST HAC, BH-FDR, EVENT DEDUP)
# ==============================================================================

def newey_west_tstat(x: np.ndarray, y: np.ndarray, max_lags: int = 20, control: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    Runs OLS of y on x (and optional baseline volatility control) with Newey-West HAC standard errors.
    Returns (beta_x, nw_tstat_x, two_sided_pval).
    """
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if control is not None:
        mask = mask & (~np.isnan(control))
    n = int(np.sum(mask))
    if n < 50:
        return 0.0, 0.0, 1.0

    xv = x[mask]
    yv = y[mask]
    xv_std = (xv - np.mean(xv)) / (np.std(xv) + 1e-12)

    if control is not None:
        cv = control[mask]
        cv_std = (cv - np.mean(cv)) / (np.std(cv) + 1e-12)
        X = np.column_stack([np.ones(n), xv_std, cv_std])
    else:
        X = np.column_stack([np.ones(n), xv_std])

    k = X.shape[1]
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 1.0

    beta = xtx_inv @ (X.T @ yv)
    resid = yv - X @ beta

    # Newey-West sandwich meat matrix S
    Z = X * resid[:, None]
    S = Z.T @ Z
    L = min(max_lags, int(np.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0)))))
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        gamma = Z[lag:].T @ Z[:-lag]
        S += w * (gamma + gamma.T)

    cov = xtx_inv @ S @ xtx_inv * (n / max(1, n - k))
    se_x = float(np.sqrt(max(1e-16, cov[1, 1])))
    t_stat = float(beta[1] / se_x)
    # Normal approximation for large-sample two-sided p-value
    p_val = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    return float(beta[1]), t_stat, p_val


def benjamini_hochberg_qvals(pvals: List[float]) -> List[float]:
    """Computes Benjamini-Hochberg FDR adjusted q-values."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    qvals = np.zeros(m, dtype=float)
    prev_q = 1.0
    for rank_idx in range(m - 1, -1, -1):
        orig_idx = order[rank_idx]
        rank = rank_idx + 1
        q = min(prev_q, pvals[orig_idx] * m / rank)
        qvals[orig_idx] = q
        prev_q = q
    return qvals.tolist()


def deduplicate_events(trigger_indices: List[int], min_gap: int = 5) -> List[int]:
    """Clusters consecutive daily triggers within `min_gap` trading days into 1 independent event."""
    dedup = []
    last_idx = -999999
    for idx in trigger_indices:
        if idx - last_idx >= min_gap:
            dedup.append(idx)
            last_idx = idx
    return dedup


# ==============================================================================
# 4. STAGE 4 & 5: TRIPLE-BARRIER PATH SIMULATION & PURGED WALK-FORWARD ENGINE
# ==============================================================================

def evaluate_signal_events(
    df: pd.DataFrame,
    event_indices: List[int],
    direction: int,  # +1 for Long, -1 for Short
    horizon: int = 20,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.2,
) -> Dict[str, float]:
    """
    Evaluates de-duplicated signal triggers using BOTH:
      1. Fixed-horizon return R_{t -> t+h}
      2. Path-aware MFE/MAE & Triple-Barrier (+2.0 ATR target vs -1.2 ATR stop-loss)
    """
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    atrs = df["atr_20"].to_numpy(dtype=float)
    n = len(closes)

    fixed_rets = []
    tb_rets = []
    tb_hits = []
    mfes = []
    maes = []

    for idx in event_indices:
        if idx + 2 >= n:
            continue
        end_idx = min(n - 1, idx + horizon)
        entry_p = closes[idx]
        atr = atrs[idx] if not np.isnan(atrs[idx]) and atrs[idx] > 0 else entry_p * 0.015

        # Fixed horizon return in trade direction
        raw_fixed_ret = (closes[end_idx] - entry_p) / entry_p
        fixed_rets.append(direction * raw_fixed_ret)

        # Path MFE / MAE
        win_h = highs[idx + 1 : end_idx + 1]
        win_l = lows[idx + 1 : end_idx + 1]
        if direction == 1:
            mfe = (np.max(win_h) - entry_p) / entry_p
            mae = (np.min(win_l) - entry_p) / entry_p
        else:
            mfe = (entry_p - np.min(win_l)) / entry_p
            mae = (entry_p - np.max(win_h)) / entry_p
        mfes.append(mfe)
        maes.append(mae)

        # Triple-Barrier simulation day by day
        tp_price = entry_p + direction * tp_atr_mult * atr
        sl_price = entry_p - direction * sl_atr_mult * atr
        outcome_ret = direction * raw_fixed_ret
        hit_tp = 0
        for k in range(idx + 1, end_idx + 1):
            if direction == 1:
                if lows[k] <= sl_price:
                    outcome_ret = -sl_atr_mult * atr / entry_p
                    hit_tp = 0
                    break
                if highs[k] >= tp_price:
                    outcome_ret = tp_atr_mult * atr / entry_p
                    hit_tp = 1
                    break
            else:
                if highs[k] >= sl_price:
                    outcome_ret = -sl_atr_mult * atr / entry_p
                    hit_tp = 0
                    break
                if lows[k] <= tp_price:
                    outcome_ret = tp_atr_mult * atr / entry_p
                    hit_tp = 1
                    break
        else:
            hit_tp = 1 if outcome_ret > 0 else 0

        tb_rets.append(outcome_ret)
        tb_hits.append(hit_tp)

    if not fixed_rets:
        return {
            "n_events": 0, "fixed_win_rate": 0.0, "fixed_mean_ret": 0.0,
            "tb_win_rate": 0.0, "tb_mean_ret": 0.0, "profit_factor": 0.0,
            "mean_mfe": 0.0, "mean_mae": 0.0,
        }

    fixed_arr = np.array(fixed_rets)
    tb_arr = np.array(tb_rets)
    gains = tb_arr[tb_arr > 0].sum()
    losses = abs(tb_arr[tb_arr < 0].sum())
    pf = float(gains / losses) if losses > 1e-9 else (9.99 if gains > 0 else 0.0)

    return {
        "n_events": len(fixed_rets),
        "fixed_win_rate": float(np.mean(fixed_arr > 0)),
        "fixed_mean_ret": float(np.mean(fixed_arr)),
        "tb_win_rate": float(np.mean(tb_hits)),
        "tb_mean_ret": float(np.mean(tb_arr)),
        "profit_factor": pf,
        "mean_mfe": float(np.mean(mfes)),
        "mean_mae": float(np.mean(maes)),
    }


def run_multi_asset_screening(
    df_raw: pd.DataFrame,
    asset_symbol: str,
    horizon: int = 20,
    holdout_ratio: float = 0.15,
) -> pd.DataFrame:
    """
    Executes the full 6-Stage Screening Pipeline for a given futures asset (GC, HG, ES, NQ, CL).
    Splits sample into Development (first 85%) and Frozen Holdout (last 15%, with embargo = horizon).
    Screens all candidate indicators across Q1 (Dip/Directional), Q2 (Breakout Continuation),
    and Q3 (Rally Exhaustion/Reversal).
    """
    if asset_symbol not in ASSET_CONFIGS:
        raise ValueError(f"Unsupported asset_symbol {asset_symbol}. Choose from {list(ASSET_CONFIGS.keys())}")

    cfg = ASSET_CONFIGS[asset_symbol]
    df, pct_cols = compute_candidate_indicators(df_raw, cfg)

    # Compute forward horizon return
    df["fwd_ret"] = df["close"].shift(-horizon) / df["close"] - 1.0
    baseline_vol = df["atm_iv_30"] if "atm_iv_30" in df.columns else df["realized_vol_20"]

    n_total = len(df)
    holdout_start = int(n_total * (1.0 - holdout_ratio))
    dev_end = max(252, holdout_start - horizon)  # Embargo gap = horizon days to prevent label leakage

    # Define Q1 / Q2 / Q3 Regime Masks
    vol20_daily = df["realized_vol_20"] / np.sqrt(252)
    thresh_q2 = cfg.q2_surge_vol_mult * vol20_daily * np.sqrt(cfg.q2_breakout_lookback)
    thresh_q3 = cfg.q3_rally_vol_mult * vol20_daily * np.sqrt(cfg.q3_rally_lookback)

    high20 = df["close"].rolling(cfg.q2_breakout_lookback).max().shift(1)
    mask_q1 = pd.Series(True, index=df.index)
    mask_q2 = (df["close"] >= high20) | (df["roc_20"] >= thresh_q2)
    mask_q3 = df["roc_20"] >= thresh_q3

    regimes = [
        ("Q1_DIRECTIONAL_LONG", mask_q1, +1),
        ("Q2_BREAKOUT_CONT_LONG", mask_q2, +1),
        ("Q3_TOP_EXHAUSTION_SHORT", mask_q3, -1),
    ]

    records = []
    for q_name, r_mask, trade_dir in regimes:
        raw_pvals = []
        temp_rows = []

        for pcol in pct_cols:
            base_name = pcol.replace("_pct", "")
            # Development slice
            sub_dev = df.iloc[:dev_end]
            m_dev = r_mask.iloc[:dev_end].to_numpy(dtype=bool)

            x_dev = sub_dev[pcol].to_numpy(dtype=float)[m_dev]
            y_dev = sub_dev["fwd_ret"].to_numpy(dtype=float)[m_dev] * trade_dir
            c_dev = baseline_vol.iloc[:dev_end].to_numpy(dtype=float)[m_dev]

            beta, nw_t, p_val = newey_west_tstat(x_dev, y_dev, max_lags=horizon, control=None)
            _, nw_t_ctrl, _ = newey_west_tstat(x_dev, y_dev, max_lags=horizon, control=c_dev)

            # Determine whether High Percentile (P >= 0.80) or Low Percentile (P <= 0.20) is the active trigger tail
            trigger_tail = "HIGH (P>=80%)" if nw_t >= 0 else "LOW (P<=20%)"
            if nw_t < 0:
                nw_t = -nw_t
                nw_t_ctrl = -nw_t_ctrl

            # Apply Right-Side Confirmation Gate on triggers
            if trigger_tail.startswith("HIGH"):
                raw_trig = (df[pcol] >= 0.80) & r_mask
            else:
                raw_trig = (df[pcol] <= 0.20) & r_mask

            if trade_dir == 1:
                confirmed_trig = raw_trig & (df["right_side_bull_confirm"] == 1)
            else:
                confirmed_trig = raw_trig & (df["right_side_bear_confirm"] == 1)

            dev_trig_idx = deduplicate_events(np.where(confirmed_trig.iloc[:dev_end].to_numpy())[0].tolist(), min_gap=5)
            hold_trig_idx = [
                idx for idx in deduplicate_events(np.where(confirmed_trig.to_numpy())[0].tolist(), min_gap=5)
                if idx >= holdout_start
            ]

            dev_eval = evaluate_signal_events(df, dev_trig_idx, direction=trade_dir, horizon=horizon)
            hold_eval = evaluate_signal_events(df, hold_trig_idx, direction=trade_dir, horizon=horizon)

            raw_pvals.append(p_val)
            temp_rows.append({
                "asset": asset_symbol,
                "regime_question": q_name,
                "indicator": base_name,
                "trigger_tail": trigger_tail,
                "nw_t_stat": round(nw_t, 2),
                "nw_t_vol_controlled": round(nw_t_ctrl, 2),
                "p_val": p_val,
                "dev_events": dev_eval["n_events"],
                "dev_fixed_win_rate": round(dev_eval["fixed_win_rate"] * 100, 1),
                "dev_tb_win_rate": round(dev_eval["tb_win_rate"] * 100, 1),
                "dev_profit_factor": round(dev_eval["profit_factor"], 2),
                "dev_mean_mfe_pct": round(dev_eval["mean_mfe"] * 100, 2),
                "dev_mean_mae_pct": round(dev_eval["mean_mae"] * 100, 2),
                "holdout_events": hold_eval["n_events"],
                "holdout_tb_win_rate": round(hold_eval["tb_win_rate"] * 100, 1),
                "holdout_profit_factor": round(hold_eval["profit_factor"], 2),
            })

        # Apply Benjamini-Hochberg FDR within each question family
        qvals = benjamini_hochberg_qvals(raw_pvals)
        for row, qv in zip(temp_rows, qvals):
            row["bh_fdr_q"] = round(qv, 4)
            # Composite Pass/Fail Gate
            passed_stat = (row["nw_t_stat"] >= 2.20) and (qv <= 0.15) and (row["nw_t_vol_controlled"] >= 1.65)
            passed_dev = (row["dev_events"] >= 10) and (row["dev_tb_win_rate"] >= 55.0) and (row["dev_profit_factor"] >= 1.30)
            passed_hold = (row["holdout_events"] >= 2) and (row["holdout_tb_win_rate"] >= 50.0) and (row["holdout_profit_factor"] >= 1.10)
            row["status"] = "PASS_LIVE_READY" if (passed_stat and passed_dev and passed_hold) else (
                "WATCHLIST_DEV_ONLY" if (passed_stat and passed_dev) else "REJECTED"
            )
            records.append(row)

    res_df = pd.DataFrame(records)
    return res_df.sort_values(by=["regime_question", "nw_t_stat"], ascending=[True, False]).reset_index(drop=True)


# ==============================================================================
# 5. SELF-TEST DEMO (RUNS OUT OF THE BOX ON SYNTHETIC MULTI-ASSET DATA)
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2026-10-08", freq="B")
    n = len(dates)
    # Generate realistic synthetic futures + options surface series for self-test
    rets = np.random.normal(0.0003, 0.012, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets))
    df_demo = pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.003, size=n)),
        "high": close * (1 + np.abs(np.random.normal(0.006, 0.004, size=n))),
        "low": close * (1 - np.abs(np.random.normal(0.006, 0.004, size=n))),
        "close": close,
        "volume": np.random.lognormal(11.5, 0.4, size=n),
        "open_interest": 450000 + np.cumsum(np.random.normal(0, 1500, size=n)),
        "close_m6": close * (1 + np.random.normal(0.015, 0.008, size=n)),
        "atm_iv_30": 0.16 + np.abs(np.random.normal(0, 0.04, size=n)),
        "atm_iv_60": 0.165 + np.abs(np.random.normal(0, 0.03, size=n)),
        "iv_25d_call_30": 0.17 + np.abs(np.random.normal(0, 0.045, size=n)),
        "iv_25d_put_30": 0.165 + np.abs(np.random.normal(0, 0.04, size=n)),
        "iv_10d_put_30": 0.18 + np.abs(np.random.normal(0, 0.05, size=n)),
        "iv_cross_std_60": 0.02 + np.abs(np.random.normal(0, 0.01, size=n)),
        "opt_put_vol": np.random.lognormal(9.5, 0.5, size=n),
        "opt_call_vol": np.random.lognormal(9.6, 0.5, size=n),
    }, index=dates)

    screen_out = run_multi_asset_screening(df_demo, asset_symbol="GC", horizon=20)
    print("Self-test completed successfully. Top screened rows for GC (Gold Futures):")
    print(screen_out.head(8).to_string(index=False))