"""Strategy Choice Util."""

import streamlit as st

from components.strategies.gradient_momentum import GradientMomentumStrategy
from components.strategies.mean_reversion import PairsTradingStrategyJob
from components.strategies.momentum_strategy import MomentumStrategyJob
from components.strategies.volatility_breakout import VolatilityBreakoutStrategyJob


def render_momentum_params():
    col1, col2, col3 = st.columns(3)

    with col1:
        top_n = st.slider(
            "Top N Performers",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Number of top-performing stocks to include in the portfolio.",
        )
        momentum_threshold = st.slider(
            "Momentum Threshold",
            min_value=0.0,
            max_value=0.10,
            value=0.02,
            step=0.005,
            help="Minimum required momentum (e.g., return %) for a stock to qualify.",
        )

    with col2:
        base_window = st.slider(
            "Base Lookback Window",
            min_value=10,
            max_value=250,
            value=20,
            step=5,
            help="Number of days used to calculate momentum.",
        )
        ma_window = st.slider(
            "Moving Average Window",
            min_value=10,
            max_value=250,
            value=50,
            step=5,
            help="Window size for the moving average filter (used for trend confirmation).",
        )

    with col3:
        max_holding_days = st.slider(
            "Max Holding Days",
            min_value=1,
            max_value=365,
            value=60,
            step=1,
            help="Maximum number of days a position is held before forced exit.",
        )
        min_atr_window = st.slider(
            "Minimum ATR Window",
            min_value=5,
            max_value=60,
            value=14,
            step=1,
            help="ATR window for volatility filtering or position sizing.",
        )

    return {
        "top_n": top_n,
        "momentum_threshold": momentum_threshold,
        "max_holding_days": max_holding_days,
        "base_window": base_window,
        "ma_window": ma_window,
        "min_atr_window": min_atr_window,
    }


def render_volatility_params():
    col1, col2, col3 = st.columns(3)

    with col1:
        atr_window = st.slider(
            "ATR Window",
            min_value=5,
            max_value=100,
            value=14,
            step=1,
            help="Number of days to compute the Average True Range (ATR). Used as a measure of market volatility.",
        )

    with col2:
        ma_window = st.slider(
            "Moving Average Window",
            min_value=10,
            max_value=200,
            value=20,
            step=5,
            help="Window length for moving average trend filter. Filters trades to align with trend.",
        )

    with col3:
        breakout_threshold = st.slider(
            "Breakout Threshold (*ATR)",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="Multiplier on ATR to define breakout range. Price must exceed this multiple of ATR to trigger a trade.",
        )

    return {
        "atr_window": atr_window,
        "ma_window": ma_window,
        "breakout_threshold": breakout_threshold,
    }


def render_pairs_params():
    col1, col2, col3 = st.columns(3)

    with col1:
        window = st.slider(
            "Lookback Window",
            min_value=10,
            max_value=250,
            value=60,
            step=5,
            help="Number of days used to compute z-score, cointegration, and Hurst statistics for pair selection.",
        )

        z_score_threshold = st.slider(
            "Z-score Entry Threshold",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Z-score at which a trade is entered. Higher values reduce signal frequency.",
        )

    with col2:
        min_hurst = st.slider(
            "Minimum Hurst Exponent",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Minimum Hurst exponent required for mean-reverting behavior. Values below 0.5 indicate anti-persistence.",
        )

        min_coint_pval = st.slider(
            "Max Cointegration p-value",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            step=0.005,
            help="Maximum p-value allowed for the cointegration test. Lower means stricter filtering for pair selection.",
        )

    with col3:
        min_adf_pval = st.slider(
            "Max ADF p-value",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            step=0.005,
            help="Maximum p-value allowed for the Augmented Dickey-Fuller test, which checks for stationarity of spread.",
        )

    return {
        "window": window,
        "z_score_threshold": z_score_threshold,
        "min_hurst": min_hurst,
        "min_coint_pval": min_coint_pval,
        "min_adf_pval": min_adf_pval,
    }


def render_gradient_momentum_params():
    col1, col2, col3 = st.columns(3)

    with col1:
        lookback_window = st.slider(
            "Lookback Window",
            min_value=5,
            max_value=120,
            value=20,
            step=5,
            help="Number of days used to calculate price momentum (angle of price vector).",
        )

        rsi_window = st.slider(
            "RSI Window",
            min_value=5,
            max_value=50,
            value=14,
            step=1,
            help="Lookback period for RSI to identify overbought/oversold conditions.",
        )

        angle_calculation_day_offset = st.slider(
            "Angle Offset (days)",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            help="Offset in days to calculate angle vs current price. 0 means use the most recent price.",
        )

    with col2:
        up_trend_angle_threshold_deg = st.slider(
            "Uptrend Angle Threshold (°)",
            min_value=0,
            max_value=90,
            value=50,
            step=1,
            help="Minimum angle (in degrees) to confirm an uptrend. Higher values = steeper trends.",
        )

        down_trend_angle_threshold_deg = st.slider(
            "Downtrend Angle Threshold (°)",
            min_value=-90,
            max_value=0,
            value=-50,
            step=1,
            help="Maximum angle (in degrees) to confirm a downtrend. Lower values = steeper drops.",
        )

    with col3:
        top_n = st.slider(
            "Top N Performers",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Number of top-ranked assets (by angle) to include in the portfolio.",
        )

        max_holding_days = st.slider(
            "Max Holding Days",
            min_value=1,
            max_value=365,
            value=60,
            step=1,
            help="Maximum number of days a position is held before forced exit.",
        )

    return {
        "lookback_window": lookback_window,
        "rsi_window": rsi_window,
        "angle_calculation_day_offset": angle_calculation_day_offset,
        "up_trend_angle_threshold_deg": up_trend_angle_threshold_deg,
        "down_trend_angle_threshold_deg": down_trend_angle_threshold_deg,
        "top_n": top_n,
        "max_holding_days": max_holding_days,
    }


STRATEGY_REGISTRY = {
    "Momentum": {
        "class": MomentumStrategyJob,
        "ui_func": render_momentum_params,
        "help": "Selects the strongest-performing stocks over a recent period and holds them while momentum persists. "
        "Ideal for capturing trending moves.",
    },
    "Gradient Momentum": {
        "class": GradientMomentumStrategy,
        "ui_func": render_gradient_momentum_params,
        "help": "Uses the slope (angle) of price moves to detect strong trends, "
        "while filtering out overbought setups using RSI for cleaner entries.",
    },
    "Volatility Breakout": {
        "class": VolatilityBreakoutStrategyJob,
        "ui_func": render_volatility_params,
        "help": "Looks for sudden price moves beyond recent volatility levels, "
        "aiming to catch explosive trends after periods of quiet consolidation.",
    },
    "Pairs Trading": {
        "class": PairsTradingStrategyJob,
        "ui_func": render_pairs_params,
        "help": "Trades two related assets (like A/B) when their prices diverge abnormally. "
        "Bets they'll revert to their historical balance over time.",
    },
}
