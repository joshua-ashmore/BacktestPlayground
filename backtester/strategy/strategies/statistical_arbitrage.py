"""Statistical Arbitrage Strategy."""

import datetime
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Any, List, Literal, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller

from backtester.market_data.market import MarketSnapshot
from backtester.static_data import Directions, Symbol
from backtester.strategy.strategy import AbstractStrategyConfig


class StatisticalArbitrageParameter(BaseModel):
    """Statistical Arbitrage Parameter."""

    index: tuple[Symbol, Symbol]
    symbol_one: Symbol
    symbol_two: Symbol
    ols_hist_mean: float
    ols_hist_std: float
    symbol_one_vol_std: float
    symbol_two_vol_std: float
    volume_vol_ratio: float
    adf_value: float
    adf_p_value: float
    adf_crit_value: float
    beta: float
    correlation: float
    correlation_p_value: float
    meets_conditions: bool
    position_held: bool
    position_entered: datetime.datetime | None = None
    signal: Directions | None
    symbol_one_position_held: float
    symbol_two_position_held: float
    previously_traded: bool
    half_life: float
    sharpe_ratio: float
    final_zscore: float
    ou_mu: float
    ou_theta: float
    ou_sigma: float
    ecm_alpha: float
    residuals: list[float]


class StatisticalArbitrageStrategyIntent(BaseModel):
    """Trade Intent Model for Statistical Arbitrage Strategy."""

    entry_date: datetime.datetime
    signal: Directions
    e_t: float
    z_score: float
    beta: float
    lower_bound: float
    upper_bound: float
    ticker_one_price: float
    ticker_two_price: float
    parameter: StatisticalArbitrageParameter


class StatisticalArbitrageStrategy(AbstractStrategyConfig):
    """Stastical Arbitrage Strategy."""

    strategy_name: Literal["statistical arbitrage"] = "statistical arbitrage"
    parameters: list[StatisticalArbitrageParameter] = []
    position_held: bool = False

    def half_life(self, spread):
        """Half life calculation."""
        delta_spread = np.diff(spread)
        lagged_spread = spread[:-1]
        beta = np.polyfit(lagged_spread, delta_spread, 1)[0]
        hl = -np.log(2) / beta
        return hl

    def generate_ols_and_adf(self, x: List[float], y: List[float]):
        """Compute OLS, ADF, and beta."""
        x_series = pd.Series(x)
        y_series = pd.Series(y)
        beta = np.cov(x, y)[0, 1] / np.var(y)
        residuals = x_series - beta * y_series
        adf_result = adfuller(residuals, maxlag=1)
        return residuals, adf_result, beta

    def calibrate_ou(self, spread: np.ndarray) -> tuple[float, float, float]:
        """
        Calibrates the OU process from a spread time series.
        Returns:
            theta: Speed of mean reversion
            mu: Long-term mean
            sigma: Volatility
        """
        spread = np.array(spread)
        X = spread[:-1]
        Y = spread[1:]

        # Linear regression: Y = alpha + beta * X + error
        beta, alpha = np.polyfit(X, Y, 1)
        dt = 1

        theta = -np.log(beta) / dt if beta < 1 else 0  # Mean-reversion speed
        mu = alpha / (1 - beta) if beta != 1 else 0  # Long-term mean
        residuals = Y - (alpha + beta * X)
        sigma_hat = (
            np.std(residuals) * np.sqrt(2 * theta / (1 - beta**2)) if theta > 0 else 0
        )

        return theta, mu, sigma_hat

    def estimate_ecm(self, x: np.ndarray, y: np.ndarray, resid: np.ndarray) -> float:
        """
        Estimates the ECM adjustment speed (alpha).
        ECM: Δy_t = α * resid_(t-1) + β * Δx_t + ε_t
        Returns:
            alpha: Speed of error correction (should be negative and significant)
        """
        import pandas as pd
        import statsmodels.api as sm

        df = pd.DataFrame({"y": y, "x": x, "resid": resid})

        df["delta_y"] = df["y"].diff()
        df["delta_x"] = df["x"].diff()
        df["lag_resid"] = df["resid"].shift(1)

        df = df.dropna()

        X_ecm = sm.add_constant(df[["lag_resid", "delta_x"]])
        model = sm.OLS(df["delta_y"], X_ecm).fit()

        alpha = model.params["lag_resid"] if "lag_resid" in model.params else 0
        return alpha

    def evaluate_pair(
        self,
        pair: Tuple[str, str],
        close_data: dict[str, list[float]],
        volume_data: dict[str, list[float]],
        min_final_zscore: float = 0,
    ):
        s1, s2 = pair
        x_close = close_data.get(s1)
        y_close = close_data.get(s2)
        x_volume = volume_data.get(s1)
        y_volume = volume_data.get(s2)

        if not x_close or not y_close or not x_volume or not y_volume:
            return None
        if len(x_close) != len(y_close):
            return None

        x_close = np.array(x_close)
        y_close = np.array(y_close)
        x_volume = np.array(x_volume)
        y_volume = np.array(y_volume)

        # Filter by volume volatility
        x_vol_std = np.std(x_volume)
        y_vol_std = np.std(y_volume)
        if x_vol_std < 10_000 or y_vol_std < 10_000:
            return None
        vol_ratio = max(x_vol_std / y_vol_std, y_vol_std / x_vol_std)
        if vol_ratio > 2:
            return None

        # Correlation check
        correlation, corr_p = pearsonr(x_close, y_close)
        if correlation < 0.6:
            return None

        # Run OLS + ADF in both directions
        res1, adf1, beta1 = self.generate_ols_and_adf(x_close, y_close)
        res2, adf2, beta2 = self.generate_ols_and_adf(y_close, x_close)

        best_first = adf1[0] < adf2[0]
        adf, beta, resid = (adf1, beta1, res1) if best_first else (adf2, beta2, res2)
        x_series, y_series = (x_close, y_close) if best_first else (y_close, x_close)

        # Spread stats
        mean_spread = np.mean(resid)
        std_spread = np.std(resid)
        final_zscore = (
            (resid.iloc[-1] - mean_spread) / std_spread if std_spread > 0 else 0
        )
        if final_zscore < min_final_zscore:
            return None

        # Half-life
        half_life = self.half_life(resid)
        if half_life < 1 / 24 or half_life > 30:
            return None

        # Sharpe Ratio
        daily_spread_returns = np.diff(resid) / resid[:-1]
        if np.std(daily_spread_returns) == 0:
            return None
        sharpe_ratio = (
            np.mean(daily_spread_returns) / np.std(daily_spread_returns) * np.sqrt(252)
        )
        if sharpe_ratio < 0.5:
            return None

        # OU calibration
        ou_theta, ou_mu, ou_sigma = self.calibrate_ou(np.array(resid))
        if ou_theta < 0.01 or ou_sigma / ou_theta > 5:
            return None

        # ECM estimation (speed of error correction)
        ecm_alpha = self.estimate_ecm(x_series, y_series, resid)
        if ecm_alpha > -0.01:  # weak or no correction
            return None

        # Final statistical filters
        if not (1 < beta < 2):
            return None
        if adf[0] > adf[4]["1%"] or adf[1] > 0.01:
            return None
        if correlation < 0.8 or corr_p > 0.01:
            return None

        symbol_one, symbol_two = (s1, s2) if best_first else (s2, s1)
        return {
            "index": (symbol_one, symbol_two),
            "symbol_one": symbol_one,
            "symbol_two": symbol_two,
            "ols_hist_mean": mean_spread,
            "ols_hist_std": std_spread,
            "symbol_one_vol_std": x_vol_std,
            "symbol_two_vol_std": y_vol_std,
            "volume_vol_ratio": vol_ratio,
            "adf_value": round(adf[0], 2),
            "adf_p_value": round(adf[1], 5),
            "adf_crit_value": round(adf[4]["1%"], 2),
            "beta": round(beta, 2),
            "correlation": round(correlation, 2),
            "correlation_p_value": round(corr_p, 5),
            "meets_conditions": True,
            "position_held": False,
            "signal": None,
            "symbol_one_position_held": 0,
            "symbol_two_position_held": 0,
            "previously_traded": False,
            "half_life": half_life,
            "sharpe_ratio": sharpe_ratio,
            "final_zscore": final_zscore,
            "ou_mu": ou_mu,
            "ou_theta": ou_theta,
            "ou_sigma": ou_sigma,
            "ecm_alpha": ecm_alpha,
            "residuals": resid,
        }

    def generate_parameters_optimised(
        self,
        market_snapshot: MarketSnapshot,
        current_date: datetime.datetime,
        window: int,
    ):
        """Generate parameters optimised."""
        min_date = current_date - datetime.timedelta(days=window)
        max_date = current_date - datetime.timedelta(days=1)

        symbols = [
            instrument.symbol
            for instrument in self.instruments
            if instrument.symbol in list(market_snapshot.data.keys())
        ]

        # Precache close and volume data
        close_data = {
            symbol: market_snapshot.get(
                symbol=symbol, variable="close", min_date=min_date, max_date=max_date
            )
            for symbol in symbols
        }
        volume_data = {
            symbol: market_snapshot.get(
                symbol=symbol, variable="volume", min_date=min_date, max_date=max_date
            )
            for symbol in symbols
        }

        pairs = list(combinations(symbols, 2))

        parameters: list[StatisticalArbitrageParameter] = []
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(
                    self.evaluate_pair,
                    pair,
                    close_data,
                    volume_data,
                )
                for pair in pairs
            ]
            for f in futures:
                result = f.result()
                if result:
                    parameters.append(StatisticalArbitrageParameter(**result))

        tickers_held = [
            param.index for param in self.parameters if param.position_held is True
        ]

        # If we hold open positions, preserve them and discard the rest
        if tickers_held:
            self.parameters = [
                param for param in self.parameters if param.index in tickers_held
            ]
            self.parameters.extend(parameters)
        else:
            # If no open positions, just use the new parameters
            self.parameters = parameters

        return self.parameters

    async def logic(
        self,
        parameter: StatisticalArbitrageParameter,
        market_snapshot: MarketSnapshot,
        min_date: datetime.datetime,
        max_date: datetime.datetime,
        signal_date: datetime.datetime,
        window: int,
    ) -> dict[str, Any] | None:
        """Check trading logic for a given pair and return signal if conditions are met."""

        # ols_hist_mean = np.mean(parameter.residuals[-5:])
        # ols_hist_std = np.std(parameter.residuals[-5:])
        # lower_bound = ols_hist_mean - std_dev_multiplier * ols_hist_std
        # upper_bound = ols_hist_mean + std_dev_multiplier * ols_hist_std

        symbols = [parameter.symbol_one, parameter.symbol_two]

        # Retrieve close price history
        close_data = {
            symbol: market_snapshot.get(
                symbol=symbol,
                variable="close",
                min_date=min_date,
                max_date=max_date,
                with_timestamps=True,
            )
            for symbol in symbols
        }

        try:
            df_one = pd.DataFrame(
                close_data[parameter.symbol_one], columns=["datetime", "price"]
            ).set_index("datetime")
            df_two = pd.DataFrame(
                close_data[parameter.symbol_two], columns=["datetime", "price"]
            ).set_index("datetime")
        except Exception:
            return None

        if df_one.empty or df_two.empty or len(df_one) != len(df_two):
            return None

        # Align on timestamps
        df = pd.concat([df_one, df_two], axis=1, join="inner")
        df.columns = [parameter.symbol_one, parameter.symbol_two]

        # Get most recent prices
        last_price_one = df[parameter.symbol_one].iloc[-1]
        last_price_two = df[parameter.symbol_two].iloc[-1]

        try:
            price_one_series = df[parameter.symbol_one].iloc[-window:]
            price_two_series = df[parameter.symbol_two].iloc[-window:]

            beta = OLS(price_one_series, price_two_series).fit().params[0]
        except Exception:
            return None

        # Calculate spread
        # e_t = last_price_one - beta * last_price_two

        ols_hist_mean = np.mean(parameter.residuals[-window:])
        ols_hist_std = np.std(parameter.residuals[-window:])

        lower_bound = -1.0
        upper_bound = 1.0

        e_t = last_price_one - beta * last_price_two
        z_score = (e_t - ols_hist_mean) / ols_hist_std

        signal = None
        if not parameter.position_held and not parameter.previously_traded:
            if z_score < lower_bound:
                signal = "open-buy"  # Long spread: buy asset1, sell asset2
                parameter.position_held = True
            elif z_score > upper_bound:
                signal = "open-sell"  # Short spread: sell asset1, buy asset2
                parameter.position_held = True
        else:
            if abs(z_score) < 0.2 and not parameter.previously_traded:
                signal = "close"
                parameter.previously_traded = True
                # parameter.position_held = False
            else:
                signal = "no signal"

        if signal:
            return {
                "entry_date": signal_date,
                "signal": signal,
                "e_t": e_t,
                "z-score": z_score,
                "beta": beta,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "ticker_one_price": last_price_one,
                "ticker_two_price": last_price_two,
                "parameter": parameter,
            }

        return None

    # async def evaluate(
    def evaluate(
        self,
        parameter: StatisticalArbitrageParameter,
        market_snapshot: MarketSnapshot,
        signal_date: datetime.datetime,
        ols_regression_window: int = 60,
        z_score_calculation_window: int = 20,
    ) -> StatisticalArbitrageStrategyIntent | None:
        """Check trading logic for a given pair and return intent if conditions are met."""
        min_date = signal_date - datetime.timedelta(days=ols_regression_window)
        max_date = signal_date - datetime.timedelta(days=1)

        close_data = {
            symbol: market_snapshot.get(
                symbol=symbol,
                variable="close",
                min_date=min_date,
                max_date=max_date,
                with_timestamps=True,
            )
            for symbol in [parameter.symbol_one, parameter.symbol_two]
        }

        try:
            df_one = pd.DataFrame(
                close_data[parameter.symbol_one], columns=["datetime", "price"]
            ).set_index("datetime")
            df_two = pd.DataFrame(
                close_data[parameter.symbol_two], columns=["datetime", "price"]
            ).set_index("datetime")
        except Exception:
            return None

        if df_one.empty or df_two.empty or len(df_one) != len(df_two):
            return None

        # Align on timestamps
        df = pd.concat([df_one, df_two], axis=1, join="inner")
        df.columns = [parameter.symbol_one, parameter.symbol_two]

        # Get most recent prices
        last_price_one = df[parameter.symbol_one].iloc[-1]
        last_price_two = df[parameter.symbol_two].iloc[-1]

        try:
            price_one_series = df[parameter.symbol_one].iloc[-ols_regression_window:]
            price_two_series = df[parameter.symbol_two].iloc[-ols_regression_window:]

            beta = OLS(price_one_series, price_two_series).fit().params.iloc[0]
        except Exception:
            return None

        # ols_hist_mean = np.mean(parameter.residuals[-z_score_calculation_window:])
        # ols_hist_std = np.std(parameter.residuals[-z_score_calculation_window:])
        residuals = price_one_series - beta * price_two_series
        ols_hist_mean = residuals[-z_score_calculation_window:].mean()
        ols_hist_std = residuals[-z_score_calculation_window:].std()

        lower_bound = -1.0
        upper_bound = 1.0

        e_t = last_price_one - beta * last_price_two
        z_score = (e_t - ols_hist_mean) / ols_hist_std

        recent_spread = residuals[-5:]
        slope = np.polyfit(range(5), recent_spread, 1)[0]

        if not parameter.position_held and not parameter.previously_traded:
            if z_score < lower_bound and slope > 0:
                parameter.signal = Directions.BUY
                parameter.position_held = True
                parameter.position_entered = signal_date
                parameter.symbol_one_position_held = 0
                parameter.symbol_two_position_held = 0
            elif z_score > upper_bound and slope < 0:
                parameter.signal = Directions.SELL
                parameter.position_held = True
                parameter.position_entered = signal_date
                parameter.symbol_one_position_held = 0
                parameter.symbol_two_position_held = 0
        elif parameter.position_held:
            rolling_mean = residuals.rolling(window=z_score_calculation_window).mean()
            rolling_std = residuals.rolling(window=z_score_calculation_window).std()
            historical_zscores = (residuals[:-1] - rolling_mean[:-1]) / rolling_std[:-1]
            historical_zscores = historical_zscores.dropna()
            close_percentile = 10 if signal_date.weekday() < 4 else 20
            dynamic_close_threshold = np.percentile(
                np.abs(historical_zscores), close_percentile
            )

            # direction = 1 if parameter.signal == Directions.BUY else -1
            # exit_price_one = market_snapshot.get(
            #     parameter.symbol_one, variable="close", dates=signal_date
            # )[0]
            # exit_price_two = market_snapshot.get(
            #     parameter.symbol_two, variable="close", dates=signal_date
            # )[0]
            # entry_price_one = market_snapshot.get(
            #     parameter.symbol_one, variable="close", dates=parameter.position_entered
            # )[0]
            # entry_price_two = market_snapshot.get(
            #     parameter.symbol_two, variable="close", dates=parameter.position_entered
            # )[0]
            # pnl = direction * (
            #     (ticker_one_current_price - ticker_one_entry_price)
            #     - (ticker_two_current_price - ticker_two_entry_price) * parameter.beta
            # )
            # notional = 100
            # pnl = 0
            # if parameter.signal == Directions.SELL:
            #     pnl = notional * (
            #         (exit_price_two - entry_price_two) * beta
            #         - (exit_price_one - entry_price_one)
            #     )
            # elif parameter.signal == Directions.BUY:
            #     pnl = notional * (
            #         (entry_price_two - exit_price_two) * beta
            #         - (entry_price_one - exit_price_one)
            #     )

            if parameter.position_entered != signal_date and (
                abs(z_score) <= dynamic_close_threshold
                or abs(z_score) < 0.75
                or (signal_date - parameter.position_entered).days >= 15
                # or pnl < -5
            ):
                parameter.signal = Directions.CLOSE
                parameter.previously_traded = True
                parameter.position_held = False

        if parameter.signal:
            return StatisticalArbitrageStrategyIntent(
                entry_date=signal_date,
                signal=parameter.signal,
                e_t=e_t,
                z_score=z_score,
                beta=beta,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                ticker_one_price=last_price_one,
                ticker_two_price=last_price_two,
                parameter=parameter,
            )

        return None
