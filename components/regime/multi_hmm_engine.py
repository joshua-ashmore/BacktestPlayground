"""Multi-HMM Regime Engine."""

from datetime import date, timedelta
from typing import Dict, Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA

from components.job.base_model import StrategyJob
from components.regime.base_model import AbstractRegimeEngine


class MultiHMMEngine(AbstractRegimeEngine):
    """HMM Strategy Selector using separate HMMs per strategy group."""

    engine_type: Literal["multi_hmm"] = "multi_hmm"

    preferred_state_idx: Dict[str, int] = {
        "trending": 0,
        "volatile": 1,
        "mean_reverting": 2,
    }

    def detect_regime_probabilities(
        self, job: StrategyJob, previous_date: date
    ) -> Dict[str, float]:
        """Detect probabilities for each strategy based on its own HMM."""

        strategy_scores = {}
        for strategy, strategy_model in self.strategy_map.items():
            tickers = strategy_model.symbols
            if tickers == []:
                strategy_scores[strategy] = 0
                continue
            preferred_state = self.preferred_state_idx[strategy]

            # Step 1: Load market data
            data = pd.concat(
                [
                    pd.Series(
                        job.market_snapshot.get(
                            symbol=symbol,
                            variable="close",
                            min_date=previous_date + timedelta(days=-3 * 365),
                            max_date=previous_date,
                            with_timestamps=True,
                        ),
                        name=symbol,
                    )
                    for symbol in tickers
                ],
                axis=1,
            ).dropna()

            if data.shape[0] < 100:
                continue  # skip if not enough data

            # Step 2: Log returns and volatility
            returns = np.log(data / data.shift(1)).dropna()
            vol = returns.rolling(window=5).std()
            features = pd.concat([returns, vol.add_suffix("_vol")], axis=1).dropna()

            # Step 3: PCA
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(features)

            # Step 4: HMM training
            model = GaussianHMM(
                n_components=3, covariance_type="full", n_iter=1000, random_state=42
            )
            model.fit(pca_features)

            # Step 5: Inference
            posteriors = model.predict_proba(pca_features)
            next_prob = posteriors[-1] @ model.transmat_

            # Step 6: Score based on preferred state
            confidence = max(next_prob)
            score = next_prob[preferred_state] * confidence
            strategy_scores[strategy] = score

        # Normalize scores to sum to 1
        total = sum(strategy_scores.values())
        normalized_scores = {
            k: v / total if total > 0 else 0.0 for k, v in strategy_scores.items()
        }

        return normalized_scores
