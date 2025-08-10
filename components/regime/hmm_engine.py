"""Hidden Markov Model Regime Detection Engine."""

from datetime import date, timedelta
from typing import Dict, Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA

from components.job.base_model import StrategyJob
from components.regime.base_model import AbstractRegimeEngine


class HMMRegimeEngine(AbstractRegimeEngine):
    """HMM Regime Engine."""

    engine_type: Literal["hmm"] = "hmm"

    num_of_training_dates: int = 365

    def detect_regime_probabilities(
        self, job: StrategyJob, previous_date: date
    ) -> Dict[str, float]:
        """Detect regime probabilities."""

        # Step 1: Load market data into multivariate dataframe
        data = pd.concat(
            [
                pd.Series(
                    job.market_snapshot.get(
                        symbol=symbol,
                        variable="close",
                        min_date=previous_date
                        + timedelta(days=-self.num_of_training_dates),
                        max_date=previous_date,
                        with_timestamps=True,
                    ),
                    name=symbol,
                )
                for symbol in job.tickers
            ],
            axis=1,
        ).dropna()

        returns = np.log(data / data.shift(1)).dropna()

        # Step 2: Feature engineering
        rolling_vol = returns.rolling(window=5).std()
        features = pd.concat([returns, rolling_vol.add_suffix("_vol")], axis=1).dropna()

        # Step 3: Dimensionality reduction
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(features)

        # Step 4: Train Gaussian HMM
        model = GaussianHMM(
            n_components=3, covariance_type="full", n_iter=1000, random_state=42
        )
        model.startprob_ = np.array([0.33, 0.33, 0.34])
        model.transmat_ = np.array(
            [[0.85, 0.10, 0.05], [0.10, 0.80, 0.10], [0.05, 0.10, 0.85]]
        )
        model.fit(pca_features)

        # Step 5: Regime inference
        posteriors = model.predict_proba(pca_features)
        next_prob = posteriors[-1] @ model.transmat_

        # Step 6: Label mapping (based on regime statistics)
        means = model.means_.mean(axis=1)
        vols = np.sqrt([np.trace(cov) / cov.shape[0] for cov in model.covars_])

        label_map = {}
        for i in range(len(means)):
            if vols[i] > np.percentile(vols, 75):
                label_map[i] = "volatile"
            elif means[i] > np.percentile(means, 66):
                label_map[i] = "trending"
            else:
                label_map[i] = "mean_reverting"

        # Step 7: Map predicted probabilities to labels
        regime_probs = {label_map[i]: next_prob[i] for i in range(len(next_prob))}

        return regime_probs
