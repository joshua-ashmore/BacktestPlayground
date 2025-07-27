# BacktestPlayground

BacktestPlayground is an interactive, Streamlit-based backtesting tool for exploring and visualizing momentum trading strategies. Built in a weekend, it allows users to define custom strategies, select tickers and benchmarks, and evaluate results in a sleek browser-based interface.

## Features

- Define your own strategy parameters (stop loss, take profit, position sizing, etc.)
- Select universe of tickers to trade
- Choose from common benchmarks (e.g., SPY, QQQ, BTC)
- Interactive equity curve visualization (with optional benchmark overlay)
- Portfolio statistics: Sharpe, max drawdown, win rate, turnover, and more
- View a trade log with entry/exit data and PnL per trade

## Tech Stack

- **Python 3.12+**
- [Streamlit](https://streamlit.io/)
- [Altair](https://altair-viz.github.io/)
- [Pydantic](https://docs.pydantic.dev/)
- Modular backend with `Orchestrator`, `Trade`, and `MarketSnapshot` models

## How to Run

```bash
# Clone the repo
git clone https://github.com/joshua-ashmore/BacktestPlayground.git
cd backtestplayground

# Install requirements
poetry install

# Launch the app
streamlit run frontend/main.py
