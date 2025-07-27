"""Streamlit File."""

from datetime import datetime

import streamlit as st

from engine.orchestrator import OrchestratorConfig
from frontend.db_utils import (
    get_summary_df,
    get_timeseries_df,
    load_latest_strategy_config,
)
from frontend.st_utils import generate_charts, generate_config, generate_layout

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Portfolio Dashboard")
st.title("Portfolio Performance Dashboard")

summary_df = get_summary_df()

if summary_df.empty:
    st.warning("No portfolio data found in the database.")
    st.stop()


# Show dropdown of runs with strategy name + date range
def strat_name(x):
    strat_name = summary_df.loc[summary_df["id"] == x, "strategy_name"].values[0]
    strat_start_date = summary_df.loc[summary_df["id"] == x, "start_date"].values[0]
    strat_end_date = summary_df.loc[summary_df["id"] == x, "end_date"].values[0]
    return f"{strat_name}: {strat_start_date} → {strat_end_date}"


selected_summary_id = st.selectbox(
    "Select Strategy Run",
    options=summary_df["id"],
    format_func=lambda x: strat_name(x),
)

# Get selected row
selected_summary = summary_df[summary_df["id"] == selected_summary_id].iloc[0]

config = OrchestratorConfig(
    **load_latest_strategy_config(selected_summary["strategy_name"])
)
generate_config(config=config)

# --- Layout ---
generate_layout(selected_summary=selected_summary)
# --- Time series plot ---
st.markdown("---")

ts_df = get_timeseries_df(selected_summary_id)
generate_charts(ts_df)

# Optional: show raw data tables if user wants
if st.checkbox("Show Raw Summary Data"):
    st.dataframe(summary_df)

# Footer
st.markdown("---")
st.caption(
    f"Strategy last updated: {selected_summary['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
)
st.caption(f"Page last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
