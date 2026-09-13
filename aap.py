import json
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Delivery Network Intelligence", layout="wide")
st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
    padding: 15px;
    border-radius: 12px;
}

h1 {
    color: #0f172a;
}

h2, h3 {
    color: #1e3a8a;
}

.stDataFrame {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
@st.cache_data
def load_data():
    hubs = pd.read_csv("artifacts/bottleneck_hubs.csv")
    corridors = pd.read_csv("artifacts/delayed_corridors.csv")
    metrics = pd.read_csv("artifacts/model_metrics.csv")
    predictions = pd.read_csv("artifacts/test_predictions.csv")
    route_modes = pd.read_csv("artifacts/route_mode_recommendations.csv")
    return hubs, corridors, metrics, predictions, route_modes

hubs, corridors, metrics, predictions, route_modes = load_data()
st.title("Delivery Network Intelligence Dashboard")

overview, network, models, decisions, memo = st.tabs(
    ["Overview", "Network", "ETA Models", "Route Decisions", "Strategy Memo"]
)
with overview:
    st.title("Delivery Network Intelligence")
    overall = metrics[metrics["segment"] == "overall"].set_index("model")

    col1, col2, col3 = st.columns(3)
    col1.metric(
    "GraphSAGE MAE",
    f"{overall.loc['graphsage', 'mae_minutes']:.1f} min"
    )

    col2.metric(
    "Within 15% Accuracy",
    f"{overall.loc['graphsage', 'within_15_percent']:.1f}%"
    )

    col3.metric(
    "MAE Improvement",
    f"{100 * (overall.loc['tabular_baseline', 'mae_minutes'] - overall.loc['graphsage', 'mae_minutes']) / overall.loc['tabular_baseline', 'mae_minutes']:.1f}%"
    )

    st.subheader("ETA Model Comparison")

    st.write("Mean Absolute Error (minutes)")
    st.bar_chart(overall["mae_minutes"])

    st.write("Predictions Within 15% of Actual")
    st.bar_chart(overall["within_15_percent"])


with network:
    st.subheader("Network Bottleneck Map")

    st.image(
        "artifacts/network_bottlenecks.svg",
        use_column_width=True
    )

    st.subheader("Bottleneck Hub Ranking")

    top_n = st.sidebar.slider(
        "Number of hubs to display",
        min_value=5,
        max_value=20,
        value=10
    )

    top_hubs = hubs.head(top_n).set_index("facility_name")

    st.bar_chart(top_hubs["bottleneck_score"])

    

    st.subheader("Top-3 Hub Upgrade Scenario")

    with open(
        "artifacts/hub_upgrade_summary.json",
        encoding="utf-8"
    ) as file:
        upgrade = json.load(file)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Projected Late-Delivery Reduction",
        f"{upgrade['late_delivery_reduction_pct']:.2f}%"
    )

    col2.metric(
        "Movement Minutes Saved",
        f"{upgrade['movement_minutes_saved']:,.0f}"
    )

    col3.metric(
        "Scenario Revenue Recovered",
        f"₹{upgrade['scenario_revenue_recovered']:,.0f}"
    )

    st.subheader("Delayed Corridors")
    st.dataframe(corridors.head(10))


with models:
    st.subheader("Model Metrics")
    st.dataframe(metrics)

    st.sidebar.header("Filters")

    route_filter = st.sidebar.multiselect(
        "Route Type",
        options=predictions["route_type"].unique(),
        default=predictions["route_type"].unique()
    )

    filtered_predictions = predictions[
        predictions["route_type"].isin(route_filter)
    ]

    st.subheader("ETA Prediction Explorer")
    st.dataframe(filtered_predictions.head(100))

    st.download_button(
        label="Download Filtered Predictions",
        data=filtered_predictions.to_csv(index=False).encode("utf-8"),
        file_name="filtered_predictions.csv",
        mime="text/csv",
    )


with decisions:
    st.subheader("FTL versus Carting Recommendations")

    st.dataframe(
        route_modes[
            [
                "source_center",
                "destination_center",
                "time_bucket",
                "recommended_route_type",
                "median_ftl_time_saving_min",
                "median_break_even_ftl_premium",
                "evidence_level",
            ]
        ].head(25)
    )
st.subheader("Operations Strategy Memo")

with st.expander("View Recommendations"):
    memo = Path(
        "artifacts/network_operations_strategy_memo.md"
    ).read_text(encoding="utf-8")

    st.markdown(memo)