"""MeleeMatchup — Streamlit App

Interactive predictions for competitive Melee tournament sets.
"""

import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import shap
import streamlit as st
import xgboost as xgb

APP_DATA = Path(__file__).parent / "data" / "app"
CORE_FEATURES = [
    "elo_diff",
    "p1_expected",
    "p1_sets_played",
    "p2_sets_played",
    "recent_wr_diff",
    "p1_h2h_wr",
    "h2h_total",
    "seed_diff",
    "num_attendees",
]


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(APP_DATA / "model.json"))
    return model


@st.cache_data
def load_players():
    df = pd.read_parquet(APP_DATA / "players.parquet")
    return df[df["sets_played"] >= 5].sort_values("elo", ascending=False).reset_index(drop=True)


@st.cache_data
def load_h2h():
    return pd.read_parquet(APP_DATA / "h2h.parquet")


@st.cache_resource
def load_shap_explainer():
    model = load_model()
    explainer = shap.TreeExplainer(model)
    return explainer


FEATURE_LABELS = {
    "elo_diff": "Elo Rating Gap",
    "p1_expected": "Expected Win Rate (Elo)",
    "p1_sets_played": "P1 Experience",
    "p2_sets_played": "P2 Experience",
    "recent_wr_diff": "Recent Form Gap",
    "p1_h2h_wr": "Head-to-Head Record",
    "h2h_total": "H2H Sets Played",
    "seed_diff": "Seed Differential",
    "num_attendees": "Tournament Size",
}


def get_h2h(h2h_df, pid1, pid2):
    """Look up head-to-head record between two players."""
    pa, pb = min(pid1, pid2), max(pid1, pid2)
    row = h2h_df[(h2h_df["player_a"] == pa) & (h2h_df["player_b"] == pb)]
    if row.empty:
        return 0, 0
    r = row.iloc[0]
    if pid1 == pa:
        return int(r["a_wins"]), int(r["b_wins"])
    return int(r["b_wins"]), int(r["a_wins"])


def elo_expected(ra, rb):
    return 1.0 / (1.0 + math.pow(10, (rb - ra) / 400))


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MeleeMatchup", page_icon="🎮", layout="wide")

model = load_model()
players = load_players()
h2h_df = load_h2h()
explainer = load_shap_explainer()

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Match Predictor", "Elo Leaderboard"], index=0)

if page == "Match Predictor":
    st.title("MeleeMatchup")
    st.markdown("*Predict who wins a competitive Melee set*")

    # Player selection
    player_options = players["gamer_tag"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Player 1")
        p1_name = st.selectbox("Select player", player_options, index=0, key="p1")
    with col2:
        st.subheader("Player 2")
        default_p2 = min(1, len(player_options) - 1)
        p2_name = st.selectbox("Select player", player_options, index=default_p2, key="p2")

    # Tournament context
    attendees = st.slider("Tournament size (attendees)", 50, 2000, 300)

    if p1_name == p2_name:
        st.warning("Select two different players.")
    else:
        p1 = players[players["gamer_tag"] == p1_name].iloc[0]
        p2 = players[players["gamer_tag"] == p2_name].iloc[0]

        # Head-to-head
        p1_h2h_w, p2_h2h_w = get_h2h(h2h_df, p1["player_id"], p2["player_id"])
        h2h_total = p1_h2h_w + p2_h2h_w
        p1_h2h_wr = p1_h2h_w / h2h_total if h2h_total > 0 else 0.5

        # Build feature vector
        features = pd.DataFrame(
            [
                {
                    "elo_diff": p1["elo"] - p2["elo"],
                    "p1_expected": elo_expected(p1["elo"], p2["elo"]),
                    "p1_sets_played": p1["sets_played"],
                    "p2_sets_played": p2["sets_played"],
                    "recent_wr_diff": p1["recent_wr"] - p2["recent_wr"],
                    "p1_h2h_wr": p1_h2h_wr,
                    "h2h_total": h2h_total,
                    "seed_diff": 0,  # No seed info for ad-hoc prediction
                    "num_attendees": attendees,
                }
            ]
        )

        prob = model.predict_proba(features[CORE_FEATURES])[0]
        p1_prob = prob[1]
        p2_prob = prob[0]

        # Display prediction
        st.divider()

        # Main prediction display
        if p1_prob > p2_prob:
            fav, fav_prob, fav_elo = p1_name, p1_prob, p1["elo"]
            dog, dog_prob, dog_elo = p2_name, p2_prob, p2["elo"]
        else:
            fav, fav_prob, fav_elo = p2_name, p2_prob, p2["elo"]
            dog, dog_prob, dog_elo = p1_name, p1_prob, p1["elo"]

        col_l, col_m, col_r = st.columns([2, 1, 2])
        with col_l:
            st.metric(p1_name, f"{p1_prob:.1%}", help="Win probability")
        with col_m:
            vs_html = "<h2 style='text-align:center; padding-top:20px;'>vs</h2>"
            st.markdown(vs_html, unsafe_allow_html=True)
        with col_r:
            st.metric(p2_name, f"{p2_prob:.1%}", help="Win probability")

        # Progress bar showing matchup
        st.progress(p1_prob)

        # Detail cards
        st.divider()
        st.subheader("Matchup Breakdown")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Elo Ratings**")
            st.write(f"{p1_name}: **{p1['elo']:.0f}**")
            st.write(f"{p2_name}: **{p2['elo']:.0f}**")
            diff = abs(p1["elo"] - p2["elo"])
            st.caption(f"Difference: {diff:.0f} points")

        with c2:
            st.markdown("**Recent Form (last 30 sets)**")
            st.write(f"{p1_name}: **{p1['recent_wr']:.0%}** win rate")
            st.write(f"{p2_name}: **{p2['recent_wr']:.0%}** win rate")

        with c3:
            st.markdown("**Head-to-Head**")
            if h2h_total > 0:
                st.write(f"{p1_name}: **{p1_h2h_w}** wins")
                st.write(f"{p2_name}: **{p2_h2h_w}** wins")
                st.caption(f"{h2h_total} total sets played")
            else:
                st.write("No recorded history")

        # Overall records
        st.divider()
        st.subheader("Career Records")
        rec1, rec2 = st.columns(2)
        with rec1:
            p1_rec = f"{int(p1['wins'])}W - {int(p1['losses'])}L ({p1['win_rate']:.1%})"
            st.write(f"**{p1_name}**: {p1_rec} — {int(p1['total_sets'])} sets")
        with rec2:
            p2_rec = f"{int(p2['wins'])}W - {int(p2['losses'])}L ({p2['win_rate']:.1%})"
            st.write(f"**{p2_name}**: {p2_rec} — {int(p2['total_sets'])} sets")

        # SHAP explanation
        st.divider()
        st.subheader("Why This Prediction?")
        st.caption(
            "SHAP values show how each feature pushes the prediction "
            "toward or away from a Player 1 win."
        )

        shap_values = explainer.shap_values(features[CORE_FEATURES])
        sv = shap_values[0]  # Single prediction

        # Build a DataFrame for the horizontal bar chart
        shap_df = pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS.get(f, f) for f in CORE_FEATURES],
                "SHAP Value": sv,
            }
        )
        shap_df["abs"] = shap_df["SHAP Value"].abs()
        shap_df = shap_df.sort_values("abs", ascending=True).drop(columns="abs")
        shap_df["Color"] = shap_df["SHAP Value"].apply(
            lambda x: f"Favors {p1_name}" if x > 0 else f"Favors {p2_name}"
        )

        fig = px.bar(
            shap_df,
            x="SHAP Value",
            y="Feature",
            color="Color",
            orientation="h",
            color_discrete_map={
                f"Favors {p1_name}": "#636EFA",
                f"Favors {p2_name}": "#EF553B",
            },
        )
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            legend_title_text="",
            xaxis_title="Impact on Prediction (log-odds)",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)


elif page == "Elo Leaderboard":
    st.title("Melee Elo Leaderboard")
    st.markdown("*Custom Elo ratings computed from 550k+ tournament sets (2018–2026)*")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_sets = st.slider("Minimum sets played", 10, 200, 50)
    with col2:
        top_n = st.slider("Show top N", 25, 500, 100)

    filtered = players[players["sets_played"] >= min_sets].head(top_n).copy()
    filtered = filtered.reset_index(drop=True)
    filtered.index = filtered.index + 1  # 1-indexed rank

    display_cols = ["gamer_tag", "elo", "sets_played", "recent_wr", "wins", "losses", "win_rate"]
    display = filtered[display_cols].copy()
    display.columns = ["Player", "Elo", "Sets Played", "Recent Form", "Wins", "Losses", "Win Rate"]
    display["Elo"] = display["Elo"].apply(lambda x: f"{x:.0f}")
    display["Recent Form"] = display["Recent Form"].apply(lambda x: f"{x:.0%}")
    display["Win Rate"] = display["Win Rate"].apply(lambda x: f"{x:.1%}")
    display["Wins"] = display["Wins"].astype(int)
    display["Losses"] = display["Losses"].astype(int)

    st.dataframe(display, use_container_width=True, height=min(top_n * 35 + 38, 800))
