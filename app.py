import streamlit as st
import json
import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from src.btl_model import TennisBTLModel

# Page config
st.set_page_config(
    page_title="Tennis Betting Model",
    page_icon="🎾",
    layout="wide"
)

# Initialize model
@st.cache_resource
def load_model():
    data_path = os.path.join(parent_dir, "Data")
    model = TennisBTLModel(data_path)
    ratings_file = os.path.join(parent_dir, "player_ratings.json")
    
    if True:
        model.load_ratings("player_ratings.json")
    else:
        st.warning("Ratings file not found. Building model from scratch...")
        model.load_data('both')
        model.update_ratings('men')
        model.update_ratings('women')
        model.save_ratings(ratings_file)
    
    return model

# Load model
model = load_model()

# Title
st.title("🎾 Tennis Betting Model")
st.markdown("Elo-based model with surface and match format adjustments")

# Sidebar for inputs
with st.sidebar:
    st.header("Match Details")
    
    # Gender selection
    gender = st.radio("Gender", ["Men", "Women"])
    gender_key = 'men' if gender == "Men" else 'women'
    
    # Get player list
    if gender_key in model.ratings and model.ratings[gender_key]:
        players = sorted(model.ratings[gender_key].keys())
    else:
        st.error(f"No {gender} player ratings found. Please run train_model.py first.")
        st.stop()
    
    # Player selection
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Player 1", players, key="p1")
    with col2:
        player2 = st.selectbox("Player 2", players, key="p2", 
                              index=1 if len(players) > 1 else 0)
    
    # Match format
    best_of = st.radio("Match Format", [3, 5], 
                      format_func=lambda x: f"Best of {x} sets")
    
    # Surface selection
    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
    
    # Tournament info
    tournament = st.text_input("Tournament", placeholder="e.g., Australian Open")
    
    st.header("Market Odds")
    col1, col2 = st.columns(2)
    with col1:
        psw = st.number_input("Player 1 Odds", min_value=1.01, value=2.0, step=0.01)
    with col2:
        psl = st.number_input("Player 2 Odds", min_value=1.01, value=2.0, step=0.01)
    
    # Calculate button
    calculate = st.button("Calculate Odds", type="primary", use_container_width=True)

# Main content
if calculate and player1 and player2 and player1 != player2:
    st.header(f"{player1} vs {player2}")
    if tournament:
        st.subheader(f"{tournament} - {surface} Court - Best of {best_of}")
    else:
        st.subheader(f"{surface} Court - Best of {best_of}")
    
    # Get model predictions
    odds = model.generate_betting_odds(player1, player2, surface, best_of, gender_key)
    
    # Display ratings and probabilities
    col1, col2, col3 = st.columns(3)
    
    with col1:
        p1_rating = model.ratings[gender_key].get(player1, 1500)
        p2_rating = model.ratings[gender_key].get(player2, 1500)
        st.metric("Player 1 Rating", f"{p1_rating:.0f}")
        st.metric("Player 2 Rating", f"{p2_rating:.0f}")
        st.metric("Rating Difference", f"{abs(p1_rating - p2_rating):.0f}")
    
    with col2:
        st.metric(f"{player1} Win %", f"{odds['win_probability'][player1]*100:.1f}%")
        st.metric(f"{player2} Win %", f"{odds['win_probability'][player2]*100:.1f}%")
        
        # Surface adjustments
        p1_surf_adj = model.get_player_surface_adjustment(player1, surface, gender_key)
        p2_surf_adj = model.get_player_surface_adjustment(player2, surface, gender_key)
        if p1_surf_adj != 0 or p2_surf_adj != 0:
            st.caption(f"Surface adj: {player1} {p1_surf_adj:+.0f}, {player2} {p2_surf_adj:+.0f}")
    
    with col3:
        st.metric(f"{player1} Model Odds", f"{odds['moneyline'][player1]:.2f}")
        st.metric(f"{player2} Model Odds", f"{odds['moneyline'][player2]:.2f}")
    
    st.divider()
    
    # Edge calculation
    st.subheader("Betting Edge Analysis")
    
    # Calculate implied probabilities from market
    market_implied_p1 = 1 / psw
    market_implied_p2 = 1 / psl
    market_margin = market_implied_p1 + market_implied_p2 - 1
    
    # Remove margin proportionally
    market_true_p1 = market_implied_p1 / (market_implied_p1 + market_implied_p2)
    market_true_p2 = market_implied_p2 / (market_implied_p1 + market_implied_p2)
    
    # Calculate edges
    edge_p1 = odds['win_probability'][player1] - market_true_p1
    edge_p2 = odds['win_probability'][player2] - market_true_p2
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Market Margin: {market_margin*100:.1f}%**")
    
    with col2:
        if edge_p1 > 0.02:
            st.success(f"**{player1} Edge: +{edge_p1*100:.2f}%**")
            kelly = edge_p1 / (psw - 1)
            st.caption(f"Kelly: {kelly*100:.1f}%")
        else:
            st.error(f"**{player1} Edge: {edge_p1*100:.2f}%**")
    
    with col3:
        if edge_p2 > 0.02:
            st.success(f"**{player2} Edge: +{edge_p2*100:.2f}%**")
            kelly = edge_p2 / (psl - 1)
            st.caption(f"Kelly: {kelly*100:.1f}%")
        else:
            st.error(f"**{player2} Edge: {edge_p2*100:.2f}%**")
    
    st.divider()
    
    # Set betting predictions
    st.subheader("Set Betting Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Set Score Probabilities**")
        set_probs = odds['set_betting']
        for score, prob in sorted(set_probs.items()):
            col_a, col_b, col_c = st.columns([2, 2, 1])
            with col_a:
                st.text(f"{score}:")
            with col_b:
                st.progress(prob)
            with col_c:
                st.text(f"{prob*100:.1f}%")
    
    with col2:
        st.write("**Total Sets Market**")
        if best_of == 3:
            st.metric("Over 2.5 Sets", f"{odds['total_sets']['over_2.5']*100:.1f}%")
            st.metric("Under 2.5 Sets", f"{odds['total_sets']['under_2.5']*100:.1f}%")
        else:
            st.metric("Over 3.5 Sets", f"{odds['total_sets']['over_3.5']*100:.1f}%")
            st.metric("Under 3.5 Sets", f"{odds['total_sets']['under_3.5']*100:.1f}%")
    
    # Recommendation
    st.divider()
    st.subheader("💡 Betting Recommendation")
    
    recommendations = []
    
    # Moneyline recommendations
    if edge_p1 > 0.05:
        recommendations.append(f"✅ **Strong Bet: {player1}** at {psw:.2f} (Edge: {edge_p1*100:.1f}%)")
    elif edge_p1 > 0.03:
        recommendations.append(f"✓ **Bet: {player1}** at {psw:.2f} (Edge: {edge_p1*100:.1f}%)")
    
    if edge_p2 > 0.05:
        recommendations.append(f"✅ **Strong Bet: {player2}** at {psl:.2f} (Edge: {edge_p2*100:.1f}%)")
    elif edge_p2 > 0.03:
        recommendations.append(f"✓ **Bet: {player2}** at {psl:.2f} (Edge: {edge_p2*100:.1f}%)")
    
    if not recommendations:
        st.warning("⚠️ No significant edge detected on moneyline.")
    else:
        for rec in recommendations:
            st.write(rec)
    
    # Set betting opportunities
    st.write("\n**Set Betting Opportunities:**")
    if abs(odds['win_probability'][player1] - 0.5) < 0.15:
        if best_of == 3:
            st.info("Consider betting on 3 total sets (2-1 scoreline)")
        else:
            st.info("Consider betting on 4+ total sets in a competitive match")
    else:
        favorite = player1 if odds['win_probability'][player1] > 0.5 else player2
        if best_of == 3:
            st.info(f"Consider {favorite} to win 2-0")
        else:
            st.info(f"Consider {favorite} to win 3-0 or 3-1")

elif calculate:
    st.error("Please select two different players")

# Footer
st.divider()
with st.expander("ℹ️ Model Information"):
    st.markdown("""
    **Model Details:**
    - Elo-based rating system (start: 1500)
    - Surface-specific adjustments based on win rates
    - Match format adjustments (best of 3 vs 5)
    - Higher K-factor for Grand Slams
    - Time-weighted recent matches
    
    **Edge Calculation:**
    - Removes bookmaker margin from market odds
    - Compares true probabilities with model
    - Kelly Criterion for bet sizing
    - Recommends bets with >3% edge
    """)

# Add refresh button
if st.button("🔄 Rebuild Model", help="Re-train model with latest data"):
    st.cache_resource.clear()
    st.rerun()
