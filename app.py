import streamlit as st
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import re

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from src.btl_model import TennisBTLModel

# Page config
st.set_page_config(
    page_title="Tennis Betting Model",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-high { color: #28a745; font-weight: bold; }
    .prediction-medium { color: #ffc107; font-weight: bold; }
    .prediction-low { color: #dc3545; font-weight: bold; }
    .betting-market {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.3rem 0;
        border-left: 4px solid #007bff;
    }
    .strong-bet { background-color: #d4edda; border-left-color: #28a745; }
    .value-bet { background-color: #fff3cd; border-left-color: #ffc107; }
    .no-bet { background-color: #f8d7da; border-left-color: #dc3545; }
    </style>
""", unsafe_allow_html=True)

# Initialize model and load tournament data
@st.cache_resource
def load_model_and_tournaments():
    data_path = os.path.join(os.path.dirname(__file__), "Data")
    model = TennisBTLModel(data_path)
    
    # Load existing model ratings
    if os.path.exists("player_ratings.json"):
        model.load_ratings("player_ratings.json")
    elif os.path.exists("enhanced_player_ratings.json"):
        model.load_enhanced_ratings("enhanced_player_ratings.json")
    else:
        st.error("❌ No ratings file found! Please run train_model.py first to generate player_ratings.json")
        st.stop()
    
    # Initialize enhanced attributes if they don't exist
    if not hasattr(model, 'surface_expertise'):
        model.surface_expertise = {'men': {}, 'women': {}}
    if not hasattr(model, 'recent_form'):
        model.recent_form = {'men': {}, 'women': {}}
    
    # Load tournament data
    tournaments = {'men': {}, 'women': {}}
    
    for gender in ['men', 'women']:
        folder_name = 'Mens' if gender == 'men' else 'Womens'
        folder_path = os.path.join(data_path, folder_name)
        
        all_tournaments = []
        
        for year in range(2020, 2026):  # Recent years for tournament frequency
            file_path = os.path.join(folder_path, f"{year}.xlsx")
            if os.path.exists(file_path):
                try:
                    df = pd.read_excel(file_path)
                    
                    # Find tournament column
                    tournament_col = None
                    for col in df.columns:
                        if 'tournament' in col.lower():
                            tournament_col = col
                            break
                    
                    if tournament_col:
                        # Get surface and series/tier info
                        surface_col = None
                        series_col = None
                        for col in df.columns:
                            if 'surface' in col.lower():
                                surface_col = col
                            elif 'series' in col.lower() or 'tier' in col.lower():
                                series_col = col
                        
                        for _, row in df.iterrows():
                            tournament = row.get(tournament_col, '')
                            surface = row.get(surface_col, 'Hard') if surface_col else 'Hard'
                            series = row.get(series_col, 'ATP 250' if gender == 'men' else 'WTA 250') if series_col else ('ATP 250' if gender == 'men' else 'WTA 250')
                            
                            if pd.notna(tournament) and tournament.strip():
                                tournament = tournament.strip()
                                all_tournaments.append({
                                    'name': tournament,
                                    'surface': surface,
                                    'series': series
                                })
                except Exception as e:
                    continue
        
        # Create tournament frequency mapping
        if all_tournaments:
            # Count tournament frequencies
            tournament_counts = Counter(t['name'] for t in all_tournaments)
            
            # Create tournament info mapping
            tournament_info = {}
            for tournament_data in all_tournaments:
                name = tournament_data['name']
                if name not in tournament_info:
                    tournament_info[name] = {
                        'surface': tournament_data['surface'],
                        'series': tournament_data['series'],
                        'frequency': tournament_counts[name]
                    }
            
            # Sort by frequency
            sorted_tournaments = sorted(tournament_info.items(), 
                                      key=lambda x: x[1]['frequency'], 
                                      reverse=True)
            
            tournaments[gender] = dict(sorted_tournaments)
    
    return model, tournaments

# Utility functions for betting markets
def calculate_set_betting_odds(p1_prob, best_of=3):
    """Calculate set betting probabilities and odds"""
    p2_prob = 1 - p1_prob
    
    if best_of == 3:
        # Calculate individual set probabilities (simplified model)
        if p1_prob > 0.7:  # Strong favorite
            probs = {
                '2-0': 0.45 * p1_prob,
                '2-1': 0.55 * p1_prob,
                '1-2': 0.55 * p2_prob,
                '0-2': 0.45 * p2_prob
            }
        elif p1_prob < 0.3:  # Strong underdog
            probs = {
                '2-0': 0.25 * p1_prob,
                '2-1': 0.75 * p1_prob,
                '1-2': 0.75 * p2_prob,
                '0-2': 0.25 * p2_prob
            }
        else:  # Close match
            probs = {
                '2-0': 0.30 * p1_prob,
                '2-1': 0.70 * p1_prob,
                '1-2': 0.70 * p2_prob,
                '0-2': 0.30 * p2_prob
            }
    else:  # Best of 5
        if p1_prob > 0.7:
            probs = {
                '3-0': 0.25 * p1_prob,
                '3-1': 0.35 * p1_prob,
                '3-2': 0.40 * p1_prob,
                '2-3': 0.40 * p2_prob,
                '1-3': 0.35 * p2_prob,
                '0-3': 0.25 * p2_prob
            }
        elif p1_prob < 0.3:
            probs = {
                '3-0': 0.15 * p1_prob,
                '3-1': 0.25 * p1_prob,
                '3-2': 0.60 * p1_prob,
                '2-3': 0.60 * p2_prob,
                '1-3': 0.25 * p2_prob,
                '0-3': 0.15 * p2_prob
            }
        else:
            probs = {
                '3-0': 0.18 * p1_prob,
                '3-1': 0.32 * p1_prob,
                '3-2': 0.50 * p1_prob,
                '2-3': 0.50 * p2_prob,
                '1-3': 0.32 * p2_prob,
                '0-3': 0.18 * p2_prob
            }
    
    # Convert to odds
    odds = {score: 1/prob if prob > 0 else 999 for score, prob in probs.items()}
    
    return probs, odds

def calculate_total_sets_odds(p1_prob, best_of=3):
    """Calculate total sets over/under odds"""
    # Estimate probability of going distance based on how close the match is
    closeness = 1 - abs(p1_prob - 0.5) * 2  # 0 = one-sided, 1 = perfectly even
    
    if best_of == 3:
        # Over 2.5 sets (3 sets total)
        over_2_5 = 0.3 + (closeness * 0.4)  # 30-70% chance depending on closeness
        under_2_5 = 1 - over_2_5
        
        probs = {'over_2.5': over_2_5, 'under_2.5': under_2_5}
        odds = {'over_2.5': 1/over_2_5, 'under_2.5': 1/under_2_5}
    else:
        # Over/under 3.5 and 4.5 sets
        over_4_5 = 0.2 + (closeness * 0.5)  # 20-70% for 5 sets
        over_3_5 = 0.4 + (closeness * 0.4)  # 40-80% for 4+ sets
        
        probs = {
            'over_4.5': over_4_5,
            'under_4.5': 1 - over_4_5,
            'over_3.5': over_3_5,
            'under_3.5': 1 - over_3_5
        }
        odds = {market: 1/prob for market, prob in probs.items()}
    
    return probs, odds

def calculate_games_handicap(p1_prob, rating_diff):
    """Calculate games handicap betting"""
    # Estimate games advantage based on rating difference and win probability
    if rating_diff < 50:
        spreads = [-2.5, -1.5, 1.5, 2.5]
    elif rating_diff < 100:
        spreads = [-3.5, -2.5, 2.5, 3.5]
    else:
        spreads = [-4.5, -3.5, 3.5, 4.5]
    
    handicap_probs = {}
    for spread in spreads:
        if spread < 0:  # Favorite giving games
            # Stronger player giving games
            if p1_prob > 0.5:
                prob = max(0.1, p1_prob - abs(spread) * 0.08)
            else:
                prob = min(0.9, (1-p1_prob) + abs(spread) * 0.08)
        else:  # Underdog getting games
            if p1_prob < 0.5:
                prob = min(0.9, p1_prob + spread * 0.08)
            else:
                prob = max(0.1, (1-p1_prob) - spread * 0.08)
        
        handicap_probs[f'{spread:+.1f}'] = prob
    
    handicap_odds = {spread: 1/prob for spread, prob in handicap_probs.items()}
    
    return handicap_probs, handicap_odds

# Load model and tournaments
model, tournaments = load_model_and_tournaments()

# Helper function to safely get enhanced data
def safe_get_surface_expertise(model, gender, surface, player):
    """Safely get surface expertise, return 0 if not available"""
    try:
        return model.surface_expertise[gender].get(surface, {}).get(player, 0)
    except (AttributeError, KeyError):
        return 0

def safe_get_recent_form(model, gender, player):
    """Safely get recent form, return 0 if not available"""
    try:
        return model.recent_form[gender].get(player, 0)
    except (AttributeError, KeyError):
        return 0

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎾 Match Analysis", "🏆 All Betting Markets", "👤 Player Deep Dive"])

with tab1:
    # Main prediction interface
    st.title("🎾 Tennis Betting Model")
    
    # Model status indicator
    has_enhanced = hasattr(model, 'surface_expertise') and hasattr(model, 'recent_form')
    if has_enhanced:
        st.success("✅ model loaded with surface expertise and recent form")
    else:
        st.warning("⚠️ Basic model loaded. For enhanced features, run train_model.py to generate enhanced_player_ratings.json")
    
    st.markdown("Elo-based model with tournament integration and comprehensive betting markets")

    # Sidebar for inputs
    with st.sidebar:
        st.header("Match Configuration")
        
        # Gender selection
        gender = st.radio("🚹🚺 Gender", ["Men", "Women"], horizontal=True)
        gender_key = 'men' if gender == "Men" else 'women'
        
        # Get player list
        if gender_key in model.ratings and model.ratings[gender_key]:
            players = sorted(model.ratings[gender_key].keys())
        else:
            st.error(f"No {gender} player ratings found. Please run train_model.py first.")
            st.stop()
        
        # Player selection
        st.subheader("👥 Players")
        player1 = st.selectbox("Player 1", players, key="p1")
        player2 = st.selectbox("Player 2", players, key="p2", 
                              index=1 if len(players) > 1 else 0)
        
        # Tournament selection
        st.subheader("🏆 Tournament Selection")
        
        # Get tournament list for this gender
        tournament_list = list(tournaments[gender_key].keys()) if tournaments[gender_key] else []
        
        if tournament_list:
            # Add "Custom" option
            tournament_options = ["Custom Tournament"] + tournament_list
            selected_tournament = st.selectbox(
                "Tournament", 
                tournament_options,
                help="Tournaments sorted by frequency (most common first)"
            )
            
            if selected_tournament == "Custom Tournament":
                # Manual selection
                surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
                
                if gender == "Men":
                    tier_options = ["Grand Slam", "Masters 1000", "ATP 500", "ATP 250", "Challenger"]
                else:
                    tier_options = ["Grand Slam", "WTA 1000", "WTA 500", "WTA 250", "ITF"]
                
                tournament_tier = st.selectbox("Tournament Tier", tier_options)
                tournament_name = st.text_input("Tournament Name", placeholder="e.g., Custom Event")
            else:
                # Auto-populate from tournament data
                tournament_info = tournaments[gender_key][selected_tournament]
                surface = tournament_info['surface']
                tournament_tier = tournament_info['series']
                tournament_name = selected_tournament
                
                st.info(f"**Auto-selected:**\n- Surface: {surface}\n- Tier: {tournament_tier}")
        else:
            # Fallback to manual
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
            
            if gender == "Men":
                tier_options = ["Grand Slam", "Masters 1000", "ATP 500", "ATP 250", "Challenger"]
            else:
                tier_options = ["Grand Slam", "WTA 1000", "WTA 500", "WTA 250", "ITF"]
            
            tournament_tier = st.selectbox("Tournament Tier", tier_options)
            tournament_name = st.text_input("Tournament Name", placeholder="e.g., Australian Open")
        
        # Match details
        st.subheader("🎾 Match Details")
        
        best_of = st.radio("Match Format", [3, 5], 
                          format_func=lambda x: f"Best of {x} sets", horizontal=True)
        
        round_options = ["1st Round", "2nd Round", "3rd Round", "Round of 16",
                        "Quarterfinals", "Semifinals", "Final"]
        selected_round = st.selectbox("Tournament Round", round_options, index=3)
        
        # Market odds
        st.subheader("💰 Market Odds")
        col1, col2 = st.columns(2)
        with col1:
            market_odds_p1 = st.number_input(f"{player1[:10]}... Odds", 
                                           min_value=1.01, value=2.0, step=0.01)
        with col2:
            market_odds_p2 = st.number_input(f"{player2[:10]}... Odds", 
                                           min_value=1.01, value=2.0, step=0.01)
        
        # Calculate button
        calculate = st.button("🔮 Analyze Match", type="primary", use_container_width=True)

    # Main prediction display
    if calculate and player1 and player2 and player1 != player2:
        
        # Header
        st.header(f"{player1} vs {player2}")
        if tournament_name:
            st.subheader(f"🏆 {tournament_name} - {selected_round}")
        else:
            st.subheader(f"🏆 {tournament_tier} - {selected_round}")
        
        st.markdown(f"**📍 {surface} Court • 🎾 Best of {best_of} Sets**")
        
        # Get predictions - use enhanced method if available
        if hasattr(model, 'predict_match_enhanced'):
            prediction = model.predict_match_enhanced(player1, player2, surface, best_of, gender_key, selected_round, tournament_tier)
            p1_win_prob = prediction['probability']
        else:
            p1_win_prob = model.calculate_match_probability(player1, player2, surface, best_of, gender_key)
        
        # Display main metrics
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            p1_rating = model.ratings[gender_key].get(player1, 1500)
            st.metric(f"{player1} Rating", f"{p1_rating:.0f}")
        
        with col2:
            p2_rating = model.ratings[gender_key].get(player2, 1500)
            st.metric(f"{player2} Rating", f"{p2_rating:.0f}")
        
        with col3:
            rating_diff = abs(p1_rating - p2_rating)
            st.metric("Rating Difference", f"{rating_diff:.0f}")
        
        with col4:
            if rating_diff > 200:
                confidence = "High"
                color = "🟢"
            elif rating_diff > 100:
                confidence = "Medium"
                color = "🟡"
            else:
                confidence = "Low"
                color = "🔴"
            st.metric("Confidence", f"{color} {confidence}")
        
        # Win probabilities
        st.markdown("---")
        st.subheader("📊 Win Probabilities & Moneyline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(f"{player1} Win %", f"{p1_win_prob*100:.1f}%")
            st.progress(p1_win_prob)
            model_odds_p1 = 1 / p1_win_prob if p1_win_prob > 0 else 999
            st.metric(f"{player1} Model Odds", f"{model_odds_p1:.2f}")
        
        with col2:
            p2_win_prob = 1 - p1_win_prob
            st.metric(f"{player2} Win %", f"{p2_win_prob*100:.1f}%")
            st.progress(p2_win_prob)
            model_odds_p2 = 1 / p2_win_prob if p2_win_prob > 0 else 999
            st.metric(f"{player2} Model Odds", f"{model_odds_p2:.2f}")
        
        # Edge calculation
        st.subheader("Edge Analysis")
        
        market_implied_p1 = 1 / market_odds_p1
        market_implied_p2 = 1 / market_odds_p2
        market_margin = market_implied_p1 + market_implied_p2 - 1
        
        market_true_p1 = market_implied_p1 / (market_implied_p1 + market_implied_p2)
        market_true_p2 = market_implied_p2 / (market_implied_p1 + market_implied_p2)
        
        edge_p1 = p1_win_prob - market_true_p1
        edge_p2 = p2_win_prob - market_true_p2
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Market Margin: {market_margin*100:.1f}%**")
        
        with col2:
            if edge_p1 > 0.05:
                st.success(f"**{player1} Edge: +{edge_p1*100:.2f}%**")
                kelly = edge_p1 / (market_odds_p1 - 1)
                st.caption(f"Kelly: {kelly*100:.1f}%")
            elif edge_p1 > 0.02:
                st.warning(f"**{player1} Edge: +{edge_p1*100:.2f}%**")
            else:
                st.error(f"**{player1} Edge: {edge_p1*100:.2f}%**")
        
        with col3:
            if edge_p2 > 0.05:
                st.success(f"**{player2} Edge: +{edge_p2*100:.2f}%**")
                kelly = edge_p2 / (market_odds_p2 - 1)
                st.caption(f"Kelly: {kelly*100:.1f}%")
            elif edge_p2 > 0.02:
                st.warning(f"**{player2} Edge: +{edge_p2*100:.2f}%**")
            else:
                st.error(f"**{player2} Edge: {edge_p2*100:.2f}%**")

    elif calculate:
        st.error("❌ Please select two different players")
    else:
        st.info("👆 Configure match details in the sidebar and click 'Analyze Match' to get predictions")

with tab2:
    st.header("Betting Markets")
    
    if 'calculate' in locals() and calculate and 'player1' in locals() and 'player2' in locals() and player1 != player2:
        
        st.markdown(f"### {player1} vs {player2} - All Markets")
        
        # Calculate all betting markets
        set_probs, set_odds = calculate_set_betting_odds(p1_win_prob, best_of)
        total_probs, total_odds = calculate_total_sets_odds(p1_win_prob, best_of)
        handicap_probs, handicap_odds = calculate_games_handicap(p1_win_prob, rating_diff)
        
        # Create columns for different market types
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Set Betting")
            st.write("*Exact set score predictions*")
            
            for score, prob in set_probs.items():
                odds_val = set_odds[score]
                
                # Determine recommendation color
                if prob > 0.3:
                    market_class = "strong-bet"
                    icon = "🟢"
                elif prob > 0.2:
                    market_class = "value-bet"
                    icon = "🟡"
                else:
                    market_class = "no-bet"
                    icon = "🔴"
                
                st.markdown(f"""
                <div class="betting-market {market_class}">
                    <strong>{icon} {score}</strong><br/>
                    <span style="font-size: 0.9em;">
                        Probability: {prob*100:.1f}% | Odds: {odds_val:.2f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("⚡ Games Handicap")
            st.write("*Games spread betting*")
            
            for spread, prob in handicap_probs.items():
                odds_val = handicap_odds[spread]
                
                if prob > 0.55:
                    market_class = "strong-bet"
                    icon = "🟢"
                elif prob > 0.52:
                    market_class = "value-bet"
                    icon = "🟡"
                else:
                    market_class = "no-bet"
                    icon = "🔴"
                
                player_name = player1 if float(spread) > 0 else player2
                st.markdown(f"""
                <div class="betting-market {market_class}">
                    <strong>{icon} {player_name} {spread}</strong><br/>
                    <span style="font-size: 0.9em;">
                        Probability: {prob*100:.1f}% | Odds: {odds_val:.2f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Total Sets")
            st.write("*Over/under total sets*")
            
            for market, prob in total_probs.items():
                odds_val = total_odds[market]
                
                if prob > 0.55:
                    market_class = "strong-bet"
                    icon = "🟢"
                elif prob > 0.52:
                    market_class = "value-bet"
                    icon = "🟡"
                else:
                    market_class = "no-bet"
                    icon = "🔴"
                
                st.markdown(f"""
                <div class="betting-market {market_class}">
                    <strong>{icon} {market.replace('_', ' ').title()}</strong><br/>
                    <span style="font-size: 0.9em;">
                        Probability: {prob*100:.1f}% | Odds: {odds_val:.2f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("🏅 Player Performance")
            st.write("*Advanced player markets*")
            
            # Calculate some advanced metrics
            p1_to_win_set = min(0.85, p1_win_prob + 0.1)  # Probability to win at least one set
            p2_to_win_set = min(0.85, p2_win_prob + 0.1)
            
            # Break/hold percentages (estimated)
            p1_break_prob = 0.15 + (p1_win_prob - 0.5) * 0.3
            p2_break_prob = 0.15 + (p2_win_prob - 0.5) * 0.3
            
            performance_markets = {
                f"{player1} to win a set": p1_to_win_set,
                f"{player2} to win a set": p2_to_win_set,
                f"{player1} break of serve": max(0.05, p1_break_prob),
                f"{player2} break of serve": max(0.05, p2_break_prob)
            }
            
            for market, prob in performance_markets.items():
                odds_val = 1 / prob if prob > 0 else 999
                
                if prob > 0.6:
                    market_class = "strong-bet"
                    icon = "🟢"
                elif prob > 0.55:
                    market_class = "value-bet"
                    icon = "🟡"
                else:
                    market_class = "no-bet"
                    icon = "🔴"
                
                st.markdown(f"""
                <div class="betting-market {market_class}">
                    <strong>{icon} {market}</strong><br/>
                    <span style="font-size: 0.9em;">
                        Probability: {prob*100:.1f}% | Odds: {odds_val:.2f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        # Market Summary
        st.markdown("---")
        st.subheader("📈 Market Summary & Recommendations")
        
        # Find best value bets across all markets
        all_markets = []
        
        # Add set betting
        for score, prob in set_probs.items():
            all_markets.append(('Set Betting', f'{score}', prob, set_odds[score]))
        
        # Add totals
        for market, prob in total_probs.items():
            all_markets.append(('Total Sets', market.replace('_', ' ').title(), prob, total_odds[market]))
        
        # Add handicaps
        for spread, prob in handicap_probs.items():
            player_name = player1 if float(spread) > 0 else player2
            all_markets.append(('Handicap', f'{player_name} {spread}', prob, handicap_odds[spread]))
        
        # Sort by value (probability * odds - 1)
        value_markets = [(market_type, bet, prob, odds, prob * odds - 1) 
                        for market_type, bet, prob, odds in all_markets]
        value_markets.sort(key=lambda x: x[4], reverse=True)
        
        st.write("**🎯 Top Value Bets:**")
        for i, (market_type, bet, prob, odds, value) in enumerate(value_markets[:5]):
            if value > 0.1:
                st.success(f"{i+1}. **{market_type}**: {bet} | {prob*100:.1f}% @ {odds:.2f} | Value: +{value*100:.1f}%")
            elif value > 0.05:
                st.warning(f"{i+1}. **{market_type}**: {bet} | {prob*100:.1f}% @ {odds:.2f} | Value: +{value*100:.1f}%")
            else:
                st.info(f"{i+1}. **{market_type}**: {bet} | {prob*100:.1f}% @ {odds:.2f} | Value: {value*100:+.1f}%")
    
    else:
        st.info("👆 Please configure and analyze a match in the 'Match Analysis' tab first")

with tab3:
    st.header("👤 Player Deep Dive Analysis")
    
    # Player selection for analysis
    analysis_gender = st.selectbox("Select Gender", ["Men", "Women"], key="analysis_gender")
    analysis_gender_key = 'men' if analysis_gender == "Men" else 'women'
    
    if analysis_gender_key in model.ratings and model.ratings[analysis_gender_key]:
        analysis_players = sorted(model.ratings[analysis_gender_key].keys())
        selected_player = st.selectbox("Select Player for Analysis", analysis_players, key="analysis_player")
        
        if st.button("🔍 Analyze Player", type="primary"):
            
            # Load all match data for this player
            all_matches = []
            folder_name = 'Mens' if analysis_gender_key == 'men' else 'Womens'
            folder_path = os.path.join(os.path.dirname(__file__), "Data", folder_name)
            
            for year in range(2016, 2026):
                file_path = os.path.join(folder_path, f"{year}.xlsx")
                if os.path.exists(file_path):
                    try:
                        df = pd.read_excel(file_path)
                        
                        # Find column names
                        winner_col = None
                        loser_col = None
                        for col in df.columns:
                            if 'winner' in col.lower():
                                winner_col = col
                            elif 'loser' in col.lower():
                                loser_col = col
                        
                        if winner_col and loser_col:
                            # Filter for this player
                            player_matches = df[
                                (df[winner_col] == selected_player) | 
                                (df[loser_col] == selected_player)
                            ].copy()
                            
                            if len(player_matches) > 0:
                                player_matches['Year'] = year
                                player_matches['Is_Winner'] = player_matches[winner_col] == selected_player
                                all_matches.append(player_matches)
                    except Exception as e:
                        continue
            
            if all_matches:
                combined_matches = pd.concat(all_matches, ignore_index=True)
                
                # Display comprehensive stats
                st.subheader(f"📊 {selected_player} - Career Analysis")
                
                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                
                total_matches = len(combined_matches)
                total_wins = combined_matches['Is_Winner'].sum()
                win_rate = total_wins / total_matches if total_matches > 0 else 0
                
                with col1:
                    st.metric("Total Matches", f"{total_matches:,}")
                
                with col2:
                    st.metric("Career Wins", f"{total_wins:,}")
                
                with col3:
                    st.metric("Win Rate", f"{win_rate*100:.1f}%")
                
                with col4:
                    current_rating = model.ratings[analysis_gender_key].get(selected_player, 1500)
                    st.metric("Current Rating", f"{current_rating:.0f}")
                
                # Surface analysis
                st.subheader("🏟️ Surface Performance")
                
                surface_col = None
                for col in combined_matches.columns:
                    if 'surface' in col.lower():
                        surface_col = col
                        break
                
                if surface_col:
                    surface_stats = combined_matches.groupby(surface_col).agg({
                        'Is_Winner': ['count', 'sum', 'mean']
                    }).round(3)
                    
                    surface_stats.columns = ['Matches', 'Wins', 'Win Rate']
                    surface_stats = surface_stats.reset_index()
                    
                    # Create surface performance chart
                    if len(surface_stats) > 0:
                        fig = px.bar(surface_stats, x=surface_col, y='Win Rate', 
                                   title=f"{selected_player} - Win Rate by Surface",
                                   text='Win Rate')
                        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                        fig.update_layout(yaxis_title="Win Rate", xaxis_title="Surface")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(surface_stats, use_container_width=True)
                
                # Year-by-year performance
                st.subheader("📅 Year-by-Year Performance")
                
                yearly_stats = combined_matches.groupby('Year').agg({
                    'Is_Winner': ['count', 'sum', 'mean']
                }).round(3)
                
                yearly_stats.columns = ['Matches', 'Wins', 'Win Rate']
                yearly_stats = yearly_stats.reset_index()
                
                # Performance over time chart
                if len(yearly_stats) > 1:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=yearly_stats['Year'],
                        y=yearly_stats['Win Rate'],
                        mode='lines+markers',
                        name='Win Rate',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=yearly_stats['Year'],
                        y=yearly_stats['Matches'],
                        name='Matches Played',
                        yaxis='y2',
                        opacity=0.3,
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_player} - Performance Over Time",
                        xaxis_title="Year",
                        yaxis=dict(title="Win Rate", side="left"),
                        yaxis2=dict(title="Matches Played", side="right", overlaying="y"),
                        legend=dict(x=0.7, y=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(yearly_stats, use_container_width=True)
                
                # Tournament analysis
                st.subheader("🏆 Tournament Performance")
                
                tournament_col = None
                for col in combined_matches.columns:
                    if 'tournament' in col.lower():
                        tournament_col = col
                        break
                
                if tournament_col:
                    # Count tournament wins and appearances
                    tournament_wins = combined_matches[combined_matches['Is_Winner'] == True][tournament_col].value_counts()
                    tournament_appearances = combined_matches[tournament_col].value_counts()
                    
                    tournament_summary = pd.DataFrame({
                        'Appearances': tournament_appearances,
                        'Wins': tournament_wins.reindex(tournament_appearances.index, fill_value=0)
                    }).fillna(0)
                    
                    tournament_summary['Win Rate'] = tournament_summary['Wins'] / tournament_summary['Appearances']
                    tournament_summary = tournament_summary.sort_values('Appearances', ascending=False)
                    
                    st.write(f"**Most Played Tournaments (Top 10):**")
                    st.dataframe(tournament_summary.head(10), use_container_width=True)
                    
                    # Tournament wins
                    if tournament_wins.sum() > 0:
                        st.write(f"**🏆 Tournament Titles ({int(tournament_wins.sum())} total):**")
                        titles_df = tournament_wins.reset_index()
                        titles_df.columns = ['Tournament', 'Titles']
                        st.dataframe(titles_df.head(10), use_container_width=True)
                
                # Recent form
                st.subheader("🔥 Recent Form Analysis")
                
                # Sort by date if possible
                date_col = None
                for col in combined_matches.columns:
                    if 'date' in col.lower() or 'data' in col.lower():
                        date_col = col
                        break
                
                if date_col:
                    combined_matches[date_col] = pd.to_datetime(combined_matches[date_col], errors='coerce')
                    recent_matches = combined_matches.sort_values(date_col, ascending=False).head(20)
                else:
                    recent_matches = combined_matches.tail(20)
                
                recent_win_rate = recent_matches['Is_Winner'].mean()
                recent_wins = recent_matches['Is_Winner'].sum()
                recent_total = len(recent_matches)
                
                st.metric("Last 20 Matches", f"{recent_wins}/{recent_total} ({recent_win_rate*100:.1f}%)")
            
            else:
                st.warning(f"No match data found for {selected_player}")
    
    else:
        st.error("No player data available. Please run the model training first.")

# Footer with tools
st.markdown("---")
with st.sidebar:
    st.markdown("---")
    st.subheader("🛠️ Tools")
    
    if st.button("🔄 Refresh Model"):
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("📊 Model Stats"):
        if 'gender_key' in locals() and model.ratings[gender_key]:
            ratings_list = list(model.ratings[gender_key].values())
            st.write(f"**{gender} Players:** {len(ratings_list):,}")
            st.write(f"**Rating Range:** {min(ratings_list):.0f} - {max(ratings_list):.0f}")
            st.write(f"**Average Rating:** {np.mean(ratings_list):.0f}")