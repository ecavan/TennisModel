import pandas as pd
import numpy as np
import os
import math
from datetime import datetime, timedelta
import json
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class TennisBTLModel:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.matches_data = {'men': pd.DataFrame(), 'women': pd.DataFrame()}
        self.ratings = {'men': {}, 'women': {}}
        self.surface_adjustments = {'men': {}, 'women': {}}
        self.recent_form = {'men': {}, 'women': {}}
        self.surface_expertise = {'men': {}, 'women': {}}
        self.k_factor = 32  # Base Elo K-factor
        
        # Tournament tier weights
        self.tournament_weights = {
            'men': {
                'Grand Slam': 2.0,
                'Masters': 1.6,
                'Masters 1000': 1.6,
                'ATP 500': 1.3,
                'ATP 250': 1.0,
                'Challenger': 0.7,
                'Futures': 0.5
            },
            'women': {
                'Grand Slam': 2.0,
                'WTA Premier Mandatory': 1.8,
                'WTA Premier 5': 1.6,
                'WTA Premier': 1.4,
                'WTA International': 1.0,
                'WTA 1000': 1.8,
                'WTA 500': 1.4,
                'WTA 250': 1.0,
                'ITF': 0.6
            }
        }
        
        # Round importance weights
        self.round_weights = {
            'Final': 2.0,
            'Finals': 2.0,
            'Semifinals': 1.7,
            'Semi-Finals': 1.7,
            'Quarterfinals': 1.4,
            'Quarter-Finals': 1.4,
            '4th Round': 1.2,
            'Round of 16': 1.2,
            '3rd Round': 1.1,
            '2nd Round': 1.0,
            '1st Round': 0.9,
            'Qualifying': 0.6
        }
        
    def load_data(self, gender: str = 'both'):
        """Load tennis match data from Excel files"""
        genders = ['men', 'women'] if gender == 'both' else [gender]
        
        for g in genders:
            folder_name = 'Mens' if g == 'men' else 'Womens'
            folder_path = os.path.join(self.data_path, folder_name)
            
            all_matches = []
            for year in range(2016, 2026):
                file_path = os.path.join(folder_path, f"{year}.xlsx")
                if os.path.exists(file_path):
                    try:
                        df = pd.read_excel(file_path)
                        df = df.dropna()
                        df['Year'] = year
                        all_matches.append(df)
                        print(f"Loaded {len(df)} matches from {g} {year}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
            
            if all_matches:
                combined_df = pd.concat(all_matches, ignore_index=True)
                # Sort by date
                if 'Data' in combined_df.columns:
                    combined_df['Date'] = pd.to_datetime(combined_df['Data'], errors='coerce')
                elif 'Date' in combined_df.columns:
                    combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
                
                combined_df = combined_df.sort_values('Date', na_position='last')
                self.matches_data[g] = combined_df
                print(f"Total {g} matches loaded: {len(combined_df)}")
    
    def initialize_ratings_with_rankings(self, gender: str = 'men'):
        """Initialize ratings using official rankings and base Elo"""
        df = self.matches_data[gender]
        
        # Get latest rankings for each player
        player_rankings = {}
        
        for _, match in df.iterrows():
            for player_type in ['Winner', 'Loser']:
                player = match[player_type]
                rank_col = f"{player_type[0]}Rank"  # WRank or LRank
                points_col = f"{player_type[0]}Pts"  # WPts or LPts
                
                if pd.notna(match.get(rank_col)) and pd.notna(match.get(points_col)):
                    rank = match[rank_col]
                    points = match[points_col]
                    
                    if player not in player_rankings or match['Date'] > player_rankings[player]['date']:
                        player_rankings[player] = {
                            'rank': rank,
                            'points': points,
                            'date': match['Date']
                        }
        
        # Initialize all players first with base rating
        all_players = pd.concat([df['Winner'], df['Loser']]).unique()
        for player in all_players:
            self.ratings[gender][player] = 1500
        
        # Adjust ratings based on rankings where available
        for player, data in player_rankings.items():
            rank = data['rank']
            
            # Rank-based rating (logarithmic scale)
            if rank <= 10:
                rank_rating = 1900 - (rank - 1) * 30  # Top 10: 1900-1630
            elif rank <= 50:
                rank_rating = 1630 - (rank - 10) * 10  # 11-50: 1630-1230
            elif rank <= 100:
                rank_rating = 1230 - (rank - 50) * 6   # 51-100: 1230-930
            else:
                rank_rating = max(1000, 930 - (rank - 100) * 2)  # 100+: decreasing
            
            self.ratings[gender][player] = max(1000, min(2000, rank_rating))
        
        print(f"Initialized {len(all_players)} {gender} players with enhanced ratings")
    
    def get_tournament_weight(self, series, tier, gender):
        """Calculate tournament importance multiplier"""
        weights = self.tournament_weights[gender]
        
        # Check series for men, tier for women
        tournament_info = series if gender == 'men' else tier
        
        if pd.isna(tournament_info):
            return 1.0
            
        tournament_str = str(tournament_info).strip()
        
        # Flexible matching
        for key, weight in weights.items():
            if key.lower() in tournament_str.lower():
                return weight
        
        return 1.0  # Default weight
    
    def get_round_weight(self, round_name):
        """Calculate round importance multiplier"""
        if pd.isna(round_name):
            return 1.0
            
        round_lower = str(round_name).lower()
        
        for key, weight in self.round_weights.items():
            if key.lower() in round_lower:
                return weight
        
        return 1.0  # Default weight
    
    def calculate_time_weight(self, match_date, current_date=None):
        """Calculate weight based on match recency"""
        if current_date is None:
            current_date = datetime.now()
        
        if pd.isna(match_date):
            return 0.1
            
        days_ago = (current_date - match_date).days
        
        # Exponential decay: 50% weight after 1 year
        half_life_days = 365
        return 0.5 ** (days_ago / half_life_days)
    
    def calculate_surface_expertise(self, player, surface, gender):
        """Calculate player's surface-specific rating adjustment"""
        matches = self.matches_data[gender]
        player_matches = matches[
            ((matches['Winner'] == player) | (matches['Loser'] == player)) &
            (matches['Surface'] == surface)
        ].copy()
        
        if len(player_matches) < 5:
            return 0  # Not enough data
        
        # Calculate time-weighted performance
        total_weighted_wins = 0
        total_weight = 0
        opponent_ratings = []
        
        for _, match in player_matches.iterrows():
            is_winner = match['Winner'] == player
            opponent = match['Loser'] if is_winner else match['Winner']
            opponent_rating = self.ratings[gender].get(opponent, 1500)
            opponent_ratings.append(opponent_rating)
            
            time_weight = self.calculate_time_weight(match['Date'])
            total_weighted_wins += (1 if is_winner else 0) * time_weight
            total_weight += time_weight
        
        if total_weight == 0:
            return 0
        
        weighted_win_rate = total_weighted_wins / total_weight
        avg_opponent_rating = np.mean(opponent_ratings) if opponent_ratings else 1500
        
        # Adjust based on opponent quality
        quality_factor = (avg_opponent_rating - 1500) / 300  # Normalize
        surface_adjustment = (weighted_win_rate - 0.5) * 150 * (1 + quality_factor * 0.2)
        
        return max(-100, min(100, surface_adjustment))  # Cap the adjustment
    
    def get_recent_form(self, player, gender, days=90):
        """Calculate player's recent form"""
        cutoff_date = datetime.now() - timedelta(days=days)
        matches = self.matches_data[gender]
        
        recent_matches = matches[
            ((matches['Winner'] == player) | (matches['Loser'] == player)) &
            (matches['Date'] >= cutoff_date)
        ]
        
        if len(recent_matches) == 0:
            return 0
        
        form_score = 0
        total_weight = 0
        
        for _, match in recent_matches.iterrows():
            is_winner = match['Winner'] == player
            time_weight = self.calculate_time_weight(match['Date'])
            tournament_weight = self.get_tournament_weight(
                match.get('Series'), match.get('Tier'), gender
            )
            
            # Weight recent wins/losses by tournament importance
            weight = time_weight * tournament_weight
            form_score += (1 if is_winner else 0) * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0
            
        return (form_score / total_weight - 0.5) * 100  # -50 to +50 adjustment
    
    def calculate_dynamic_k_factor(self, rating1, rating2, tournament_weight, round_weight):
        """Calculate dynamic K-factor based on multiple factors"""
        base_k = self.k_factor
        
        # Adjust based on rating levels (newer players get higher K)
        avg_rating = (rating1 + rating2) / 2
        if avg_rating < 1300:
            rating_multiplier = 1.5
        elif avg_rating < 1500:
            rating_multiplier = 1.2
        else:
            rating_multiplier = 1.0
        
        # Adjust for tournament and round importance
        importance_multiplier = (tournament_weight + round_weight) / 2
        
        return base_k * rating_multiplier * importance_multiplier
    
    def calculate_expected_score(self, rating1: float, rating2: float, 
                               surface_adj: float = 0, form_adj: float = 0,
                               round_adj: float = 0) -> float:
        """Calculate expected score using enhanced Elo formula"""
        total_adj = surface_adj + form_adj + round_adj
        rating_diff = rating1 - rating2 + total_adj
        return 1 / (1 + 10**(-rating_diff/400))
    
    def update_ratings_enhanced(self, gender: str = 'men'):
        """Enhanced rating update with multiple factors"""
        df = self.matches_data[gender]
        
        # Initialize ratings if needed
        if not self.ratings[gender]:
            self.initialize_ratings_with_rankings(gender)
        
        print(f"Processing {len(df)} matches for enhanced {gender} ratings...")
        
        # Calculate surface expertise for all players
        surfaces = df['Surface'].unique()
        for surface in surfaces:
            if pd.notna(surface):
                for player in self.ratings[gender].keys():
                    adj = self.calculate_surface_expertise(player, surface, gender)
                    if surface not in self.surface_expertise[gender]:
                        self.surface_expertise[gender][surface] = {}
                    self.surface_expertise[gender][surface][player] = adj
        
        # Process matches chronologically
        for idx, match in df.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            
            if pd.isna(winner) or pd.isna(loser):
                continue
            
            # Get current ratings
            winner_rating = self.ratings[gender].get(winner, 1500)
            loser_rating = self.ratings[gender].get(loser, 1500)
            
            # Get adjustments
            surface = str(match.get('Surface', 'Hard')).strip()
            winner_surface_adj = self.surface_expertise[gender].get(surface, {}).get(winner, 0)
            loser_surface_adj = self.surface_expertise[gender].get(surface, {}).get(loser, 0)
            
            # Recent form adjustments
            winner_form = self.get_recent_form(winner, gender) * 0.3  # Scale down form impact
            loser_form = self.get_recent_form(loser, gender) * 0.3
            
            # Round importance (pressure factor)
            round_weight = self.get_round_weight(match.get('Round', ''))
            round_adj = (round_weight - 1.0) * 20  # Scale pressure effect
            
            # Tournament and round weights for K-factor
            tournament_weight = self.get_tournament_weight(
                match.get('Series'), match.get('Tier'), gender
            )
            
            # Calculate expected score with all adjustments
            expected_winner = self.calculate_expected_score(
                winner_rating + winner_surface_adj + winner_form,
                loser_rating + loser_surface_adj + loser_form,
                round_adj=round_adj if winner_rating > loser_rating else -round_adj
            )
            
            # Dynamic K-factor
            k_factor = self.calculate_dynamic_k_factor(
                winner_rating, loser_rating, tournament_weight, round_weight
            )
            
            # Time decay for older matches
            time_weight = self.calculate_time_weight(match['Date'])
            k_factor *= time_weight
            
            # Update ratings
            rating_change = k_factor * (1 - expected_winner)
            
            self.ratings[gender][winner] = winner_rating + rating_change
            self.ratings[gender][loser] = loser_rating - rating_change
        
        # Update recent form for all players
        for player in self.ratings[gender].keys():
            self.recent_form[gender][player] = self.get_recent_form(player, gender)
        
        print(f"Enhanced ratings updated for {gender}")
        
        # Print top 10
        top_players = sorted(self.ratings[gender].items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 {gender} players:")
        for i, (player, rating) in enumerate(top_players, 1):
            form = self.recent_form[gender].get(player, 0)
            print(f"  {i}. {player}: {rating:.0f} (Form: {form:+.1f})")
    
    def predict_match_enhanced(self, player1: str, player2: str, 
                             surface: str = 'Hard', best_of: int = 3,
                             gender: str = 'men', round_name: str = '',
                             tournament_tier: str = '') -> Dict:
        """Enhanced match prediction with all factors"""
        ratings = self.ratings[gender]
        
        if player1 not in ratings or player2 not in ratings:
            return {'probability': 0.5, 'confidence': 'Low - Missing player data'}
        
        rating1 = ratings[player1]
        rating2 = ratings[player2]
        
        # Surface adjustments
        surface_adj1 = self.surface_expertise[gender].get(surface, {}).get(player1, 0)
        surface_adj2 = self.surface_expertise[gender].get(surface, {}).get(player2, 0)
        
        # Recent form
        form1 = self.recent_form[gender].get(player1, 0) * 0.3
        form2 = self.recent_form[gender].get(player2, 0) * 0.3
        
        # Round pressure adjustment
        round_weight = self.get_round_weight(round_name)
        pressure_factor = (round_weight - 1.0) * 15
        
        # Best of adjustment (5-set matches favor consistency)
        best_of_adj = 10 if best_of == 5 else 0
        if rating1 < rating2:
            best_of_adj = -best_of_adj
        
        # Calculate probability
        total_adj1 = surface_adj1 + form1 + (pressure_factor if rating1 > rating2 else -pressure_factor)
        total_adj2 = surface_adj2 + form2
        
        probability = self.calculate_expected_score(
            rating1 + total_adj1,
            rating2 + total_adj2,
            round_adj=best_of_adj
        )
        
        # Confidence calculation
        rating_diff = abs(rating1 - rating2)
        data_confidence = min(len(self.matches_data[gender]), 1000) / 1000
        
        if rating_diff > 200:
            confidence = 'High'
        elif rating_diff > 100:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        return {
            'probability': probability,
            'confidence': confidence,
            'adjustments': {
                'surface': {'p1': surface_adj1, 'p2': surface_adj2},
                'form': {'p1': form1, 'p2': form2},
                'round_pressure': pressure_factor,
                'format': best_of_adj
            }
        }
    
    def generate_betting_odds_enhanced(self, player1: str, player2: str, 
                                     surface: str = 'Hard', best_of: int = 3,
                                     gender: str = 'men', round_name: str = '',
                                     tournament_tier: str = '', margin: float = 0.05) -> Dict:
        """Generate enhanced betting odds with detailed analysis"""
        prediction = self.predict_match_enhanced(
            player1, player2, surface, best_of, gender, round_name, tournament_tier
        )
        
        p1_win = prediction['probability']
        p2_win = 1 - p1_win
        
        # Add betting margin
        p1_win_adj = p1_win / (1 - margin)
        p2_win_adj = p2_win / (1 - margin)
        
        # Convert to decimal odds
        odds = {
            'moneyline': {
                player1: 1 / p1_win_adj if p1_win_adj > 0 else 99.99,
                player2: 1 / p2_win_adj if p2_win_adj > 0 else 99.99
            },
            'win_probability': {
                player1: p1_win,
                player2: p2_win
            },
            'confidence': prediction['confidence'],
            'adjustments': prediction['adjustments']
        }
        
        # Enhanced set betting based on match dynamics
        if best_of == 3:
            if abs(p1_win - 0.5) < 0.15:  # Close match
                odds['set_betting'] = {
                    f'{player1} 2-0': 0.25 if p1_win > 0.5 else 0.18,
                    f'{player1} 2-1': 0.25 if p1_win > 0.5 else 0.32,
                    f'{player2} 2-1': 0.32 if p1_win > 0.5 else 0.25,
                    f'{player2} 2-0': 0.18 if p1_win > 0.5 else 0.25
                }
            else:  # One-sided
                odds['set_betting'] = {
                    f'{player1} 2-0': 0.40 if p1_win > 0.5 else 0.12,
                    f'{player1} 2-1': 0.20 if p1_win > 0.5 else 0.28,
                    f'{player2} 2-1': 0.28 if p1_win > 0.5 else 0.20,
                    f'{player2} 2-0': 0.12 if p1_win > 0.5 else 0.40
                }
        
        return odds
    
    def get_surface_record(self, player: str, surface: str, gender: str) -> str:
        """Get player's record on specific surface"""
        matches = self.matches_data[gender]
        
        # Find actual column names for Winner and Loser
        winner_col = None
        loser_col = None
        surface_col = None
        
        for col in matches.columns:
            if 'winner' in col.lower() or col == 'Winner':
                winner_col = col
            elif 'loser' in col.lower() or col == 'Loser':
                loser_col = col
            elif 'surface' in col.lower() or col == 'Surface':
                surface_col = col
        
        if not winner_col or not loser_col or not surface_col:
            return "No data - missing columns"
        
        try:
            surface_matches = matches[
                ((matches[winner_col] == player) | (matches[loser_col] == player)) &
                (matches[surface_col] == surface)
            ]
            
            wins = len(surface_matches[surface_matches[winner_col] == player])
            total = len(surface_matches)
            
            if total == 0:
                return "No data"
            
            win_pct = (wins / total) * 100
            return f"{wins}-{total-wins} ({win_pct:.1f}%)"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def save_enhanced_ratings(self, filepath: str):
        """Save enhanced ratings and data to JSON file"""
        data = {
            'ratings': self.ratings,
            'surface_expertise': self.surface_expertise,
            'recent_form': self.recent_form,
            'tournament_weights': self.tournament_weights,
            'round_weights': self.round_weights,
            'last_updated': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_enhanced_ratings(self, filepath: str):
        """Load enhanced ratings and data from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.ratings = data.get('ratings', {})
        self.surface_expertise = data.get('surface_expertise', {})
        self.recent_form = data.get('recent_form', {})
        if 'tournament_weights' in data:
            self.tournament_weights = data['tournament_weights']
        if 'round_weights' in data:
            self.round_weights = data['round_weights']