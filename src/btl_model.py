import pandas as pd
import numpy as np
import os
from datetime import datetime
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
        self.series_adjustments = {'men': {}, 'women': {}}
        self.best_of_adjustments = {3: 0, 5: 0}  # Adjustment for match format
        self.k_factor = 32  # Elo K-factor
        
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
    
    def initialize_ratings(self, gender: str = 'men'):
        """Initialize all players with base Elo rating"""
        df = self.matches_data[gender]
        all_players = pd.concat([df['Winner'], df['Loser']]).unique()
        
        # Initialize with 1500 Elo
        for player in all_players:
            self.ratings[gender][player] = 1500
        
        print(f"Initialized {len(all_players)} {gender} players with 1500 rating")
    
    def calculate_expected_score(self, rating1: float, rating2: float, 
                               surface_adj: float = 0, series_adj: float = 0,
                               best_of_adj: float = 0) -> float:
        """Calculate expected score using Elo formula with adjustments"""
        rating_diff = rating1 - rating2 + surface_adj + series_adj + best_of_adj
        return 1 / (1 + 10**(-rating_diff/400))
    
    def update_ratings(self, gender: str = 'men'):
        """Update ratings using Elo system with adjustments"""
        df = self.matches_data[gender]
        
        # Initialize ratings if needed
        if not self.ratings[gender]:
            self.initialize_ratings(gender)
        
        # Track surface and series performance
        surface_wins = {}
        surface_total = {}
        series_wins = {}
        series_total = {}
        
        print(f"Processing {len(df)} matches for {gender}...")
        
        for idx, match in df.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            
            if pd.isna(winner) or pd.isna(loser):
                continue
            
            # Get current ratings
            winner_rating = self.ratings[gender].get(winner, 1500)
            loser_rating = self.ratings[gender].get(loser, 1500)
            
            # Get adjustments
            surface = str(match.get('Surface', 'Hard')).strip().title()
            series = str(match.get('Series', 'Unknown') if gender == 'men' else match.get('Tier', 'Unknown'))
            best_of = int(match.get('Best of', 3))
            
            # Track surface/series stats
            if surface not in surface_wins:
                surface_wins[surface] = {winner: 0, loser: 0}
                surface_total[surface] = {winner: 0, loser: 0}
            
            if series not in series_wins:
                series_wins[series] = {winner: 0, loser: 0}
                series_total[series] = {winner: 0, loser: 0}
            
            # Update tracking
            for player in [winner, loser]:
                if player not in surface_wins[surface]:
                    surface_wins[surface][player] = 0
                    surface_total[surface][player] = 0
                if player not in series_wins[series]:
                    series_wins[series][player] = 0
                    series_total[series][player] = 0
            
            surface_wins[surface][winner] += 1
            surface_total[surface][winner] += 1
            surface_total[surface][loser] += 1
            
            series_wins[series][winner] += 1
            series_total[series][winner] += 1
            series_total[series][loser] += 1
            
            # Calculate adjustments (simple percentage-based)
            winner_surface_adj = 0
            loser_surface_adj = 0
            if surface_total[surface][winner] > 5:
                winner_surface_adj = (surface_wins[surface][winner] / surface_total[surface][winner] - 0.5) * 100
            if surface_total[surface][loser] > 5:
                loser_surface_adj = (surface_wins[surface][loser] / surface_total[surface][loser] - 0.5) * 100
            
            # Best of adjustment (5-set matches favor stronger players slightly)
            best_of_adj = 10 if best_of == 5 else 0
            
            # Calculate expected score
            expected_winner = self.calculate_expected_score(
                winner_rating, loser_rating,
                winner_surface_adj - loser_surface_adj,
                0,  # Series adjustment simplified for now
                best_of_adj if winner_rating > loser_rating else -best_of_adj
            )
            
            # Update ratings
            k_factor = self.k_factor
            
            # Adjust K-factor for importance (Grand Slams get higher K)
            if series in ['Grand Slam', 'WTA Premier Mandatory', 'WTA Premier 5']:
                k_factor *= 1.5
            
            # Actual scores (1 for win, 0 for loss)
            rating_change = k_factor * (1 - expected_winner)
            
            self.ratings[gender][winner] = winner_rating + rating_change
            self.ratings[gender][loser] = loser_rating - rating_change
        
        # Calculate final surface adjustments
        for surface in surface_total:
            player_adjs = {}
            for player in surface_total[surface]:
                if surface_total[surface][player] > 10:
                    win_rate = surface_wins[surface][player] / surface_total[surface][player]
                    player_adjs[player] = (win_rate - 0.5) * 200  # Scale adjustment
            self.surface_adjustments[gender][surface] = player_adjs
        
        print(f"Ratings updated for {gender}")
        
        # Print top 10
        top_players = sorted(self.ratings[gender].items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 {gender} players:")
        for i, (player, rating) in enumerate(top_players, 1):
            print(f"  {i}. {player}: {rating:.0f}")
    
    def get_player_surface_adjustment(self, player: str, surface: str, gender: str) -> float:
        """Get a player's adjustment for a specific surface"""
        if surface in self.surface_adjustments[gender]:
            return self.surface_adjustments[gender][surface].get(player, 0)
        return 0
    
    def calculate_match_probability(self, player1: str, player2: str, 
                                  surface: str = 'Hard', best_of: int = 3,
                                  gender: str = 'men') -> float:
        """Calculate probability of player1 beating player2"""
        ratings = self.ratings[gender]
        
        if player1 not in ratings or player2 not in ratings:
            return 0.5
        
        rating1 = ratings[player1]
        rating2 = ratings[player2]
        
        # Get surface adjustments
        surface_adj1 = self.get_player_surface_adjustment(player1, surface, gender)
        surface_adj2 = self.get_player_surface_adjustment(player2, surface, gender)
        
        # Best of adjustment
        best_of_adj = 10 if best_of == 5 else 0
        if rating1 < rating2:
            best_of_adj = -best_of_adj
        
        # Calculate probability
        return self.calculate_expected_score(
            rating1, rating2,
            surface_adj1 - surface_adj2,
            0,  # Series adjustment not used in prediction
            best_of_adj
        )
    
    def generate_betting_odds(self, player1: str, player2: str, 
                            surface: str = 'Hard', best_of: int = 3,
                            gender: str = 'men', margin: float = 0.05) -> Dict:
        """Generate various betting odds for a match"""
        # Get base win probability
        p1_win = self.calculate_match_probability(player1, player2, surface, best_of, gender)
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
            }
        }
        
        # Set betting predictions based on win probability and match format
        if best_of == 3:
            # Best of 3 sets
            if abs(p1_win - 0.5) < 0.1:  # Close match
                odds['set_betting'] = {
                    '2-0': 0.25 if p1_win > 0.5 else 0.20,
                    '2-1': 0.45,
                    '1-2': 0.20 if p1_win > 0.5 else 0.25,
                    '0-2': 0.10 if p1_win > 0.5 else 0.15
                }
            else:  # One-sided
                odds['set_betting'] = {
                    '2-0': 0.40 if p1_win > 0.5 else 0.15,
                    '2-1': 0.35,
                    '1-2': 0.15 if p1_win > 0.5 else 0.35,
                    '0-2': 0.10 if p1_win > 0.5 else 0.15
                }
        else:
            # Best of 5 sets
            if abs(p1_win - 0.5) < 0.1:  # Close match
                odds['set_betting'] = {
                    '3-0': 0.15 if p1_win > 0.5 else 0.10,
                    '3-1': 0.25 if p1_win > 0.5 else 0.20,
                    '3-2': 0.30,
                    '2-3': 0.20 if p1_win > 0.5 else 0.25,
                    '1-3': 0.10 if p1_win > 0.5 else 0.15,
                    '0-3': 0.05 if p1_win > 0.5 else 0.10
                }
            else:  # One-sided
                odds['set_betting'] = {
                    '3-0': 0.30 if p1_win > 0.5 else 0.10,
                    '3-1': 0.35 if p1_win > 0.5 else 0.15,
                    '3-2': 0.20,
                    '2-3': 0.10 if p1_win > 0.5 else 0.25,
                    '1-3': 0.05 if p1_win > 0.5 else 0.20,
                    '0-3': 0.02 if p1_win > 0.5 else 0.15
                }
        
        # Normalize set betting probabilities
        total_prob = sum(odds['set_betting'].values())
        for score in odds['set_betting']:
            odds['set_betting'][score] /= total_prob
        
        # Total sets over/under
        if best_of == 3:
            odds['total_sets'] = {
                'over_2.5': odds['set_betting'].get('2-1', 0) + odds['set_betting'].get('1-2', 0),
                'under_2.5': odds['set_betting'].get('2-0', 0) + odds['set_betting'].get('0-2', 0)
            }
        else:
            over_3_5 = (odds['set_betting'].get('3-2', 0) + odds['set_betting'].get('2-3', 0) +
                       odds['set_betting'].get('3-1', 0) + odds['set_betting'].get('1-3', 0))
            odds['total_sets'] = {
                'over_3.5': over_3_5,
                'under_3.5': 1 - over_3_5
            }
        
        return odds
    
    def save_ratings(self, filepath: str):
        """Save player ratings to JSON file"""
        data = {
            'ratings': self.ratings,
            'surface_adjustments': self.surface_adjustments,
            'series_adjustments': self.series_adjustments,
            'last_updated': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_ratings(self, filepath: str):
        """Load player ratings from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.ratings = data['ratings']
        self.surface_adjustments = data.get('surface_adjustments', {})
        self.series_adjustments = data.get('series_adjustments', {})