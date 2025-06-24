#!/usr/bin/env python3
"""
Enhanced Tennis Model Backtest Script with Tier Analysis
Backtests the tennis model against historical data calculating edge across different tournament tiers.
Analyzes performance for: moneyline, spread, and total sets markets.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.btl_model import TennisBTLModel

class EnhancedTennisModelBacktest:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = TennisBTLModel(data_path)
        self.results = {'men': [], 'women': []}
        self.tier_analysis = {'men': {}, 'women': {}}
        
    def detect_column_names(self, df):
        """Detect actual column names in the dataframe"""
        column_map = {}
        
        # Find Winner/Loser columns
        for col in df.columns:
            if 'winner' in col.lower() or col == 'Winner':
                column_map['winner'] = col
            elif 'loser' in col.lower() or col == 'Loser':
                column_map['loser'] = col
            elif 'surface' in col.lower() or col == 'Surface':
                column_map['surface'] = col
            elif any(x in col.lower() for x in ['date', 'data']):
                column_map['date'] = col
            elif 'round' in col.lower() or col == 'Round':
                column_map['round'] = col
            elif 'series' in col.lower() or col == 'Series':
                column_map['series'] = col
            elif 'tier' in col.lower() or col == 'Tier':
                column_map['tier'] = col
            elif 'best' in col.lower() and 'of' in col.lower():
                column_map['best_of'] = col
        
        # For tier analysis, use either 'tier' or 'series' column
        if 'tier' not in column_map and 'series' in column_map:
            column_map['tier'] = column_map['series']
        
        # Find betting odds columns
        betting_cols = {}
        for col in df.columns:
            if 'b365w' in col.lower():
                betting_cols['b365_winner'] = col
            elif 'b365l' in col.lower():
                betting_cols['b365_loser'] = col
            elif 'psw' in col.lower():
                betting_cols['ps_winner'] = col
            elif 'psl' in col.lower():
                betting_cols['ps_loser'] = col
            # Add more betting markets
            elif 'b365_over' in col.lower() or 'b365over' in col.lower():
                betting_cols['b365_over'] = col
            elif 'b365_under' in col.lower() or 'b365under' in col.lower():
                betting_cols['b365_under'] = col
            elif any(x in col.lower() for x in ['pinnover', 'ps_over', 'psover']):
                betting_cols['ps_over'] = col
            elif any(x in col.lower() for x in ['pinnunder', 'ps_under', 'psunder']):
                betting_cols['ps_under'] = col
        
        # Find set score columns
        set_cols = {}
        for i in range(1, 6):  # W1, L1, W2, L2, etc.
            for col in df.columns:
                if col.lower() == f'w{i}':
                    set_cols[f'w{i}'] = col
                elif col.lower() == f'l{i}':
                    set_cols[f'l{i}'] = col
        
        return column_map, betting_cols, set_cols
    

    def calculate_total_sets(self, row, set_cols):
        """Calculate total sets played in match"""
        total_sets = 0
        for i in range(1, 6):
            w_col = set_cols.get(f'w{i}')
            l_col = set_cols.get(f'l{i}')
            
            if w_col and l_col and w_col in row and l_col in row:
                if pd.notna(row[w_col]) and pd.notna(row[l_col]):
                    if row[w_col] > 0 or row[l_col] > 0:
                        total_sets += 1
                    else:
                        break
                else:
                    break
            else:
                break
        
        return max(2, total_sets)  # Minimum 2 sets
    
    def get_betting_probabilities(self, row, betting_cols):
        """Extract betting probabilities from odds"""
        probs = {}
        
        # Bet365 moneyline odds
        if 'b365_winner' in betting_cols and 'b365_loser' in betting_cols:
            b365w = betting_cols['b365_winner']
            b365l = betting_cols['b365_loser']
            
            if b365w in row and b365l in row and pd.notna(row[b365w]) and pd.notna(row[b365l]):
                if row[b365w] > 0 and row[b365l] > 0:
                    prob_w = 1 / row[b365w]
                    prob_l = 1 / row[b365l]
                    total = prob_w + prob_l
                    if total > 0:
                        probs['b365_winner_prob'] = prob_w / total
                        probs['b365_loser_prob'] = prob_l / total
                        probs['b365_margin'] = total - 1
        
        # Pinnacle moneyline odds
        if 'ps_winner' in betting_cols and 'ps_loser' in betting_cols:
            psw = betting_cols['ps_winner']
            psl = betting_cols['ps_loser']
            
            if psw in row and psl in row and pd.notna(row[psw]) and pd.notna(row[psl]):
                if row[psw] > 0 and row[psl] > 0:
                    prob_w = 1 / row[psw]
                    prob_l = 1 / row[psl]
                    total = prob_w + prob_l
                    if total > 0:
                        probs['ps_winner_prob'] = prob_w / total
                        probs['ps_loser_prob'] = prob_l / total
                        probs['ps_margin'] = total - 1
        
        # Bet365 totals odds
        if 'b365_over' in betting_cols and 'b365_under' in betting_cols:
            b365_over = betting_cols['b365_over']
            b365_under = betting_cols['b365_under']
            
            if b365_over in row and b365_under in row and pd.notna(row[b365_over]) and pd.notna(row[b365_under]):
                if row[b365_over] > 0 and row[b365_under] > 0:
                    prob_over = 1 / row[b365_over]
                    prob_under = 1 / row[b365_under]
                    total = prob_over + prob_under
                    if total > 0:
                        probs['b365_over_prob'] = prob_over / total
                        probs['b365_under_prob'] = prob_under / total
                        probs['b365_totals_margin'] = total - 1
        
        # Pinnacle totals odds
        if 'ps_over' in betting_cols and 'ps_under' in betting_cols:
            ps_over = betting_cols['ps_over']
            ps_under = betting_cols['ps_under']
            
            if ps_over in row and ps_under in row and pd.notna(row[ps_over]) and pd.notna(row[ps_under]):
                if row[ps_over] > 0 and row[ps_under] > 0:
                    prob_over = 1 / row[ps_over]
                    prob_under = 1 / row[ps_under]
                    total = prob_over + prob_under
                    if total > 0:
                        probs['ps_over_prob'] = prob_over / total
                        probs['ps_under_prob'] = prob_under / total
                        probs['ps_totals_margin'] = total - 1
        
        return probs
    
    def run_backtest(self, gender='both', test_years=[2024, 2025]):
        """Run the backtest for specified gender and years"""
        genders = ['men', 'women'] if gender == 'both' else [gender]
        
        for g in genders:
            print(f"\n🔬 Running backtest for {g}...")
            folder_name = 'Mens' if g == 'men' else 'Womens'
            
            # Load historical data for ratings initialization
            print(f"Loading historical data to initialize ratings...")
            self.model.load_data(g)
            if g == 'men':
                self.model.update_ratings_enhanced('men')
            else:
                self.model.update_ratings_enhanced('women')
            
            all_results = []
            tier_counts = {}
            
            # Process each test year
            for year in test_years:
                file_path = os.path.join(self.data_path, folder_name, f"{year}.xlsx")
                
                if not os.path.exists(file_path):
                    print(f"❌ {year} data not found for {g}: {file_path}")
                    continue
                
                print(f"Processing {year} data...")
                
                # Load test data
                df_test = pd.read_excel(file_path)
                print(f"Initial data shape: {df_test.shape}")
                
                # Don't drop all NaN rows immediately - be more selective
                essential_cols = ['Winner', 'Loser', 'Surface', 'Series']  # Keep essential columns
                df_test = df_test.dropna(subset=[col for col in essential_cols if col in df_test.columns])
                
                # Detect column names
                column_map, betting_cols, set_cols = self.detect_column_names(df_test)
                print(f"Detected columns: {column_map}")
                print(f"Betting columns: {betting_cols}")
                
                # Use tier names directly from data
                if 'tier' in column_map:
                    df_test['RawTier'] = df_test[column_map['tier']].astype(str).fillna('Unknown')
                    # Debug: Print first few tier values
                    print(f"Sample tier values: {df_test['RawTier'].head(10).tolist()}")
                    tier_distribution = df_test['RawTier'].value_counts()
                    print(f"Tier distribution: {tier_distribution.to_dict()}")
                    for tier, count in tier_distribution.items():
                        tier_counts[tier] = tier_counts.get(tier, 0) + count
                else:
                    print("No tier column found!")
                    df_test['RawTier'] = 'Unknown'
                    tier_counts['Unknown'] = tier_counts.get('Unknown', 0) + len(df_test)
                
                # Sort data by date if available
                if 'date' in column_map:
                    df_test['Date'] = pd.to_datetime(df_test[column_map['date']], errors='coerce')
                    df_test = df_test.sort_values('Date', na_position='last')
                
                # Process each match chronologically
                year_results = self.process_matches(df_test, column_map, betting_cols, set_cols, g)
                all_results.extend(year_results)
                
                print(f"✅ Processed {len(year_results)} predictions for {year}")
            
            self.results[g] = all_results
            print(f"✅ Backtest complete for {g}: {len(all_results)} total predictions")
            print(f"Tier distribution: {tier_counts}")
    
    def process_matches(self, df, column_map, betting_cols, set_cols, gender):
        """Process matches and generate predictions"""
        results = []
        
        for idx, match in df.iterrows():
            if idx % 500 == 0:
                print(f"  Processed {idx}/{len(df)} matches...")
            
            try:
                # Get match details
                winner = match[column_map['winner']] if 'winner' in column_map else None
                loser = match[column_map['loser']] if 'loser' in column_map else None
                surface = match[column_map['surface']] if 'surface' in column_map else 'Hard'
                round_name = match[column_map['round']] if 'round' in column_map else ''
                tier = str(match.get('RawTier', 'Unknown')) if pd.notna(match.get('RawTier')) else 'Unknown'
                
                # Debug tier assignment occasionally
                if idx % 500 == 0:
                    print(f"  Match {idx}: {winner} vs {loser}, Tier: {tier}")
                
                if pd.isna(winner) or pd.isna(loser):
                    continue
                
                # Determine best of format
                best_of = 3  # Default
                if 'best_of' in column_map and column_map['best_of'] in match:
                    best_of = match[column_map['best_of']] if pd.notna(match[column_map['best_of']]) else 3
                
                # Calculate total sets
                total_sets = self.calculate_total_sets(match, set_cols)
                
                # Get betting probabilities
                betting_probs = self.get_betting_probabilities(match, betting_cols)
                
                # Make predictions using current ratings
                if winner in self.model.ratings[gender] and loser in self.model.ratings[gender]:
                    # Model prediction for moneyline
                    prediction = self.model.predict_match_enhanced(
                        winner, loser, surface, best_of, gender, round_name
                    )
                    model_prob_winner = prediction['probability']
                    
                    # Predict total sets using model probability and other factors
                    # Use the model's match probability to inform set predictions
                    match_closeness = min(model_prob_winner, 1 - model_prob_winner)  # How close the match is (0.5 = very close, 0 = very one-sided)
                    
                    # Surface effect on set length (clay tends to have longer matches)
                    surface_multiplier = 1.2 if surface.lower() == 'clay' else 1.0
                    
                    # Round effect (later rounds tend to be closer)
                    round_multiplier = 1.1 if any(keyword in round_name.lower() for keyword in ['final', 'semi', 'quarter']) else 1.0
                    
                    if best_of == 3:
                        # Base probability that match goes over 2.5 sets
                        # More even matches (closeness closer to 0.5) are more likely to go 3 sets
                        base_prob_over_2_5 = 0.3 + (match_closeness * 1.4)  # Range: 0.3 to 0.7
                        pred_over_2_5 = min(0.8, base_prob_over_2_5 * surface_multiplier * round_multiplier)
                    else:
                        # For 5-set matches, predict over/under 3.5 sets
                        base_prob_over_3_5 = 0.25 + (match_closeness * 1.0)  # Range: 0.25 to 0.75  
                        pred_over_3_5 = min(0.8, base_prob_over_3_5 * surface_multiplier * round_multiplier)
                    
                    # Store result for winner perspective
                    result = {
                        'date': match.get('Date', datetime.now()),
                        'player': winner,
                        'opponent': loser,
                        'surface': surface,
                        'round': round_name,
                        'tier': tier,
                        'best_of': best_of,
                        'total_sets': total_sets,
                        'player_rating': self.model.ratings[gender][winner],
                        'opponent_rating': self.model.ratings[gender][loser],
                        'rating_diff': self.model.ratings[gender][winner] - self.model.ratings[gender][loser],
                        'model_prob_win': model_prob_winner,
                        'actual_win': 1,  # Winner always wins
                        'actual_total_sets': total_sets
                    }
                    
                    # Add set predictions - Initialize all columns to avoid KeyError later
                    result['pred_over_2_5'] = None
                    result['actual_over_2_5'] = None
                    result['pred_over_3_5'] = None
                    result['actual_over_3_5'] = None
                    
                    if best_of == 3:
                        result['pred_over_2_5'] = pred_over_2_5
                        result['actual_over_2_5'] = 1 if total_sets > 2 else 0
                    else:
                        result['pred_over_3_5'] = pred_over_3_5 if best_of == 5 else None
                        result['actual_over_3_5'] = 1 if total_sets > 3 else 0
                    
                    # Add betting odds probabilities
                    result.update(betting_probs)
                    
                    results.append(result)
                    
                    # Store result for loser perspective
                    result_loser = result.copy()
                    result_loser.update({
                        'player': loser,
                        'opponent': winner,
                        'player_rating': self.model.ratings[gender][loser],
                        'opponent_rating': self.model.ratings[gender][winner],
                        'rating_diff': self.model.ratings[gender][loser] - self.model.ratings[gender][winner],
                        'model_prob_win': 1 - model_prob_winner,
                        'actual_win': 0,  # This player lost
                    })
                    
                    # Swap betting probs for loser perspective
                    if 'b365_winner_prob' in betting_probs:
                        result_loser['b365_winner_prob'] = betting_probs['b365_loser_prob']
                        result_loser['b365_loser_prob'] = betting_probs['b365_winner_prob']
                    if 'ps_winner_prob' in betting_probs:
                        result_loser['ps_winner_prob'] = betting_probs['ps_loser_prob']
                        result_loser['ps_loser_prob'] = betting_probs['ps_winner_prob']
                    
                    results.append(result_loser)
                
                # Update ratings based on actual match result
                self._update_single_match_rating(match, column_map, gender)
                
            except Exception as e:
                print(f"Error processing match {idx}: {e}")
                continue
        
        return results
    
    def _update_single_match_rating(self, match, column_map, gender):
        """Update ratings for a single match"""
        try:
            winner = match[column_map['winner']]
            loser = match[column_map['loser']]
            
            if pd.isna(winner) or pd.isna(loser):
                return
            
            # Get current ratings
            winner_rating = self.model.ratings[gender].get(winner, 1500)
            loser_rating = self.model.ratings[gender].get(loser, 1500)
            
            # Simple Elo update
            expected_winner = 1 / (1 + 10**((loser_rating - winner_rating)/400))
            k_factor = 32
            
            rating_change = k_factor * (1 - expected_winner)
            
            self.model.ratings[gender][winner] = winner_rating + rating_change
            self.model.ratings[gender][loser] = loser_rating - rating_change
            
        except Exception as e:
            pass
    
    def calculate_tier_metrics(self, gender):
        """Calculate metrics broken down by tier and market"""
        if not self.results[gender]:
            return {}
        
        df = pd.DataFrame(self.results[gender])
        tier_metrics = {}
        
        # Get unique tiers
        tiers = df['tier'].unique()
        
        for tier in tiers:
            tier_df = df[df['tier'] == tier]
            if len(tier_df) < 10:  # Skip tiers with too few matches
                continue
            
            tier_metrics[tier] = {}
            
            # Moneyline metrics
            moneyline_metrics = self.calculate_moneyline_metrics(tier_df)
            tier_metrics[tier]['moneyline'] = moneyline_metrics
            
            # Set totals metrics
            set_metrics = self.calculate_set_metrics(tier_df)
            tier_metrics[tier]['sets'] = set_metrics
            
            # Betting comparison
            betting_metrics = self.calculate_betting_comparison(tier_df)
            tier_metrics[tier]['vs_bookmakers'] = betting_metrics
            
        return tier_metrics
    
    def calculate_moneyline_metrics(self, df):
        """Calculate moneyline prediction metrics"""
        if len(df) == 0:
            return {}
        
        try:
            accuracy = (df['actual_win'] == (df['model_prob_win'] > 0.5)).mean()
            log_loss_score = log_loss(df['actual_win'], df['model_prob_win'])
            brier_score = brier_score_loss(df['actual_win'], df['model_prob_win'])
            auc_score = roc_auc_score(df['actual_win'], df['model_prob_win'])
            
            # Calculate edge (difference in accuracy vs random)
            edge_vs_random = accuracy - 0.5
            
            # Calculate performance by confidence levels
            df['confidence'] = np.abs(df['model_prob_win'] - 0.5)
            confidence_bins = pd.qcut(df['confidence'], q=3, labels=['Low', 'Medium', 'High'])
            confidence_performance = df.groupby(confidence_bins).apply(
                lambda x: (x['actual_win'] == (x['model_prob_win'] > 0.5)).mean()
            ).to_dict()
            
            return {
                'sample_size': len(df),
                'accuracy': accuracy,
                'edge_vs_random': edge_vs_random,
                'log_loss': log_loss_score,
                'brier_score': brier_score,
                'auc': auc_score,
                'confidence_performance': confidence_performance
            }
        except Exception as e:
            return {'error': str(e), 'sample_size': len(df)}
    
    def calculate_set_metrics(self, df):
        """Calculate set total prediction metrics"""
        metrics = {}
        
        # Over/Under 2.5 sets (for 3-set matches)
        if 'pred_over_2_5' in df.columns:
            over_2_5_df = df[df['pred_over_2_5'].notna()]
            if len(over_2_5_df) > 0:
                try:
                    accuracy = (over_2_5_df['actual_over_2_5'] == (over_2_5_df['pred_over_2_5'] > 0.5)).mean()
                    log_loss_score = log_loss(over_2_5_df['actual_over_2_5'], over_2_5_df['pred_over_2_5'])
                    edge_vs_random = accuracy - 0.5
                    
                    metrics['over_2_5'] = {
                        'sample_size': len(over_2_5_df),
                        'accuracy': accuracy,
                        'edge_vs_random': edge_vs_random,
                        'log_loss': log_loss_score
                    }
                except:
                    metrics['over_2_5'] = {'error': 'Calculation failed', 'sample_size': len(over_2_5_df)}
        
        # Over/Under 3.5 sets (for 5-set matches)
        if 'pred_over_3_5' in df.columns:
            over_3_5_df = df[df['pred_over_3_5'].notna()]
            if len(over_3_5_df) > 0:
                try:
                    accuracy = (over_3_5_df['actual_over_3_5'] == (over_3_5_df['pred_over_3_5'] > 0.5)).mean()
                    log_loss_score = log_loss(over_3_5_df['actual_over_3_5'], over_3_5_df['pred_over_3_5'])
                    edge_vs_random = accuracy - 0.5
                    
                    metrics['over_3_5'] = {
                        'sample_size': len(over_3_5_df),
                        'accuracy': accuracy,
                        'edge_vs_random': edge_vs_random,
                        'log_loss': log_loss_score
                    }
                except:
                    metrics['over_3_5'] = {'error': 'Calculation failed', 'sample_size': len(over_3_5_df)}
        
        return metrics
    
    def calculate_betting_comparison(self, df):
        """Compare model performance vs bookmakers"""
        comparison = {}
        
        for bookmaker in ['b365', 'ps']:
            # Moneyline comparison
            prob_col = f'{bookmaker}_winner_prob'
            if prob_col in df.columns and df[prob_col].notna().sum() > 10:
                valid_mask = df[prob_col].notna()
                betting_df = df[valid_mask]
                
                model_acc = (betting_df['actual_win'] == (betting_df['model_prob_win'] > 0.5)).mean()
                betting_acc = (betting_df['actual_win'] == (betting_df[prob_col] > 0.5)).mean()
                
                try:
                    model_log_loss = log_loss(betting_df['actual_win'], betting_df['model_prob_win'])
                    betting_log_loss = log_loss(betting_df['actual_win'], betting_df[prob_col])
                except:
                    model_log_loss = betting_log_loss = np.nan
                
                comparison[f'{bookmaker}_moneyline'] = {
                    'model_accuracy': model_acc,
                    'bookmaker_accuracy': betting_acc,
                    'edge': model_acc - betting_acc,
                    'model_log_loss': model_log_loss,
                    'bookmaker_log_loss': betting_log_loss,
                    'sample_size': len(betting_df)
                }
            
            # Totals comparison
            over_prob_col = f'{bookmaker}_over_prob'
            if over_prob_col in df.columns and df[over_prob_col].notna().sum() > 10:
                valid_mask = df[over_prob_col].notna() & df['pred_over_2_5'].notna()
                betting_df = df[valid_mask]
                
                if len(betting_df) > 0:
                    model_acc = (betting_df['actual_over_2_5'] == (betting_df['pred_over_2_5'] > 0.5)).mean()
                    betting_acc = (betting_df['actual_over_2_5'] == (betting_df[over_prob_col] > 0.5)).mean()
                    
                    comparison[f'{bookmaker}_totals'] = {
                        'model_accuracy': model_acc,
                        'bookmaker_accuracy': betting_acc,
                        'edge': model_acc - betting_acc,
                        'sample_size': len(betting_df)
                    }
        
        return comparison
    
    def create_tier_visualizations(self, gender):
        """Create tier-based performance visualizations"""
        if not self.results[gender]:
            print(f"No results available for {gender}")
            return
        
        tier_metrics = self.calculate_tier_metrics(gender)
        if not tier_metrics:
            print(f"No tier metrics available for {gender}")
            return
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'{gender.title()} Tennis Model Performance by Tournament Tier', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        tiers = list(tier_metrics.keys())
        moneyline_data = []
        sets_data = []
        edge_data = []
        
        for tier in tiers:
            if 'moneyline' in tier_metrics[tier]:
                ml_metrics = tier_metrics[tier]['moneyline']
                moneyline_data.append({
                    'Tier': tier,
                    'Accuracy': ml_metrics.get('accuracy', 0),
                    'Edge': ml_metrics.get('edge_vs_random', 0),
                    'Sample_Size': ml_metrics.get('sample_size', 0),
                    'AUC': ml_metrics.get('auc', 0.5)
                })
            
            if 'sets' in tier_metrics[tier]:
                set_metrics = tier_metrics[tier]['sets']
                if 'over_2_5' in set_metrics:
                    sets_data.append({
                        'Tier': tier,
                        'Market': 'Over 2.5 Sets',
                        'Accuracy': set_metrics['over_2_5'].get('accuracy', 0),
                        'Edge': set_metrics['over_2_5'].get('edge_vs_random', 0),
                        'Sample_Size': set_metrics['over_2_5'].get('sample_size', 0)
                    })
                if 'over_3_5' in set_metrics:
                    sets_data.append({
                        'Tier': tier,
                        'Market': 'Over 3.5 Sets',
                        'Accuracy': set_metrics['over_3_5'].get('accuracy', 0),
                        'Edge': set_metrics['over_3_5'].get('edge_vs_random', 0),
                        'Sample_Size': set_metrics['over_3_5'].get('sample_size', 0)
                    })
        
        # 1. Moneyline Accuracy by Tier
        if moneyline_data:
            ml_df = pd.DataFrame(moneyline_data)
            bars = axes[0, 0].bar(ml_df['Tier'], ml_df['Accuracy'])
            axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
            axes[0, 0].set_title('Moneyline Accuracy by Tier')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add sample size annotations
            for bar, size in zip(bars, ml_df['Sample_Size']):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'n={size}', ha='center', va='bottom', fontsize=8)
        
        # 2. Edge vs Random by Tier
        if moneyline_data:
            bars = axes[0, 1].bar(ml_df['Tier'], ml_df['Edge'], 
                                 color=['green' if x > 0 else 'red' for x in ml_df['Edge']])
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[0, 1].set_title('Edge vs Random (Moneyline)')
            axes[0, 1].set_ylabel('Edge (accuracy - 0.5)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. AUC by Tier
        if moneyline_data:
            bars = axes[0, 2].bar(ml_df['Tier'], ml_df['AUC'])
            axes[0, 2].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.5)')
            axes[0, 2].set_title('AUC by Tier')
            axes[0, 2].set_ylabel('AUC Score')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Set Totals Performance
        if sets_data:
            sets_df = pd.DataFrame(sets_data)
            pivot_df = sets_df.pivot(index='Tier', columns='Market', values='Accuracy')
            pivot_df.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
            axes[1, 0].set_title('Set Totals Accuracy by Tier')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Bookmaker Comparison (if available)
        self.plot_bookmaker_comparison(axes[1, 1], tier_metrics, 'Moneyline vs Bookmakers')
        
        # 6. Sample Size Distribution
        if moneyline_data:
            axes[1, 2].pie(ml_df['Sample_Size'], labels=ml_df['Tier'], autopct='%1.1f%%')
            axes[1, 2].set_title('Sample Size Distribution by Tier')
        
        # 7. Edge Comparison: Model vs Bookmakers
        self.plot_edge_comparison(axes[2, 0], tier_metrics)
        
        # 8. Performance Consistency
        self.plot_performance_consistency(axes[2, 1], gender)
        
        # 9. Tier Difficulty Analysis
        self.plot_tier_difficulty(axes[2, 2], tier_metrics)
        
        plt.tight_layout()
        plt.savefig(f'plots/{gender}_tier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_bookmaker_comparison(self, ax, tier_metrics, title):
        """Plot comparison with bookmakers"""
        comparison_data = []
        
        for tier, metrics in tier_metrics.items():
            if 'vs_bookmakers' in metrics:
                for market, comp in metrics['vs_bookmakers'].items():
                    if 'edge' in comp:
                        comparison_data.append({
                            'Tier': tier,
                            'Market': market,
                            'Edge': comp['edge'],
                            'Sample_Size': comp.get('sample_size', 0)
                        })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            # Plot edge vs bookmakers
            for market in comp_df['Market'].unique():
                market_data = comp_df[comp_df['Market'] == market]
                ax.plot(market_data['Tier'], market_data['Edge'], marker='o', label=market)
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.set_title(title)
            ax.set_ylabel('Edge vs Bookmaker')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No bookmaker\ncomparison data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    def plot_edge_comparison(self, ax, tier_metrics):
        """Plot edge comparison across tiers"""
        edge_data = []
        for tier, metrics in tier_metrics.items():
            if 'moneyline' in metrics:
                edge_data.append({
                    'Tier': tier,
                    'Moneyline_Edge': metrics['moneyline'].get('edge_vs_random', 0)
                })
            
            if 'sets' in metrics and 'over_2_5' in metrics['sets']:
                if edge_data and edge_data[-1]['Tier'] == tier:
                    edge_data[-1]['Sets_Edge'] = metrics['sets']['over_2_5'].get('edge_vs_random', 0)
                else:
                    edge_data.append({
                        'Tier': tier,
                        'Sets_Edge': metrics['sets']['over_2_5'].get('edge_vs_random', 0)
                    })
        
        if edge_data:
            edge_df = pd.DataFrame(edge_data)
            x = np.arange(len(edge_df))
            width = 0.35
            
            if 'Moneyline_Edge' in edge_df.columns:
                ax.bar(x - width/2, edge_df['Moneyline_Edge'], width, label='Moneyline', alpha=0.8)
            if 'Sets_Edge' in edge_df.columns:
                ax.bar(x + width/2, edge_df['Sets_Edge'], width, label='Sets', alpha=0.8)
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.set_title('Edge Comparison by Market')
            ax.set_ylabel('Edge vs Random')
            ax.set_xticks(x)
            ax.set_xticklabels(edge_df['Tier'], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def plot_performance_consistency(self, ax, gender):
        """Plot performance consistency over time"""
        df = pd.DataFrame(self.results[gender])
        if 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
            monthly_performance = df.groupby(['month', 'tier']).apply(
                lambda x: (x['actual_win'] == (x['model_prob_win'] > 0.5)).mean()
            ).unstack(fill_value=np.nan)
            
            if len(monthly_performance) > 1:
                monthly_performance.plot(ax=ax, marker='o')
                ax.set_title('Monthly Performance by Tier')
                ax.set_ylabel('Accuracy')
                ax.tick_params(axis='x', rotation=45)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient data\nfor time analysis', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Monthly Performance by Tier')
    
    def plot_tier_difficulty(self, ax, tier_metrics):
        """Plot tier difficulty analysis"""
        difficulty_data = []
        for tier, metrics in tier_metrics.items():
            if 'moneyline' in metrics:
                # Use inverse of edge as difficulty proxy (lower edge = more difficult)
                difficulty = 1 / (metrics['moneyline'].get('edge_vs_random', 0.01) + 0.01)
                difficulty_data.append({
                    'Tier': tier,
                    'Difficulty': difficulty,
                    'Sample_Size': metrics['moneyline'].get('sample_size', 0)
                })
        
        if difficulty_data:
            diff_df = pd.DataFrame(difficulty_data)
            bars = ax.bar(diff_df['Tier'], diff_df['Difficulty'])
            ax.set_title('Prediction Difficulty by Tier')
            ax.set_ylabel('Difficulty Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    def generate_tier_report(self):
        """Generate comprehensive tier analysis report"""
        print("\n" + "="*100)
        print("🎾 ENHANCED TENNIS MODEL TIER ANALYSIS REPORT")
        print("="*100)
        print(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analysis Period: Out-of-sample backtesting")
        print(f"Model: Enhanced Elo with surface expertise and recent form")
        
        for gender in ['men', 'women']:
            if not self.results[gender]:
                continue
                
            print(f"\n📊 {gender.upper()} TIER ANALYSIS")
            print("-" * 80)
            
            tier_metrics = self.calculate_tier_metrics(gender)
            
            # Overall summary
            total_predictions = len(self.results[gender])
            overall_accuracy = np.mean([r['actual_win'] == (r['model_prob_win'] > 0.5) for r in self.results[gender]])
            
            print(f"Total Predictions: {total_predictions:,}")
            print(f"Overall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
            print(f"Overall Edge vs Random: {overall_accuracy - 0.5:+.3f} ({(overall_accuracy - 0.5)*100:+.1f}%)")
            
            # Tier-by-tier analysis
            print(f"\n🏆 TIER-BY-TIER BREAKDOWN:")
            print("-" * 50)
            
            for tier, metrics in tier_metrics.items():
                print(f"\n{tier}:")
                
                # Moneyline performance
                if 'moneyline' in metrics:
                    ml = metrics['moneyline']
                    print(f"  📈 MONEYLINE:")
                    print(f"    Sample Size: {ml.get('sample_size', 0):,}")
                    print(f"    Accuracy: {ml.get('accuracy', 0):.3f} ({ml.get('accuracy', 0)*100:.1f}%)")
                    print(f"    Edge vs Random: {ml.get('edge_vs_random', 0):+.3f} ({ml.get('edge_vs_random', 0)*100:+.1f}%)")
                    print(f"    AUC: {ml.get('auc', 0.5):.3f}")
                    print(f"    Log Loss: {ml.get('log_loss', 0):.4f}")
                
                # Set totals performance
                if 'sets' in metrics:
                    sets = metrics['sets']
                    print(f"  🎯 SET TOTALS:")
                    
                    if 'over_2_5' in sets:
                        o25 = sets['over_2_5']
                        if 'accuracy' in o25:
                            print(f"    Over 2.5 Sets:")
                            print(f"      Sample Size: {o25.get('sample_size', 0):,}")
                            print(f"      Accuracy: {o25.get('accuracy', 0):.3f} ({o25.get('accuracy', 0)*100:.1f}%)")
                            print(f"      Edge vs Random: {o25.get('edge_vs_random', 0):+.3f} ({o25.get('edge_vs_random', 0)*100:+.1f}%)")
                    
                    if 'over_3_5' in sets:
                        o35 = sets['over_3_5']
                        if 'accuracy' in o35:
                            print(f"    Over 3.5 Sets:")
                            print(f"      Sample Size: {o35.get('sample_size', 0):,}")
                            print(f"      Accuracy: {o35.get('accuracy', 0):.3f} ({o35.get('accuracy', 0)*100:.1f}%)")
                            print(f"      Edge vs Random: {o35.get('edge_vs_random', 0):+.3f} ({o35.get('edge_vs_random', 0)*100:+.1f}%)")
                
                # Bookmaker comparison
                if 'vs_bookmakers' in metrics:
                    print(f"  🏦 VS BOOKMAKERS:")
                    for market, comp in metrics['vs_bookmakers'].items():
                        if 'edge' in comp:
                            print(f"    {market.replace('_', ' ').title()}:")
                            print(f"      Model Accuracy: {comp.get('model_accuracy', 0):.3f}")
                            print(f"      Bookmaker Accuracy: {comp.get('bookmaker_accuracy', 0):.3f}")
                            print(f"      Edge vs Bookmaker: {comp.get('edge', 0):+.3f} ({comp.get('edge', 0)*100:+.1f}%)")
                            print(f"      Sample Size: {comp.get('sample_size', 0):,}")
            
            # Create visualizations
            self.create_tier_visualizations(gender)
            
            # Performance ranking
            print(f"\n🥇 TIER PERFORMANCE RANKING (by moneyline accuracy):")
            print("-" * 50)
            
            tier_ranking = []
            for tier, metrics in tier_metrics.items():
                if 'moneyline' in metrics and 'accuracy' in metrics['moneyline']:
                    tier_ranking.append((tier, metrics['moneyline']['accuracy'], 
                                       metrics['moneyline']['sample_size']))
            
            tier_ranking.sort(key=lambda x: x[1], reverse=True)
            for i, (tier, acc, size) in enumerate(tier_ranking, 1):
                print(f"  {i}. {tier}: {acc:.3f} ({acc*100:.1f}%) - {size:,} predictions")
        
        # Cross-gender comparison
        if self.results['men'] and self.results['women']:
            print(f"\n⚡ CROSS-GENDER COMPARISON")
            print("-" * 50)
            
            men_metrics = self.calculate_tier_metrics('men')
            women_metrics = self.calculate_tier_metrics('women')
            
            common_tiers = set(men_metrics.keys()) & set(women_metrics.keys())
            
            for tier in common_tiers:
                print(f"\n{tier}:")
                men_acc = men_metrics[tier]['moneyline'].get('accuracy', 0)
                women_acc = women_metrics[tier]['moneyline'].get('accuracy', 0)
                
                print(f"  Men's Accuracy: {men_acc:.3f} ({men_acc*100:.1f}%)")
                print(f"  Women's Accuracy: {women_acc:.3f} ({women_acc*100:.1f}%)")
                print(f"  Difference: {men_acc - women_acc:+.3f} ({(men_acc - women_acc)*100:+.1f}%)")
        
        print(f"\n💾 Visualizations saved to plots/ directory")
        print(f"✅ Enhanced tier analysis complete!")

def main():
    print("🎾 Enhanced Tennis Model Backtest with Tier Analysis")
    print("=" * 60)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Initialize backtest
    data_path = os.path.join(os.path.dirname(__file__), "Data")
    backtest = EnhancedTennisModelBacktest(data_path)
    
    # Run backtest
    backtest.run_backtest('both', test_years=[2024, 2025])
    
    # Generate comprehensive report
    backtest.generate_tier_report()
    
    print("\n✅ Enhanced backtest complete!")

if __name__ == "__main__":
    main()