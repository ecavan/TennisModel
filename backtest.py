#!/usr/bin/env python3
"""
Tennis Model Backtest Script
Backtests the tennis model against 2025 data without using future information.
Compares model predictions vs actual outcomes and betting odds.
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

class TennisModelBacktest:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = TennisBTLModel(data_path)
        self.results = {'men': [], 'women': []}
        
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
        
        # Bet365 odds
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
        
        # Pinnacle odds
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
        
        return probs
    
    def run_backtest(self, gender='both'):
        """Run the backtest for specified gender"""
        genders = ['men', 'women'] if gender == 'both' else [gender]
        
        for g in genders:
            print(f"\n🔬 Running backtest for {g}...")
            folder_name = 'Mens' if g == 'men' else 'Womens'
            file_path = os.path.join(self.data_path, folder_name, "2025.xlsx")
            
            if not os.path.exists(file_path):
                print(f"❌ 2025 data not found for {g}: {file_path}")
                continue
            
            # Load 2025 data
            df_2025 = pd.read_excel(file_path)
            df_2025 = df_2025.dropna()
            
            # Detect column names
            column_map, betting_cols, set_cols = self.detect_column_names(df_2025)
            print(f"Detected columns: {column_map}")
            print(f"Betting columns: {betting_cols}")
            
            # Load historical data (2016-2024) to initialize ratings
            print(f"Loading historical data to initialize ratings...")
            self.model.load_data(g)
            if g == 'men':
                self.model.update_ratings_enhanced('men')
            else:
                self.model.update_ratings_enhanced('women')
            
            print(f"Initial ratings calculated. Starting backtest...")
            
            # Sort 2025 data by date
            if 'date' in column_map:
                df_2025['Date'] = pd.to_datetime(df_2025[column_map['date']], errors='coerce')
                df_2025 = df_2025.sort_values('Date', na_position='last')
            
            backtest_results = []
            
            # Process each match chronologically
            for idx, match in df_2025.iterrows():
                if idx % 100 == 0:
                    print(f"  Processed {idx}/{len(df_2025)} matches...")
                
                # Get match details
                try:
                    winner = match[column_map['winner']] if 'winner' in column_map else None
                    loser = match[column_map['loser']] if 'loser' in column_map else None
                    surface = match[column_map['surface']] if 'surface' in column_map else 'Hard'
                    round_name = match[column_map['round']] if 'round' in column_map else ''
                    
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
                    
                    # Make prediction using current ratings (no future data)
                    if winner in self.model.ratings[g] and loser in self.model.ratings[g]:
                        # Model prediction
                        prediction = self.model.predict_match_enhanced(
                            winner, loser, surface, best_of, g, round_name
                        )
                        model_prob_winner = prediction['probability']
                        
                        # Predict total sets
                        # Simple heuristic: closer ratings = more sets
                        rating_diff = abs(self.model.ratings[g][winner] - self.model.ratings[g][loser])
                        if best_of == 3:
                            if rating_diff < 100:
                                pred_over_2_5 = 0.65  # Close match likely goes 3 sets
                            else:
                                pred_over_2_5 = 0.35  # One-sided likely 2 sets
                        else:
                            if rating_diff < 100:
                                pred_over_3_5 = 0.55  # Close 5-set match
                            else:
                                pred_over_3_5 = 0.40  # One-sided 5-set match
                        
                        # Store result
                        result = {
                            'date': match.get('Date', datetime.now()),
                            'winner': winner,
                            'loser': loser,
                            'surface': surface,
                            'round': round_name,
                            'best_of': best_of,
                            'total_sets': total_sets,
                            'winner_rating': self.model.ratings[g][winner],
                            'loser_rating': self.model.ratings[g][loser],
                            'model_prob_winner': model_prob_winner,
                            'model_prob_loser': 1 - model_prob_winner,
                            'actual_winner': 1,  # Winner always wins by definition
                            'actual_total_sets': total_sets
                        }
                        
                        # Add set predictions
                        if best_of == 3:
                            result['pred_over_2_5'] = pred_over_2_5
                            result['actual_over_2_5'] = 1 if total_sets > 2 else 0
                        else:
                            result['pred_over_3_5'] = pred_over_3_5
                            result['actual_over_3_5'] = 1 if total_sets > 3 else 0
                        
                        # Add betting odds probabilities
                        result.update(betting_probs)
                        
                        backtest_results.append(result)
                        
                        # Now add the loser's perspective (they lost)
                        result_loser = result.copy()
                        result_loser.update({
                            'winner': loser,
                            'loser': winner,
                            'winner_rating': self.model.ratings[g][loser],
                            'loser_rating': self.model.ratings[g][winner],
                            'model_prob_winner': 1 - model_prob_winner,
                            'model_prob_loser': model_prob_winner,
                            'actual_winner': 0,  # This player lost
                        })
                        
                        # Swap betting probs for loser perspective
                        if 'b365_winner_prob' in betting_probs:
                            result_loser['b365_winner_prob'] = betting_probs['b365_loser_prob']
                            result_loser['b365_loser_prob'] = betting_probs['b365_winner_prob']
                        if 'ps_winner_prob' in betting_probs:
                            result_loser['ps_winner_prob'] = betting_probs['ps_loser_prob']
                            result_loser['ps_loser_prob'] = betting_probs['ps_winner_prob']
                        
                        backtest_results.append(result_loser)
                    
                    # Update ratings based on actual match result
                    self._update_single_match_rating(match, column_map, g)
                    
                except Exception as e:
                    print(f"Error processing match {idx}: {e}")
                    continue
            
            self.results[g] = backtest_results
            print(f"✅ Backtest complete for {g}: {len(backtest_results)} predictions")
    
    def _update_single_match_rating(self, match, column_map, gender):
        """Update ratings for a single match"""
        try:
            winner = match[column_map['winner']]
            loser = match[column_map['loser']]
            surface = match[column_map['surface']] if 'surface' in column_map else 'Hard'
            
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
            pass  # Skip problematic matches
    
    def calculate_metrics(self, gender):
        """Calculate backtest metrics"""
        if not self.results[gender]:
            return {}
        
        df = pd.DataFrame(self.results[gender])
        
        # Basic accuracy
        accuracy = (df['actual_winner'] == (df['model_prob_winner'] > 0.5)).mean()
        
        # Probabilistic metrics
        try:
            log_loss_score = log_loss(df['actual_winner'], df['model_prob_winner'])
            brier_score = brier_score_loss(df['actual_winner'], df['model_prob_winner'])
            auc_score = roc_auc_score(df['actual_winner'], df['model_prob_winner'])
        except:
            log_loss_score = brier_score = auc_score = np.nan
        
        # Betting comparison (if available)
        betting_metrics = {}
        for bookmaker in ['b365', 'ps']:
            prob_col = f'{bookmaker}_winner_prob'
            if prob_col in df.columns and df[prob_col].notna().sum() > 0:
                valid_mask = df[prob_col].notna()
                betting_df = df[valid_mask]
                
                if len(betting_df) > 0:
                    betting_accuracy = (betting_df['actual_winner'] == (betting_df[prob_col] > 0.5)).mean()
                    try:
                        betting_log_loss = log_loss(betting_df['actual_winner'], betting_df[prob_col])
                        betting_brier = brier_score_loss(betting_df['actual_winner'], betting_df[prob_col])
                    except:
                        betting_log_loss = betting_brier = np.nan
                    
                    betting_metrics[bookmaker] = {
                        'accuracy': betting_accuracy,
                        'log_loss': betting_log_loss,
                        'brier_score': betting_brier,
                        'sample_size': len(betting_df)
                    }
        
        # Set predictions
        set_metrics = {}
        if 'pred_over_2_5' in df.columns:
            valid_sets = df['pred_over_2_5'].notna()
            if valid_sets.sum() > 0:
                set_df = df[valid_sets]
                set_accuracy = (set_df['actual_over_2_5'] == (set_df['pred_over_2_5'] > 0.5)).mean()
                try:
                    set_log_loss = log_loss(set_df['actual_over_2_5'], set_df['pred_over_2_5'])
                except:
                    set_log_loss = np.nan
                set_metrics['over_2_5'] = {
                    'accuracy': set_accuracy,
                    'log_loss': set_log_loss,
                    'sample_size': len(set_df)
                }
        
        if 'pred_over_3_5' in df.columns:
            valid_sets = df['pred_over_3_5'].notna()
            if valid_sets.sum() > 0:
                set_df = df[valid_sets]
                set_accuracy = (set_df['actual_over_3_5'] == (set_df['pred_over_3_5'] > 0.5)).mean()
                try:
                    set_log_loss = log_loss(set_df['actual_over_3_5'], set_df['pred_over_3_5'])
                except:
                    set_log_loss = np.nan
                set_metrics['over_3_5'] = {
                    'accuracy': set_accuracy,
                    'log_loss': set_log_loss,
                    'sample_size': len(set_df)
                }
        
        return {
            'sample_size': len(df),
            'accuracy': accuracy,
            'log_loss': log_loss_score,
            'brier_score': brier_score,
            'auc': auc_score,
            'betting_comparison': betting_metrics,
            'set_predictions': set_metrics
        }
    
    def create_visualizations(self, gender):
        """Create backtest visualizations"""
        if not self.results[gender]:
            print(f"No results available for {gender}")
            return
        
        df = pd.DataFrame(self.results[gender])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{gender.title()} Tennis Model Backtest Results', fontsize=16, fontweight='bold')
        
        # 1. Calibration Plot
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                df['actual_winner'], df['model_prob_winner'], n_bins=10
            )
            axes[0, 0].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            axes[0, 0].plot([0, 1], [0, 1], "k:", alpha=0.7, label="Perfect Calibration")
            axes[0, 0].set_xlabel('Mean Predicted Probability')
            axes[0, 0].set_ylabel('Fraction of Positives')
            axes[0, 0].set_title('Calibration Plot')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        except:
            axes[0, 0].text(0.5, 0.5, 'Calibration plot\nnot available', ha='center', va='center')
            axes[0, 0].set_title('Calibration Plot')
        
        # 2. Prediction Distribution
        axes[0, 1].hist(df['model_prob_winner'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
        axes[0, 1].set_xlabel('Model Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Prediction Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Accuracy by Confidence
        try:
            confidence_bins = pd.cut(np.abs(df['model_prob_winner'] - 0.5), bins=5)
            accuracy_by_conf = df.groupby(confidence_bins).apply(
                lambda x: (x['actual_winner'] == (x['model_prob_winner'] > 0.5)).mean()
            )
            accuracy_by_conf.plot(kind='bar', ax=axes[0, 2], rot=45)
            axes[0, 2].set_title('Accuracy by Confidence Level')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].grid(True, alpha=0.3)
        except:
            axes[0, 2].text(0.5, 0.5, 'Confidence analysis\nnot available', ha='center', va='center')
            axes[0, 2].set_title('Accuracy by Confidence Level')
        
        # 4. Model vs Betting Odds Comparison
        betting_comparison_data = []
        for bookmaker in ['b365', 'ps']:
            prob_col = f'{bookmaker}_winner_prob'
            if prob_col in df.columns and df[prob_col].notna().sum() > 10:
                valid_mask = df[prob_col].notna()
                betting_df = df[valid_mask]
                
                model_acc = (betting_df['actual_winner'] == (betting_df['model_prob_winner'] > 0.5)).mean()
                betting_acc = (betting_df['actual_winner'] == (betting_df[prob_col] > 0.5)).mean()
                
                betting_comparison_data.append({
                    'Bookmaker': bookmaker.upper(),
                    'Model': model_acc,
                    'Bookmaker_Odds': betting_acc
                })
        
        if betting_comparison_data:
            comp_df = pd.DataFrame(betting_comparison_data)
            x = np.arange(len(comp_df))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, comp_df['Model'], width, label='Model', alpha=0.8)
            axes[1, 0].bar(x + width/2, comp_df['Bookmaker_Odds'], width, label='Bookmaker', alpha=0.8)
            axes[1, 0].set_xlabel('Bookmaker')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Model vs Bookmaker Accuracy')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(comp_df['Bookmaker'])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No betting odds\navailable', ha='center', va='center')
            axes[1, 0].set_title('Model vs Bookmaker Accuracy')
        
        # 5. Performance Over Time
        try:
            df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
            monthly_acc = df.groupby('month').apply(
                lambda x: (x['actual_winner'] == (x['model_prob_winner'] > 0.5)).mean()
            )
            if len(monthly_acc) > 1:
                monthly_acc.plot(ax=axes[1, 1], marker='o')
                axes[1, 1].set_title('Model Accuracy Over Time')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor time analysis', ha='center', va='center')
                axes[1, 1].set_title('Model Accuracy Over Time')
        except:
            axes[1, 1].text(0.5, 0.5, 'Time analysis\nnot available', ha='center', va='center')
            axes[1, 1].set_title('Model Accuracy Over Time')
        
        # 6. Set Predictions (if available)
        set_data = []
        if 'pred_over_2_5' in df.columns:
            valid_sets = df['pred_over_2_5'].notna()
            if valid_sets.sum() > 0:
                set_df = df[valid_sets]
                acc_2_5 = (set_df['actual_over_2_5'] == (set_df['pred_over_2_5'] > 0.5)).mean()
                set_data.append({'Type': 'Over 2.5 Sets', 'Accuracy': acc_2_5})
        
        if 'pred_over_3_5' in df.columns:
            valid_sets = df['pred_over_3_5'].notna()
            if valid_sets.sum() > 0:
                set_df = df[valid_sets]
                acc_3_5 = (set_df['actual_over_3_5'] == (set_df['pred_over_3_5'] > 0.5)).mean()
                set_data.append({'Type': 'Over 3.5 Sets', 'Accuracy': acc_3_5})
        
        if set_data:
            set_df_plot = pd.DataFrame(set_data)
            set_df_plot.plot(x='Type', y='Accuracy', kind='bar', ax=axes[1, 2], rot=45)
            axes[1, 2].set_title('Set Prediction Accuracy')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No set prediction\ndata available', ha='center', va='center')
            axes[1, 2].set_title('Set Prediction Accuracy')
        
        plt.tight_layout()
        plt.savefig(f'{gender}_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive backtest report"""
        print("\n" + "="*80)
        print("🎾 TENNIS MODEL BACKTEST REPORT")
        print("="*80)
        print(f"Backtest Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Period: 2025 (out-of-sample)")
        print(f"Model: Enhanced Elo with surface expertise and recent form")
        
        for gender in ['men', 'women']:
            print(f"\n📊 {gender.upper()} RESULTS")
            print("-" * 50)
            
            metrics = self.calculate_metrics(gender)
            
            if not metrics:
                print("❌ No results available")
                continue
            
            print(f"Sample Size: {metrics['sample_size']:,} predictions")
            print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
            print(f"Log Loss: {metrics['log_loss']:.4f}")
            print(f"Brier Score: {metrics['brier_score']:.4f}")
            print(f"AUC: {metrics['auc']:.3f}")
            
            # Betting comparison
            if metrics['betting_comparison']:
                print(f"\n🏦 Betting Odds Comparison:")
                for bookmaker, stats in metrics['betting_comparison'].items():
                    print(f"  {bookmaker.upper()}:")
                    print(f"    Model Accuracy: {metrics['accuracy']:.3f}")
                    print(f"    Bookmaker Accuracy: {stats['accuracy']:.3f}")
                    print(f"    Model Log Loss: {metrics['log_loss']:.4f}")
                    print(f"    Bookmaker Log Loss: {stats['log_loss']:.4f}")
                    edge = metrics['accuracy'] - stats['accuracy']
                    print(f"    Model Edge: {edge:+.3f} ({edge*100:+.1f}%)")
                    print(f"    Sample Size: {stats['sample_size']:,}")
            
            # Set predictions
            if metrics['set_predictions']:
                print(f"\n🎯 Set Prediction Performance:")
                for pred_type, stats in metrics['set_predictions'].items():
                    print(f"  {pred_type.replace('_', ' ').title()}:")
                    print(f"    Accuracy: {stats['accuracy']:.3f}")
                    print(f"    Log Loss: {stats['log_loss']:.4f}")
                    print(f"    Sample Size: {stats['sample_size']:,}")
            
            # Create visualizations
            self.create_visualizations(gender)
        
        # Overall comparison
        print(f"\n🔍 OVERALL ASSESSMENT")
        print("-" * 50)
        
        men_metrics = self.calculate_metrics('men')
        women_metrics = self.calculate_metrics('women')
        
        if men_metrics and women_metrics:
            print(f"Men's Accuracy: {men_metrics['accuracy']:.3f}")
            print(f"Women's Accuracy: {women_metrics['accuracy']:.3f}")
            
            combined_accuracy = (
                men_metrics['accuracy'] * men_metrics['sample_size'] + 
                women_metrics['accuracy'] * women_metrics['sample_size']
            ) / (men_metrics['sample_size'] + women_metrics['sample_size'])
            
            print(f"Combined Accuracy: {combined_accuracy:.3f}")
            
            # Model performance assessment
            if combined_accuracy > 0.55:
                assessment = "🟢 EXCELLENT - Model shows strong predictive power"
            elif combined_accuracy > 0.53:
                assessment = "🟡 GOOD - Model shows meaningful edge over random"
            elif combined_accuracy > 0.51:
                assessment = "🟠 FAIR - Model shows slight edge, needs improvement"
            else:
                assessment = "🔴 POOR - Model needs significant improvement"
            
            print(f"\nPerformance Assessment: {assessment}")
        
        print(f"\n💾 Visualizations saved as:")
        print(f"  - men_backtest_results.png")
        print(f"  - women_backtest_results.png")

def main():
    print("🎾 Tennis Model Backtest")
    print("=" * 40)
    
    # Initialize backtest
    data_path = os.path.join(os.path.dirname(__file__), "Data")
    backtest = TennisModelBacktest(data_path)
    
    # Run backtest
    backtest.run_backtest('both')
    
    # Generate report
    backtest.generate_report()
    
    print("\n✅ Backtest complete!")

if __name__ == "__main__":
    main()